"""会话图片文件的配额、租约与生命周期；不获取或改写入站事件。"""

import asyncio
import logging
import math
import time
import weakref
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable
from uuid import uuid4

from .image_resources import ImageFetchError, ImageTooLarge, fetch_image
from .message_context import ImageReference


MIB = 1024 * 1024


@dataclass(frozen=True)
class CacheLimits:
    max_image_bytes: int = 30 * MIB
    max_count: int = 20
    max_total_bytes: int = 75 * MIB
    ttl_seconds: float = 7200

    def __post_init__(self):
        if min(self.max_image_bytes, self.max_count, self.max_total_bytes) <= 0:
            raise ValueError("Image cache size and count limits must be positive")
        if not math.isfinite(self.ttl_seconds) or self.ttl_seconds <= 0:
            raise ValueError("Image cache lifetime must be positive and finite")


@dataclass(frozen=True)
class CachedImage:
    path: Path
    size_bytes: int
    generation: int


@dataclass
class _File:
    chat_id: str
    ref: ImageReference
    path: Path
    size_bytes: int = 0
    generation: int = 0
    cached_at: float | None = None
    expires_at: float | None = None
    order: int = 0
    leases: int = 0
    available: bool = False
    pending: bool = True


@dataclass
class _Job:
    task: asyncio.Task
    waiters: int = 0


class _CapacityUnavailable(Exception):
    pass


class SessionImageCache:
    """文件账本包含在途 .part 与租约延迟删除的文件，不借缓存淘汰超配额。

    仅在同一事件循环中使用。状态更新均为同步片段；各会话下载锁与插件
    的聊天锁无关，下载期间清理和转述回写仍可推进。
    """

    def __init__(
        self, root: Path, *, limits: CacheLimits | Callable[[], CacheLimits] = CacheLimits(),
        fetch=fetch_image, clock=time.monotonic, logger=None,
    ):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._limits_source = limits
        self._fetch = fetch
        self._clock = clock
        self._logger = logger or logging.getLogger(__name__)
        self._refs = weakref.WeakValueDictionary()
        self._owners: dict[int, tuple[str, weakref.ReferenceType]] = {}
        self._files: dict[Path, _File] = {}
        self._current: dict[tuple[str, str], _File] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._jobs: dict[tuple[str, str], _Job] = {}
        self._completed: dict[tuple[str, str], int] = {}
        self._failures: dict[tuple[str, str], str] = {}
        self._unlink_warned: set[Path] = set()
        self._order = 0
        self._closed = False
        # 此目录专属于本服务；只处理自己命名的平铺文件，不递归删除目录。
        for path in self.root.glob("oncue_*"):
            if path.suffix in {".cache", ".part"} and (path.is_file() or path.is_symlink()):
                try:
                    path.unlink()
                except OSError:
                    raise RuntimeError("Unable to clean abandoned image cache files") from None

    def _limits(self) -> CacheLimits:
        return self._limits_source() if callable(self._limits_source) else self._limits_source

    def register(self, chat_id: str, ref: ImageReference) -> ImageReference:
        """仅同会话、同图片 ID 复用；不把相同 URL 猜作相同图片。"""
        if self._closed:
            return ref
        key = (chat_id, ref.image_id)
        canonical = self._refs.get(key)
        if canonical is None:
            owner = self._owners.get(id(ref))
            if owner is not None and owner[0] != chat_id:
                # 即使调用方误用同一个对象，两个会话也不能共享资源生命周期。
                ref = replace(ref, generation=0, unavailable_reason="")
            self._refs[key] = ref
            identity = id(ref)

            def release(reference):
                current = self._owners.get(identity)
                if current is not None and current[1] is reference:
                    self._owners.pop(identity, None)

            self._owners[identity] = (chat_id, weakref.ref(ref, release))
            return ref
        if canonical is not ref:
            if ref.source:
                canonical.source = ref.source
            if getattr(ref, "local_source", ""):
                canonical.local_source = ref.local_source
        return canonical

    @staticmethod
    def _key(file: _File) -> tuple[str, str]:
        return file.chat_id, file.ref.image_id

    def _delete(self, file: _File) -> bool:
        if file.leases:
            return False
        try:
            file.path.unlink(missing_ok=True)
        except OSError as exc:
            if file.path not in self._unlink_warned:
                self._logger.warning(
                    f"[OnCue][图片缓存] 清理文件失败: 会话={file.chat_id} | 图片={file.ref.image_id} | {type(exc).__name__}"
                )
                self._unlink_warned.add(file.path)
            return False
        self._files.pop(file.path, None)
        self._unlink_warned.discard(file.path)
        return True

    def _retire(self, file: _File) -> None:
        file.available = False
        key = self._key(file)
        if self._current.get(key) is file:
            self._current.pop(key, None)
            file.ref.unavailable_reason = "本地缓存已被清理"
        self._delete(file)

    def _usage(self, chat_id: str) -> tuple[int, int]:
        files = [file for file in self._files.values() if file.chat_id == chat_id]
        return len(files), sum(file.size_bytes for file in files)

    def _make_room(
        self, chat_id: str, limits: CacheLimits, *, count: int = 0, size: int = 0,
        protected: _File | None = None,
    ) -> bool:
        while True:
            used_count, used_size = self._usage(chat_id)
            if used_count + count <= limits.max_count and used_size + size <= limits.max_total_bytes:
                return True
            candidates = [
                file for file in self._files.values()
                if file.chat_id == chat_id and file.available and not file.leases and file is not protected
            ]
            if not candidates:
                return False
            self._retire(min(candidates, key=lambda file: file.order))

    def cleanup(self) -> None:
        now = self._clock()
        limits = self._limits()
        for file in list(self._files.values()):
            if file.pending:  # 写入中的文件由任务清理，仍计入配额。
                continue
            if file.available and (
                now >= file.expires_at or file.size_bytes > limits.max_image_bytes or not file.path.is_file()
            ):
                self._retire(file)
            elif not file.available:
                self._delete(file)
        for chat_id in {file.chat_id for file in self._files.values()}:
            # 配置缩小时，旧租约不能让超配额文件一直接受新读者。
            # 退休文件不再可读，物理文件仍留在账本中，直到已有读者完成。
            retained = [
                file for file in self._files.values()
                if file.chat_id == chat_id and (file.available or file.pending)
            ]
            used_count = len(retained)
            used_size = sum(file.size_bytes for file in retained)
            for file in sorted(
                (file for file in retained if file.available), key=lambda file: file.order,
            ):
                if used_count <= limits.max_count and used_size <= limits.max_total_bytes:
                    break
                self._retire(file)
                used_count -= 1
                used_size -= file.size_bytes
        for states in (self._completed, self._failures):
            for key in list(states):
                if key not in self._refs:
                    states.pop(key, None)
        live_chats = (
            {key[0] for key in self._refs}
            | {key[0] for key in self._jobs}
            | {file.chat_id for file in self._files.values()}
        )
        for chat_id, lock in list(self._locks.items()):
            if chat_id not in live_chats and not lock.locked():
                self._locks.pop(chat_id, None)

    def _cached(self, chat_id: str, ref: ImageReference) -> _File | None:
        file = self._current.get((chat_id, ref.image_id))
        if file is not None and file.available:
            return file
        return None

    @staticmethod
    def _view(file: _File) -> CachedImage:
        return CachedImage(file.path, file.size_bytes, file.generation)

    def _failure(self, chat_id: str, ref: ImageReference, reason: str, stage: str) -> None:
        # 重新获取失败不抹掉已经确认的本地清理事实；成功后才恢复可用。
        if reason != "获取失败" or ref.unavailable_reason != "本地缓存已被清理":
            ref.unavailable_reason = reason
        key = (chat_id, ref.image_id)
        signature = reason + ":" + stage
        if self._failures.get(key) != signature:
            self._logger.warning(
                f"[OnCue][图片缓存] 获取失败: 会话={chat_id} | 图片={ref.image_id} | 阶段={stage} | 原因={reason}"
            )
            self._failures[key] = signature

    async def _download(self, chat_id: str, ref: ImageReference, source: str) -> CachedImage | None:
        lock = self._locks.setdefault(chat_id, asyncio.Lock())
        async with lock:
            if self._closed:
                return None
            self.cleanup()
            cached = self._cached(chat_id, ref)
            if cached is not None:
                return self._view(cached)
            file = None
            output = None
            succeeded = False

            def write(chunk: bytes) -> None:
                nonlocal file, output
                if not chunk:
                    return
                if self._closed:
                    raise asyncio.CancelledError()
                limits = self._limits()
                previous = file.size_bytes if file is not None else 0
                if previous + len(chunk) > limits.max_image_bytes:
                    raise ImageTooLarge(None)
                if not self._make_room(
                    chat_id, limits, count=int(file is None), size=len(chunk), protected=file,
                ):
                    raise _CapacityUnavailable()
                if file is None:
                    path = self.root / ("oncue_" + uuid4().hex + ".part")
                    output = path.open("xb", buffering=0)
                    file = _File(chat_id, ref, path)
                    self._files[path] = file
                try:
                    # 先记足本次写入额度；磁盘故障时宁可高估，不漏算部分写入。
                    file.size_bytes += len(chunk)
                    written = output.write(chunk)
                    if written != len(chunk):
                        raise OSError("Incomplete cache write")
                except OSError:
                    # 失败删除也可能失败，账本仍须记住磁盘实际占用。
                    try:
                        file.size_bytes = file.path.stat().st_size
                    except OSError:
                        pass
                    raise

            try:
                await self._fetch(source, write, self._limits().max_image_bytes)
                if file is None or file.size_bytes == 0:
                    raise ImageFetchError()
                limits = self._limits()
                if file.size_bytes > limits.max_image_bytes:
                    raise ImageTooLarge(file.size_bytes)
                if not self._make_room(chat_id, limits, protected=file):
                    raise _CapacityUnavailable()
                output.close()
                output = None
                final = file.path.with_suffix(".cache")
                file.path.replace(final)
                self._files.pop(file.path)
                file.path = final
                self._files[final] = file
                now = self._clock()
                self._order += 1
                ref.generation += 1
                file.generation = ref.generation
                file.cached_at = now
                file.expires_at = now + self._limits().ttl_seconds
                file.order = self._order
                file.available = True
                file.pending = False
                ref.size_bytes = file.size_bytes
                ref.unavailable_reason = ""
                key = (chat_id, ref.image_id)
                self._current[key] = file
                self._completed.pop(key, None)
                self._failures.pop(key, None)
                succeeded = True
                return self._view(file)
            except ImageTooLarge as exc:
                ref.size_bytes = exc.size_bytes
                self._failure(chat_id, ref, "超过插件大小限制", "大小校验")
                return None
            except ImageFetchError as exc:
                reason = "云端缓存已被清理" if exc.reason == "cloudclean" else "获取失败"
                self._failure(chat_id, ref, reason, "资源读取")
                return None
            except _CapacityUnavailable:
                self._failure(chat_id, ref, "获取失败", "缓存容量不足或文件仍在使用")
                return None
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._failure(chat_id, ref, "获取失败", type(exc).__name__)
                return None
            finally:
                if output is not None:
                    try:
                        output.close()
                    except OSError:
                        pass
                if file is not None and not succeeded:
                    file.available = False
                    file.pending = False
                    self._delete(file)

    async def ensure(self, chat_id: str, ref: ImageReference, *, source: str | None = None) -> CachedImage | None:
        if self._closed:
            return None
        ref = self.register(chat_id, ref)
        self.cleanup()
        cached = self._cached(chat_id, ref)
        if cached is not None:
            return self._view(cached)
        key = (chat_id, ref.image_id)
        job = self._jobs.get(key)
        if job is None:
            task = asyncio.create_task(self._download(chat_id, ref, ref.source if source is None else source))
            job = _Job(task)
            self._jobs[key] = job
        job.waiters += 1
        try:
            return await asyncio.shield(job.task)
        finally:
            job.waiters -= 1
            if job.waiters == 0:
                if self._jobs.get(key) is job:
                    self._jobs.pop(key, None)
                if not job.task.done():
                    job.task.cancel()
                    await asyncio.gather(job.task, return_exceptions=True)

    @asynccontextmanager
    async def acquire(self, chat_id: str, ref: ImageReference, *, source: str | None = None):
        result = await self.ensure(chat_id, ref, source=source)
        file = self._current.get((chat_id, ref.image_id))
        self.cleanup()
        if (
            result is None or self._closed or file is None or not file.available
            or file.generation != result.generation
        ):
            yield None
            return
        file.leases += 1
        try:
            yield self._view(file)
        finally:
            file.leases -= 1
            if not file.available:
                self._delete(file)

    def description_completed(self, chat_id: str, ref: ImageReference, generation: int, text: str) -> bool:
        key = (chat_id, ref.image_id)
        canonical = self._refs.get(key)
        owner = self._owners.get(id(ref))
        if (
            self._closed or canonical is None or canonical.generation != generation
            or (owner is not None and owner[0] != chat_id)
            or self._completed.get(key) == generation or not text.strip()
        ):
            return False
        self.cleanup()
        canonical.description = text.strip()
        self._completed[key] = generation
        file = self._cached(chat_id, canonical)
        if file is not None and file.generation == generation:
            now = self._clock()
            file.expires_at = now + max(0.0, file.expires_at - now) * 0.3
        return True

    async def close(self) -> None:
        self._closed = True
        tasks = [job.task for job in self._jobs.values()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._jobs.clear()
        self._locks.clear()
        for file in list(self._files.values()):
            self._retire(file)
        self._completed.clear()
        self._failures.clear()
        self._refs.clear()
        self._owners.clear()
