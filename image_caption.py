"""单图转述的轮次预算、共享任务与全局并发；不触发聊天回复。"""

import asyncio
import logging
import weakref
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from .image_cache import SessionImageCache
from .message_context import ImageReference


def clip_description(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    if max_chars <= 0:
        return ""
    marker = "…（描述已截断）…"
    if max_chars <= len(marker):
        return "…"
    available = max_chars - len(marker)
    head, tail = (available + 1) // 2, available // 2
    return text[:head] + marker + (text[-tail:] if tail else "")


class CaptionRound:
    """一次初判、额外决策和正文共用的额度；调用方持有到该轮结束。"""

    def __init__(self, limit: int = 3, is_active: Callable[[], bool] = lambda: True):
        self.limit = max(0, int(limit))
        self.is_active = is_active
        self.used = 0
        self.closed = False
        self._seen: dict[tuple[str, str], asyncio.Future[str]] = {}
        self._jobs: set[_CaptionJob] = set()

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    def active(self) -> bool:
        if self.closed:
            return False
        try:
            return bool(self.is_active())
        except Exception:
            return False

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for job in list(self._jobs):
            job.service._release(job, self)
        self._jobs.clear()


@dataclass(eq=False)
class _CaptionJob:
    service: "ImageCaptionService"
    chat_id: str
    ref: ImageReference
    max_chars: int
    # 弱引用不让遗失收尾的原生生成轮次被队列永久保活。
    consumers: dict[weakref.ReferenceType, asyncio.Future[str]] = field(default_factory=dict)
    state: str = "queued"
    task: asyncio.Task | None = None

    @property
    def key(self) -> tuple[str, str]:
        return self.chat_id, self.ref.image_id


class ImageCaptionService:
    """同一事件循环内使用；每轮 Future 独立取消，底层生成任务仍共享。

    并发配置在请求、任务完成和 refresh() 时读取。运行中的生成不会因
    轮次退出或并发下调而被强停；只有关闭整个服务才取消运行任务。
    """

    def __init__(
        self, cache: SessionImageCache, generate, *,
        concurrency: int | Callable[[], int] = 2, logger=None,
    ):
        self.cache = cache
        self.generate = generate
        self._concurrency = concurrency
        self._logger = logger or logging.getLogger(__name__)
        self._jobs: dict[tuple[str, str], _CaptionJob] = {}
        self._queue: deque[_CaptionJob] = deque()
        self._running: set[asyncio.Task] = set()
        self._closed = False
        self._loop = None

    def _limit(self) -> int:
        try:
            value = self._concurrency() if callable(self._concurrency) else self._concurrency
            return max(0, int(value))
        except (TypeError, ValueError, OverflowError):
            return 0

    def _warning(self, job: _CaptionJob, stage: str, reason: str) -> None:
        self._logger.warning(
            f"[OnCue][图片转述] 会话={job.chat_id} | 图片={job.ref.image_id} | 阶段={stage} | {reason}"
        )

    def request(
        self, chat_id: str, ref: ImageReference, budget: CaptionRound, max_chars: int,
    ) -> asyncio.Future[str] | None:
        """只新建任务时消耗额度；同轮失败不重试，既有描述直接复用。"""
        if self._closed or not budget.active() or max_chars <= 0:
            return None
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise RuntimeError("Image caption service must stay on one event loop")
        ref = self.cache.register(chat_id, ref)
        key = (chat_id, ref.image_id)
        if key in budget._seen:
            return budget._seen[key]
        if ref.description:
            result = loop.create_future()
            result.set_result(ref.description)
            budget._seen[key] = result
            return result
        job = self._jobs.get(key)
        if job is None:
            if budget.remaining == 0:
                return None
            job = _CaptionJob(self, chat_id, ref, max(1, int(max_chars)))
            self._jobs[key] = job
            self._queue.append(job)
            budget.used += 1

        result = loop.create_future()
        budget._seen[key] = result
        budget._jobs.add(job)

        def collected(reference):
            if not loop.is_closed():
                try:
                    loop.call_soon_threadsafe(self._collected, job, reference)
                except RuntimeError:
                    pass  # 事件循环可能刚在另一线程关闭。

        reference = weakref.ref(budget, collected)
        job.consumers[reference] = result

        def cancelled(future):
            consumer = reference()
            if future.cancelled() and consumer is not None:
                self._release(job, consumer)

        result.add_done_callback(cancelled)
        self.refresh()
        return result

    def _collected(self, job: _CaptionJob, reference: weakref.ReferenceType) -> None:
        result = job.consumers.pop(reference, None)
        if result is not None and not result.done():
            result.cancel()
        if job.state == "queued" and not self._prune(job):
            self._finish(job, cancelled=True)
        self.refresh()

    def _prune(self, job: _CaptionJob) -> bool:
        for reference, result in list(job.consumers.items()):
            consumer = reference()
            if consumer is None or not consumer.active() or result.cancelled():
                job.consumers.pop(reference, None)
                if consumer is not None:
                    consumer._jobs.discard(job)
                if not result.done():
                    result.cancel()
        return bool(job.consumers)

    def _release(self, job: _CaptionJob, budget: CaptionRound) -> None:
        budget._jobs.discard(job)
        for reference, result in list(job.consumers.items()):
            if reference() is budget:
                job.consumers.pop(reference, None)
                if not result.done():
                    result.cancel()
        if job.state == "queued" and not self._prune(job):
            self._finish(job, cancelled=True)
        self.refresh()

    def refresh(self) -> None:
        """按当前需求和并发配置推进 FIFO 队列，不增加模型重试。"""
        if self._closed:
            return
        while self._queue:
            job = self._queue[0]
            if job.state != "queued":
                self._queue.popleft()
                continue
            if not self._prune(job):
                self._queue.popleft()
                self._finish(job, cancelled=True)
                continue
            if len(self._running) >= self._limit():
                break
            self._queue.popleft()
            job.state = "running"
            task = asyncio.create_task(self._run(job))
            job.task = task
            self._running.add(task)
            task.add_done_callback(lambda completed, job=job: self._done(job, completed))

    def _finish(self, job: _CaptionJob, text: str = "", *, cancelled: bool = False) -> None:
        if job.state == "finished":
            return
        job.state = "finished"
        if self._jobs.get(job.key) is job:
            self._jobs.pop(job.key, None)
        for reference, result in list(job.consumers.items()):
            consumer = reference()
            if consumer is not None:
                consumer._jobs.discard(job)
            if not result.done():
                result.cancel() if cancelled else result.set_result(text)
        job.consumers.clear()

    def _done(self, job: _CaptionJob, task: asyncio.Task) -> None:
        self._running.discard(task)
        error = None if task.cancelled() else task.exception()
        if error is not None:
            self._warning(job, "任务收尾", type(error).__name__)
            self._finish(job)
        self.refresh()

    async def _run(self, job: _CaptionJob) -> None:
        result = ""
        cancelled = False
        stage = "资源读取"
        try:
            if not self._prune(job):
                cancelled = True
                return
            if job.ref.description:
                result = job.ref.description
                return
            async with self.cache.acquire(job.chat_id, job.ref) as cached:
                if cached is None:
                    self._warning(job, stage, "图片当前不可读取，未发起转述")
                    return
                if self._closed or not self._prune(job):
                    cancelled = True
                    return
                stage = "模型转述"
                text = await self.generate(job.chat_id, job.ref, cached, job.max_chars)
                if self._closed:
                    cancelled = True
                    return
                if not isinstance(text, str):
                    raise TypeError("Caption must be text")
                text = clip_description(text, job.max_chars)
                if not text:
                    self._warning(job, stage, "模型未返回描述")
                    return
                if not self._closed and self.cache.description_completed(
                    job.chat_id, job.ref, cached.generation, text,
                ):
                    result = text
        except asyncio.CancelledError:
            cancelled = True
            raise
        except Exception as exc:
            self._warning(job, stage, type(exc).__name__)
        finally:
            self._finish(job, result, cancelled=cancelled)

    async def close(self) -> None:
        self._closed = True
        for job in list(self._queue):
            self._finish(job, cancelled=True)
        self._queue.clear()
        tasks = list(self._running)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for job in list(self._jobs.values()):
            self._finish(job, cancelled=True)
        self._jobs.clear()
