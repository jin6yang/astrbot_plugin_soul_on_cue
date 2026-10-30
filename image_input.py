"""按请求准备图片占位、转述和临时视觉输入，不改写历史记录。"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import logging
from dataclasses import dataclass, replace

from .image_cache import CachedImage
from .image_caption import CaptionRound, clip_description
from .message_context import ContextRow, ContextSnapshot, ImageReference, MessagePart, iter_images, render_message


@dataclass(frozen=True)
class EncodedImage:
    data_uri: str
    is_montage: bool = False


@dataclass(frozen=True)
class PreparedContext:
    snapshot: ContextSnapshot
    image_urls: list[str]
    image_ids: tuple[str, ...]


async def encode_cached_image(
    cached: CachedImage, *, max_size: int = 1024, logger=None,
) -> EncodedImage | None:
    """在调用方租约内生成内存预览；取消时先等工作线程退出再归还租约。"""
    logger = logger or logging.getLogger(__name__)

    def encode() -> EncodedImage:
        from astrbot.core.utils.media_utils import _prepare_model_image_sync

        data, montage = _prepare_model_image_sync(cached.path.read_bytes(), max_size)
        if not data or len(data) >= 512 * 1024:
            raise ValueError("Image preview exceeds its memory input limit")
        mime = "image/png" if data.startswith(b"\x89PNG\r\n\x1a\n") else "image/jpeg"
        return EncodedImage(f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}", montage)

    # Unlike a cancellable to_thread Task, this protected executor Future keeps
    # representing the actual worker lifetime while its caller is cancelled.
    worker = asyncio.get_running_loop().run_in_executor(None, encode)
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not worker.cancelled():
            worker.exception()
        raise
    except Exception as exc:
        logger.warning(f"[OnCue][图片输入] 内存预处理失败: {type(exc).__name__}")
        return None


def _walk(parts: tuple[MessagePart, ...]):
    for part in parts:
        yield part
        yield from _walk(part.children)


def _image_text(ref: ImageReference, description: str, number: int | None = None) -> str:
    fields = ["图片"]
    if ref.size_bytes is not None:
        if ref.size_bytes < 1024:
            fields.append(f"{ref.size_bytes}B")
        else:
            unit, divisor = ("KiB", 1024) if ref.size_bytes < 1024**2 else ("MiB", 1024**2)
            fields.append(f"{ref.size_bytes / divisor:.2f}".rstrip("0").rstrip(".") + unit)
    fields.append(f"已附图（图片{number}）" if number is not None else ref.status)
    if description:
        fields.append(description)
    return "[" + "，".join(fields) + "]"


def _display_parts(parts: tuple[MessagePart, ...], text_for_image) -> tuple[MessagePart, ...]:
    """只制作临时显示副本；返回的正文不会替换原 parts 中的 canonical 引用。"""
    output = []
    for part in parts:
        children = _display_parts(part.children, text_for_image)
        if part.image is not None:
            output.append(replace(part, image=None, text=text_for_image(part.image), children=children))
        else:
            output.append(replace(part, children=children))
    return tuple(output)


def _description_allowance(row: ContextRow, limit: int, numbers: dict[str, int]) -> int:
    images = tuple(iter_images(row.parts))
    baseline = render_message(_display_parts(
        row.parts, lambda ref: _image_text(ref, "", numbers.get(ref.image_id)),
    ))
    # Each nonempty description introduces one extra separator in its marker.
    return max(0, limit - len(baseline) - len(row.relations) - len(images)) // max(1, len(images))


def _image_body(parts: tuple[MessagePart, ...], limit: int) -> str:
    if limit <= 0:
        return ""
    atoms = tuple(part for part in _walk(parts) if part.kind == "image")
    text = render_message(atoms, limit)
    if len(text) <= limit:
        return text
    # The shared renderer's omission notice may itself exceed a tiny remaining
    # budget. Keep only whole markers rather than cutting an image label apart.
    kept = []
    used = 0
    for atom in atoms:
        value = atom.render().strip()
        size = len(value) + bool(kept)
        if value and used + size <= limit:
            kept.append(value)
            used += size
    return " ".join(kept)


class ImageInputPreparer:
    def __init__(self, cache, captions, *, encode=None, logger=None):
        self.cache = cache
        self.captions = captions
        self.logger = logger or logging.getLogger(__name__)
        self.encode = encode

    async def prepare(
        self, chat_id: str, snapshot: ContextSnapshot, *, mode: str, vision: bool | None,
        round: CaptionRound, focus_records: tuple[str, ...] = (), max_images: int = 5,
        message_char_limit: int = 500, image_offset: int = 0,
        skip_direct_ids: set[str] | None = None, max_size: int = 1024,
    ) -> PreparedContext:
        """在同一轮共享转述额度，逐阶段重建图片表示；不按回复速度增加期限。"""
        if mode not in {"placeholder", "describe", "mixed"}:
            raise ValueError("Unknown image input mode")
        limit = max(1, int(message_char_limit))
        max_images = max(0, int(max_images))
        image_offset = max(0, int(image_offset))
        skip_direct_ids = skip_direct_ids or set()
        rows_by_id = {row.record_id: row for row in snapshot.rows}
        referenced_rows = {row.message_id: row for row in snapshot.rows if row.message_id}
        referenced_rows.update(rows_by_id)
        focus = []
        visited = set()

        def add_focus(row: ContextRow) -> None:
            if row.record_id in visited:
                return
            visited.add(row.record_id)
            focus.append(row)
            for part in _walk(row.parts):
                if (part.kind == "quote" or part.quote_prefix) and part.reference_id:
                    referenced = referenced_rows.get(part.reference_id)
                    if referenced is not None:
                        add_focus(referenced)

        for record_id in focus_records:
            if record_id in rows_by_id:
                add_focus(rows_by_id[record_id])
        priority = focus + [row for row in reversed(snapshot.rows) if row.record_id not in visited]

        # Reserve space for actual text, relation markers, and image state labels
        # before dividing the remaining message budget between its images.
        budgets = {row.record_id: _description_allowance(row, limit, {}) for row in snapshot.rows}

        pending = {}
        caption_seen = set()
        if mode != "placeholder" and any(any(iter_images(row.parts)) for row in priority) and round.active():
            for row in priority:
                allowance = budgets[row.record_id]
                if allowance <= 0:
                    continue
                for ref in iter_images(row.parts):
                    if ref.image_id in caption_seen or ref.description:
                        continue
                    caption_seen.add(ref.image_id)
                    future = self.captions.request(chat_id, ref, round, allowance)
                    if future is not None:
                        pending[ref.image_id] = future

        urls = []
        image_ids = []
        numbers = {}
        associations = []
        montages = []
        direct_seen = set()
        if mode == "mixed" and vision is True:
            for row in focus:
                for position, ref in enumerate(iter_images(row.parts), 1):
                    if ref.image_id in direct_seen or ref.image_id in skip_direct_ids:
                        continue
                    direct_seen.add(ref.image_id)
                    if len(urls) >= max_images:
                        break
                    try:
                        async with self.cache.acquire(chat_id, ref) as cached:
                            if cached is None:
                                self.logger.warning(f"[OnCue][图片输入] 图片当前不可读取: 图片={ref.image_id}")
                                continue
                            encoded = (
                                self.encode(cached) if self.encode is not None
                                else encode_cached_image(cached, max_size=max_size, logger=self.logger)
                            )
                            if inspect.isawaitable(encoded):
                                encoded = await encoded
                            if encoded is None:
                                self.logger.warning(f"[OnCue][图片输入] 本次预览不可用，保留占位: 图片={ref.image_id}")
                                continue
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        self.logger.warning(
                            f"[OnCue][图片输入] 预览读取失败，保留占位: 图片={ref.image_id} | {type(exc).__name__}"
                        )
                        continue
                    number = image_offset + len(urls) + 1
                    numbers[ref.image_id] = number
                    urls.append(encoded.data_uri)
                    image_ids.append(ref.image_id)
                    associations.append(
                        f"图片{number}：{row.header} 中的第{position}张图片"
                        f"（图片ID={json.dumps(ref.image_id, ensure_ascii=False)}）"
                    )
                    if encoded.is_montage:
                        montages.append(number)

        descriptions = {}
        if mode == "describe" or (mode == "mixed" and vision is not True):
            keys = tuple(pending)
            results = await asyncio.gather(
                *(asyncio.shield(pending[key]) for key in keys), return_exceptions=True,
            )
            for key, result in zip(keys, results):
                if isinstance(result, str) and result.strip():
                    descriptions[key] = result.strip()
                elif isinstance(result, Exception):
                    self.logger.warning(f"[OnCue][图片输入] 本次转述不可用: 图片={key} | {type(result).__name__}")

        output_rows = []
        for row in snapshot.rows:
            allowance = _description_allowance(row, limit, numbers)

            def image_text(ref: ImageReference) -> str:
                description = "" if mode == "placeholder" else clip_description(descriptions.get(ref.image_id, ref.description), allowance)
                return _image_text(ref, description, numbers.get(ref.image_id))

            if row.parts:
                shown = _display_parts(row.parts, image_text)
                body = render_message(shown, limit)
                image_body = _image_body(shown, max(0, limit - len(row.relations)))
                output_rows.append(replace(row, body=body, image_body=image_body))
            else:
                output_rows.append(replace(row, image_body=""))
        manifest = ""
        if associations:
            manifest = "本次附图关联（编号对应本次模型输入）：\n" + "\n".join(associations)
            if montages:
                labels = "、".join(f"图片{number}" for number in montages)
                manifest += f"\n{labels}是动画部分帧按时间顺序拼合的预览，只能依据这些可见帧理解，不代表完整动画。"
        prepared = replace(snapshot, rows=tuple(output_rows), image_manifest=manifest)
        return PreparedContext(prepared, urls, tuple(image_ids))
