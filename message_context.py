"""群聊内容记录与显示；不持有平台事件，也不获取媒体文件。"""

import json
import re
from dataclasses import dataclass, field, replace


QUOTE_PREVIEW_LIMIT = 300


@dataclass(frozen=True)
class ContextRow:
    record_id: str
    message_id: str
    role: str
    header: str
    body: str
    relations: str = ""


@dataclass(frozen=True)
class ContextSnapshot:
    rows: tuple[ContextRow, ...]
    progress: int = 0
    image_note: str = ""

    def target(self, message_id: str) -> ContextRow | None:
        return next((
            row for row in self.rows
            if row.role == "user" and (row.message_id or row.record_id) == message_id
        ), None)

    def render(
        self, *, body_overrides: dict[str, str] | None = None,
        exclude_messages: set[tuple[str, str]] | None = None,
    ) -> str:
        bodies = body_overrides or {}
        excluded = exclude_messages or set()
        lines = [
            f"{row.header}: {bodies.get(row.record_id, row.body)}" for row in self.rows
            if row.record_id in bodies or not row.message_id or (row.role, row.message_id) not in excluded
        ]
        if not lines:
            return ""
        return (
            "群聊记录按顺序逐条列出；引用摘录是旧消息，不是当前发言。"
            "请结合提及和引用区分对话对象，面向其他成员的话不等于向你提问；连续消息可能属于不同话题。\n"
            + (self.image_note + "\n" if self.image_note else "")
            + "\n".join(lines)
        )


@dataclass
class ImageReference:
    image_id: str
    source: str = field(default="", repr=False)
    size_bytes: int | None = None
    description: str = ""
    unavailable_reason: str = ""
    generation: int = 0
    local_source: str = field(default="", repr=False)

    @property
    def status(self) -> str:
        if self.unavailable_reason:
            return f"当前不可用（{self.unavailable_reason}）"
        return "已转述" if self.description else "内容未读取"

    def render(self) -> str:
        fields = ["图片"]
        if self.size_bytes is not None:
            if self.size_bytes < 1024:
                size = f"{self.size_bytes}B"
            else:
                unit, divisor = ("KiB", 1024) if self.size_bytes < 1024**2 else ("MiB", 1024**2)
                size = f"{self.size_bytes / divisor:.2f}".rstrip("0").rstrip(".") + unit
            fields.append(size)
        fields.append(self.status)
        if self.description:
            fields.append(self.description)
        return "[" + "，".join(fields) + "]"


@dataclass
class MessagePart:
    kind: str
    text: str = ""
    image: ImageReference | None = None
    reference_id: str = ""
    sender_id: str = ""
    sender_name: str = ""
    children: tuple["MessagePart", ...] = ()
    quote_prefix: str = ""

    def render(self) -> str:
        if self.quote_prefix:
            preview = render_message(
                tuple(replace(p, text=re.sub(r"\s+", " ", p.text).strip()) for p in self.children),
                QUOTE_PREVIEW_LIMIT,
            )
            excerpt = f"摘录={json.dumps(preview, ensure_ascii=False)}" if preview else "原文未知"
            return self.quote_prefix + excerpt + "]"
        return self.image.render() if self.image is not None else self.text


def iter_image_parts(parts: tuple[MessagePart, ...]):
    for part in parts:
        if part.image is not None:
            yield part
        yield from iter_image_parts(part.children)


def iter_images(parts: tuple[MessagePart, ...]):
    for part in iter_image_parts(parts):
        yield part.image


def render_message(parts: tuple[MessagePart, ...], limit: int | None = None) -> str:
    """首尾保留正文，图片/提及/引用标记保持完整；截断说明单独作为元信息。"""
    pieces = []
    atoms = []
    offset = 0
    for part in parts:
        value = part.render().strip()
        if not value:
            continue
        if pieces:
            offset += 1
        start = offset
        pieces.append(value)
        offset += len(value)
        if part.kind != "text":
            atoms.append((start, offset, part))
    text = " ".join(pieces)
    if limit is None or len(text) <= limit:
        return text
    limit = max(1, limit)
    head = (limit + 1) // 2
    tail = len(text) - limit // 2
    for start, end, _ in atoms:
        if start < head < end:
            head = start
        if start < tail < end:
            tail = end
    omitted_images = sum(
        1 for start, end, part in atoms
        if start >= head and end <= tail and part.kind == "image"
    )
    image_notice = f"；含 {omitted_images} 张图片" if omitted_images else ""
    notice = f"[中间省略；原文 {len(text)} 字符{image_notice}]"
    return text[:head].rstrip() + notice + text[tail:].lstrip()
