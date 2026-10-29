"""群聊内容记录与显示；不持有平台事件，也不获取媒体文件。"""

from dataclasses import dataclass, field


@dataclass
class ImageReference:
    image_id: str
    source: str = field(default="", repr=False)
    size_bytes: int | None = None
    status: str = "内容未读取"
    description: str = ""

    def render(self) -> str:
        fields = ["图片"]
        if self.size_bytes is not None:
            fields.append(f"{self.size_bytes / (1024 * 1024):.2f}".rstrip("0").rstrip(".") + "MB")
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

    def render(self) -> str:
        return self.image.render() if self.image is not None else self.text


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
