"""准备本次请求的群聊背景，仅按明确身份排除已提供的消息。"""

from dataclasses import dataclass

from .message_context import ContextRow, ContextSnapshot


def request_background(
    snapshot: ContextSnapshot, event, req, *,
    target: ContextRow | None = None, include_history: bool = True,
) -> str:
    if not include_history:
        if target is None:
            return ""
        snapshot = ContextSnapshot((target,), snapshot.progress, snapshot.image_note, snapshot.image_manifest)
    bodies = {}
    if (req.prompt and req.prompt.strip()) or getattr(req, "image_urls", None) or getattr(req, "audio_urls", None):
        current_id = event.get_extra("oncue_record_id")
        if current_id:
            current_row = next((row for row in snapshot.rows if row.record_id == current_id), None)
            if current_row is not None:
                # 原生引用不一定含消息 ID 和成员 ID，去掉正文后仍保留这些关系。
                # 在原位置替换正文，不能把合并等待中的首条消息移到最后。
                bodies[current_id] = (
                    (current_row.relations + " " if current_row.relations else "")
                    + (current_row.image_body + " " if current_row.image_body else "")
                    + "[正文及附件见本次原生输入]"
                )

    # 普通 AstrBot 历史没有平台消息 ID。只接受调用方显式提供的身份字段，
    # 不解析正文中的 ID、原生群记录字符串或检查点，不按同文猜测。
    excluded_messages = set()
    contexts = getattr(req, "contexts", None)
    if isinstance(contexts, list):
        for message in contexts:
            if not isinstance(message, dict) or message.get("role") not in {"user", "assistant"}:
                continue
            origin = message.get("unified_msg_origin")
            if origin is not None and origin != event.unified_msg_origin:
                continue
            message_id = message.get("message_id")
            if isinstance(message_id, (str, int)) and not isinstance(message_id, bool):
                message_id = str(message_id).strip()
                if message_id and message_id != "unknown":
                    excluded_messages.add((message["role"], message_id))

    # 钩子之后原生历史仍可能被压缩或淘汰，显式目标必须随本次输入保留。
    # 当前消息仍由 bodies 指向原生当前输入，不重复正文。
    if target is not None:
        excluded_messages.discard((target.role, target.message_id))
    context = snapshot.render(body_overrides=bodies, exclude_messages=excluded_messages)
    if target is not None:
        context += (
            f"\n[本次回应目标]\n{target.header}\n"
            "请围绕这条消息自然回应；它可能早于当前原生输入，其他消息用于理解后续语境。\n[/本次回应目标]"
        )
    if not context:
        return ""
    return (
        "[群聊补充上下文]\n以下是本次参考的群聊记录。"
        "请结合整组交流自然接话，不要将所有内容视为同一个人的请求。\n"
        + context + "\n[/群聊补充上下文]"
    )


@dataclass(frozen=True)
class _OwnedPart:
    part: object
    kind: type
    image_url: object
    content: dict

    @classmethod
    def capture(cls, part):
        return cls(part, type(part), getattr(part, "image_url", None), part.model_dump_for_context())

    def unchanged(self, parts) -> bool:
        if not any(part is self.part for part in parts):
            return False
        if type(self.part) is not self.kind or getattr(self.part, "image_url", None) is not self.image_url:
            return False
        try:
            return self.part.model_dump_for_context() == self.content
        except Exception:
            # 其它插件已接手甚至改换结构时，不再处理这个对象。
            return False


def set_temporary_background(req, text: str, text_part_type, *, images=(), image_part_type=None) -> None:
    """只替换仍未被改写的自有块；被其它插件接手后不再注入该类内容。"""
    original = getattr(req, "extra_user_content_parts", None) or []
    owned = getattr(req, "_oncue_context_owner", None)
    owned_images = getattr(req, "_oncue_image_owners", ())
    text_released = getattr(req, "_oncue_context_released", False)
    images_released = getattr(req, "_oncue_images_released", False)
    replace_text = owned is not None and owned.unchanged(original)
    if owned is not None and not replace_text:
        text_released = True
    replace_images = [entry for entry in owned_images if entry.unchanged(original)]
    if len(replace_images) != len(owned_images):
        images_released = True
    # 不按 URL、正文标签或拷贝的识别字段认领对象，也不原地修改外部列表。
    parts = [part for part in original if not (
        (replace_text and part is owned.part)
        or any(part is entry.part for entry in replace_images)
    )]
    new_part = text_part_type(text=text).mark_as_temp() if text and not text_released else None
    if new_part is not None:
        parts.append(new_part)
    new_images = [
        image_part_type(image_url={"url": url}).mark_as_temp() for url in images
    ] if image_part_type and not images_released else []
    parts.extend(new_images)
    req.extra_user_content_parts = parts
    req._oncue_context_part = new_part
    req._oncue_image_parts = new_images
    req._oncue_context_owner = _OwnedPart.capture(new_part) if new_part is not None else None
    req._oncue_image_owners = tuple(_OwnedPart.capture(part) for part in new_images)
    req._oncue_context_released = text_released
    req._oncue_images_released = images_released


def add_stage_direction(req, line: str) -> None:
    if not getattr(req, "_oncue_direction_added", False):
        req.system_prompt = (req.system_prompt or "") + "\n" + line
        req._oncue_direction_added = True
