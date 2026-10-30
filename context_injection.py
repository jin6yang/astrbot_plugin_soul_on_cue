"""准备本次请求的群聊背景，仅按明确身份排除已提供的消息。"""

from .message_context import ContextRow, ContextSnapshot


def request_background(
    snapshot: ContextSnapshot, event, req, *,
    target: ContextRow | None = None, include_history: bool = True,
) -> str:
    if not include_history:
        if target is None:
            return ""
        snapshot = ContextSnapshot((target,), snapshot.progress, snapshot.image_note)
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


def set_temporary_background(req, text: str, text_part_type) -> None:
    """只替换本插件在同一请求中拥有的内容块，不动其他插件的内容。"""
    owned = getattr(req, "_oncue_context_part", None)
    parts = [part for part in (getattr(req, "extra_user_content_parts", None) or []) if part is not owned]
    new_part = text_part_type(text=text).mark_as_temp() if text else None
    if new_part is not None:
        parts.append(new_part)
    req.extra_user_content_parts = parts
    req._oncue_context_part = new_part


def add_stage_direction(req, line: str) -> None:
    if not getattr(req, "_oncue_direction_added", False):
        req.system_prompt = (req.system_prompt or "") + "\n" + line
        req._oncue_direction_added = True
