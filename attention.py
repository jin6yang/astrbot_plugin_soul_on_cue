"""临时关注的生命周期；仅保存插件状态，不调用模型或改变回复状态。"""

import json
from dataclasses import dataclass


ATTENTION_MESSAGE_LIMIT = 50
ATTENTION_MAX_SECONDS = 3 * 60 * 60
ATTENTION_NOTE_LIMIT = 200


@dataclass(frozen=True)
class AttentionNotice:
    token: int
    note: str
    message_id: str
    reason: str


@dataclass(frozen=True)
class AttentionSnapshot:
    revision: int
    active: bool
    started_at: float
    start_progress: int
    note: str
    message_id: str
    notice: AttentionNotice | None

    def render(self, now: float, progress: int) -> str:
        sections = []
        if self.active:
            count = max(0, progress - self.start_progress)
            elapsed = max(0.0, now - self.started_at) / 60
            source = f"；关联消息ID={json.dumps(self.message_id, ensure_ascii=False)}" if self.message_id else ""
            sections.append(
                "[临时关注]\n"
                f"关注事项：{json.dumps(self.note, ensure_ascii=False)}{source}。\n"
                f"本次已收到 {count} 条新消息（不含 Bot），持续 {elapsed:.0f} 分钟。"
                "关注只缩短一瞥间隔，不强制回复；保持关注不会重置消息数或时长。\n"
                "[/临时关注]"
            )
        if self.notice is not None:
            notice = self.notice
            source = f"；关联消息ID={json.dumps(notice.message_id, ensure_ascii=False)}" if notice.message_id else ""
            sections.append(
                "[临时关注退出提醒]\n"
                f"此前关注：{json.dumps(notice.note, ensure_ascii=False)}{source}。\n"
                f"强制退出原因：{notice.reason}。退出时已恢复普通一瞥。"
                "这不代表话题已经结束，也不要求自动重新关注。\n"
                "[/临时关注退出提醒]"
            )
        return "\n".join(sections)


@dataclass
class AttentionState:
    active: bool = False
    started_at: float = 0.0
    start_progress: int = 0
    note: str = ""
    message_id: str = ""
    revision: int = 0
    notice: AttentionNotice | None = None

    def snapshot(self) -> AttentionSnapshot:
        return AttentionSnapshot(
            self.revision, self.active, self.started_at, self.start_progress,
            self.note, self.message_id, self.notice,
        )

    def _clear(self) -> None:
        self.active = False
        self.started_at = 0.0
        self.start_progress = 0
        self.note = ""
        self.message_id = ""

    def expire(self, now: float, progress: int) -> bool:
        if not self.active:
            return False
        reasons = []
        if progress - self.start_progress >= ATTENTION_MESSAGE_LIMIT:
            reasons.append("已收到 50 条新消息（不含 Bot）")
        if now - self.started_at >= ATTENTION_MAX_SECONDS:
            reasons.append("已满 3 小时")
        if not reasons:
            return False
        self.revision += 1
        self.notice = AttentionNotice(self.revision, self.note, self.message_id, "；".join(reasons))
        self._clear()
        return True

    def apply(
        self, action: str, note: str, message_id: str, now: float, progress: int,
        *, expected_revision: int,
    ) -> str:
        if expected_revision != self.revision:
            return "stale"
        note = note.strip()[:ATTENTION_NOTE_LIMIT]
        message_id = message_id.strip()
        if action == "stop":
            if not self.active:
                return "none"
            self._clear()
            self.revision += 1
            return "stopped"
        if action == "keep" or (action == "start" and self.active):
            if not self.active:
                return "none"
            next_note = note or self.note
            next_message_id = message_id or self.message_id
            if (next_note, next_message_id) != (self.note, self.message_id):
                self.note, self.message_id = next_note, next_message_id
                self.revision += 1
            return "kept"
        if action not in {"start", "restart"} or not note:
            return "none"
        self.active = True
        self.started_at = now
        self.start_progress = progress
        self.note = note
        self.message_id = message_id
        self.revision += 1
        return "restarted" if action == "restart" else "started"

    def acknowledge(self, token: int) -> None:
        if self.notice is not None and self.notice.token == token:
            self.notice = None
