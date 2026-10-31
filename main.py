import asyncio
import json
import random
import re
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, StarTools
from astrbot.core.agent.message import ImageURLPart, TextPart
from astrbot.core.message.components import At, Image, Plain, Reply
from astrbot.core.message.message_event_result import ResultContentType

from .attention import ATTENTION_NOTE_LIMIT, AttentionState
from .context_injection import add_stage_direction, request_background, set_temporary_background
from .image_cache import CacheLimits, SessionImageCache
from .image_caption import CaptionRound, ImageCaptionService
from .image_input import ImageInputPreparer
from .image_models import ImageModels
from .message_context import ContextRow, ContextSnapshot, ImageReference, MessagePart, iter_image_parts, iter_images, render_message, snapshot_parts


MESSAGE_CACHE_LIMIT = 300
GLANCE_CONCURRENCY = 2

ATTENTION_DECISION_RULE = (
    "临时关注与本次是否回复独立，只影响之后的一瞥间隔。"
    "如需关注，在同一个 JSON 中给出 attention_action（start 开启、keep 保持、stop 结束、restart 重启）"
    "和 attention_note（简短说明关注什么）。start/restart 必须有说明；"
    "keep 不重置消息计数或计时，缺省为 keep；不自行指定时间，不因关注而强制回复。"
)


DEFAULT_DECISION_PROMPT = """你是群聊角色扮演机器人的"内心"。请代入以下角色，判断看到群聊对话后是否想开口。

判断原则（按优先级）：
1. 有人明确@或点名你、或直接向你提问 → 优先考虑回应，仍可根据角色和语境保持沉默
2. 话题命中角色的兴趣或专业领域、有人需要帮助、气氛需要打圆场 → 可以开口
3. 群友之间的闲聊与角色无关、刚说过话没有新信息、深夜角色在睡觉 → 保持安静
4. 只有内容未读取的图片占位，且没有文字、点名、提问、引用你或紧跟你的发言 → 默认保持安静；已提供图片或描述时，结合实际内容判断
5. 有任何犹豫 → 保持安静
6. 若这次开口需要外部知识库资料才能答好，在同一个 JSON 里给出 need_kb=true 和独立检索词 kb_query；否则 need_kb=false、kb_query 留空
7. 需要回应某条消息时，target_message_id 填本次记录中的消息ID（缺失时用内部ID），可选择较早的消息；自然接话可留空，不填写记录之外的ID
8. 是否继续关注话题与本次是否回复独立；需要时说明关注事项，不因关注而强制开口

只输出 JSON，不要输出其它内容：
{"should_reply": true或false, "speaker": "回应的角色名(单角色填角色名)", "mood": "当前心情短语", "reason": "一句话理由", "need_kb": true或false, "kb_query": "需要时填独立检索词", "target_message_id": "回应目标的消息ID或内部ID，可留空", "attention_action": "start/keep/stop/restart", "attention_note": "关注事项，可留空"}

<角色设定>
{character_card}
</角色设定>

<群聊近况>
{chat_history}
</群聊近况>

当前时间：{current_time}

<触发背景>
{entry_context}
</触发背景>"""


@dataclass
class MessageWait:
    deadline: float
    last_activity_at: float
    latest_item: dict
    reply_version: int
    changed: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class ChatFacts:
    messages: deque = field(default_factory=lambda: deque(maxlen=MESSAGE_CACHE_LIMIT))
    reply_ts: deque = field(default_factory=deque)
    pending_replies: dict[str, float] = field(default_factory=dict)
    backoffs: dict = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    decision_inflight: object | None = None
    glance_inflight: bool = False
    glance_target: dict | None = None
    message_wait: MessageWait | None = None
    incoming_seq: int = 0  # 去重后的非空群友消息序号，不受缓存淘汰影响
    reply_version: int = 0  # 点名或新回复会使在途的主动决策失效
    last_reply_ts: float = 0.0
    last_activity_ts: float = 0.0
    last_observation_activity_ts: float = 0.0
    cooldown_until: float = 0.0
    attention: AttentionState = field(default_factory=AttentionState)


class OnCuePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.chats: dict[str, ChatFacts] = {}
        self.data_dir: Path | None = None
        self.glance_file: Path | None = None
        self.glance_next_due: dict[str, float] = {}
        self.glance_last_check: dict[str, int] = {}  # 最近有效一瞥决策所见的消息序号
        self.glance_task: asyncio.Task | None = None
        self._glance_jobs: dict[str, asyncio.Task] = {}
        self._decision_card_issue: str | None = None
        self.kb_missing_warned = False
        self.kb_not_taken_warned = False
        self._terminating = False
        self._context_conflict_signatures: dict[str, tuple[str, ...]] = {}
        self.image_cache: SessionImageCache | None = None
        self.image_cache_task: asyncio.Task | None = None
        self.image_models = ImageModels(context, self._cfg_value, logger)
        self.image_captions: ImageCaptionService | None = None
        self.image_input: ImageInputPreparer | None = None

    async def initialize(self) -> None:
        try:
            self.data_dir = StarTools.get_data_dir("astrbot_plugin_soul_on_cue")
            self.image_cache = SessionImageCache(
                self.data_dir / "images", limits=self._image_cache_limits, logger=logger,
            )
            self.image_captions = ImageCaptionService(
                self.image_cache, self.image_models.caption,
                concurrency=lambda: max(1, self._cfg_int("image_caption_concurrency", 2)), logger=logger,
            )
            self.image_input = ImageInputPreparer(self.image_cache, self.image_captions, logger=logger)
            self.image_cache_task = asyncio.create_task(self._image_cleanup_loop())
            self.glance_file = self.data_dir / "glance.json"
            self._load_glance()
            if self._cfg_bool("enable", True):
                await self._decision_card()
            self.glance_task = asyncio.create_task(self._glance_loop())
        except Exception as e:
            logger.error(f"[OnCue] 初始化失败: {e}")

    async def terminate(self) -> None:
        self._terminating = True
        for chat in list(self.chats.values()):
            async with chat.lock:
                self._cancel_message_wait(chat)
        if self.image_cache_task:
            self.image_cache_task.cancel()
            try:
                await self.image_cache_task
            except asyncio.CancelledError:
                pass
            self.image_cache_task = None
        if self.glance_task:
            self.glance_task.cancel()
            try:
                await self.glance_task
            except asyncio.CancelledError:
                pass
            self.glance_task = None
        glance_jobs = list(self._glance_jobs.values())
        for task in glance_jobs:
            task.cancel()
        if glance_jobs:
            await asyncio.gather(*glance_jobs, return_exceptions=True)
        self._glance_jobs.clear()
        if self.image_cache:
            if self.image_captions:
                await self.image_captions.close()
                self.image_captions = None
            self.image_input = None
            await self.image_cache.close()
            self.image_cache = None

    def _image_mode(self) -> str:
        return {
            "不转述（仅占位）": "placeholder", "仅转述": "describe", "混合（视觉优先）": "mixed",
        }.get(self._cfg_str("image_mode", "混合（视觉优先）"), "mixed")

    def _new_image_round(self, chat: ChatFacts) -> CaptionRound:
        version = chat.reply_version
        return CaptionRound(
            max(0, self._cfg_int("image_caption_budget", 3)),
            is_active=lambda: (
                not self._terminating and self._cfg_bool("enable", True)
                and chat.reply_version == version and self._image_mode() != "placeholder"
                and self._cfg_int("image_caption_budget", 3) > 0
                and self.image_models.caption_available()
            ),
        )

    @staticmethod
    def _close_image_round(event: AstrMessageEvent) -> None:
        budget = event.get_extra("oncue_image_round")
        if isinstance(budget, CaptionRound):
            budget.close()
            event.set_extra("oncue_image_round", None)

    async def _decision_images(self, event, chat, snapshot, budget):
        if self.image_input is None or not any(any(iter_images(row.parts)) for row in snapshot.rows):
            return snapshot, None, []
        provider_id, vision = await self.image_models.decision_provider(event.unified_msg_origin)
        if not provider_id:
            return snapshot, None, []
        focus = tuple(dict.fromkeys(record for record in (
            event.get_extra("oncue_record_id"),
            next((row.record_id for row in reversed(snapshot.rows) if row.role == "user"), None),
        ) if record))
        prepared = await self._prepare_images(
            event.unified_msg_origin, snapshot, vision=vision,
            round=budget, focus_records=focus,
            max_images=max(0, self._cfg_int("image_request_count", 5)),
            message_char_limit=max(1, self._cfg_int("message_char_limit", 500)),
            max_size=self.image_models.image_max_size(event.unified_msg_origin),
        )
        return prepared.snapshot, provider_id, prepared.image_urls

    async def _prepare_images(self, chat_id, snapshot, **options):
        mode = self._image_mode()
        prepared = await self.image_input.prepare(chat_id, snapshot, mode=mode, **options)
        latest = self._image_mode()
        if latest != mode and latest in {"placeholder", "describe"}:
            # 等待期间收紧模式时重建本次表示，已发生的转述结果仍可留作缓存。
            prepared = await self.image_input.prepare(chat_id, prepared.snapshot, mode=latest, **options)
        return prepared

    def _image_cache_limits(self) -> CacheLimits:
        return CacheLimits(
            max_image_bytes=max(1, self._cfg_int("image_max_mb", 30)) * 1024**2,
            max_count=max(1, self._cfg_int("image_cache_count", 20)),
            max_total_bytes=max(1, self._cfg_int("image_cache_mb", 75)) * 1024**2,
            ttl_seconds=max(1, self._cfg_int("image_cache_minutes", 120)) * 60,
        )

    async def _image_cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(15)
            if self.image_cache:
                self.image_cache.cleanup()

    async def _retain_local_images(self, chat_id: str, parts: tuple[MessagePart, ...]) -> None:
        cache = self.image_cache
        if cache is None:
            return
        seen = set()
        for image in iter_images(parts):
            if image.local_source and image.image_id not in seen:
                seen.add(image.image_id)
                await cache.ensure(chat_id, image, source=image.local_source)

    def _chat(self, chat_id: str) -> ChatFacts:
        if chat_id not in self.chats:
            self.chats[chat_id] = ChatFacts(messages=deque(maxlen=self._message_cache_count()))
        return self.chats[chat_id]

    def _message_cache_count(self) -> int:
        return max(2, self._cfg_int("message_cache_count", MESSAGE_CACHE_LIMIT))

    def _resize_message_cache(self, chat: ChatFacts) -> None:
        capacity = self._message_cache_count()
        if chat.messages.maxlen != capacity:
            chat.messages = deque(chat.messages, maxlen=capacity)

    def _append_message(self, chat: ChatFacts, item: dict) -> bool:
        """持有会话锁时入账；重复投递不更新任何触发状态。"""
        self._resize_message_cache(chat)
        if any(m.get("record_id") == item["record_id"] for m in chat.messages):
            return False
        if item.get("role") == "user" and item.get("active"):
            chat.incoming_seq += 1
            item["seq"] = chat.incoming_seq
        chat.messages.append(item)
        return True

    def _cfg_value(self, key: str, default):
        try:
            value = self.config.get(key, None)
        except Exception:
            value = None
        if value is not None:
            return value
        for section in ("config_wake", "config_decision", "config_trigger", "config_character", "config_context"):
            try:
                obj = self.config.get(section, {})
            except Exception:
                obj = {}
            if isinstance(obj, dict) and key in obj:
                return obj.get(key, default)
            try:
                dotted = self.config.get(f"{section}.{key}", None)
            except Exception:
                dotted = None
            if dotted is not None:
                return dotted
        return default

    def _cfg_str(self, key: str, default: str = "") -> str:
        value = self._cfg_value(key, default)
        return default if value is None else str(value)

    def _cfg_int(self, key: str, default: int) -> int:
        try:
            return int(self._cfg_value(key, default))
        except Exception:
            return default

    def _cfg_bool(self, key: str, default: bool) -> bool:
        value = self._cfg_value(key, default)
        return default if value is None else bool(value)

    def _check_context_conflict(self, event: AstrMessageEvent) -> None:
        umo = event.unified_msg_origin
        get_config = getattr(self.context, "get_config", None)
        if not callable(get_config):
            return
        try:
            settings = get_config(umo=umo).get("provider_ltm_settings", {})
            if not isinstance(settings, dict):
                return
            causes = []
            details = []
            if self._cfg_bool("enable_context_injection", True) and settings.get("group_icl_enable", False):
                causes.append("群聊消息记录注入上下文")
                details.append("本插件的上下文整理与 AstrBot「群聊消息记录注入上下文」同时启用，可能重复注入群聊记录")
            active_reply = settings.get("active_reply", {})
            if isinstance(active_reply, dict) and active_reply.get("enable", False):
                causes.append("主动回复")
                details.append("AstrBot「主动回复」已启用，可能绕过本插件的参与决策，并额外注入群聊记录")
            native_caption = get_config(umo=umo).get("provider_settings", {}).get("default_image_caption_provider_id")
            if native_caption:
                causes.append("原生图片转述")
                details.append("原生「图片转述模型」未留空，可能与插件图片模式或转述调用重复；请留空并在插件内选择转述模型")
        except Exception as exc:
            logger.debug(f"[OnCue] 无法检查会话上下文配置: {umo} | {exc}")
            return
        signature = tuple(causes)
        if not signature:
            self._context_conflict_signatures.pop(umo, None)
        elif self._context_conflict_signatures.get(umo) != signature:
            logger.warning(
                f"[OnCue][配置冲突] 会话={umo} | {'；'.join(details)}。请关闭上述原生设置。"
            )
            self._context_conflict_signatures[umo] = signature

    def _load_glance(self) -> None:
        self.glance_next_due = {}
        if not self.glance_file or not self.glance_file.exists():
            return
        try:
            data = json.loads(self.glance_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for chat_id, ts in data.get("next_due", {}).items():
                    self.glance_next_due[str(chat_id)] = float(ts)
        except Exception as e:
            logger.error(f"[OnCue] 读取 GLANCE 持久化失败: {e}")

    def _save_glance(self) -> None:
        if not self.glance_file:
            return
        try:
            self.glance_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {"next_due": self.glance_next_due}
            self.glance_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error(f"[OnCue] 写入 GLANCE 持久化失败: {e}")

    def _glance_interval_seconds(self, chat: ChatFacts | None = None) -> float:
        focused = chat is not None and chat.attention.active
        prefix = "attention_glance" if focused else "glance"
        default_min, default_max = (3, 8) if focused else (15, 35)
        minutes_min = self._cfg_int(f"{prefix}_min_minutes", default_min)
        minutes_max = self._cfg_int(f"{prefix}_max_minutes", default_max)
        if minutes_min <= 0:
            minutes_min = default_min
        if minutes_max <= 0:
            minutes_max = default_max
        if minutes_min > minutes_max:
            minutes_min, minutes_max = minutes_max, minutes_min
        return random.uniform(minutes_min, minutes_max) * 60.0

    def _schedule_glance(self, chat_id: str, base_ts: float, min_ts: float = 0.0) -> None:
        due = base_ts + self._glance_interval_seconds(self.chats.get(chat_id))
        if min_ts > 0:
            due = max(due, min_ts)
        self.glance_next_due[chat_id] = due
        self._save_glance()

    def _refresh_attention(self, chat_id: str, chat: ChatFacts, now: float) -> None:
        """持有会话锁；到期只更新插件状态，不调用模型或改变观测期。"""
        enabled = self._cfg_bool("enable", True) and self._cfg_bool("enable_glance", False)
        if chat.attention.expire(now, chat.incoming_seq):
            logger.info(f"[OnCue][临时关注] 已强制退出: {chat_id} | {chat.attention.notice.reason}")
            if enabled:
                self._schedule_glance(chat_id, now)
        if not enabled and chat.attention.active:
            chat.attention.apply("stop", "", "", now, chat.incoming_seq, expected_revision=chat.attention.revision)

    def _commit_attention(self, chat_id, chat, decision, view, snapshot, now):
        """仅提交当前有效的最终决策；过期关注的旧结果不恢复状态。"""
        if not decision["_valid"]:
            return
        if view.notice is not None:
            chat.attention.acknowledge(view.notice.token)
        if not self._cfg_bool("enable_glance", False):
            return
        action = decision["attention_action"]
        source = None
        if action == "restart" or (action == "start" and not chat.attention.active):
            source = decision.get("_target") or next((row for row in reversed(snapshot.rows) if row.role == "user"), None)
        message_id = (source.message_id or source.record_id) if source else ""
        result = chat.attention.apply(
            action, decision["attention_note"], message_id, now, chat.incoming_seq,
            expected_revision=view.revision,
        )
        if result in {"started", "restarted"}:
            due = now + self._glance_interval_seconds(chat)
            self.glance_next_due[chat_id] = min(self.glance_next_due.get(chat_id, due), due)
            self._save_glance()
        elif result == "stopped":
            self._schedule_glance(chat_id, now)

    def _ensure_glance_schedule(self, chat_id: str, now: float) -> None:
        if not self._cfg_bool("enable_glance", False):
            return
        if chat_id not in self.glance_next_due:
            self._schedule_glance(chat_id, now)
            due_in = max(0.0, self.glance_next_due.get(chat_id, now) - now)
            logger.info(f"[OnCue][GLANCE] 已排程: {chat_id} | {due_in / 60:.1f} 分钟后")

    async def _glance_loop(self) -> None:
        while True:
            await asyncio.sleep(15)
            await self._glance_tick()

    async def _glance_tick(self) -> None:
        now = time.time()
        for chat_id, chat in list(self.chats.items()):
            async with chat.lock:
                self._refresh_attention(chat_id, chat, now)
        if self._terminating or not self._cfg_bool("enable", True) or not self._cfg_bool("enable_glance", False):
            return
        # 至多两个会话同时运行，按已排定的时间先后领取空位；不在轮询中等待模型。
        for chat_id, due in sorted(self.glance_next_due.items(), key=lambda item: item[1]):
            if len(self._glance_jobs) >= GLANCE_CONCURRENCY or due > now:
                break
            if chat_id in self._glance_jobs:
                continue
            chat = self._chat(chat_id)
            async with chat.lock:
                if chat.glance_inflight or chat.decision_inflight or chat.message_wait is not None:
                    continue
            task = asyncio.create_task(self._glance_job(chat_id))
            self._glance_jobs[chat_id] = task
            task.add_done_callback(lambda completed, key=chat_id: self._finish_glance_job(key, completed))

    def _finish_glance_job(self, chat_id: str, task: asyncio.Task) -> None:
        # 协程首次运行前也可能被取消，不能只依赖它自身的 finally。
        if self._glance_jobs.get(chat_id) is task:
            self._glance_jobs.pop(chat_id, None)

    async def _glance_job(self, chat_id: str) -> None:
        try:
            await self._glance_due(chat_id)
        except Exception as exc:
            logger.error(f"[OnCue] GLANCE 执行失败 {chat_id}: {exc}")
            self._schedule_glance(chat_id, time.time())

    def _create_glance_event(self, chat: ChatFacts):
        if chat.glance_target is None:
            return None
        from .native_reply import GlanceEvent

        async def can_send():
            return await self._glance_can_send(chat, event)

        event = GlanceEvent(self.context, chat.glance_target, can_send)
        return event

    async def _glance_can_send(self, chat: ChatFacts, event: AstrMessageEvent) -> bool:
        async with chat.lock:
            decision = event.get_extra("oncue_decision")
            return bool(
                decision and self._cfg_bool("enable", True) and not self._terminating
                and self._cfg_bool("enable_glance", False)
                and decision.get("_reply_version") == chat.reply_version
            )

    async def _run_native_reply(self, event: AstrMessageEvent) -> None:
        from .native_reply import run_native_reply

        await run_native_reply(self.context, event)

    async def _glance_due(self, chat_id: str) -> None:
        chat = self._chat(chat_id)
        async with chat.lock:
            self._refresh_attention(chat_id, chat, time.time())
            if chat.glance_inflight:
                return
            chat.glance_inflight = True
        event = None
        try:
            event = self._create_glance_event(chat)
            if event is None:
                return
            if await self._decide(event, chat, "你刚忙完自己的事，顺手瞄了一眼群聊。", "glance"):
                await self._run_native_reply(event)
        finally:
            if event is not None:
                self._close_image_round(event)
            try:
                # 原生工具发送/部分分段发送可能没有最终钩子，只补记有发送凭据的内容。
                if event is not None and event._has_send_oper and event.get_extra("oncue_decision"):
                    delivered = list(getattr(event, "delivered", []))
                    tool_texts = event.get_extra("_send_message_to_user_current_session_plain_texts", [])
                    if isinstance(tool_texts, list):
                        delivered.extend(Plain(t) for t in tool_texts if isinstance(t, str) and t.strip())
                    event.set_result(event.chain_result(delivered or [Plain("[已发送消息]")]))
                    await self.mark_reply_sent(event)
            finally:
                async with chat.lock:
                    if event is not None:
                        decision = event.get_extra("oncue_decision")
                        if isinstance(decision, dict):
                            chat.pending_replies.pop(decision.get("_reservation"), None)
                            event.set_extra("oncue_decision", None)
                    chat.glance_inflight = False
                    self._schedule_glance(chat_id, time.time(), min_ts=chat.cooldown_until + 1)

    def _kb_plugin(self):
        try:
            md = self.context.get_registered_star("astrbot_plugin_external_knowledgebase")
        except Exception:
            md = None
        if not md or not getattr(md, "activated", True):
            return None
        return getattr(md, "star_cls", None)

    def _kb_available(self) -> bool:
        plugin = self._kb_plugin()
        if not plugin:
            return False
        try:
            return bool(
                callable(getattr(plugin, "is_taken_over", None))
                and plugin.is_taken_over()
                and callable(getattr(plugin, "retrieve_text", None))
            )
        except Exception:
            return False

    async def _kb_retrieve(self, query: str) -> str:
        query = (query or "").strip()
        if not query:
            return ""
        plugin = self._kb_plugin()
        if not plugin:
            if not self.kb_missing_warned:
                self.kb_missing_warned = True
                logger.warning("[OnCue] 决策要求查知识库，但未找到已启用的外部知识库插件，已忽略")
            return ""
        try:
            taken_over = callable(getattr(plugin, "is_taken_over", None)) and plugin.is_taken_over()
        except Exception:
            taken_over = False
        if not taken_over:
            if not self.kb_not_taken_warned:
                self.kb_not_taken_warned = True
                logger.warning("[OnCue] 决策要求查知识库，但知识库插件未设为“被外部接管”，已忽略以避免重复注入")
            return ""
        try:
            text = await plugin.retrieve_text(query)
            if text:
                logger.info(f"[OnCue] 知识库已检索: {len(text)} chars | {query[:60]}")
            return text or ""
        except Exception as e:
            logger.error(f"[OnCue] 知识库检索失败，按无资料处理: {e}")
            return ""

    def _report_decision_card_issue(self, issue: str) -> None:
        # 相同配置问题只报一次；配置恢复或问题变化后可再次提示。
        if issue != self._decision_card_issue:
            logger.error(
                f"[OnCue][决策卡不可用] {issue}；已跳过 LLM 决策。"
                "请在插件设置「角色设定 → 决策卡」中选择有效且有内容的人格。"
            )
            self._decision_card_issue = issue

    async def _decision_card(self) -> str | None:
        persona_id = self._cfg_str("persona_id").strip()
        if not persona_id:
            self._report_decision_card_issue("未选择决策卡")
            return None
        try:
            persona = self.context.persona_manager.get_persona_v3_by_id(persona_id)
        except Exception as e:
            self._report_decision_card_issue(f"读取决策卡失败（人格 ID={persona_id}）：{e}")
            return None
        if persona is None:
            self._report_decision_card_issue(f"选中的人格不存在或已被删除（人格 ID={persona_id}）")
            return None
        card = persona.get("prompt")
        if not isinstance(card, str) or not card.strip():
            self._report_decision_card_issue(f"选中的人格没有有效内容（人格 ID={persona_id}）")
            return None
        if self._decision_card_issue is not None:
            logger.info(f"[OnCue] 决策卡已恢复可用（人格 ID={persona_id}）")
            self._decision_card_issue = None
        return card

    @staticmethod
    def _participant_label(sender_id, name="", self_id="") -> str:
        sender_id = str(sender_id or "").strip()
        name = re.sub(r"\s+", " ", str(name or "")).strip()
        if sender_id in {"0", "unknown"}:
            sender_id = ""
        if sender_id and sender_id == str(self_id):
            return "Bot（你）"
        label = f"成员「{name}」" if name and name != sender_id else "成员"
        return f"{label}（ID={sender_id}）" if sender_id else f"{label}（身份未知）"

    def _mention_text(self, comp: At, self_id: str) -> str:
        target_id = getattr(comp, "qq", "")
        if str(target_id) == "all":
            return "[提及全体成员]"
        target = self._participant_label(target_id, getattr(comp, "name", ""), self_id)
        return f"[提及 {target}]"

    def _mention_part(self, comp: At, self_id: str) -> MessagePart:
        return MessagePart(
            "mention", self._mention_text(comp, self_id),
            reference_id=str(getattr(comp, "qq", "") or ""),
        )

    @staticmethod
    def _platform_message_id(value) -> str:
        message_id = "" if value is None else str(value).strip()
        # AstrBot 的部分适配器用 unknown 表示缺失 ID，不能用于去重或引用恢复。
        return "" if message_id == "unknown" else message_id

    @staticmethod
    def _image_part(comp: Image, image_id: str) -> MessagePart:
        sources = [getattr(comp, name, None) for name in ("url", "file", "path")]
        sources = [source for source in sources if isinstance(source, str) and source]
        local = ""
        for source in sources:
            if source.startswith(("http://", "https://", "base64://", "data:")):
                continue
            try:
                if source.startswith("file://") or Path(source).is_file():
                    local = source
                    break
            except (OSError, ValueError):
                pass
        return MessagePart("image", image=ImageReference(
            image_id, sources[0] if sources else "", local_source=local,
        ))

    def _quote_parts(self, comp: Reply, self_id: str, record_id: str) -> tuple[MessagePart, ...]:
        parts = []
        for index, quoted in enumerate(getattr(comp, "chain", None) or []):
            if isinstance(quoted, Plain):
                parts.append(MessagePart("text", quoted.text or ""))
            elif isinstance(quoted, Image):
                parts.append(self._image_part(quoted, f"{record_id}:quoted:{index}"))
            elif isinstance(quoted, At):
                parts.append(self._mention_part(quoted, self_id))
            elif isinstance(quoted, Reply):
                # 引文只展开一层，避免嵌套引用不断放大上下文。
                parts.append(MessagePart("quote", "[嵌套引用]"))
            else:
                parts.append(MessagePart("attachment", "[附件]"))
        if not render_message(tuple(parts)):
            fallback = str(getattr(comp, "message_str", "") or getattr(comp, "text", "") or "").strip()
            parts = [MessagePart("text", fallback)] if fallback else []
        return tuple(parts)

    def _reply_part(self, comp: Reply, self_id: str, chat: ChatFacts | None, record_id: str) -> MessagePart:
        message_id = self._platform_message_id(getattr(comp, "id", None))
        sender_id = str(getattr(comp, "sender_id", "") or "").strip()
        if sender_id in {"0", "unknown"}:
            sender_id = ""
        sender_name = getattr(comp, "sender_nickname", "") or ""
        children = self._quote_parts(comp, self_id, record_id)
        content = render_message(children)
        # 只用同一会话中 ID 精确匹配的原消息补齐，不能按昵称或相邻位置猜测。
        if chat is not None and message_id:
            source = next((m for m in reversed(chat.messages) if m.get("message_id") == message_id), None)
            if source is not None:
                sender_id = sender_id or source["sender_id"]
                sender_name = sender_name or source["sender_name"]
                if not content:
                    # 缓存回填也只展开一层，不能沿连续引用持有已淘汰的整棵消息树。
                    children = tuple(
                        MessagePart("quote", "[嵌套引用]") if p.kind == "quote" else p
                        for p in source["parts"]
                    )
                else:
                    # 已明确引用同一条原消息，且资源唯一匹配时复用逐图身份。
                    originals = [p.image for p in source["parts"] if p.image is not None]
                    for quoted in iter_image_parts(children):
                        matches = [image for image in originals if image.source and image.source == quoted.image.source]
                        if len(matches) == 1:
                            quoted.image = matches[0]
        author = self._participant_label(sender_id, sender_name, self_id)
        reference = f"消息ID={json.dumps(message_id, ensure_ascii=False)}" if message_id else "消息ID未知"
        return MessagePart(
            "quote",
            reference_id=message_id, sender_id=sender_id, sender_name=str(sender_name), children=children,
            quote_prefix=f"[引用 {reference}；作者={author}；",
        )

    def _event_to_item(self, event: AstrMessageEvent, chat: ChatFacts | None = None) -> dict:
        parts = []
        pure_text_parts = []
        obj = getattr(event, "message_obj", None)
        message_id = self._platform_message_id(getattr(obj, "message_id", None))
        record_id = event.get_extra("oncue_record_id")
        if not record_id:
            record_id = f"platform:{message_id}" if message_id else "internal:" + uuid4().hex
            event.set_extra("oncue_record_id", record_id)
        only_plain = True
        self_id = str(event.get_self_id())
        for index, comp in enumerate(event.get_messages()):
            if not isinstance(comp, Plain):
                only_plain = False
            if isinstance(comp, Plain):
                text = (comp.text or "").strip()
                if text:
                    parts.append(MessagePart("text", text))
                    pure_text_parts.append(text)
            elif isinstance(comp, Image):
                parts.append(self._image_part(comp, f"{record_id}:{index}"))
            elif isinstance(comp, At):
                parts.append(self._mention_part(comp, self_id))
            elif isinstance(comp, Reply):
                parts.append(self._reply_part(comp, self_id, chat, f"{record_id}:{index}"))
            else:
                parts.append(MessagePart("attachment", "[附件]"))
        parts = tuple(parts)
        if self.image_cache:
            for part in iter_image_parts(parts):
                part.image = self.image_cache.register(event.unified_msg_origin, part.image)
        text = render_message(parts)
        pure_text = " ".join(pure_text_parts).strip()
        sender = getattr(getattr(event, "message_obj", None), "sender", None)
        sender_name = getattr(sender, "nickname", "") or str(event.get_sender_id())
        now = time.time()
        return {
            "message_id": message_id,
            "record_id": record_id,
            "parts": parts,
            "ts": now,
            "time_str": datetime.fromtimestamp(now).strftime("%H:%M:%S"),
            "sender_id": str(event.get_sender_id()),
            "sender_name": str(sender_name),
            "role": "user",
            "text": text or "[空消息]",
            "pure_text": pure_text,
            "is_pure_text": only_plain and bool(pure_text),
            "active": bool(text),
        }

    def _wake_names(self) -> list[str]:
        return [x.strip() for x in self._cfg_str("wake_names").split("|") if x.strip()]

    def _is_forced(self, event: AstrMessageEvent, item: dict) -> bool:
        if event.is_at_or_wake_command:
            return True
        plain = item["pure_text"]
        return bool(plain) and any(name in plain for name in self._wake_names())

    def _is_observing(self, chat: ChatFacts, now: float) -> bool:
        if chat.last_reply_ts <= 0:
            return False
        timeout = self._cfg_int("observation_timeout", 120)
        if timeout <= 0:
            return False
        base = chat.last_reply_ts
        if self._cfg_bool("observation_refresh", True):
            base = max(chat.last_reply_ts, chat.last_observation_activity_ts)
        return now - base < timeout

    def _prune_reply_window(self, chat: ChatFacts, now: float) -> None:
        window = self._cfg_int("reply_window_seconds", 300)
        while chat.reply_ts and now - chat.reply_ts[0] > window:
            chat.reply_ts.popleft()
        # 有些失败路径不会触发发送回调，待发送占位最多保留一个计数窗口。
        for token, until in list(chat.pending_replies.items()):
            if now >= until:
                chat.pending_replies.pop(token, None)

    def _reply_window_allowed(self, chat: ChatFacts, now: float) -> bool:
        self._prune_reply_window(chat, now)
        return len(chat.reply_ts) + len(chat.pending_replies) < self._cfg_int("max_replies_per_window", 5)

    def _decision_current(self, chat: ChatFacts, reply_version: int, now: float) -> bool:
        """持有会话锁时复核，避免应用网络调用期间已过时的决策。"""
        return (
            self._cfg_bool("enable", True)
            and not self._terminating
            and chat.reply_version == reply_version
            and now >= chat.cooldown_until
            and self._reply_window_allowed(chat, now)
        )

    def _trigger_ready(self, chat: ChatFacts, key: str, now: float) -> bool:
        state = chat.backoffs.get(key)
        return not state or now >= float(state.get("until", 0.0))

    def _trigger_reject(self, chat: ChatFacts, key: str, now: float) -> int:
        no_reply_cd = self._cfg_int("no_reply_cooldown", 20)
        state = chat.backoffs.setdefault(key, {"count": 0, "until": 0.0})
        state["count"] = int(state.get("count", 0)) + 1
        mult = min(2 ** state["count"], self._cfg_int("trigger_backoff_max", 4))
        state["until"] = now + no_reply_cd * mult
        return mult

    def _trigger_success(self, chat: ChatFacts, key: str | None) -> None:
        if key:
            chat.backoffs.pop(key, None)

    def _context_snapshot(self, chat: ChatFacts, self_id: str = "") -> ContextSnapshot:
        if self.image_cache:
            self.image_cache.cleanup()
        self._resize_message_cache(chat)
        count = max(1, self._cfg_int("context_message_count", 50))
        limit = max(1, self._cfg_int("message_char_limit", 500))
        rows = list(chat.messages)[-count:]
        records = []
        for m in rows:
            sender = "Bot（你）" if m.get("role") == "assistant" else self._participant_label(m["sender_id"], m["sender_name"])
            message_id = m.get("message_id", "")
            label = "消息ID" if message_id else "内部ID"
            reference = f" / {label}={json.dumps(message_id or m['record_id'], ensure_ascii=False)}"
            text = render_message(m["parts"], limit) or "[空消息]"
            relations = []
            for part in m["parts"]:
                if part.kind == "mention":
                    relations.append(part)
                elif part.kind == "quote":
                    target = f"消息ID={json.dumps(part.reference_id, ensure_ascii=False)}" if part.reference_id else "消息ID未知"
                    author = self._participant_label(part.sender_id, part.sender_name, self_id)
                    relations.append(MessagePart("quote", f"[引用 {target}；作者={author}]"))
            records.append(ContextRow(
                m["record_id"], message_id, m["role"], f"[{sender} / {m['time_str']}{reference}]", text,
                render_message(tuple(relations), limit),
                snapshot_parts(m["parts"]),
            ))
        has_images = any(any(iter_images(m["parts"])) for m in rows)
        note = ""
        if has_images:
            max_mb = self._image_cache_limits().max_image_bytes / 1024**2
            note = (
                f"每张图片 {max_mb:g} MiB 是插件的大小限制，不是平台限制；"
                "图片状态是本次快照，并非永久状态。已有描述来自模型，不等于完整原图或确定事实。"
            )
        return ContextSnapshot(tuple(records), chat.incoming_seq, note)

    def _history_text(self, chat: ChatFacts) -> str:
        return self._context_snapshot(chat).render()

    def _record_reply(self, chat: ChatFacts, text: str, now: float, parts: tuple[MessagePart, ...] | None = None) -> None:
        """在持有会话锁时记录发送结果；自身发言只供决策参考。"""
        self._cancel_message_wait(chat)
        self._append_message(chat, {
            "message_id": "",
            "record_id": "internal:" + uuid4().hex,
            "parts": parts if parts is not None else (MessagePart("text", text),),
            "ts": now,
            "time_str": datetime.fromtimestamp(now).strftime("%H:%M:%S"),
            "sender_id": "assistant",
            "sender_name": "Bot（你）",
            "role": "assistant",
            "text": text,
            "pure_text": "",
            "is_pure_text": False,
            "active": False,
        })
        chat.reply_ts.append(now)
        chat.reply_version += 1
        chat.last_reply_ts = now
        chat.cooldown_until = max(chat.cooldown_until, now + self._cfg_int("reply_cooldown", 10))

    async def _build_prompt(self, chat_history: str, entry_context: str) -> str | None:
        template = self._cfg_str("decision_prompt") or DEFAULT_DECISION_PROMPT
        card = await self._decision_card()
        if card is None:
            return None
        mapping = {
            "{character_card}": card,
            "{current_time}": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "{chat_history}": chat_history or "（暂无）",
            "{entry_context}": entry_context,
        }
        template = re.sub(
            r"\{(?:character_card|current_time|chat_history|entry_context)\}",
            lambda match: mapping[match.group()], template,
        )
        if self._kb_available():
            template += "\n\n知识库联动：如果这次开口需要外部资料才能答好，在同一个 JSON 里增加 \"need_kb\": true 和 \"kb_query\": \"独立检索词\"；否则省略或设为 false。kb_query 不要照抄群聊原话。"
        return template

    async def _llm_decision(self, umo: str, prompt: str, *, provider_id: str | None = None, image_urls=None) -> str:
        provider_id = provider_id or self._cfg_str("analyzer_provider").strip()
        try:
            if not provider_id:
                provider_id = await self.context.get_current_chat_provider_id(umo=umo)
            options = {"image_urls": image_urls} if image_urls else {}
            resp = await self.context.llm_generate(chat_provider_id=provider_id, prompt=prompt, **options)
            return resp.completion_text or ""
        except Exception as e:
            logger.error(f"[OnCue] 决策 LLM 调用失败，本次按沉默处理: {e}")
            return ""

    def _parse_decision(self, raw: str) -> dict:
        text = (raw or "").strip()
        candidate = ""
        block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
        if block:
            candidate = block.group(1)
        else:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                candidate = text[start : end + 1]
        data = {}
        if candidate:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    data = parsed
            except Exception:
                data = {}
        should_reply = data.get("should_reply")
        valid = isinstance(should_reply, bool) or str(should_reply).strip().lower() in {
            "true", "false", "1", "0", "yes", "no", "y", "n",
        }
        if not isinstance(should_reply, bool):
            should_reply = str(should_reply).strip().lower() in {"true", "1", "yes", "y"}
        need_kb = data.get("need_kb", False)
        if not isinstance(need_kb, bool):
            need_kb = str(need_kb).strip().lower() in {"true", "1", "yes", "y"}
        target = data.get("target_message_id")
        target_valid = target is None or (isinstance(target, (str, int)) and not isinstance(target, bool))
        attention_action = data.get("attention_action", "keep")
        attention_note = data.get("attention_note", "")
        decision = {
            "_valid": valid and target_valid,
            "should_reply": should_reply and target_valid,
            "speaker": str(data.get("speaker", "") or "").strip(),
            "mood": str(data.get("mood", "") or "").strip(),
            "reason": str(data.get("reason", "") or "").strip(),
            "need_kb": need_kb,
            "kb_query": str(data.get("kb_query", "") or "").strip(),
            "target_message_id": str(target).strip() if target is not None and target_valid else "",
            "attention_action": attention_action.strip().lower() if isinstance(attention_action, str) else "",
            "attention_note": attention_note.strip()[:ATTENTION_NOTE_LIMIT] if isinstance(attention_note, str) else "",
        }
        if decision["need_kb"] and not decision["kb_query"]:
            decision["need_kb"] = False
        if decision["should_reply"] and decision["reason"].lower() in {"", "无", "没有", "none", "null"}:
            decision["should_reply"] = False
            decision["reason"] = "正向理由为空，代码兜底不回复"
        return decision

    def _target_decoration_conflict(self, event: AstrMessageEvent, target: ContextRow | None) -> bool:
        if target is None or target.record_id == event.get_extra("oncue_record_id"):
            return False
        get_config = getattr(self.context, "get_config", None)
        if not callable(get_config):
            return False
        try:
            settings = get_config(umo=event.unified_msg_origin).get("platform_settings", {})
            conflicts = []
            if settings.get("reply_with_mention", False):
                conflicts.append("自动 @ 回复")
            if settings.get("reply_with_quote", False):
                conflicts.append("引用回复")
        except Exception as exc:
            logger.warning(f"[OnCue] 无法检查回应目标的原生发送设置，本次保持沉默: {exc}")
            return True
        if conflicts:
            logger.warning(
                f"[OnCue][回应目标冲突] 会话={event.unified_msg_origin} | AstrBot「{'、'.join(conflicts)}」"
                "仍指向触发消息，无法安全回应另一条消息，本次保持沉默。请关闭上述原生设置；插件不改写原事件身份。"
            )
        return bool(conflicts)

    @staticmethod
    def _cancel_message_wait(chat: ChatFacts) -> None:
        """持有会话锁时取消等待，唤醒原事件以正常结束处理。"""
        pending = chat.message_wait
        chat.message_wait = None
        if pending is not None:
            pending.changed.set()

    async def _wait_for_messages(self, event: AstrMessageEvent, chat: ChatFacts, pending: MessageWait) -> None:
        try:
            while True:
                async with chat.lock:
                    if (
                        chat.message_wait is not pending
                        or self._terminating
                        or not self._cfg_bool("enable", True)
                        or chat.reply_version != pending.reply_version
                    ):
                        return
                    quiet = max(0, self._cfg_int("message_wait_seconds", 2))
                    remaining = min(pending.last_activity_at + quiet, pending.deadline) - time.monotonic()
                    if remaining > 0:
                        pending.changed.clear()
                if remaining <= 0:
                    await self._decide(event, chat, "", None, wait_batch=pending)
                    continue
                # 只挂起首次触发的事件，不占会话锁；新消息的事件继续正常处理。
                try:
                    await asyncio.wait_for(pending.changed.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    pass
        finally:
            async with chat.lock:
                if chat.message_wait is pending:
                    self._cancel_message_wait(chat)

    async def _decide(
        self, event: AstrMessageEvent, chat: ChatFacts, entry_context: str, backoff_key: str | None,
        *, wait_batch: MessageWait | None = None, summoned_version: int | None = None,
    ) -> bool:
        summoned = summoned_version is not None
        async with chat.lock:
            now = time.time()
            self._refresh_attention(event.unified_msg_origin, chat, now)
            if self._terminating or not self._cfg_bool("enable", True):
                return False
            if summoned and chat.reply_version != summoned_version:
                return False
            if wait_batch is not None:
                if chat.message_wait is not wait_batch or chat.reply_version != wait_batch.reply_version:
                    return False
                quiet = max(0, self._cfg_int("message_wait_seconds", 2))
                if time.monotonic() < min(wait_batch.last_activity_at + quiet, wait_batch.deadline):
                    return False
                # 等待到决策的交接在同一把锁内完成；失败时也结束本轮等待。
                chat.message_wait = None
                if self._is_observing(chat, now):
                    trigger = ("你刚才已经开口过，现在还在观测窗内；判断这一幕是否值得再接话。", None)
                else:
                    trigger = self._stat_trigger_locked(chat, wait_batch.latest_item, now)
                if not trigger:
                    return False
                entry_context, backoff_key = trigger
            elif chat.message_wait is not None:
                return False
            if not summoned and (
                chat.decision_inflight or now < chat.cooldown_until
                or (chat.glance_inflight and backoff_key != "glance")
            ):
                return False
            if backoff_key == "glance" and chat.incoming_seq <= self.glance_last_check.get(event.unified_msg_origin, 0):
                return False
            if backoff_key and not self._trigger_ready(chat, backoff_key, now):
                return False
            if not summoned and not self._reply_window_allowed(chat, now):
                return False
            source = "summoned" if summoned else backoff_key or "observe"
            snapshot = self._context_snapshot(chat, str(event.get_self_id()))
            reply_version = chat.reply_version
            owner = object()
            chat.decision_inflight = owner
            image_round = self._new_image_round(chat)
            event.set_extra("oncue_image_round", image_round)
        accepted = False

        def current(now: float) -> bool:
            # 仅在持有会话锁时调用；模型与知识库请求均在锁外。
            return (
                chat.decision_inflight is owner
                and self._cfg_bool("enable", True) and not self._terminating
                and chat.reply_version == reply_version
                and (backoff_key != "glance" or self._cfg_bool("enable_glance", False))
                and (summoned or self._decision_current(chat, reply_version, now))
            )

        try:
            for attempt in range(2):
                async with chat.lock:
                    now = time.time()
                    self._refresh_attention(event.unified_msg_origin, chat, now)
                    if not current(now) or (attempt and not self._cfg_bool("enable_extra_decision", True)):
                        return False
                    attention_view = chat.attention.snapshot()
                    attention_context = attention_view.render(now, snapshot.progress)
                if self.image_input is not None and any(any(iter_images(row.parts)) for row in snapshot.rows):
                    if await self._decision_card() is None:
                        return False
                snapshot, image_provider, image_urls = await self._decision_images(event, chat, snapshot, image_round)
                prompt = await self._build_prompt(snapshot.render(), entry_context)
                if prompt is None:
                    return False
                if self._cfg_bool("enable_glance", False):
                    prompt += "\n\n" + ATTENTION_DECISION_RULE + "\n" + (attention_context or "[临时关注] 当前未开启，采用普通一瞥间隔。")
                elif attention_context:
                    prompt += "\n\n" + attention_context + "\n定时一瞥当前已关闭，不安排后续关注检查。"
                async with chat.lock:
                    if not current(time.time()) or (attempt and not self._cfg_bool("enable_extra_decision", True)):
                        return False
                logger.info(f"[OnCue] 决策调用: source={source} | 第 {attempt + 1} 次 | 已发送 {len(chat.reply_ts)} 待发送 {len(chat.pending_replies)} 上限 {self._cfg_int('max_replies_per_window', 5)}")
                image_options = {"provider_id": image_provider, "image_urls": image_urls} if image_provider else {}
                raw = await self._llm_decision(event.unified_msg_origin, prompt, **image_options)
                decision = self._parse_decision(raw)
                target_id = decision["target_message_id"]
                decision["_target"] = snapshot.target(target_id) if target_id else None
                if target_id and decision["_target"] is None:
                    decision.update(_valid=False, should_reply=False, reason="回应目标不在本次群聊快照中")
                    logger.warning(f"[OnCue] 决策回应目标无效，本次保持沉默: {target_id}")
                async with chat.lock:
                    if not current(time.time()):
                        logger.info(f"[OnCue] 会话状态已变化，忽略本次决策: source={source}")
                        return False
                    if (
                        attempt == 0 and decision["_valid"]
                        and self._cfg_bool("enable_extra_decision", True)
                        and chat.incoming_seq > snapshot.progress
                    ):
                        added = chat.incoming_seq - snapshot.progress
                        snapshot = self._context_snapshot(chat, str(event.get_self_id()))
                        entry_context += (
                            f"\n初判快照之后新增了 {added} 条群友消息。这是本轮唯一一次额外决策，"
                            "请结合本次提供的群聊记录重新判断，以本次结论为准；可选择记录中较早的回应目标。"
                        )
                        continue
                break
            # 原生装饰器没有单请求目标覆盖接口，不能让正文回应旧消息却 @/引用触发者。
            if (
                decision["should_reply"] and backoff_key != "glance"
                and self._target_decoration_conflict(event, decision["_target"])
            ):
                decision.update(_valid=False, should_reply=False, reason="原生自动提及或引用与回应目标冲突")
            # 初判被替代时不提前检索知识库，也不占用回复名额。
            if decision["should_reply"] and decision.get("need_kb"):
                decision["kb_text"] = await self._kb_retrieve(decision.get("kb_query", ""))
            async with chat.lock:
                now = time.time()
                self._refresh_attention(event.unified_msg_origin, chat, now)
                if not current(now):
                    logger.info(f"[OnCue] 会话状态已变化，忽略本次决策: source={source}")
                    return False
                self._commit_attention(event.unified_msg_origin, chat, decision, attention_view, snapshot, now)
                if backoff_key == "glance" and decision["_valid"]:
                    self.glance_last_check[event.unified_msg_origin] = snapshot.progress
                if decision["should_reply"]:
                    token = uuid4().hex
                    chat.pending_replies[token] = now + max(1, self._cfg_int("reply_window_seconds", 300))
                    decision["_reservation"] = token
                    decision["_backoff_key"] = backoff_key
                    decision["_reply_version"] = reply_version
                    decision["_chat_context"] = snapshot
                    chat.cooldown_until = max(chat.cooldown_until, now + self._cfg_int("reply_cooldown", 10))
                    event.set_extra("oncue_decision", decision)
                    accepted = True
                    event.is_at_or_wake_command = True
                    logger.info(f"[OnCue] 决定回复: source={source} | {decision['reason']}")
                    return True
                mult = self._trigger_reject(chat, backoff_key, now) if backoff_key else 1
                cooldown = self._cfg_int("no_reply_cooldown", 20) * mult
                chat.cooldown_until = max(chat.cooldown_until, now + cooldown)
                logger.info(f"[OnCue] 决定沉默: source={source} | 冷却 {cooldown}s | {decision['reason'] or raw[:80]}")
                return False
        finally:
            if not accepted:
                image_round.close()
                if event.get_extra("oncue_image_round") is image_round:
                    event.set_extra("oncue_image_round", None)
            async with chat.lock:
                if chat.decision_inflight is owner:
                    chat.decision_inflight = None

    def _stat_trigger_locked(self, chat: ChatFacts, item: dict, now: float) -> tuple[str, str] | None:
        if self._cfg_bool("enable_echo_trigger", False) and item.get("is_pure_text", False):
            window = max(2, min(self._cfg_int("echo_message_count", 5), chat.messages.maxlen))
            threshold = max(2, min(self._cfg_int("echo_threshold", 3), window))
            norm = re.sub(r"\s+", " ", item["pure_text"]).strip()
            if norm:
                # 先截取窗口；混合消息和自身回复占位置，但不贡献复读人数。
                recent = list(chat.messages)[-window:]
                participants = {
                    m["sender_id"] for m in recent
                    if m.get("role", "user") == "user" and m.get("is_pure_text", False)
                    and re.sub(r"\s+", " ", m["pure_text"]).strip() == norm
                }
                count = len(participants)
                key = f"echo:{norm}"
                if count >= threshold and self._trigger_ready(chat, key, now):
                    logger.info(f"[OnCue] 触发边命中: echo | 最近 {len(recent)} 条 | 人数 {count}/{threshold} | {norm[:40]}")
                    preview = render_message(
                        (MessagePart("text", norm),), max(1, self._cfg_int("message_char_limit", 500)),
                    )
                    return (
                        f"最近 {len(recent)} 条群聊消息中，有 {count} 名不同成员重复了同一句纯文本："
                        f"{json.dumps(preview, ensure_ascii=False)}。请结合角色性格和聊天语境，判断是否想参与。",
                        key,
                    )
        if self._cfg_bool("enable_dense_trigger", False):
            window = self._cfg_int("dense_window", 120)
            recent = [m for m in chat.messages if m.get("role", "user") == "user" and now - m["ts"] <= window and m["active"]]
            participants = {m["sender_id"] for m in recent}
            if len(recent) >= self._cfg_int("dense_message_threshold", 10) and len(participants) >= self._cfg_int("dense_participant_threshold", 3):
                key = "dense"
                if self._trigger_ready(chat, key, now):
                    logger.info(f"[OnCue] 触发边命中: dense | 消息 {len(recent)}/{self._cfg_int('dense_message_threshold', 10)} | 人数 {len(participants)}/{self._cfg_int('dense_participant_threshold', 3)}")
                    return "群里讨论突然变得热烈。", key
        return None

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE, priority=-10)
    async def on_message(self, event: AstrMessageEvent):
        if self.image_cache:
            self.image_cache.cleanup()
        await self._on_group_message(event)
        if self.image_cache and not self._terminating and self._cfg_bool("enable", True):
            chat = self.chats.get(event.unified_msg_origin)
            record_id = event.get_extra("oncue_record_id")
            if chat is not None and record_id:
                item = next((m for m in chat.messages if m["record_id"] == record_id), None)
                if item is not None:
                    # 在入站事件资源被框架清理前复制，仅保留已有本地资源。
                    try:
                        await self._retain_local_images(event.unified_msg_origin, item["parts"])
                        async with chat.lock:
                            decision = event.get_extra("oncue_decision")
                            if isinstance(decision, dict) and (
                                self._terminating or not self._cfg_bool("enable", True)
                                or ("_reply_version" in decision and decision["_reply_version"] != chat.reply_version)
                            ):
                                self._discard_decision(event, chat)
                    except BaseException:
                        async with chat.lock:
                            self._discard_decision(event, chat)
                        raise

    @staticmethod
    def _discard_decision(event: AstrMessageEvent, chat: ChatFacts) -> None:
        """持有会话锁时释放本事件名额，不影响新事件的占用。"""
        decision = event.get_extra("oncue_decision")
        OnCuePlugin._close_image_round(event)
        if isinstance(decision, dict):
            chat.pending_replies.pop(decision.get("_reservation"), None)
            event.set_extra("oncue_decision", None)
            event.is_at_or_wake_command = False

    async def _on_group_message(self, event: AstrMessageEvent):
        if self._terminating or not self._cfg_bool("enable", True):
            return
        if str(event.get_sender_id()) == str(event.get_self_id()):
            return
        self._check_context_conflict(event)
        chat = self._chat(event.unified_msg_origin)
        pending = None
        summoned = False
        summoned_version = None
        async with chat.lock:
            now = time.time()
            observing = self._is_observing(chat, now)
            item = self._event_to_item(event, chat)
            if not self._append_message(chat, item):
                event.stop_event()
                return
            self._refresh_attention(event.unified_msg_origin, chat, now)
            if self._cfg_bool("enable_glance", False):
                # 仅保存路由信息，不持有入站事件、原消息或媒体临时文件。
                chat.glance_target = {
                    "umo": event.unified_msg_origin,
                    "self_id": event.get_self_id(),
                    "group_id": event.get_group_id(),
                    "platform_meta": event.platform_meta,
                    "session_isolated": event.get_extra("_session_isolated", False),
                }
            if item["active"]:
                chat.last_activity_ts = now
                # 静默期活动只更新群聊活跃度，不能重新开启已经过期的观测窗。
                if observing and self._cfg_bool("observation_refresh", True):
                    chat.last_observation_activity_ts = now
            self._ensure_glance_schedule(event.unified_msg_origin, now)
            if self._is_forced(event, item):
                self._cancel_message_wait(chat)
                chat.reply_version += 1
                if self._cfg_bool("force_reply_when_summoned", True):
                    event.set_extra("oncue_decision", {
                        "should_reply": True, "speaker": "", "mood": "", "reason": "被@/点名直接回复",
                        "_reply_version": chat.reply_version,
                        "_chat_context": self._context_snapshot(chat, str(event.get_self_id())),
                    })
                    event.is_at_or_wake_command = True
                    logger.info(f"[OnCue] 点名直接回复: {item['sender_name']} | {item['pure_text'][:60]}")
                    return
                # 只关闭默认 LLM 放行，保留其它插件处理；决策同意后再唤醒。
                event.is_at_or_wake_command = False
                summoned = True
                summoned_version = chat.reply_version
            elif chat.message_wait is not None:
                if item["active"]:
                    chat.message_wait.latest_item = item
                    chat.message_wait.last_activity_at = time.monotonic()
                    chat.message_wait.changed.set()
                return
            if summoned:
                trigger = ("有人@、引用、使用唤醒前缀或昵称呼唤你。优先考虑回应，但仍可按角色性格和语境保持沉默。", None)
            elif observing:
                trigger = ("你刚才已经开口过，现在还在观测窗内；判断这一幕是否值得再接话。", None)
            else:
                trigger = self._stat_trigger_locked(chat, item, now)
            if trigger and not summoned and self._cfg_int("message_wait_seconds", 2) > 0:
                if (
                    not item["active"] or chat.decision_inflight or chat.glance_inflight or now < chat.cooldown_until
                    or not self._reply_window_allowed(chat, now)
                ):
                    return
                started = time.monotonic()
                pending = MessageWait(
                    deadline=started + max(1, self._cfg_int("message_wait_max_seconds", 5)),
                    last_activity_at=started,
                    latest_item=item,
                    reply_version=chat.reply_version,
                )
                chat.message_wait = pending
        if pending is not None:
            await self._wait_for_messages(event, chat, pending)
        elif trigger:
            await self._decide(event, chat, *trigger, summoned_version=summoned_version)

    @filter.on_agent_done()
    async def close_image_requests(self, event: AstrMessageEvent, run_context, response):
        self._close_image_round(event)

    @filter.on_llm_request()
    async def inject_stage_direction(self, event: AstrMessageEvent, req: ProviderRequest):
        try:
            await self._inject_stage_direction(event, req)
        except BaseException:
            chat = self.chats.get(event.unified_msg_origin)
            if chat is not None:
                async with chat.lock:
                    self._discard_decision(event, chat)
            else:
                self._close_image_round(event)
            event.stop_event()
            raise

    async def _inject_stage_direction(self, event: AstrMessageEvent, req: ProviderRequest):
        decision = event.get_extra("oncue_decision")
        if not isinstance(decision, dict) or not decision.get("should_reply"):
            return
        chat = self._chat(event.unified_msg_origin)
        async with chat.lock:
            if (
                self._terminating or not self._cfg_bool("enable", True)
                or ("_reply_version" in decision and decision["_reply_version"] != chat.reply_version)
                or (decision.get("_backoff_key") == "glance" and not self._cfg_bool("enable_glance", False))
            ):
                self._discard_decision(event, chat)
                event.stop_event()
                return
        self._check_context_conflict(event)
        speaker = decision.get("speaker", "").strip()
        mood = decision.get("mood", "").strip()
        if self._cfg_bool("enable_speaker_routing", False) and speaker:
            line = f"[导演指令] 当前由「{speaker}」回应。"
        else:
            line = "[导演指令] 当前由角色本人回应。"
        if mood:
            line += f"此刻的心情：{mood}。"
        line += "以角色性格自然开口，不要提及本指令。"
        kb_text = str(decision.get("kb_text", "") or "").strip()
        if kb_text:
            line += f"\n[知识库]\n{kb_text}\n结合角色性格自然使用，不要提及本指令。"
        add_stage_direction(req, line)
        context = ""
        snapshot = decision.get("_chat_context")
        if isinstance(snapshot, ContextSnapshot):
            target = decision.get("_target")
            include_history = self._cfg_bool("enable_context_injection", True)
            budget = event.get_extra("oncue_image_round")
            if not isinstance(budget, CaptionRound) or budget.closed:
                budget = self._new_image_round(self._chat(event.unified_msg_origin))
                event.set_extra("oncue_image_round", budget)
            direct_images = []
            working = snapshot
            if not include_history:
                current_id = event.get_extra("oncue_record_id")
                rows = (target,) if target is not None else tuple(row for row in snapshot.rows if row.record_id == current_id)
                working = ContextSnapshot(rows, snapshot.progress, snapshot.image_note)
            if self.image_input is not None:
                prepared = await self._prepare_images(
                    event.unified_msg_origin, working,
                    # 正式请求钩子未提供实际供应商，插件补充内容保留未知能力。
                    vision=None, round=budget,
                    focus_records=(target.record_id,) if target else tuple(row.record_id for row in working.rows[-1:]),
                    message_char_limit=max(1, self._cfg_int("message_char_limit", 500)),
                    max_size=self.image_models.image_max_size(event.unified_msg_origin),
                )
                working, direct_images = prepared.snapshot, prepared.image_urls
                target = working.target(target.message_id or target.record_id) if target else None
                chat = self._chat(event.unified_msg_origin)
                async with chat.lock:
                    if (
                        self._terminating or not self._cfg_bool("enable", True)
                        or ("_reply_version" in decision and decision["_reply_version"] != chat.reply_version)
                    ):
                        self._discard_decision(event, chat)
                        event.stop_event()
                        return
            context = request_background(
                working, event, req, target=target, include_history=True,
            )
            if not include_history and target is None and not any(row.image_body for row in working.rows):
                context = ""
            set_temporary_background(req, context, TextPart, images=direct_images, image_part_type=ImageURLPart)
        else:
            set_temporary_background(req, context, TextPart)

    @filter.on_decorating_result()
    async def mark_streaming_reply_sent(self, event: AstrMessageEvent):
        # AstrBot 的流式发送路径可能不触发 after_message_sent。
        result = event.get_result()
        if result and result.result_content_type == ResultContentType.STREAMING_FINISH:
            await self.mark_reply_sent(event)

    @filter.after_message_sent()
    async def mark_reply_sent(self, event: AstrMessageEvent):
        self._close_image_round(event)
        decision = event.get_extra("oncue_decision")
        if not isinstance(decision, dict) or not decision.get("should_reply"):
            return
        chat = self._chat(event.unified_msg_origin)
        async with chat.lock:
            # 同一事件可能经过多个完成入口，取锁后再次检查，避免重复入账。
            decision = event.get_extra("oncue_decision")
            if not isinstance(decision, dict) or not decision.get("should_reply"):
                return
            chat.pending_replies.pop(decision.get("_reservation"), None)
            event.set_extra("oncue_decision", None)
            result = event.get_result()
            # 非空结果只是待发送内容，不能单独作为成功发送的依据。
            if not getattr(event, "_has_send_oper", False) or not result:
                return
            parts = []
            has_body = False
            for comp in result.chain or []:
                if isinstance(comp, Plain):
                    if (comp.text or "").strip():
                        parts.append(MessagePart("text", comp.text.strip()))
                        has_body = True
                elif isinstance(comp, Image):
                    parts.append(self._image_part(comp, "internal:" + uuid4().hex))
                    has_body = True
                elif isinstance(comp, At):
                    parts.append(self._mention_part(comp, str(event.get_self_id())))
                elif isinstance(comp, Reply):
                    parts.append(self._reply_part(comp, str(event.get_self_id()), chat, "internal:" + uuid4().hex))
                else:
                    parts.append(MessagePart("attachment", "[附件]"))
                    has_body = True
            if not has_body:
                return
            parts = tuple(parts)
            text = render_message(parts)
            if not text:
                return
            now = time.time()
            self._record_reply(chat, text, now, parts)
            self._trigger_success(chat, decision.get("_backoff_key"))
            logger.info(f"[OnCue] 回复已发送: 进入观测 {self._cfg_int('observation_timeout', 120)}s | 冷却 {self._cfg_int('reply_cooldown', 10)}s")
        if self.image_cache and not self._terminating:
            for part in iter_image_parts(parts):
                part.image = self.image_cache.register(event.unified_msg_origin, part.image)
            await self._retain_local_images(event.unified_msg_origin, parts)
