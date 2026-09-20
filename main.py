import asyncio
import hashlib
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
from astrbot.core.message.components import At, Image, Plain, Reply
from astrbot.core.message.message_event_result import MessageChain, ResultContentType


DEFAULT_DECISION_PROMPT = """你是群聊角色扮演机器人的"内心"。请代入以下角色，判断看到群聊对话后是否想开口。

<角色设定>
{character_card}
</角色设定>

当前时间：{current_time}

<触发背景>
{entry_context}
</触发背景>

<群聊近况>
{chat_history}
</群聊近况>

判断原则（按优先级）：
1. 有人明确@或点名你、或直接向你提问 → 必须回应
2. 话题命中角色的兴趣或专业领域、有人需要帮助、气氛需要打圆场 → 可以开口
3. 群友之间的闲聊与角色无关、刚说过话没有新信息、深夜角色在睡觉 → 保持安静
4. 只有 [图片]/表情包且没有文字、点名、提问、引用你或紧跟你的发言 → 默认保持安静（[图片] 只代表群里有人活动，不是开口理由）
5. 有任何犹豫 → 保持安静
6. 若这次开口需要外部知识库资料才能答好，在同一个 JSON 里给出 need_kb=true 和独立检索词 kb_query；否则 need_kb=false、kb_query 留空

只输出 JSON，不要输出其它内容：
{"should_reply": true或false, "speaker": "回应的角色名(单角色填角色名)", "mood": "当前心情短语", "reason": "一句话理由", "need_kb": true或false, "kb_query": "需要时填独立检索词"}"""

GLANCE_GENERATE_PROMPT = """你要以群聊角色的身份自然开口说一句话。

<角色设定>
{character_card}
</角色设定>

当前时间：{current_time}

<导演指令>
{stage_direction}
</导演指令>

<群聊近况>
{chat_history}
</群聊近况>

要求：只输出要发到群里的正文，不要解释，不要提及导演指令，不要复读别人刚说过的话。"""

@dataclass
class ChatFacts:
    messages: deque = field(default_factory=lambda: deque(maxlen=50))
    reply_ts: deque = field(default_factory=deque)
    pending_replies: dict[str, float] = field(default_factory=dict)
    backoffs: dict = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_reply_ts: float = 0.0
    last_activity_ts: float = 0.0
    last_observation_activity_ts: float = 0.0
    cooldown_until: float = 0.0


class OnCuePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.chats: dict[str, ChatFacts] = {}
        self.data_dir: Path | None = None
        self.glance_file: Path | None = None
        self.glance_next_due: dict[str, float] = {}
        self.glance_last_check: dict[str, float] = {}
        self.glance_task: asyncio.Task | None = None
        self.persona_cache: dict[str, tuple[float, str]] = {}
        self.card_cache_file: Path | None = None
        self.card_cache: dict[str, dict] = {}
        self.card_fail_until: dict[str, float] = {}
        self.card_op_lock = asyncio.Lock()
        self.condense_prompt_warned = False
        self.kb_missing_warned = False
        self.kb_not_taken_warned = False

    async def initialize(self) -> None:
        try:
            self.data_dir = StarTools.get_data_dir("astrbot_plugin_soul_on_cue")
            self.glance_file = self.data_dir / "glance.json"
            self.card_cache_file = self.data_dir / "persona_cards.json"
            self._load_glance()
            self._load_card_cache()
            self.glance_task = asyncio.create_task(self._glance_loop())
        except Exception as e:
            logger.error(f"[OnCue] 初始化 GLANCE 失败: {e}")

    async def terminate(self) -> None:
        if self.glance_task:
            self.glance_task.cancel()
            try:
                await self.glance_task
            except asyncio.CancelledError:
                pass
            self.glance_task = None

    def _chat(self, chat_id: str) -> ChatFacts:
        if chat_id not in self.chats:
            self.chats[chat_id] = ChatFacts()
        return self.chats[chat_id]

    def _cfg_value(self, key: str, default):
        try:
            value = self.config.get(key, None)
        except Exception:
            value = None
        if value is not None:
            return value
        for section in ("config_wake", "config_decision", "config_condense", "config_trigger", "config_character"):
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

    def _load_card_cache(self) -> None:
        self.card_cache = {}
        if not self.card_cache_file or not self.card_cache_file.exists():
            return
        try:
            data = json.loads(self.card_cache_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, dict) and value.get("card"):
                        self.card_cache[str(key)] = value
        except Exception as e:
            logger.error(f"[OnCue] 读取决策卡缓存失败: {e}")

    def _save_card_cache(self) -> None:
        if not self.card_cache_file:
            return
        try:
            self.card_cache_file.parent.mkdir(parents=True, exist_ok=True)
            items = sorted(self.card_cache.items(), key=lambda kv: float(kv[1].get("created_ts", 0.0)), reverse=True)[:100]
            payload = {key: value for key, value in items}
            self.card_cache_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error(f"[OnCue] 写入决策卡缓存失败: {e}")

    def _card_key(self, raw_card: str) -> str:
        return hashlib.sha256(raw_card.encode("utf-8")).hexdigest()

    async def _decision_card(self, umo: str, platform_name: str = "") -> str:
        raw = await self._character_card(umo, platform_name)
        mode = self._cfg_str("decision_card_mode", "raw")
        max_chars = max(200, self._cfg_int("decision_card_max_chars", 1000))
        if mode != "condense" or not raw.strip():
            return raw
        condense_prompt = self._cfg_str("decision_card_condense_prompt")
        if "{raw_card}" not in condense_prompt:
            if not self.condense_prompt_warned:
                self.condense_prompt_warned = True
                logger.error("[OnCue] 已选择自动浓缩，但浓缩提示词为空或缺少 {raw_card}，本次及后续直接使用原始角色卡")
            return raw
        key = self._card_key(condense_prompt + "\n" + str(max_chars) + "\n" + raw)
        cached = self.card_cache.get(key)
        if cached and cached.get("card"):
            return str(cached["card"])
        async with self.card_op_lock:
            cached = self.card_cache.get(key)
            if cached and cached.get("card"):
                return str(cached["card"])
            now = time.time()
            if now < self.card_fail_until.get(key, 0.0):
                return raw
            provider_id = self._cfg_str("decision_card_condense_provider").strip() or self._cfg_str("analyzer_provider").strip()
            try:
                if not provider_id:
                    provider_id = await self.context.get_current_chat_provider_id(umo=umo)
                prompt = condense_prompt.replace("{max_chars}", str(max_chars)).replace("{raw_card}", raw)
                resp = await self.context.llm_generate(chat_provider_id=provider_id, prompt=prompt)
                card = (resp.completion_text or "").strip()
                if card.startswith("```"):
                    card = card.strip("`").strip()
                if len(card) > max_chars:
                    logger.warning(f"[OnCue] 自动浓缩结果 {len(card)} 字超过目标 {max_chars} 字，按不截断保留")
                if not card or card == raw:
                    raise ValueError("浓缩结果为空")
                self.card_cache[key] = {"card": card, "created_ts": now, "chars": len(card)}
                self._save_card_cache()
                logger.info(f"[OnCue] 决策卡已自动浓缩: {len(raw)} -> {len(card)} chars")
                return card
            except Exception as e:
                self.card_fail_until[key] = now + 600
                logger.error(f"[OnCue] 决策卡自动浓缩失败，10 分钟内回退原始卡: {e}")
                return raw

    def _glance_interval_seconds(self) -> float:
        minutes_min = self._cfg_int("glance_min_minutes", 15)
        minutes_max = self._cfg_int("glance_max_minutes", 40)
        if minutes_min <= 0:
            minutes_min = 15
        if minutes_max <= 0:
            minutes_max = 40
        if minutes_min > minutes_max:
            minutes_min, minutes_max = minutes_max, minutes_min
        return random.uniform(minutes_min, minutes_max) * 60.0

    def _schedule_glance(self, chat_id: str, base_ts: float, min_ts: float = 0.0) -> None:
        due = base_ts + self._glance_interval_seconds()
        if min_ts > 0:
            due = max(due, min_ts)
        self.glance_next_due[chat_id] = due
        self._save_glance()

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
            if not self._cfg_bool("enable", True) or not self._cfg_bool("enable_glance", False):
                continue
            now = time.time()
            due_chats = [chat_id for chat_id, ts in self.glance_next_due.items() if ts <= now]
            for chat_id in due_chats:
                try:
                    await self._glance_due(chat_id)
                except Exception as e:
                    logger.error(f"[OnCue] GLANCE 执行失败 {chat_id}: {e}")
                    self._schedule_glance(chat_id, time.time())

    async def _glance_due(self, chat_id: str) -> None:
        chat = self._chat(chat_id)
        async with chat.lock:
            now = time.time()
            last_check = self.glance_last_check.get(chat_id, 0.0)
            self.glance_last_check[chat_id] = now
            cooldown_left = max(0.0, chat.cooldown_until - now)
            logger.info(f"[OnCue][GLANCE] 到期: {chat_id} | 冷却剩 {cooldown_left:.0f}s | 窗口回复 {len(chat.reply_ts)}")
            if chat.last_activity_ts <= last_check:
                logger.info(f"[OnCue][GLANCE] 跳过: 自上次瞥屏后无新消息 | {chat_id}")
                self._schedule_glance(chat_id, now)
                return
            if now < chat.cooldown_until:
                logger.info(f"[OnCue][GLANCE] 跳过: 冷却中剩 {cooldown_left:.0f}s | {chat_id}")
                self._schedule_glance(chat_id, now, min_ts=chat.cooldown_until + 1)
                return
            if not self._trigger_ready(chat, "glance", now):
                until = float(chat.backoffs.get("glance", {}).get("until", now))
                logger.info(f"[OnCue][GLANCE] 跳过: 触发退避中剩 {max(0.0, until - now):.0f}s | {chat_id}")
                self._schedule_glance(chat_id, now, min_ts=until + 1)
                return
            if not self._reply_window_allowed(chat, now):
                logger.info(f"[OnCue][GLANCE] 跳过: 回复窗口已满 | 已发送 {len(chat.reply_ts)} 待发送 {len(chat.pending_replies)} 上限 {self._cfg_int('max_replies_per_window', 5)} | {chat_id}")
                self._schedule_glance(chat_id, now)
                return
            logger.info(f"[OnCue][GLANCE] 进入决策 | {chat_id}")
            prompt = await self._build_prompt(chat, "你刚忙完自己的事，顺手瞄了一眼群聊。", chat_id, chat_id.split(":", 1)[0])
            raw = await self._llm_decision(chat_id, prompt)
            decision = self._parse_decision(raw)
            if decision["should_reply"] and decision.get("need_kb"):
                decision["kb_text"] = await self._kb_retrieve(decision.get("kb_query", ""))
            if not decision["should_reply"]:
                now = time.time()
                mult = self._trigger_reject(chat, "glance", now)
                cooldown = self._cfg_int("no_reply_cooldown", 20) * mult
                chat.cooldown_until = max(chat.cooldown_until, now + cooldown)
                logger.info(f"[OnCue][GLANCE] 决定沉默: {decision['reason']} | 退避 x{mult} 冷却 {cooldown}s")
                self._schedule_glance(chat_id, now)
                return
            text = await self._glance_generate(chat, chat_id, decision)
            text = text.strip()
            if not text:
                now = time.time()
                mult = self._trigger_reject(chat, "glance", now)
                cooldown = self._cfg_int("no_reply_cooldown", 20) * mult
                chat.cooldown_until = max(chat.cooldown_until, now + cooldown)
                logger.info(f"[OnCue][GLANCE] 决定回复但生成为空 | 退避 x{mult} 冷却 {cooldown}s")
                self._schedule_glance(chat_id, now)
                return
            ok = await self.context.send_message(chat_id, MessageChain([Plain(text)]))
            now = time.time()
            if not ok:
                logger.warning(f"[OnCue] GLANCE 未找到可用平台，发送失败: {chat_id}")
                self._schedule_glance(chat_id, now)
                return
            self._record_reply(chat, text, now)
            self._trigger_success(chat, "glance")
            await self._append_assistant_history(chat_id, text)
            logger.info(f"[OnCue][GLANCE] 已主动开口: {text[:60]}")
            self._schedule_glance(chat_id, now)

    async def _append_assistant_history(self, umo: str, text: str) -> None:
        try:
            conv_mgr = self.context.conversation_manager
            cid = await conv_mgr.get_curr_conversation_id(umo)
            if not cid:
                return
            conv = await conv_mgr.get_conversation(umo, cid)
            if not conv:
                return
            history = conv.history or []
            if isinstance(history, str):
                history = json.loads(history or "[]")
            if not isinstance(history, list):
                return
            history.append({"role": "assistant", "content": text})
            await conv_mgr.update_conversation(umo, cid, history=history)
        except Exception as e:
            logger.error(f"[OnCue] GLANCE 写入历史失败: {e}")

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

    async def _character_card(self, umo: str, platform_name: str = "") -> str:
        manual = self._cfg_str("character_card")
        source = self._cfg_str("persona_source", "persona_id")
        if source == "persona_id":
            selected_persona_id = self._cfg_str("persona_id").strip()
            if not selected_persona_id:
                return manual
            selected_persona = self.context.persona_manager.get_persona_v3_by_id(selected_persona_id)
            if selected_persona and selected_persona.get("prompt"):
                card = str(selected_persona.get("prompt"))
                self.persona_cache[umo] = (time.time(), card)
                logger.info(f"[OnCue] 使用显式人格: {selected_persona_id} (chars={len(card)})")
                return card
            logger.error(f"[OnCue] 指定人格 ID 无可用 prompt，回退手工角色卡: persona_id={selected_persona_id}")
            return manual
        if source != "auto":
            return manual
        cached = self.persona_cache.get(umo)
        now = time.time()
        if cached and cached[1] and now - cached[0] < 10:
            return cached[1]
        try:
            conv_mgr = self.context.conversation_manager
            cid = await conv_mgr.get_curr_conversation_id(umo)
            conversation_persona_id = None
            if cid:
                conv = await conv_mgr.get_conversation(umo, cid)
                conversation_persona_id = getattr(conv, "persona_id", None) if conv else None
            cfg = self.context.get_config(umo=umo).get("provider_settings", {})
            default_persona_id = cfg.get("default_personality")
            persona_id, persona, forced_persona_id, _ = await self.context.persona_manager.resolve_selected_persona(
                umo=umo,
                conversation_persona_id=conversation_persona_id,
                platform_name=platform_name or umo.split(":", 1)[0],
                provider_settings=cfg,
            )
            if persona and persona.get("prompt"):
                card = str(persona.get("prompt"))
                self.persona_cache[umo] = (now, card)
                return card
            if conversation_persona_id == "[%None]":
                logger.info(f"[OnCue] persona auto 为空: 当前会话显式无人格; cid={cid}; resolved={persona_id}")
                return manual
            if conversation_persona_id:
                conv_persona = self.context.persona_manager.get_persona_v3_by_id(conversation_persona_id)
                if conv_persona and conv_persona.get("prompt"):
                    card = str(conv_persona.get("prompt"))
                    self.persona_cache[umo] = (now, card)
                    return card
            default_persona = await self.context.persona_manager.get_default_persona_v3(umo)
            if default_persona and default_persona.get("prompt") and default_persona.get("name") != "default":
                card = str(default_persona.get("prompt"))
                self.persona_cache[umo] = (now, card)
                return card
            personas_v3_count = len(getattr(self.context.persona_manager, "personas_v3", []) or [])
            logger.info(
                "[OnCue] persona auto 未取到角色卡，回退 manual: "
                f"cid={cid}; conversation_persona_id={conversation_persona_id}; "
                f"default_personality={default_persona_id}; resolved={persona_id}; "
                f"forced={forced_persona_id}; personas_v3={personas_v3_count}"
            )
        except Exception as e:
            logger.error(f"[OnCue] 自动读取人格失败，回退 manual: {e}")
        return manual

    def _event_to_item(self, event: AstrMessageEvent) -> dict:
        parts = []
        pure_text_parts = []
        image_count = 0
        other_count = 0
        for comp in event.get_messages():
            if isinstance(comp, Plain):
                text = (comp.text or "").strip()
                if text:
                    parts.append(text)
                    pure_text_parts.append(text)
            elif isinstance(comp, Image):
                image_count += 1
            elif isinstance(comp, At):
                name = getattr(comp, "name", "") or getattr(comp, "qq", "")
                if name:
                    parts.append(f"[At:{name}]")
            else:
                other_count += 1
        text = " ".join(parts).strip()
        pure_text = " ".join(pure_text_parts).strip()
        if image_count:
            text = (text + " " if text else "") + " ".join(["[图片]"] * image_count)
        if other_count:
            text = (text + " " if text else "") + " ".join(["[附件]"] * other_count)
        sender = getattr(getattr(event, "message_obj", None), "sender", None)
        sender_name = getattr(sender, "nickname", "") or str(event.get_sender_id())
        return {
            "ts": time.time(),
            "time_str": datetime.now().strftime("%H:%M:%S"),
            "sender_id": str(event.get_sender_id()),
            "sender_name": str(sender_name),
            "role": "user",
            "text": text or "[空消息]",
            "pure_text": pure_text,
            "active": bool(text or image_count or other_count),
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

    def _history_text(self, chat: ChatFacts) -> str:
        count = max(1, self._cfg_int("context_message_count", 20))
        rows = list(chat.messages)[-count:]
        return "\n".join(f"[{m['sender_name']}/{m['time_str']}]: {m['text']}" for m in rows)

    def _record_reply(self, chat: ChatFacts, text: str, now: float) -> None:
        """在持有会话锁时记录发送结果；自身发言只供决策参考。"""
        chat.messages.append({
            "ts": now,
            "time_str": datetime.fromtimestamp(now).strftime("%H:%M:%S"),
            "sender_id": "assistant",
            "sender_name": "Bot（你）",
            "role": "assistant",
            "text": text,
            "pure_text": "",
            "active": False,
        })
        chat.reply_ts.append(now)
        chat.last_reply_ts = now
        chat.cooldown_until = max(chat.cooldown_until, now + self._cfg_int("reply_cooldown", 10))

    async def _build_prompt(self, chat: ChatFacts, entry_context: str, umo: str, platform_name: str = "") -> str:
        template = self._cfg_str("decision_prompt") or DEFAULT_DECISION_PROMPT
        card = await self._decision_card(umo, platform_name) or "（未填写角色卡；按谨慎、不插话处理）"
        mapping = {
            "{character_card}": card,
            "{current_time}": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "{chat_history}": self._history_text(chat) or "（暂无）",
            "{entry_context}": entry_context,
        }
        for key, value in mapping.items():
            template = template.replace(key, value)
        if self._kb_available():
            template += "\n\n知识库联动：如果这次开口需要外部资料才能答好，在同一个 JSON 里增加 \"need_kb\": true 和 \"kb_query\": \"独立检索词\"；否则省略或设为 false。kb_query 不要照抄群聊原话。"
        return template

    async def _glance_generate(self, chat: ChatFacts, umo: str, decision: dict) -> str:
        speaker = decision.get("speaker", "").strip()
        mood = decision.get("mood", "").strip()
        if self._cfg_bool("enable_speaker_routing", False) and speaker:
            stage = f"当前由「{speaker}」回应。"
        else:
            stage = "当前由角色本人回应。"
        if mood:
            stage += f"此刻的心情：{mood}。"
        kb_text = str(decision.get("kb_text", "") or "").strip()
        if kb_text:
            stage += f"\n可参考知识库：\n{kb_text}"
        provider_id = await self.context.get_current_chat_provider_id(umo=umo)
        card = await self._character_card(umo, umo.split(":", 1)[0])
        prompt = GLANCE_GENERATE_PROMPT.replace("{character_card}", card or "（未填写角色卡）")
        prompt = prompt.replace("{current_time}", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        prompt = prompt.replace("{stage_direction}", stage)
        prompt = prompt.replace("{chat_history}", self._history_text(chat) or "（暂无）")
        try:
            resp = await self.context.llm_generate(chat_provider_id=provider_id, prompt=prompt)
            return resp.completion_text or ""
        except Exception as e:
            logger.error(f"[OnCue] GLANCE 生成失败: {e}")
            return ""

    async def _llm_decision(self, umo: str, prompt: str) -> str:
        provider_id = self._cfg_str("analyzer_provider").strip()
        try:
            if not provider_id:
                provider_id = await self.context.get_current_chat_provider_id(umo=umo)
            resp = await self.context.llm_generate(chat_provider_id=provider_id, prompt=prompt)
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
        should_reply = data.get("should_reply", False)
        if not isinstance(should_reply, bool):
            should_reply = str(should_reply).strip().lower() in {"true", "1", "yes", "y"}
        need_kb = data.get("need_kb", False)
        if not isinstance(need_kb, bool):
            need_kb = str(need_kb).strip().lower() in {"true", "1", "yes", "y"}
        decision = {
            "should_reply": should_reply,
            "speaker": str(data.get("speaker", "") or "").strip(),
            "mood": str(data.get("mood", "") or "").strip(),
            "reason": str(data.get("reason", "") or "").strip(),
            "need_kb": need_kb,
            "kb_query": str(data.get("kb_query", "") or "").strip(),
        }
        if decision["need_kb"] and not decision["kb_query"]:
            decision["need_kb"] = False
        if decision["should_reply"] and decision["reason"].lower() in {"", "无", "没有", "none", "null"}:
            decision["should_reply"] = False
            decision["reason"] = "正向理由为空，代码兜底不回复"
        return decision

    async def _decide_locked(self, event: AstrMessageEvent, chat: ChatFacts, entry_context: str, backoff_key: str | None) -> bool:
        now = time.time()
        if now < chat.cooldown_until:
            return False
        if backoff_key and not self._trigger_ready(chat, backoff_key, now):
            return False
        if not self._reply_window_allowed(chat, now):
            return False
        source = backoff_key or "observe"
        logger.info(f"[OnCue] 决策调用: source={source} | 已发送 {len(chat.reply_ts)} 待发送 {len(chat.pending_replies)} 上限 {self._cfg_int('max_replies_per_window', 5)}")
        prompt = await self._build_prompt(chat, entry_context, event.unified_msg_origin, event.get_platform_name())
        raw = await self._llm_decision(event.unified_msg_origin, prompt)
        decision = self._parse_decision(raw)
        if decision["should_reply"] and decision.get("need_kb"):
            decision["kb_text"] = await self._kb_retrieve(decision.get("kb_query", ""))
        now = time.time()
        if decision["should_reply"]:
            token = uuid4().hex
            chat.pending_replies[token] = now + max(1, self._cfg_int("reply_window_seconds", 300))
            decision["_reservation"] = token
            decision["_backoff_key"] = backoff_key
            chat.cooldown_until = max(chat.cooldown_until, now + self._cfg_int("reply_cooldown", 10))
            event.set_extra("oncue_decision", decision)
            event.is_at_or_wake_command = True
            logger.info(f"[OnCue] 决定回复: source={source} | {decision['reason']}")
            return True
        mult = self._trigger_reject(chat, backoff_key, now) if backoff_key else 1
        cooldown = self._cfg_int("no_reply_cooldown", 20) * mult
        chat.cooldown_until = max(chat.cooldown_until, now + cooldown)
        logger.info(f"[OnCue] 决定沉默: source={source} | 冷却 {cooldown}s | {decision['reason'] or raw[:80]}")
        return False

    async def _maybe_stat_trigger_locked(self, event: AstrMessageEvent, chat: ChatFacts, item: dict, now: float) -> None:
        if self._cfg_bool("enable_echo_trigger", False) and item["pure_text"] and item["text"] == item["pure_text"]:
            window = self._cfg_int("echo_window", 120)
            threshold = self._cfg_int("echo_threshold", 3)
            norm = re.sub(r"\s+", " ", item["pure_text"]).strip()
            if norm:
                count = sum(1 for m in chat.messages if m.get("role", "user") == "user" and now - m["ts"] <= window and re.sub(r"\s+", " ", m["pure_text"]).strip() == norm)
                key = f"echo:{norm}"
                if count >= threshold and self._trigger_ready(chat, key, now):
                    logger.info(f"[OnCue] 触发边命中: echo | {count}/{threshold} | {norm[:40]}")
                    await self._decide_locked(event, chat, "群里多人在复读同一句话。", key)
                    return
        if self._cfg_bool("enable_dense_trigger", False):
            window = self._cfg_int("dense_window", 120)
            recent = [m for m in chat.messages if m.get("role", "user") == "user" and now - m["ts"] <= window and m["active"]]
            participants = {m["sender_id"] for m in recent}
            if len(recent) >= self._cfg_int("dense_message_threshold", 10) and len(participants) >= self._cfg_int("dense_participant_threshold", 3):
                key = "dense"
                if self._trigger_ready(chat, key, now):
                    logger.info(f"[OnCue] 触发边命中: dense | 消息 {len(recent)}/{self._cfg_int('dense_message_threshold', 10)} | 人数 {len(participants)}/{self._cfg_int('dense_participant_threshold', 3)}")
                    await self._decide_locked(event, chat, "群里讨论突然变得热烈。", key)

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE, priority=-10)
    async def on_message(self, event: AstrMessageEvent):
        if not self._cfg_bool("enable", True):
            return
        if str(event.get_sender_id()) == str(event.get_self_id()):
            return
        chat = self._chat(event.unified_msg_origin)
        async with chat.lock:
            now = time.time()
            observing = self._is_observing(chat, now)
            item = self._event_to_item(event)
            chat.messages.append(item)
            if item["active"]:
                chat.last_activity_ts = now
                # 静默期活动只更新群聊活跃度，不能重新开启已经过期的观测窗。
                if observing and self._cfg_bool("observation_refresh", True):
                    chat.last_observation_activity_ts = now
            self._ensure_glance_schedule(event.unified_msg_origin, now)
            if self._is_forced(event, item):
                if self._cfg_bool("force_reply_when_summoned", True):
                    event.set_extra("oncue_decision", {"should_reply": True, "speaker": "", "mood": "", "reason": "被@/点名强制唤醒"})
                    event.is_at_or_wake_command = True
                    logger.info(f"[OnCue] 强制唤醒: {item['sender_name']} | {item['pure_text'][:60]}")
                elif event.is_at_or_wake_command:
                    event.stop_event()
                    logger.warning(f"[OnCue] 已 veto 被唤醒事件: {item['sender_name']} | {item['pure_text'][:60]}")
                return
            if observing:
                await self._decide_locked(event, chat, "你刚才已经开口过，现在还在观测窗内；判断这一幕是否值得再接话。", None)
            else:
                await self._maybe_stat_trigger_locked(event, chat, item, now)

    @filter.on_llm_request()
    async def inject_stage_direction(self, event: AstrMessageEvent, req: ProviderRequest):
        if not self._cfg_bool("enable", True):
            return
        decision = event.get_extra("oncue_decision")
        if not isinstance(decision, dict) or not decision.get("should_reply"):
            return
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
        req.system_prompt += "\n" + line

    @filter.on_decorating_result()
    async def mark_streaming_reply_sent(self, event: AstrMessageEvent):
        # AstrBot 的流式发送路径可能不触发 after_message_sent。
        result = event.get_result()
        if result and result.result_content_type == ResultContentType.STREAMING_FINISH:
            await self.mark_reply_sent(event)

    @filter.after_message_sent()
    async def mark_reply_sent(self, event: AstrMessageEvent):
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
            for comp in result.chain or []:
                if isinstance(comp, Plain):
                    if (comp.text or "").strip():
                        parts.append(comp.text.strip())
                elif isinstance(comp, Image):
                    parts.append("[图片]")
                elif not isinstance(comp, (At, Reply)):
                    parts.append("[附件]")
            text = " ".join(parts)
            if not text:
                return
            now = time.time()
            self._record_reply(chat, text, now)
            self._trigger_success(chat, decision.get("_backoff_key"))
            logger.info(f"[OnCue] 回复已发送: 进入观测 {self._cfg_int('observation_timeout', 120)}s | 冷却 {self._cfg_int('reply_cooldown', 10)}s")
