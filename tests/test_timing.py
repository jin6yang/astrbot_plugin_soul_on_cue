"""时序回归测试：替换 AstrBot API 和外部调用，直接运行插件逻辑。"""

import asyncio
import importlib.util
import json
import logging
import sys
import unittest
from enum import Enum, auto
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, patch


class Plain:
    def __init__(self, text):
        self.text = text


class Star:
    def __init__(self, context):
        self.context = context


class ResultContentType(Enum):
    GENERAL_RESULT = auto()
    LLM_RESULT = auto()
    STREAMING_RESULT = auto()
    STREAMING_FINISH = auto()


def _event(text="hello", *, summoned=False):
    extras = {}
    event = SimpleNamespace(
        unified_msg_origin="test:GroupMessage:1",
        is_at_or_wake_command=summoned,
        message_obj=SimpleNamespace(sender=SimpleNamespace(nickname="Alice")),
        get_sender_id=lambda: "alice",
        get_self_id=lambda: "bot",
        get_platform_name=lambda: "test",
        get_messages=lambda: [Plain(text)] if text else [],
        get_extra=extras.get,
        set_extra=extras.__setitem__,
        _has_send_oper=False,
        result=SimpleNamespace(chain=[], result_content_type=ResultContentType.LLM_RESULT),
    )
    event.get_result = lambda: event.result
    return event


def _load_plugin():
    def decorator(*args, **kwargs):
        return lambda handler: handler

    api = {
        "astrbot": {},
        "astrbot.api": {"AstrBotConfig": dict, "logger": logging.getLogger(__name__)},
        "astrbot.api.event": {
            "AstrMessageEvent": object,
            "filter": SimpleNamespace(
                EventMessageType=SimpleNamespace(GROUP_MESSAGE=1),
                event_message_type=decorator,
                on_llm_request=decorator,
                on_decorating_result=decorator,
                after_message_sent=decorator,
            ),
        },
        "astrbot.api.provider": {"ProviderRequest": object},
        "astrbot.api.star": {"Context": object, "Star": Star, "StarTools": object},
        "astrbot.core": {},
        "astrbot.core.message": {},
        "astrbot.core.message.components": {
            "Plain": Plain,
            "Image": type("Image", (), {}),
            "At": type("At", (), {}),
            "Reply": type("Reply", (), {}),
        },
        "astrbot.core.message.message_event_result": {"MessageChain": list, "ResultContentType": ResultContentType},
    }
    modules = {}
    for name, attributes in api.items():
        modules[name] = ModuleType(name)
        modules[name].__dict__.update(attributes)
    path = Path(__file__).resolve().parents[1] / "main.py"
    spec = importlib.util.spec_from_file_location("oncue_timing_tests", path)
    module = importlib.util.module_from_spec(spec)
    modules[spec.name] = module
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


plugin_module = _load_plugin()


class TimingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.now = 1000.0
        clock = patch.object(plugin_module, "time", SimpleNamespace(time=lambda: self.now))
        clock.start()
        self.addCleanup(clock.stop)
        self.plugin = plugin_module.OnCuePlugin(
            SimpleNamespace(send_message=AsyncMock(return_value=True)),
            {"enable_glance": False, "enable_dense_trigger": False, "enable_echo_trigger": False},
        )
        self.event = _event()
        self.chat_id = self.event.unified_msg_origin
        self.chat = self.plugin._chat(self.chat_id)
        self.plugin._build_prompt = AsyncMock(return_value="prompt")
        self.plugin._llm_decision = AsyncMock(return_value='{"should_reply": false}')
        self.plugin._glance_generate = AsyncMock(return_value="reply")
        self.plugin._append_assistant_history = AsyncMock()
        self.plugin._glance_interval_seconds = lambda: 900.0

    async def test_messages_cannot_start_observation_without_a_reply(self):
        await self.plugin.on_message(self.event)
        self.assertFalse(self.plugin._is_observing(self.chat, self.now))
        self.plugin._llm_decision.assert_not_awaited()
        self.assertEqual(len(self.chat.messages), 1)
        self.assertEqual(self.chat.last_activity_ts, self.now)

    async def test_expired_observation_stays_silent_across_new_messages(self):
        self.chat.last_reply_ts = 100.0
        for timestamp in (1000.0, 1001.0, 1100.0):
            self.now = timestamp
            await self.plugin.on_message(_event())
            self.assertFalse(self.plugin._is_observing(self.chat, self.now))
        self.plugin._llm_decision.assert_not_awaited()
        self.assertEqual(len(self.chat.messages), 3)
        self.assertEqual(self.chat.last_activity_ts, 1100.0)

    async def test_activity_refreshes_only_an_unexpired_observation(self):
        self.chat.last_reply_ts = 1000.0
        for timestamp in (1100.0, 1200.0):
            self.now = timestamp
            await self.plugin.on_message(_event())
        self.assertTrue(self.plugin._is_observing(self.chat, 1319.0))
        for timestamp in (1320.0, 1330.0):
            self.now = timestamp
            await self.plugin.on_message(_event())
            self.assertFalse(self.plugin._is_observing(self.chat, self.now))
        self.assertEqual(self.plugin._llm_decision.await_count, 2)

    async def test_disabled_refresh_keeps_the_original_deadline(self):
        self.plugin.config["observation_refresh"] = False
        self.chat.last_reply_ts = 1000.0
        for timestamp in (1100.0, 1120.0, 1130.0):
            self.now = timestamp
            await self.plugin.on_message(_event())
        self.assertEqual(self.plugin._llm_decision.await_count, 1)
        self.assertFalse(self.plugin._is_observing(self.chat, self.now))

    async def test_empty_message_does_not_extend_observation(self):
        self.chat.last_reply_ts = 1000.0
        self.now = 1100.0
        await self.plugin.on_message(_event(""))
        self.assertFalse(self.plugin._is_observing(self.chat, 1120.0))
        self.assertEqual(self.chat.last_activity_ts, 0.0)

    async def test_summoned_reply_can_start_a_new_observation(self):
        self.chat.last_reply_ts = 100.0
        event = _event(summoned=True)
        await self.plugin.on_message(event)
        self.assertTrue(event.get_extra("oncue_decision")["should_reply"])
        self.assertFalse(self.plugin._is_observing(self.chat, self.now))
        self.now = 1030.0
        event._has_send_oper = True
        event.result.chain = [Plain("reply")]
        await self.plugin.mark_reply_sent(event)
        self.assertTrue(self.plugin._is_observing(self.chat, 1149.0))
        self.assertFalse(self.plugin._is_observing(self.chat, 1150.0))
        self.plugin._llm_decision.assert_not_awaited()

    async def test_message_checks_expiry_after_waiting_for_chat_lock(self):
        self.chat.last_reply_ts = 890.0
        async with self.chat.lock:
            pending = asyncio.create_task(self.plugin.on_message(self.event))
            await asyncio.sleep(0)
            self.assertFalse(pending.done())
            self.now = 1020.0
        await pending
        self.plugin._llm_decision.assert_not_awaited()
        self.assertFalse(self.plugin._is_observing(self.chat, self.now))

    async def test_slow_positive_decision_and_kb_keep_full_cooldown(self):
        async def decide(*args):
            self.now += 30.0
            return json.dumps({"should_reply": True, "reason": "topic", "need_kb": True, "kb_query": "query"})

        async def retrieve(*args):
            self.now += 15.0
            return "knowledge"

        self.plugin._llm_decision.side_effect = decide
        self.plugin._kb_retrieve = AsyncMock(side_effect=retrieve)
        self.assertTrue(await self.plugin._decide_locked(self.event, self.chat, "observe", None))
        self.assertEqual(self.chat.cooldown_until, 1055.0)
        self.assertEqual(list(self.chat.reply_ts), [])
        self.assertEqual(len(self.chat.pending_replies), 1)
        self.now = 1054.0
        self.assertFalse(await self.plugin._decide_locked(_event(), self.chat, "observe", None))
        self.plugin._llm_decision.assert_awaited_once()
        self.now = 1055.0
        self.assertTrue(await self.plugin._decide_locked(_event(), self.chat, "observe", None))
        self.assertEqual(self.plugin._llm_decision.await_count, 2)

    async def test_slow_negative_decision_starts_cooldown_and_backoff_on_completion(self):
        async def decide(*args):
            self.now = 1030.0
            return '{"should_reply": false}'

        self.plugin._llm_decision.side_effect = decide
        self.assertFalse(await self.plugin._decide_locked(self.event, self.chat, "dense", "dense"))
        self.assertEqual(self.chat.cooldown_until, 1070.0)
        self.assertFalse(self.plugin._trigger_ready(self.chat, "dense", 1069.0))
        self.assertTrue(self.plugin._trigger_ready(self.chat, "dense", 1070.0))

    async def test_glance_slow_send_uses_completion_time(self):
        async def decide(*args):
            self.now += 15.0
            return '{"should_reply": true, "reason": "topic"}'

        async def generate(*args):
            self.now += 25.0
            return "reply"

        async def send(*args):
            self.now += 20.0
            return True

        self.chat.last_activity_ts = 950.0
        self.plugin._llm_decision.side_effect = decide
        self.plugin._glance_generate.side_effect = generate
        self.plugin.context.send_message.side_effect = send
        await self.plugin._glance_due(self.chat_id)
        self.assertEqual(self.chat.last_reply_ts, 1060.0)
        self.assertEqual(list(self.chat.reply_ts), [1060.0])
        self.assertEqual(self.chat.cooldown_until, 1070.0)
        self.assertEqual(self.plugin.glance_next_due[self.chat_id], 1960.0)
        self.plugin._append_assistant_history.assert_awaited_once_with(self.chat_id, "reply")

    async def test_glance_negative_decision_uses_completion_time(self):
        async def decide(*args):
            self.now = 1030.0
            return '{"should_reply": false}'

        self.chat.last_activity_ts = 950.0
        self.plugin._llm_decision.side_effect = decide
        await self.plugin._glance_due(self.chat_id)
        self.assertEqual(self.chat.cooldown_until, 1070.0)
        self.assertEqual(self.chat.backoffs["glance"]["until"], 1070.0)
        self.assertEqual(self.plugin.glance_next_due[self.chat_id], 1930.0)
        self.plugin._glance_generate.assert_not_awaited()

    async def test_glance_empty_generation_uses_completion_time(self):
        async def generate(*args):
            self.now = 1050.0
            return ""

        self.chat.last_activity_ts = 950.0
        self.plugin._llm_decision.return_value = '{"should_reply": true, "reason": "topic"}'
        self.plugin._glance_generate.side_effect = generate
        await self.plugin._glance_due(self.chat_id)
        self.assertEqual(self.chat.cooldown_until, 1090.0)
        self.assertEqual(self.chat.backoffs["glance"]["until"], 1090.0)
        self.assertEqual(self.plugin.glance_next_due[self.chat_id], 1950.0)
        self.plugin.context.send_message.assert_not_awaited()

    async def test_glance_checks_cooldown_after_waiting_for_chat_lock(self):
        self.chat.last_activity_ts = 950.0
        self.chat.cooldown_until = 1010.0
        async with self.chat.lock:
            pending = asyncio.create_task(self.plugin._glance_due(self.chat_id))
            await asyncio.sleep(0)
            self.assertFalse(pending.done())
            self.now = 1020.0
        await pending
        self.plugin._llm_decision.assert_awaited_once()

    async def test_reply_callback_starts_cooldown_after_waiting_for_chat_lock(self):
        self.event.set_extra("oncue_decision", {"should_reply": True})
        self.event._has_send_oper = True
        self.event.result.chain = [Plain("reply")]
        async with self.chat.lock:
            pending = asyncio.create_task(self.plugin.mark_reply_sent(self.event))
            await asyncio.sleep(0)
            self.assertFalse(pending.done())
            self.now = 1030.0
        await pending
        self.assertEqual(self.chat.cooldown_until, 1040.0)
        self.assertEqual(self.chat.last_reply_ts, 1030.0)


if __name__ == "__main__":
    unittest.main()
