"""点名两种模式、配置读取和并发优先级。"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from test_timing import _event, plugin_module


class SummonTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.plugin = plugin_module.OnCuePlugin(SimpleNamespace(), {
            "force_reply_when_summoned": False,
            "message_wait_seconds": 0,
        })
        self.plugin._decision_card = AsyncMock(return_value="决策角色卡")
        self.plugin._llm_decision = AsyncMock(return_value='{"should_reply": true, "mood": "愉快", "reason": "想接话"}')
        self.event = _event("你想聊聊吗", summoned=True)
        self.chat = self.plugin._chat(self.event.unified_msg_origin)

    async def test_direct_mode_skips_decision_but_uses_common_prompt_hook(self):
        self.plugin.config["force_reply_when_summoned"] = True
        await self.plugin.on_message(self.event)
        self.plugin._llm_decision.assert_not_awaited()
        req = SimpleNamespace(prompt="原请求", system_prompt="AstrBot 人格")
        await self.plugin.inject_stage_direction(self.event, req)
        self.assertTrue(self.event.is_at_or_wake_command)
        self.assertTrue(req.system_prompt.startswith("AstrBot 人格"))
        self.assertIn("你想聊聊吗", req.prompt)
        self.assertNotIn("决策角色卡", req.system_prompt)

    async def test_decision_mode_can_reply_and_uses_same_direction_hook(self):
        await self.plugin.on_message(self.event)
        self.plugin._llm_decision.assert_awaited_once()
        self.assertTrue(self.event.is_at_or_wake_command)
        self.assertEqual(len(self.chat.pending_replies), 1)
        req = SimpleNamespace(prompt="原请求", system_prompt="AstrBot 人格")
        await self.plugin.inject_stage_direction(self.event, req)
        self.assertIn("愉快", req.system_prompt)
        self.assertIn("你想聊聊吗", req.prompt)

    async def test_negative_or_invalid_decision_does_not_leak_native_wake(self):
        for raw in ('{"should_reply": false}', "bad JSON"):
            with self.subTest(raw=raw):
                event = _event(summoned=True)
                self.plugin._llm_decision.return_value = raw
                await self.plugin.on_message(event)
                self.assertFalse(event.is_at_or_wake_command)
                self.assertIsNone(event.get_extra("oncue_decision"))
                self.assertFalse(event.stopped)  # 其它插件仍可处理此事件。
        self.assertEqual(self.chat.pending_replies, {})

    async def test_provider_exception_cannot_fall_through_to_native_reply(self):
        self.plugin._llm_decision.side_effect = RuntimeError("failed")
        with self.assertRaisesRegex(RuntimeError, "failed"):
            await self.plugin.on_message(self.event)
        self.assertFalse(self.event.is_at_or_wake_command)
        self.assertFalse(self.chat.decision_inflight)

    async def test_nickname_uses_decision_mode_too(self):
        self.plugin.config["wake_names"] = "小助手"
        event = _event("小助手，你觉得呢？")
        await self.plugin.on_message(event)
        self.plugin._llm_decision.assert_awaited_once()
        self.assertTrue(event.is_at_or_wake_command)

    async def test_summon_is_not_silently_dropped_by_autonomous_reply_limits(self):
        self.chat.cooldown_until = float("inf")
        self.plugin.config["max_replies_per_window"] = 0
        await self.plugin.on_message(self.event)
        self.plugin._llm_decision.assert_awaited_once()
        self.assertTrue(self.event.is_at_or_wake_command)

    async def test_old_decision_cleanup_cannot_clear_new_summon_owner(self):
        entered, release = asyncio.Event(), asyncio.Event()
        new_entered, new_release = asyncio.Event(), asyncio.Event()
        calls = 0

        async def decide(*args):
            nonlocal calls
            calls += 1
            if calls == 1:
                entered.set()
                await release.wait()
            else:
                new_entered.set()
                await new_release.wait()
            return '{"should_reply": true, "reason": "想接话"}'

        self.plugin._llm_decision.side_effect = decide
        old = _event()
        old_task = asyncio.create_task(self.plugin._decide(old, self.chat, "observe", None))
        new_task = None
        try:
            await asyncio.wait_for(entered.wait(), 1)
            new_task = asyncio.create_task(self.plugin.on_message(self.event))
            await asyncio.wait_for(new_entered.wait(), 1)
            owner = self.chat.decision_inflight
            release.set()
            await asyncio.wait_for(old_task, 1)
            self.assertIs(self.chat.decision_inflight, owner)
            self.assertIsNone(old.get_extra("oncue_decision"))
            new_release.set()
            await asyncio.wait_for(new_task, 1)
            self.assertTrue(self.event.is_at_or_wake_command)
            self.assertFalse(self.chat.decision_inflight)
        finally:
            release.set()
            new_release.set()
            await asyncio.gather(*(t for t in (old_task, new_task) if t), return_exceptions=True)

    async def test_stale_summon_waiting_for_lock_does_not_start_decision(self):
        self.chat.reply_version = 2
        self.assertFalse(await self.plugin._decide(self.event, self.chat, "点名", None, summoned_version=1))
        self.plugin._llm_decision.assert_not_awaited()

    async def test_summon_mode_reads_wake_section(self):
        self.plugin.config = {"config_wake": {"force_reply_when_summoned": False}}
        await self.plugin.on_message(self.event)
        self.plugin._llm_decision.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
