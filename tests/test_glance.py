"""验证 GLANCE 跳过或决策失败后，不会丢失尚未检查的群聊活动。"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from test_timing import _event, plugin_module


class GlanceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.now = 1000.0
        clock = patch.object(plugin_module, "time", SimpleNamespace(time=lambda: self.now))
        clock.start()
        self.addCleanup(clock.stop)
        self.plugin = plugin_module.OnCuePlugin(
            SimpleNamespace(send_message=AsyncMock(return_value=True)),
            {"enable_glance": True, "enable_dense_trigger": False, "enable_echo_trigger": False},
        )
        self.chat_id = _event().unified_msg_origin
        self.chat = self.plugin._chat(self.chat_id)
        self.chat.last_activity_ts = 950.0
        self.plugin.glance_last_check[self.chat_id] = 900.0
        self.plugin._build_prompt = AsyncMock(return_value="prompt")
        self.plugin._llm_decision = AsyncMock(return_value='{"should_reply": false}')
        self.plugin._glance_generate = AsyncMock(return_value="reply")
        self.plugin._append_assistant_history = AsyncMock()
        self.plugin._glance_interval_seconds = lambda: 900.0

    async def assert_skipped_activity_is_checked_next_time(self):
        await self.plugin._glance_due(self.chat_id)
        self.plugin._build_prompt.assert_not_awaited()
        self.plugin._llm_decision.assert_not_awaited()
        self.assertEqual(self.plugin.glance_last_check[self.chat_id], 900.0)
        self.now = self.plugin.glance_next_due[self.chat_id]
        await self.plugin._glance_due(self.chat_id)
        self.plugin._llm_decision.assert_awaited_once()
        self.assertEqual(self.plugin.glance_last_check[self.chat_id], 950.0)

    async def test_cooldown_skip_preserves_unchecked_activity(self):
        self.chat.cooldown_until = 1100.0
        await self.assert_skipped_activity_is_checked_next_time()

    async def test_backoff_skip_preserves_unchecked_activity(self):
        self.chat.backoffs["glance"] = {"count": 1, "until": 1100.0}
        await self.assert_skipped_activity_is_checked_next_time()

    async def test_full_reply_window_preserves_unchecked_activity(self):
        self.plugin.config["max_replies_per_window"] = 1
        self.chat.reply_ts.append(self.now)
        await self.assert_skipped_activity_is_checked_next_time()

    async def test_pending_reply_limit_preserves_unchecked_activity(self):
        self.plugin.config["max_replies_per_window"] = 1
        self.chat.pending_replies["pending"] = 1100.0
        await self.assert_skipped_activity_is_checked_next_time()

    async def test_no_new_activity_does_not_advance_progress_or_call_model(self):
        self.chat.last_activity_ts = 900.0
        for _ in range(2):
            await self.plugin._glance_due(self.chat_id)
            self.assertEqual(self.plugin.glance_last_check[self.chat_id], 900.0)
            self.now = self.plugin.glance_next_due[self.chat_id]
        self.plugin._llm_decision.assert_not_awaited()

    async def test_silence_checks_activity_once_and_new_message_allows_another_check(self):
        async def decide(*args):
            self.now += 30.0
            return '{"should_reply": false}'

        self.plugin._llm_decision.side_effect = decide
        await self.plugin._glance_due(self.chat_id)
        self.assertEqual(self.plugin.glance_last_check[self.chat_id], 950.0)
        self.now = self.plugin.glance_next_due[self.chat_id]
        await self.plugin._glance_due(self.chat_id)
        self.plugin._llm_decision.assert_awaited_once()
        await self.plugin.on_message(_event("new topic"))
        activity_ts = self.chat.last_activity_ts
        self.now = self.plugin.glance_next_due[self.chat_id]
        await self.plugin._glance_due(self.chat_id)
        self.assertEqual(self.plugin._llm_decision.await_count, 2)
        self.assertEqual(self.plugin.glance_last_check[self.chat_id], activity_ts)
        self.plugin.context.send_message.assert_not_awaited()

    async def test_invalid_decisions_preserve_activity_until_a_valid_decision(self):
        invalid_results = (
            "", "not JSON", "{broken}", "{}",
            '{"should_reply": null}', '{"should_reply": "maybe"}',
        )
        for raw in invalid_results:
            with self.subTest(raw=raw):
                self.plugin._llm_decision.return_value = raw
                await self.plugin._glance_due(self.chat_id)
                self.assertEqual(self.plugin.glance_last_check[self.chat_id], 900.0)
                self.now = self.plugin.glance_next_due[self.chat_id]
        self.plugin._llm_decision.return_value = '```json\n{"should_reply": "false"}\n```'
        await self.plugin._glance_due(self.chat_id)
        self.assertEqual(self.plugin.glance_last_check[self.chat_id], 950.0)
        self.assertEqual(self.plugin._llm_decision.await_count, len(invalid_results) + 1)
        self.plugin.context.send_message.assert_not_awaited()

    async def test_provider_failure_preserves_activity_for_next_scheduled_check(self):
        self.plugin.config["analyzer_provider"] = "test-provider"
        self.plugin.context.llm_generate = AsyncMock(side_effect=[
            RuntimeError("provider unavailable"),
            SimpleNamespace(completion_text='{"should_reply": false}'),
        ])
        del self.plugin._llm_decision
        with self.assertLogs(plugin_module.logger, level="ERROR"):
            await self.plugin._glance_due(self.chat_id)
        self.assertEqual(self.plugin.glance_last_check[self.chat_id], 900.0)
        self.now = self.plugin.glance_next_due[self.chat_id]
        await self.plugin._glance_due(self.chat_id)
        self.assertEqual(self.plugin.context.llm_generate.await_count, 2)
        self.assertEqual(self.plugin.glance_last_check[self.chat_id], 950.0)

    async def test_prompt_failure_does_not_mark_activity_checked(self):
        self.plugin._build_prompt.side_effect = RuntimeError("prompt failed")
        with self.assertRaisesRegex(RuntimeError, "prompt failed"):
            await self.plugin._glance_due(self.chat_id)
        self.assertEqual(self.plugin.glance_last_check[self.chat_id], 900.0)
        self.plugin._llm_decision.assert_not_awaited()

    async def test_cancelled_decision_does_not_mark_activity_checked(self):
        self.plugin._llm_decision.side_effect = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await self.plugin._glance_due(self.chat_id)
        self.assertEqual(self.plugin.glance_last_check[self.chat_id], 900.0)
        self.assertFalse(self.chat.lock.locked())

    async def test_successful_reply_does_not_make_old_activity_new_again(self):
        self.plugin._llm_decision.return_value = '{"should_reply": true, "reason": "topic"}'
        await self.plugin._glance_due(self.chat_id)
        self.assertEqual(self.plugin.glance_last_check[self.chat_id], 950.0)
        self.now = self.plugin.glance_next_due[self.chat_id]
        await self.plugin._glance_due(self.chat_id)
        self.plugin._llm_decision.assert_awaited_once()
        self.plugin.context.send_message.assert_awaited_once()

    async def test_valid_decision_is_checked_even_if_delivery_fails(self):
        self.plugin._llm_decision.return_value = '{"should_reply": true, "reason": "topic"}'
        self.plugin.context.send_message.return_value = False
        with self.assertLogs(plugin_module.logger, level="WARNING"):
            await self.plugin._glance_due(self.chat_id)
        self.assertEqual(self.plugin.glance_last_check[self.chat_id], 950.0)
        self.now = self.plugin.glance_next_due[self.chat_id]
        await self.plugin._glance_due(self.chat_id)
        self.plugin._llm_decision.assert_awaited_once()
        self.assertEqual(list(self.chat.reply_ts), [])


if __name__ == "__main__":
    unittest.main()
