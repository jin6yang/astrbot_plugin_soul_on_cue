"""验证回复缓存、发送计数和流式完成的边界。"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from test_timing import Plain, ResultContentType, _event, plugin_module


class ReplyTests(unittest.IsolatedAsyncioTestCase):
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
        self.plugin._llm_decision = AsyncMock(return_value='{"should_reply": true, "reason": "topic"}')
        self.plugin._glance_generate = AsyncMock(return_value="glance reply")
        self.plugin._append_assistant_history = AsyncMock()
        self.plugin._glance_interval_seconds = lambda: 900.0

    async def decide(self):
        return await self.plugin._decide_locked(self.event, self.chat, "observe", None)

    def sent_result(self, text="bot reply"):
        self.event._has_send_oper = True
        self.event.result.chain = [Plain(text)]

    async def test_decision_reserves_capacity_without_recording_a_sent_reply(self):
        self.assertTrue(await self.decide())
        self.assertEqual(len(self.chat.pending_replies), 1)
        self.assertEqual(list(self.chat.reply_ts), [])
        self.assertEqual(self.chat.last_reply_ts, 0.0)
        self.assertEqual(list(self.chat.messages), [])

    async def test_success_records_actual_reply_and_send_time(self):
        await self.plugin.on_message(self.event)
        self.assertTrue(await self.decide())
        self.now = 1040.0
        self.sent_result()
        await self.plugin.mark_reply_sent(self.event)
        self.assertEqual(list(self.chat.reply_ts), [1040.0])
        self.assertEqual(self.chat.pending_replies, {})
        self.assertEqual(self.chat.last_reply_ts, 1040.0)
        self.assertEqual(self.chat.cooldown_until, 1050.0)
        self.assertEqual(self.chat.last_activity_ts, 1000.0)
        self.assertEqual([row["role"] for row in self.chat.messages], ["user", "assistant"])
        history = self.plugin._history_text(self.chat)
        self.assertIn("hello", history)
        self.assertIn("Bot（你）", history)
        self.assertIn("bot reply", history)
        self.assertNotIn("prompt", history)

    async def test_nonempty_unsent_result_does_not_count_or_enter_observation(self):
        self.assertTrue(await self.decide())
        self.event.result.chain = [Plain("delivery failed")]
        await self.plugin.mark_reply_sent(self.event)
        self.assertEqual(list(self.chat.reply_ts), [])
        self.assertEqual(list(self.chat.messages), [])
        self.assertEqual(self.chat.pending_replies, {})
        self.assertFalse(self.plugin._is_observing(self.chat, self.now))
        self.assertIsNone(self.event.get_extra("oncue_decision"))

    async def test_empty_or_header_only_result_does_not_count(self):
        for chain in ([], [Plain("  ")], [plugin_module.At(), plugin_module.Reply()]):
            with self.subTest(chain=chain):
                self.event.set_extra("oncue_decision", {"should_reply": True})
                self.event._has_send_oper = True
                self.event.result.chain = chain
                await self.plugin.mark_reply_sent(self.event)
                self.assertEqual(list(self.chat.reply_ts), [])
                self.assertEqual(list(self.chat.messages), [])
                self.assertEqual(self.chat.last_reply_ts, 0.0)

    async def test_sent_image_is_cached_as_a_placeholder(self):
        self.event.set_extra("oncue_decision", {"should_reply": True})
        self.event._has_send_oper = True
        self.event.result.chain = [plugin_module.Image()]
        await self.plugin.mark_reply_sent(self.event)
        self.assertEqual(self.chat.messages[-1]["text"], "[图片]")
        self.assertEqual(list(self.chat.reply_ts), [1000.0])

    async def test_pending_reply_prevents_overbooking_and_failed_callback_releases_it(self):
        self.plugin.config["max_replies_per_window"] = 1
        self.assertTrue(await self.decide())
        self.now = 1020.0
        self.assertFalse(await self.plugin._decide_locked(_event(), self.chat, "observe", None))
        self.plugin._llm_decision.assert_awaited_once()
        await self.plugin.mark_reply_sent(self.event)
        self.assertTrue(await self.plugin._decide_locked(_event(), self.chat, "observe", None))
        self.assertEqual(list(self.chat.reply_ts), [])
        self.assertEqual(len(self.chat.pending_replies), 1)

    async def test_missing_callback_reservation_expires_without_becoming_a_reply(self):
        self.plugin.config["max_replies_per_window"] = 1
        self.assertTrue(await self.decide())
        self.assertFalse(self.plugin._reply_window_allowed(self.chat, 1299.0))
        self.assertTrue(self.plugin._reply_window_allowed(self.chat, 1300.0))
        self.assertEqual(self.chat.pending_replies, {})
        self.assertEqual(list(self.chat.reply_ts), [])

    async def test_successful_reply_counts_for_a_full_window_from_send_time(self):
        self.plugin.config["max_replies_per_window"] = 1
        await self.decide()
        self.now = 1100.0
        self.sent_result()
        await self.plugin.mark_reply_sent(self.event)
        self.assertFalse(self.plugin._reply_window_allowed(self.chat, 1350.0))
        self.assertTrue(self.plugin._reply_window_allowed(self.chat, 1401.0))

    async def test_concurrent_completion_callbacks_record_only_once(self):
        await self.decide()
        self.sent_result()
        async with self.chat.lock:
            callbacks = [asyncio.create_task(self.plugin.mark_reply_sent(self.event)) for _ in range(2)]
            await asyncio.sleep(0)
            self.assertTrue(all(not callback.done() for callback in callbacks))
        await asyncio.gather(*callbacks)
        await self.plugin.mark_reply_sent(self.event)
        self.assertEqual(len(self.chat.reply_ts), 1)
        self.assertEqual(len(self.chat.messages), 1)

    async def test_streaming_finish_records_once_without_a_normal_sent_callback(self):
        await self.decide()
        self.sent_result("streamed reply")
        self.event.result.result_content_type = ResultContentType.STREAMING_FINISH
        await self.plugin.mark_streaming_reply_sent(self.event)
        await self.plugin.mark_streaming_reply_sent(self.event)
        await self.plugin.mark_reply_sent(self.event)
        self.assertEqual(len(self.chat.reply_ts), 1)
        self.assertEqual(self.chat.messages[-1]["text"], "streamed reply")
        self.assertEqual(self.chat.pending_replies, {})

    async def test_regular_pre_send_hook_does_not_record_or_release_reservation(self):
        await self.decide()
        self.event.result.chain = [Plain("not sent yet")]
        await self.plugin.mark_streaming_reply_sent(self.event)
        self.assertEqual(list(self.chat.reply_ts), [])
        self.assertEqual(len(self.chat.pending_replies), 1)
        self.assertIsNotNone(self.event.get_extra("oncue_decision"))

    async def test_streaming_finish_without_send_marker_does_not_count(self):
        await self.decide()
        self.event.result.chain = [Plain("not delivered")]
        self.event.result.result_content_type = ResultContentType.STREAMING_FINISH
        await self.plugin.mark_streaming_reply_sent(self.event)
        self.assertEqual(list(self.chat.reply_ts), [])
        self.assertEqual(list(self.chat.messages), [])

    async def test_glance_success_updates_both_histories_without_new_user_activity(self):
        self.chat.last_activity_ts = 950.0
        await self.plugin._glance_due(self.chat_id)
        self.assertEqual(self.chat.messages[-1]["text"], "glance reply")
        self.assertEqual(self.chat.messages[-1]["role"], "assistant")
        self.assertEqual(self.chat.last_activity_ts, 950.0)
        self.assertEqual(list(self.chat.reply_ts), [1000.0])
        self.plugin._append_assistant_history.assert_awaited_once_with(self.chat_id, "glance reply")

    async def test_glance_failed_send_does_not_update_either_history(self):
        self.chat.last_activity_ts = 950.0
        self.plugin.context.send_message.return_value = False
        with self.assertLogs(plugin_module.logger, level="WARNING"):
            await self.plugin._glance_due(self.chat_id)
        self.assertEqual(list(self.chat.messages), [])
        self.assertEqual(list(self.chat.reply_ts), [])
        self.assertEqual(self.chat.last_reply_ts, 0.0)
        self.plugin._append_assistant_history.assert_not_awaited()

    async def test_bot_replies_do_not_contribute_to_echo_or_density(self):
        self.plugin.config.update(
            enable_echo_trigger=True,
            echo_threshold=3,
            enable_dense_trigger=True,
            dense_message_threshold=3,
            dense_participant_threshold=2,
        )
        for _ in range(3):
            self.plugin._record_reply(self.chat, "same text", self.now)
        event = _event("same text")
        item = self.plugin._event_to_item(event)
        self.chat.messages.append(item)
        self.plugin._decide_locked = AsyncMock(return_value=False)
        await self.plugin._maybe_stat_trigger_locked(event, self.chat, item, self.now)
        self.plugin._decide_locked.assert_not_awaited()

    async def test_trigger_backoff_resets_only_after_a_sent_reply(self):
        self.chat.backoffs["dense"] = {"count": 2, "until": 0.0}
        self.assertTrue(await self.plugin._decide_locked(self.event, self.chat, "dense", "dense"))
        self.assertIn("dense", self.chat.backoffs)
        self.sent_result()
        await self.plugin.mark_reply_sent(self.event)
        self.assertNotIn("dense", self.chat.backoffs)

    async def test_summoned_reply_is_counted_without_a_reservation(self):
        event = _event(summoned=True)
        await self.plugin.on_message(event)
        event._has_send_oper = True
        event.result.chain = [Plain("summoned reply")]
        await self.plugin.mark_reply_sent(event)
        self.assertEqual(list(self.chat.reply_ts), [1000.0])
        self.assertEqual(self.chat.pending_replies, {})
        self.assertEqual(self.chat.messages[-1]["text"], "summoned reply")


if __name__ == "__main__":
    unittest.main()
