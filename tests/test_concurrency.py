"""用暂停的外部调用验证会话锁、决策快照和并发完成边界。"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from test_timing import Plain, _event, plugin_module


REPLY = '{"should_reply": true, "reason": "topic"}'
SILENT = '{"should_reply": false}'
WITH_KB = '{"should_reply": true, "reason": "topic", "need_kb": true, "kb_query": "query"}'


class ConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.now = 1000.0
        clock = patch.object(plugin_module, "time", SimpleNamespace(time=lambda: self.now))
        clock.start()
        self.addCleanup(clock.stop)
        self.plugin = plugin_module.OnCuePlugin(
            SimpleNamespace(
                send_message=AsyncMock(return_value=True),
                get_current_chat_provider_id=AsyncMock(return_value="main"),
                llm_generate=AsyncMock(return_value=SimpleNamespace(completion_text="glance reply")),
            ),
            {"enable_glance": True, "enable_dense_trigger": False, "enable_echo_trigger": False},
        )
        self.event = _event("first topic")
        self.chat_id = self.event.unified_msg_origin
        self.chat = self.plugin._chat(self.chat_id)
        self.chat.last_reply_ts = 950.0
        self.chat.last_activity_ts = 950.0
        self.chat.messages.append(self.plugin._event_to_item(_event("old topic")))
        self.plugin._decision_card = AsyncMock(return_value="character card")
        self.plugin._character_card = AsyncMock(return_value="character card")
        self.plugin._llm_decision = AsyncMock(return_value=SILENT)
        self.plugin._kb_retrieve = AsyncMock(return_value="knowledge")
        self.plugin._append_assistant_history = AsyncMock()
        self.plugin._glance_interval_seconds = lambda: 900.0
        self.tasks = []

    async def asyncTearDown(self):
        for task in self.tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)

    def start(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.tasks.append(task)
        return task

    def block(self, mock, result):
        entered, release = asyncio.Event(), asyncio.Event()

        async def delayed(*args, **kwargs):
            entered.set()
            await release.wait()
            return result

        mock.side_effect = delayed
        return entered, release

    async def test_slow_decision_allows_cache_updates_without_a_second_decision(self):
        entered, release = self.block(self.plugin._llm_decision, SILENT)
        task = self.start(self.plugin.on_message(self.event))
        await asyncio.wait_for(entered.wait(), 1)
        self.assertFalse(self.chat.lock.locked())
        self.now = 1001.0
        await asyncio.wait_for(self.plugin.on_message(_event("second topic")), 1)
        self.assertEqual(self.chat.messages[-1]["text"], "second topic")
        self.assertEqual(self.chat.last_activity_ts, 1001.0)
        self.plugin._llm_decision.assert_awaited_once()
        self.assertFalse(task.done())
        release.set()
        await task
        self.assertFalse(self.chat.decision_inflight)
        self.now = 1030.0
        await self.plugin.on_message(_event("third topic"))
        self.assertEqual(self.plugin._llm_decision.await_count, 2)

    async def test_summon_during_decision_is_immediate_and_invalidates_old_reply(self):
        entered, release = self.block(self.plugin._llm_decision, REPLY)
        task = self.start(self.plugin.on_message(self.event))
        await asyncio.wait_for(entered.wait(), 1)
        summoned = _event("answer me", summoned=True)
        await asyncio.wait_for(self.plugin.on_message(summoned), 1)
        self.assertTrue(summoned.get_extra("oncue_decision")["should_reply"])
        self.assertFalse(task.done())
        release.set()
        await task
        self.assertIsNone(self.event.get_extra("oncue_decision"))
        self.assertFalse(self.event.is_at_or_wake_command)
        self.assertEqual(self.chat.pending_replies, {})
        self.assertFalse(self.chat.decision_inflight)

    async def test_stale_silence_does_not_add_backoff_after_a_summon(self):
        entered, release = self.block(self.plugin._llm_decision, SILENT)
        task = self.start(self.plugin._decide(self.event, self.chat, "dense", "dense"))
        await asyncio.wait_for(entered.wait(), 1)
        await asyncio.wait_for(self.plugin.on_message(_event(summoned=True)), 1)
        release.set()
        self.assertFalse(await task)
        self.assertEqual(self.chat.cooldown_until, 0.0)
        self.assertEqual(self.chat.backoffs, {})

    async def test_reply_callback_can_finish_during_decision_and_invalidate_it(self):
        entered, release = self.block(self.plugin._llm_decision, REPLY)
        task = self.start(self.plugin.on_message(self.event))
        await asyncio.wait_for(entered.wait(), 1)
        previous = _event()
        previous.set_extra("oncue_decision", {"should_reply": True})
        previous._has_send_oper = True
        previous.result.chain = [Plain("already replied")]
        self.now = 1010.0
        await asyncio.wait_for(self.plugin.mark_reply_sent(previous), 1)
        self.assertEqual(self.chat.messages[-1]["text"], "already replied")
        self.now = 1040.0  # 即使新回复的冷却已结束，旧决策仍然失效。
        release.set()
        await task
        self.assertIsNone(self.event.get_extra("oncue_decision"))
        self.assertEqual(list(self.chat.reply_ts), [1010.0])
        self.assertEqual(self.chat.pending_replies, {})

    async def test_slow_character_card_keeps_a_fixed_decision_snapshot(self):
        entered, release = self.block(self.plugin._decision_card, "character card")
        task = self.start(self.plugin.on_message(self.event))
        await asyncio.wait_for(entered.wait(), 1)
        self.assertFalse(self.chat.lock.locked())
        await asyncio.wait_for(self.plugin.on_message(_event("later topic")), 1)
        release.set()
        await task
        prompt = self.plugin._llm_decision.call_args.args[1]
        self.assertIn("first topic", prompt)
        self.assertNotIn("later topic", prompt)
        self.assertIn("later topic", self.plugin._history_text(self.chat))

    async def test_slow_knowledge_lookup_does_not_block_messages_or_summons(self):
        self.plugin._llm_decision.return_value = WITH_KB
        entered, release = self.block(self.plugin._kb_retrieve, "knowledge")
        task = self.start(self.plugin.on_message(self.event))
        await asyncio.wait_for(entered.wait(), 1)
        self.assertFalse(self.chat.lock.locked())
        await asyncio.wait_for(self.plugin.on_message(_event("later topic")), 1)
        await asyncio.wait_for(self.plugin.on_message(_event(summoned=True)), 1)
        release.set()
        await task
        self.assertIsNone(self.event.get_extra("oncue_decision"))
        self.assertEqual(self.chat.pending_replies, {})
        self.assertFalse(self.chat.decision_inflight)

    async def test_glance_waits_for_existing_event_decision_without_consuming_activity(self):
        entered, release = self.block(self.plugin._llm_decision, SILENT)
        task = self.start(self.plugin.on_message(self.event))
        await asyncio.wait_for(entered.wait(), 1)
        await asyncio.wait_for(self.plugin._glance_due(self.chat_id), 1)
        self.plugin._llm_decision.assert_awaited_once()
        self.assertNotIn(self.chat_id, self.plugin.glance_last_check)
        self.assertTrue(self.chat.decision_inflight)
        release.set()
        await task
        self.now = self.plugin.glance_next_due[self.chat_id]
        await self.plugin._glance_due(self.chat_id)
        self.assertEqual(self.plugin._llm_decision.await_count, 2)

    async def test_glance_busy_gate_is_shared_with_messages_and_other_glances(self):
        entered, release = self.block(self.plugin._llm_decision, SILENT)
        task = self.start(self.plugin._glance_due(self.chat_id))
        await asyncio.wait_for(entered.wait(), 1)
        self.now = 1010.0
        await asyncio.wait_for(self.plugin.on_message(_event("new topic")), 1)
        await asyncio.wait_for(self.plugin._glance_due(self.chat_id), 1)
        self.plugin._llm_decision.assert_awaited_once()
        self.assertTrue(self.chat.decision_inflight)
        release.set()
        await task
        self.assertEqual(self.plugin.glance_last_check[self.chat_id], 950.0)
        self.assertEqual(self.chat.last_activity_ts, 1010.0)
        self.now = self.plugin.glance_next_due[self.chat_id]
        await self.plugin._glance_due(self.chat_id)
        self.assertEqual(self.plugin._llm_decision.await_count, 2)
        self.assertEqual(self.plugin.glance_last_check[self.chat_id], 1010.0)

    async def test_glance_summon_during_analysis_skips_generation_and_progress(self):
        entered, release = self.block(self.plugin._llm_decision, REPLY)
        task = self.start(self.plugin._glance_due(self.chat_id))
        await asyncio.wait_for(entered.wait(), 1)
        await asyncio.wait_for(self.plugin.on_message(_event(summoned=True)), 1)
        release.set()
        await task
        self.plugin.context.llm_generate.assert_not_awaited()
        self.plugin.context.send_message.assert_not_awaited()
        self.assertNotIn(self.chat_id, self.plugin.glance_last_check)

    async def test_glance_generation_uses_the_same_snapshot_as_analysis(self):
        entered, release = self.block(self.plugin._llm_decision, REPLY)
        task = self.start(self.plugin._glance_due(self.chat_id))
        await asyncio.wait_for(entered.wait(), 1)
        self.now = 1010.0
        await asyncio.wait_for(self.plugin.on_message(_event("later topic")), 1)
        release.set()
        await task
        decision_prompt = self.plugin._llm_decision.call_args.args[1]
        generation_prompt = self.plugin.context.llm_generate.call_args.kwargs["prompt"]
        for prompt in (decision_prompt, generation_prompt):
            self.assertIn("old topic", prompt)
            self.assertNotIn("later topic", prompt)
        self.assertEqual(self.plugin.glance_last_check[self.chat_id], 950.0)
        self.plugin.context.send_message.assert_awaited_once()

    async def test_glance_summon_during_generation_cancels_unsent_reply(self):
        self.plugin._llm_decision.return_value = REPLY
        entered, release = self.block(
            self.plugin.context.llm_generate, SimpleNamespace(completion_text="old reply")
        )
        task = self.start(self.plugin._glance_due(self.chat_id))
        await asyncio.wait_for(entered.wait(), 1)
        self.assertFalse(self.chat.lock.locked())
        await asyncio.wait_for(self.plugin.on_message(_event(summoned=True)), 1)
        release.set()
        await task
        self.plugin.context.send_message.assert_not_awaited()
        self.assertEqual(list(self.chat.reply_ts), [])
        self.assertFalse(self.chat.decision_inflight)

    async def test_started_glance_send_does_not_block_summons_and_still_gets_recorded(self):
        self.plugin._llm_decision.return_value = REPLY
        entered, release = self.block(self.plugin.context.send_message, True)
        task = self.start(self.plugin._glance_due(self.chat_id))
        await asyncio.wait_for(entered.wait(), 1)
        self.assertFalse(self.chat.lock.locked())
        await asyncio.wait_for(self.plugin.on_message(_event(summoned=True)), 1)
        release.set()
        await task
        self.assertEqual(len(self.chat.reply_ts), 1)
        self.assertEqual(self.chat.messages[-1]["text"], "glance reply")
        self.plugin._append_assistant_history.assert_awaited_once()

    async def test_glance_history_write_does_not_hold_chat_lock(self):
        self.plugin._llm_decision.return_value = REPLY
        entered, release = self.block(self.plugin._append_assistant_history, None)
        task = self.start(self.plugin._glance_due(self.chat_id))
        await asyncio.wait_for(entered.wait(), 1)
        self.assertFalse(self.chat.lock.locked())
        await asyncio.wait_for(self.plugin.on_message(_event("new topic")), 1)
        self.assertEqual(self.chat.messages[-1]["text"], "new topic")
        release.set()
        await task
        self.assertFalse(self.chat.decision_inflight)

    async def test_cancelled_event_decision_releases_busy_flag(self):
        entered, _ = self.block(self.plugin._llm_decision, SILENT)
        task = self.start(self.plugin.on_message(self.event))
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(self.chat.decision_inflight)
        self.assertFalse(self.chat.lock.locked())
        self.plugin._llm_decision.side_effect = None
        await self.plugin.on_message(_event("try again"))
        self.assertEqual(self.plugin._llm_decision.await_count, 2)

    async def test_cancelled_glance_send_releases_busy_flag(self):
        self.plugin._llm_decision.return_value = REPLY
        entered, _ = self.block(self.plugin.context.send_message, True)
        task = self.start(self.plugin._glance_due(self.chat_id))
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(self.chat.decision_inflight)
        self.assertFalse(self.chat.lock.locked())
        self.assertEqual(list(self.chat.reply_ts), [])

    async def test_prompt_exception_releases_busy_flag_for_next_message(self):
        self.plugin._decision_card.side_effect = RuntimeError("card failure")
        with self.assertRaisesRegex(RuntimeError, "card failure"):
            await self.plugin.on_message(self.event)
        self.assertFalse(self.chat.decision_inflight)
        self.plugin._decision_card.side_effect = None
        await self.plugin.on_message(_event("try again"))
        self.plugin._llm_decision.assert_awaited_once()

    async def test_glance_generation_exception_releases_busy_flag(self):
        self.plugin._llm_decision.return_value = REPLY
        self.plugin._glance_generate = AsyncMock(side_effect=RuntimeError("generation failure"))
        with self.assertRaisesRegex(RuntimeError, "generation failure"):
            await self.plugin._glance_due(self.chat_id)
        self.assertFalse(self.chat.decision_inflight)
        self.assertFalse(self.chat.lock.locked())

    async def test_busy_flag_is_per_chat(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def decide(chat_id, prompt):
            if chat_id == self.chat_id:
                entered.set()
                await release.wait()
            return SILENT

        self.plugin._llm_decision.side_effect = decide
        task = self.start(self.plugin.on_message(self.event))
        await asyncio.wait_for(entered.wait(), 1)
        other = _event("other chat")
        other.unified_msg_origin = "test:GroupMessage:2"
        other_chat = self.plugin._chat(other.unified_msg_origin)
        other_chat.last_reply_ts = 950.0
        await asyncio.wait_for(self.plugin.on_message(other), 1)
        self.assertEqual(self.plugin._llm_decision.await_count, 2)
        self.assertTrue(self.chat.decision_inflight)
        self.assertFalse(other_chat.decision_inflight)
        release.set()
        await task

    async def test_config_disable_during_decision_prevents_late_wake(self):
        entered, release = self.block(self.plugin._llm_decision, REPLY)
        task = self.start(self.plugin.on_message(self.event))
        await asyncio.wait_for(entered.wait(), 1)
        self.plugin.config["enable"] = False
        release.set()
        await task
        self.assertIsNone(self.event.get_extra("oncue_decision"))
        self.assertEqual(self.chat.pending_replies, {})
        self.assertFalse(self.chat.decision_inflight)


if __name__ == "__main__":
    unittest.main()
