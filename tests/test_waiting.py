"""验证普通消息等待、取消、触发复核及主回复上下文。"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from test_context import at, reply
from test_timing import Plain, _event, plugin_module


POSITIVE = '{"should_reply": true, "reason": "想接话"}'


class WaitingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.now = 1000.0
        self.mono = 100.0
        clock = patch.object(plugin_module, "time", SimpleNamespace(time=lambda: self.now, monotonic=lambda: self.mono))
        clock.start()
        self.addCleanup(clock.stop)
        self.plugin = plugin_module.OnCuePlugin(
            SimpleNamespace(),
            {"enable_glance": False, "enable_echo_trigger": False, "enable_dense_trigger": False},
        )
        self.event = _event("我想换电脑")
        self.chat = self.plugin._chat(self.event.unified_msg_origin)
        self.chat.last_reply_ts = self.now - 1
        self.plugin._decision_card = AsyncMock(return_value="角色卡")
        self.plugin._llm_decision = AsyncMock(return_value=POSITIVE)
        self.plugin._glance_interval_seconds = lambda: 900.0
        self.tasks = []

    async def asyncTearDown(self):
        for task in self.tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)

    async def settle(self):
        # 让 Event.wait、wait_for 和消息处理任务完成各自的调度，不依赖机器速度。
        for _ in range(10):
            await asyncio.sleep(0)

    async def start(self, event=None):
        task = asyncio.create_task(self.plugin.on_message(event or self.event))
        self.tasks.append(task)
        await self.settle()
        return task

    async def advance(self, seconds):
        self.now += seconds
        self.mono += seconds
        for chat in self.plugin.chats.values():
            if chat.message_wait is not None:
                chat.message_wait.changed.set()
        await self.settle()

    async def finish(self, task):
        await asyncio.wait_for(task, 1)

    def member_event(self, sender, text="哈哈"):
        event = _event(text)
        event.get_sender_id = lambda: sender
        return event

    def request(self, prompt="原始消息"):
        return SimpleNamespace(prompt=prompt, system_prompt="原系统提示", image_urls=["original-image"], contexts=[])

    async def test_default_wait_reaches_quiet_deadline_without_holding_lock(self):
        task = await self.start()
        self.assertIsNotNone(self.chat.message_wait)
        self.assertFalse(self.chat.lock.locked())
        self.assertFalse(self.chat.decision_inflight)
        self.plugin._llm_decision.assert_not_awaited()
        await self.advance(1)
        self.assertFalse(task.done())
        await self.advance(1)
        await self.finish(task)
        self.plugin._llm_decision.assert_awaited_once()
        self.assertIsNone(self.chat.message_wait)

    async def test_new_messages_restart_quiet_wait_and_only_owner_is_released_to_reply(self):
        task = await self.start()
        await self.advance(1)
        second = _event("主要用来剪视频")
        await self.plugin.on_message(second)
        await self.advance(1)
        self.plugin._llm_decision.assert_not_awaited()
        third = _event("预算八千")
        await self.plugin.on_message(third)
        await self.advance(2)
        await self.finish(task)
        self.plugin._llm_decision.assert_awaited_once()
        prompt = self.plugin._llm_decision.call_args.args[1]
        for text in ("我想换电脑", "主要用来剪视频", "预算八千"):
            self.assertIn(text, prompt)
        self.assertTrue(self.event.is_at_or_wake_command)
        self.assertFalse(second.is_at_or_wake_command)
        self.assertFalse(third.is_at_or_wake_command)
        self.assertEqual(len(self.chat.pending_replies), 1)

    async def test_continuous_messages_cannot_extend_total_deadline(self):
        task = await self.start()
        deadline = self.chat.message_wait.deadline
        for index in range(4):
            await self.advance(1)
            await self.plugin.on_message(_event(f"补充 {index}"))
            self.assertEqual(self.chat.message_wait.deadline, deadline)
            self.plugin._llm_decision.assert_not_awaited()
        await self.advance(1)
        await self.finish(task)
        self.plugin._llm_decision.assert_awaited_once()

    async def test_maximum_can_be_shorter_than_quiet_wait(self):
        self.plugin.config.update(message_wait_seconds=10, message_wait_max_seconds=3)
        task = await self.start()
        await self.advance(3)
        await self.finish(task)
        self.plugin._llm_decision.assert_awaited_once()

    async def test_zero_wait_still_supplies_the_common_context(self):
        self.plugin.config["message_wait_seconds"] = 0
        await self.plugin.on_message(self.event)
        self.plugin._llm_decision.assert_awaited_once()
        self.assertIsNone(self.chat.message_wait)
        self.assertIn("_chat_context", self.event.get_extra("oncue_decision"))
        req = self.request()
        await self.plugin.inject_stage_direction(self.event, req)
        self.assertTrue(req.prompt.startswith("原始消息"))
        self.assertIn(self.event.get_extra("oncue_decision")["_chat_context"], req.prompt)

    async def test_empty_message_does_not_restart_quiet_wait(self):
        task = await self.start()
        await self.advance(1)
        await self.plugin.on_message(_event(""))
        await self.advance(1)
        await self.finish(task)
        self.plugin._llm_decision.assert_awaited_once()

    async def test_wall_clock_jump_does_not_end_wait_early(self):
        task = await self.start()
        self.now += 3600
        await self.advance(1)
        self.plugin._llm_decision.assert_not_awaited()
        self.now -= 3600
        await self.advance(1)
        await self.finish(task)
        self.plugin._llm_decision.assert_awaited_once()

    async def test_summon_cancels_wait_and_does_not_delay_forced_path(self):
        task = await self.start()
        summoned = _event("现在回答", summoned=True)
        await self.plugin.on_message(summoned)
        await self.finish(task)
        self.assertTrue(summoned.is_at_or_wake_command)
        self.assertTrue(summoned.get_extra("oncue_decision")["should_reply"])
        self.assertFalse(self.event.is_at_or_wake_command)
        self.assertIsNone(self.chat.message_wait)
        self.plugin._llm_decision.assert_not_awaited()

    async def test_decision_summon_cancels_wait_and_asks_model_immediately(self):
        task = await self.start()
        self.plugin.config["force_reply_when_summoned"] = False
        summoned = _event(summoned=True)
        await self.plugin.on_message(summoned)
        await self.finish(task)
        self.assertIsNone(self.chat.message_wait)
        self.assertTrue(summoned.is_at_or_wake_command)
        self.assertTrue(summoned.get_extra("oncue_decision")["should_reply"])
        self.plugin._llm_decision.assert_awaited_once()

    async def test_old_wait_cleanup_cannot_clear_a_new_wait(self):
        old_task = await self.start()
        old_pending = self.chat.message_wait
        await self.plugin.on_message(_event(summoned=True))
        new_task = await self.start(_event("新的交流"))
        new_pending = self.chat.message_wait
        await self.finish(old_task)
        self.assertIsNot(new_pending, old_pending)
        self.assertIs(self.chat.message_wait, new_pending)
        await self.advance(2)
        await self.finish(new_task)
        self.plugin._llm_decision.assert_awaited_once()

    async def test_sent_reply_cancels_pending_wait(self):
        task = await self.start()
        sent = _event()
        sent.set_extra("oncue_decision", {"should_reply": True})
        sent._has_send_oper = True
        sent.result.chain = [Plain("已经回复了")]
        await self.plugin.mark_reply_sent(sent)
        await self.finish(task)
        self.plugin._llm_decision.assert_not_awaited()
        self.assertIsNone(self.chat.message_wait)
        self.assertEqual(len(self.chat.reply_ts), 1)

    async def test_cancelled_owner_releases_wait_for_next_message(self):
        task = await self.start()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertIsNone(self.chat.message_wait)
        next_task = await self.start(_event("再试一次"))
        await self.advance(2)
        await self.finish(next_task)
        self.plugin._llm_decision.assert_awaited_once()

    async def test_termination_releases_all_waits_without_new_decisions(self):
        first = await self.start()
        other = _event("另一个群")
        other.unified_msg_origin = "test:GroupMessage:2"
        self.plugin._chat(other.unified_msg_origin).last_reply_ts = self.now - 1
        second = await self.start(other)
        await self.plugin.terminate()
        await self.finish(first)
        await self.finish(second)
        await self.plugin.on_message(_event("卸载后的事件"))
        self.plugin._llm_decision.assert_not_awaited()
        self.assertTrue(all(chat.message_wait is None for chat in self.plugin.chats.values()))

    async def test_prompt_failure_clears_wait_and_busy_state(self):
        self.plugin._decision_card.side_effect = RuntimeError("failed")
        task = await self.start()
        await self.advance(2)
        with self.assertRaisesRegex(RuntimeError, "failed"):
            await task
        self.assertIsNone(self.chat.message_wait)
        self.assertFalse(self.chat.decision_inflight)

    async def test_entry_gates_do_not_create_waits(self):
        for gate in ("busy", "cooldown", "capacity"):
            with self.subTest(gate=gate):
                self.chat.decision_inflight = gate == "busy"
                self.chat.cooldown_until = self.now + 30 if gate == "cooldown" else 0
                self.chat.pending_replies = {"pending": self.now + 30} if gate == "capacity" else {}
                self.plugin.config["max_replies_per_window"] = 1
                await self.plugin.on_message(_event())
                self.assertIsNone(self.chat.message_wait)
                self.plugin._llm_decision.assert_not_awaited()

    async def test_end_of_wait_rechecks_cooldown_capacity_and_enable(self):
        for gate in ("cooldown", "capacity", "disabled"):
            with self.subTest(gate=gate):
                self.chat.cooldown_until = 0
                self.chat.pending_replies.clear()
                self.plugin.config["enable"] = True
                task = await self.start(_event())
                if gate == "cooldown":
                    self.chat.cooldown_until = self.now + 30
                elif gate == "capacity":
                    self.plugin.config["max_replies_per_window"] = 1
                    self.chat.pending_replies["other"] = self.now + 30
                else:
                    self.plugin.config["enable"] = False
                await self.advance(2)
                await self.finish(task)
                self.assertIsNone(self.chat.message_wait)
                self.plugin._llm_decision.assert_not_awaited()

    async def test_observation_expiring_during_wait_does_not_force_a_decision(self):
        self.plugin.config.update(observation_refresh=False, observation_timeout=2)
        task = await self.start()
        await self.advance(2)
        await self.finish(task)
        self.plugin._llm_decision.assert_not_awaited()

    async def test_silent_chat_starts_wait_only_after_dense_threshold(self):
        self.chat.last_reply_ts = 0
        self.plugin.config.update(enable_dense_trigger=True, dense_message_threshold=3, dense_participant_threshold=2)
        await self.plugin.on_message(self.member_event("a", "第一句"))
        await self.plugin.on_message(self.member_event("b", "第二句"))
        self.assertIsNone(self.chat.message_wait)
        task = await self.start(self.member_event("c", "第三句"))
        self.assertIsNotNone(self.chat.message_wait)
        await self.advance(2)
        await self.finish(task)
        self.plugin._llm_decision.assert_awaited_once()

    async def test_dense_window_is_rechecked_after_wait(self):
        self.chat.last_reply_ts = 0
        self.plugin.config.update(enable_dense_trigger=True, dense_message_threshold=3, dense_participant_threshold=2, dense_window=1)
        await self.plugin.on_message(self.member_event("a"))
        await self.plugin.on_message(self.member_event("b"))
        task = await self.start(self.member_event("c"))
        await self.advance(2)
        await self.finish(task)
        self.plugin._llm_decision.assert_not_awaited()

    async def test_echo_trigger_and_backoff_are_rechecked_after_wait(self):
        self.chat.last_reply_ts = 0
        self.plugin.config["enable_echo_trigger"] = True
        await self.plugin.on_message(self.member_event("a"))
        await self.plugin.on_message(self.member_event("b"))
        task = await self.start(self.member_event("c"))
        self.chat.backoffs["echo:哈哈"] = {"until": self.now + 30}
        await self.advance(2)
        await self.finish(task)
        self.plugin._llm_decision.assert_not_awaited()
        self.chat.backoffs.clear()
        task = await self.start(self.member_event("d"))
        await self.plugin.on_message(self.member_event("e", "换个话题"))
        await self.advance(2)
        await self.finish(task)
        self.plugin._llm_decision.assert_not_awaited()

    async def test_glance_skips_wait_without_consuming_activity(self):
        task = await self.start()
        await self.plugin._glance_due(self.event.unified_msg_origin)
        self.assertNotIn(self.event.unified_msg_origin, self.plugin.glance_last_check)
        self.plugin._llm_decision.assert_not_awaited()
        await self.advance(2)
        await self.finish(task)
        self.plugin._llm_decision.assert_awaited_once()

    async def test_waits_are_independent_between_chats(self):
        first = await self.start()
        other = _event("另一群的第一句")
        other.unified_msg_origin = "test:GroupMessage:2"
        self.plugin._chat(other.unified_msg_origin).last_reply_ts = self.now - 1
        second = await self.start(other)
        await self.advance(1)
        await self.plugin.on_message(_event("本群有补充"))
        await self.advance(1)
        await self.finish(second)
        self.assertFalse(first.done())
        await self.advance(1)
        await self.finish(first)
        self.assertEqual(self.plugin._llm_decision.await_count, 2)

    async def test_messages_during_model_call_are_cached_without_automatic_followup(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def analyze(*args):
            entered.set()
            await release.wait()
            return POSITIVE

        self.plugin._llm_decision.side_effect = analyze
        task = await self.start()
        await self.plugin.on_message(_event("等待阶段的补充"))
        await self.advance(2)
        await asyncio.wait_for(entered.wait(), 1)
        self.assertIsNone(self.chat.message_wait)
        await self.plugin.on_message(_event("模型开始后的新消息"))
        self.assertIsNone(self.chat.message_wait)
        release.set()
        await self.finish(task)
        self.plugin._llm_decision.assert_awaited_once()
        context = self.event.get_extra("oncue_decision")["_chat_context"]
        self.assertIn("等待阶段的补充", context)
        self.assertNotIn("模型开始后的新消息", context)

    async def test_reply_request_receives_same_snapshot_and_preserves_original_request(self):
        self.event.message_obj.message_id = "first"
        task = await self.start()
        second = self.member_event("bob", "预算八千")
        second.get_messages = lambda: [reply("first"), at("carol", "小王"), Plain("预算八千")]
        await self.plugin.on_message(second)
        await self.advance(2)
        await self.finish(task)
        snapshot = self.event.get_extra("oncue_decision")["_chat_context"]
        self.assertIn(snapshot, self.plugin._llm_decision.call_args.args[1])
        req = self.request()
        await self.plugin.inject_stage_direction(self.event, req)
        await self.plugin.inject_stage_direction(self.event, req)
        self.assertIn(snapshot, req.prompt)
        self.assertIn("预算八千", req.prompt)
        self.assertIn("[引用", req.prompt)
        self.assertIn("ID=carol", req.prompt)
        self.assertTrue(req.prompt.startswith("原始消息"))
        self.assertEqual(req.prompt.count("[群聊补充上下文]"), 1)
        self.assertEqual(req.image_urls, ["original-image"])
        self.assertEqual(req.contexts, [])
        self.assertNotIn(snapshot, req.system_prompt)
        empty_prompt = self.request(None)
        await self.plugin.inject_stage_direction(self.event, empty_prompt)
        self.assertIn(snapshot, empty_prompt.prompt)

    async def test_new_message_at_admission_restarts_wait_before_model_call(self):
        original_decide = self.plugin._decide
        first_attempt = True

        async def interrupted_decide(*args, **kwargs):
            nonlocal first_attempt
            if first_attempt:
                first_attempt = False
                await self.plugin.on_message(_event("恰好到点时的补充"))
            return await original_decide(*args, **kwargs)

        self.plugin._decide = interrupted_decide
        task = await self.start()
        await self.advance(2)
        self.plugin._llm_decision.assert_not_awaited()
        self.assertIsNotNone(self.chat.message_wait)
        await self.advance(2)
        await self.finish(task)
        self.plugin._llm_decision.assert_awaited_once()
        self.assertIn("恰好到点时的补充", self.plugin._llm_decision.call_args.args[1])

    async def test_idle_timeout_also_runs_the_pending_decision(self):
        async def timeout(awaitable, timeout):
            awaitable.close()
            self.now += timeout
            self.mono += timeout
            raise asyncio.TimeoutError

        with patch.object(plugin_module.asyncio, "wait_for", side_effect=timeout):
            await self.plugin.on_message(self.event)
        self.plugin._llm_decision.assert_awaited_once()
        self.assertIsNone(self.chat.message_wait)


if __name__ == "__main__":
    unittest.main()
