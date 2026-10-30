"""验证临时关注只提交最终决策，且不会改变原生回复或逐条调用模型。"""

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from test_timing import _event, _native_pipeline, plugin_module


def result(action="keep", note="", *, reply=False):
    return json.dumps({
        "should_reply": reply, "reason": "自然接话" if reply else "暂不接话",
        "attention_action": action, "attention_note": note,
    }, ensure_ascii=False)


class AttentionIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.now = 1000.0
        clock = patch.object(plugin_module, "time", SimpleNamespace(time=lambda: self.now))
        clock.start()
        self.addCleanup(clock.stop)
        random = patch.object(plugin_module.random, "uniform", side_effect=lambda low, high: low)
        random.start()
        self.addCleanup(random.stop)
        self.plugin = plugin_module.OnCuePlugin(
            SimpleNamespace(send_message=AsyncMock(return_value=True)),
            {
                "enable_glance": True, "enable_echo_trigger": False,
                "enable_dense_trigger": False, "message_wait_seconds": 0,
                "glance_min_minutes": 15, "glance_max_minutes": 35,
                "attention_glance_min_minutes": 3, "attention_glance_max_minutes": 8,
            },
        )
        self.plugin._decision_card = AsyncMock(return_value="character card")
        self.plugin._llm_decision = AsyncMock(return_value=result())
        self.event = self.message("待关注的话题", "seed")
        self.chat_id = self.event.unified_msg_origin
        self.chat = self.plugin._chat(self.chat_id)
        self.plugin._append_message(self.chat, self.plugin._event_to_item(self.event, self.chat))
        self.tasks = []

    async def asyncTearDown(self):
        for task in self.tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)

    @staticmethod
    def message(text="群聊新消息", message_id=None, *, sender_id="alice", summoned=False):
        event = _event(text, summoned=summoned)
        event.message_obj.message_id = message_id
        event.get_sender_id = lambda: sender_id
        return event

    def start(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.tasks.append(task)
        return task

    def pause_decisions(self, *results):
        entered = [asyncio.Event() for _ in results]
        release = [asyncio.Event() for _ in results]
        index = 0

        async def decide(*args, **kwargs):
            nonlocal index
            current = index
            index += 1
            if current >= len(results):
                return result()
            entered[current].set()
            await release[current].wait()
            return results[current]

        self.plugin._llm_decision.side_effect = decide
        return entered, release

    async def wait(self, barrier):
        await asyncio.wait_for(barrier.wait(), 1)

    async def decide(self, raw=None):
        self.chat.cooldown_until = 0
        if raw is not None:
            self.plugin._llm_decision.return_value = raw
        return await self.plugin._decide(self.event, self.chat, "普通决策机会", None)

    async def begin_attention(self, note="关注事项-UNIQUE"):
        await self.decide(result("start", note))
        self.assertTrue(self.chat.attention.active)

    async def expire_by_messages(self):
        for index in range(50):
            await self.plugin.on_message(self.message(message_id=f"limit-{index}"))
        self.assertFalse(self.chat.attention.active)
        self.assertIsNotNone(self.chat.attention.notice)

    async def test_silent_decision_can_start_attention_without_reply_or_observation(self):
        reply = await self.decide(result("start", "稍后看看检修进展"))
        self.assertFalse(reply)
        self.assertTrue(self.chat.attention.active)
        self.assertEqual(self.chat.attention.note, "稍后看看检修进展")
        self.assertFalse(self.plugin._is_observing(self.chat, self.now))
        self.assertIsNone(self.event.get_extra("oncue_decision"))
        self.plugin.context.send_message.assert_not_awaited()

    async def test_reply_and_attention_are_independent_outputs(self):
        self.assertTrue(await self.decide(result("start", "关注后续", reply=True)))
        self.assertTrue(self.chat.attention.active)
        self.assertTrue(self.event.get_extra("oncue_decision")["should_reply"])
        self.assertFalse(self.plugin._is_observing(self.chat, self.now))

    async def test_initial_attention_is_not_committed_when_extra_decision_replaces_it(self):
        entered, release = self.pause_decisions(result("start", "不可提前提交"), result("stop"))
        task = self.start(self.decide())
        await self.wait(entered[0])
        await self.plugin.on_message(self.message("新消息", "extra"))
        release[0].set()
        await self.wait(entered[1])
        self.assertFalse(self.chat.attention.active)
        release[1].set()
        await task
        self.assertFalse(self.chat.attention.active)
        self.assertEqual(self.plugin._llm_decision.await_count, 2)

    async def test_extra_decision_can_start_attention_after_initial_keep(self):
        entered, release = self.pause_decisions(result(), result("start", "最终关注事项"))
        task = self.start(self.decide())
        await self.wait(entered[0])
        await self.plugin.on_message(self.message("新增上下文", "extra"))
        release[0].set()
        await self.wait(entered[1])
        release[1].set()
        await task
        self.assertTrue(self.chat.attention.active)
        self.assertEqual(self.chat.attention.note, "最终关注事项")

    async def test_invalid_decision_cannot_submit_attention_action(self):
        await self.decide('{"attention_action":"start","attention_note":"缺少有效结论"}')
        self.assertFalse(self.chat.attention.active)
        await self.decide("not JSON")
        self.assertFalse(self.chat.attention.active)

    async def test_cancelled_decision_does_not_start_attention(self):
        entered, _ = self.pause_decisions(result("start", "应被取消"))
        task = self.start(self.decide())
        await self.wait(entered[0])
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(self.chat.attention.active)
        self.assertIsNone(self.chat.decision_inflight)
        self.assertFalse(self.chat.lock.locked())

    async def test_summoned_new_reply_invalidates_older_attention_output(self):
        entered, release = self.pause_decisions(result("start", "过期关注"))
        task = self.start(self.decide())
        await self.wait(entered[0])
        await self.plugin.on_message(self.message("叫你呢", "summon", summoned=True))
        release[0].set()
        await task
        self.assertFalse(self.chat.attention.active)

    async def test_entering_attention_keeps_an_already_nearer_glance(self):
        self.plugin.glance_next_due[self.chat_id] = self.now + 30
        await self.begin_attention()
        self.assertEqual(self.plugin.glance_next_due[self.chat_id], self.now + 30)

    async def test_entering_attention_shortens_a_later_glance(self):
        self.plugin.glance_next_due[self.chat_id] = self.now + 900
        await self.begin_attention()
        self.assertEqual(self.plugin.glance_next_due[self.chat_id], self.now + 180)

    async def test_keep_and_new_messages_do_not_postpone_existing_glance(self):
        await self.begin_attention()
        due = self.plugin.glance_next_due[self.chat_id]
        started = self.chat.attention.started_at
        progress = self.chat.attention.start_progress
        self.now += 1
        await self.plugin.on_message(self.message("另一个人的消息", "later", sender_id="bob"))
        await self.decide(result("keep", "更新关注说明"))
        self.assertEqual(self.plugin.glance_next_due[self.chat_id], due)
        self.assertEqual(self.chat.attention.started_at, started)
        self.assertEqual(self.chat.attention.start_progress, progress)
        self.assertEqual(self.chat.attention.note, "更新关注说明")

    async def test_only_fifty_unique_nonempty_human_messages_force_exit(self):
        await self.begin_attention()
        baseline = self.chat.incoming_seq
        await self.plugin.on_message(self.message("重复投递", "seed"))
        await self.plugin.on_message(self.message("", "empty"))
        await self.plugin.on_message(self.message("机器人消息", "bot", sender_id="bot"))
        self.assertEqual(self.chat.incoming_seq, baseline)
        for index in range(49):
            await self.plugin.on_message(self.message(message_id=f"human-{index}"))
        self.assertTrue(self.chat.attention.active)
        await self.plugin.on_message(self.message(message_id="human-49"))
        self.assertFalse(self.chat.attention.active)
        self.assertIsNotNone(self.chat.attention.notice)
        self.assertIn("50", self.chat.attention.notice.reason)
        self.plugin._llm_decision.assert_awaited_once()

    async def test_keeping_attention_does_not_replace_its_origin_with_another_reply_target(self):
        await self.begin_attention()
        origin = self.chat.attention.message_id
        await self.plugin.on_message(self.message("另一个话题", "another-topic"))
        await self.decide(json.dumps({
            "should_reply": False, "target_message_id": "another-topic",
            "attention_action": "keep", "attention_note": "继续等待原话题后续",
        }))
        self.assertEqual(self.chat.attention.message_id, origin)
        self.assertEqual(self.chat.attention.note, "继续等待原话题后续")

    async def test_three_hour_expiry_does_not_call_model_when_nothing_is_unchecked(self):
        await self.begin_attention()
        _native_pipeline(self.plugin)
        self.plugin.glance_last_check[self.chat_id] = self.chat.incoming_seq
        self.now += 3 * 60 * 60
        await self.plugin._glance_due(self.chat_id)
        self.assertFalse(self.chat.attention.active)
        self.assertIsNotNone(self.chat.attention.notice)
        self.plugin._llm_decision.assert_awaited_once()

    async def test_exit_notice_survives_invalid_decision_then_is_consumed_once(self):
        await self.begin_attention()
        await self.expire_by_messages()
        notice = self.chat.attention.notice
        await self.decide("not JSON")
        self.assertEqual(self.chat.attention.notice, notice)
        prompt = self.plugin._llm_decision.call_args.args[1]
        self.assertIn("关注事项-UNIQUE", prompt)
        self.assertIn("50", prompt)
        self.assertIn("退出", prompt)
        await self.decide(result())
        self.assertIsNone(self.chat.attention.notice)
        await self.decide(result())
        self.assertNotIn("关注事项-UNIQUE", self.plugin._llm_decision.call_args.args[1])

    async def test_exit_notice_remains_available_through_initial_and_extra_decision(self):
        await self.begin_attention()
        await self.expire_by_messages()
        notice = self.chat.attention.notice
        entered, release = self.pause_decisions(result(), result())
        task = self.start(self.decide())
        await self.wait(entered[0])
        self.assertIn("关注事项-UNIQUE", self.plugin._llm_decision.call_args.args[1])
        await self.plugin.on_message(self.message("补判期间的新消息", "notice-extra"))
        release[0].set()
        await self.wait(entered[1])
        self.assertEqual(self.chat.attention.notice, notice)
        self.assertIn("关注事项-UNIQUE", self.plugin._llm_decision.call_args.args[1])
        release[1].set()
        await task
        self.assertIsNone(self.chat.attention.notice)

    async def test_disabled_glance_still_delivers_exit_notice_without_starting_attention(self):
        await self.begin_attention()
        await self.expire_by_messages()
        self.plugin.config["enable_glance"] = False
        await self.decide(result("restart", "不应开启"))
        prompt = self.plugin._llm_decision.call_args.args[1]
        self.assertIn("关注事项-UNIQUE", prompt)
        self.assertIn("定时一瞥当前已关闭", prompt)
        self.assertFalse(self.chat.attention.active)
        self.assertIsNone(self.chat.attention.notice)
        await self.decide(result())
        self.assertNotIn("关注事项-UNIQUE", self.plugin._llm_decision.call_args.args[1])

    async def test_cancelled_decision_does_not_consume_exit_notice(self):
        await self.begin_attention()
        await self.expire_by_messages()
        notice = self.chat.attention.notice
        entered, _ = self.pause_decisions(result())
        task = self.start(self.decide())
        await self.wait(entered[0])
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.chat.attention.notice, notice)

    async def test_superseded_decision_does_not_consume_exit_notice(self):
        await self.begin_attention()
        await self.expire_by_messages()
        notice = self.chat.attention.notice
        entered, release = self.pause_decisions(result())
        task = self.start(self.decide())
        await self.wait(entered[0])
        await self.plugin.on_message(self.message("优先回应点名", "notice-summon", summoned=True))
        release[0].set()
        await task
        self.assertEqual(self.chat.attention.notice, notice)

    async def test_old_decision_cannot_restart_attention_after_fifty_message_exit(self):
        await self.begin_attention()
        self.plugin.config["enable_extra_decision"] = False
        entered, release = self.pause_decisions(result("restart", "旧结果不得恢复关注"))
        task = self.start(self.decide())
        await self.wait(entered[0])
        await self.expire_by_messages()
        notice = self.chat.attention.notice
        release[0].set()
        await task
        self.assertFalse(self.chat.attention.active)
        self.assertEqual(self.chat.attention.notice, notice)

    async def test_old_decision_cannot_restart_attention_after_three_hour_exit(self):
        await self.begin_attention()
        entered, release = self.pause_decisions(result("restart", "旧结果不得恢复关注"))
        task = self.start(self.decide())
        await self.wait(entered[0])
        self.now += 3 * 60 * 60
        release[0].set()
        await task
        self.assertFalse(self.chat.attention.active)
        self.assertIsNotNone(self.chat.attention.notice)

    async def test_disabling_glance_during_decision_ignores_attention_action(self):
        entered, release = self.pause_decisions(result("start", "开关已关"))
        task = self.start(self.decide())
        await self.wait(entered[0])
        self.plugin.config["enable_glance"] = False
        release[0].set()
        await task
        self.assertFalse(self.chat.attention.active)

    async def test_direct_summon_does_not_add_an_attention_decision(self):
        await self.plugin.on_message(self.message("直接叫你", "summon", summoned=True))
        self.plugin._llm_decision.assert_not_awaited()
        self.assertFalse(self.chat.attention.active)

    async def test_attention_interval_uses_user_range_and_leaves_ordinary_range_separate(self):
        self.plugin.config.update(attention_glance_min_minutes=2, attention_glance_max_minutes=6)
        self.assertEqual(self.plugin._glance_interval_seconds(self.chat), 15 * 60)
        await self.begin_attention()
        self.assertEqual(self.plugin._glance_interval_seconds(self.chat), 2 * 60)
        self.assertEqual(self.plugin._glance_interval_seconds(), 15 * 60)


if __name__ == "__main__":
    unittest.main()
