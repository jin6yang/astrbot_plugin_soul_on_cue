"""用可控模型屏障验证单轮额外决策、消息进度和回应目标。"""

import asyncio
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from test_timing import Plain, _event, _native_pipeline, _request_text, plugin_module


SILENT = '{"should_reply": false}'
REPLY = '{"should_reply": true, "reason": "first choice"}'
UPDATED_REPLY = '{"should_reply": true, "reason": "updated choice"}'


class ExtraDecisionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.now = 1000.0
        clock = patch.object(plugin_module, "time", SimpleNamespace(time=lambda: self.now))
        clock.start()
        self.addCleanup(clock.stop)
        self.plugin = plugin_module.OnCuePlugin(
            SimpleNamespace(send_message=AsyncMock(return_value=True)),
            {
                "enable_glance": False,
                "enable_echo_trigger": False,
                "enable_dense_trigger": False,
                "message_wait_seconds": 0,
            },
        )
        self.plugin._decision_card = AsyncMock(return_value="character card")
        self.plugin._llm_decision = AsyncMock(return_value=REPLY)
        self.plugin._kb_retrieve = AsyncMock(return_value="knowledge")
        self.event = self.message("first message", "first")
        self.chat_id = self.event.unified_msg_origin
        self.chat = self.plugin._chat(self.chat_id)
        self.chat.last_reply_ts = 950.0
        self.tasks = []

    async def asyncTearDown(self):
        for task in self.tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)

    @staticmethod
    def message(text, message_id=None, *, sender_id="alice", summoned=False):
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
        calls = 0

        async def decide(*args):
            nonlocal calls
            index = calls
            calls += 1
            if index >= len(results):
                return SILENT
            entered[index].set()
            await release[index].wait()
            return results[index]

        self.plugin._llm_decision.side_effect = decide
        return entered, release

    async def wait(self, barrier):
        await asyncio.wait_for(barrier.wait(), 1)

    async def receive(self, event):
        await asyncio.wait_for(self.plugin.on_message(event), 1)

    @staticmethod
    def request(prompt="first message"):
        return SimpleNamespace(prompt=prompt, system_prompt="native persona", contexts=[], image_urls=[])

    async def test_new_message_after_initial_silence_can_replace_it_with_one_reply(self):
        entered, release = self.pause_decisions(SILENT, UPDATED_REPLY)
        task = self.start(self.plugin.on_message(self.event))
        await self.wait(entered[0])
        self.assertFalse(self.chat.lock.locked())
        await self.receive(self.message("new information", "second"))
        self.assertEqual(self.plugin._llm_decision.await_count, 1)
        self.assertIsNone(self.event.get_extra("oncue_decision"))
        release[0].set()
        await self.wait(entered[1])
        self.assertIn("new information", self.plugin._llm_decision.call_args.args[1])
        self.assertFalse(self.chat.lock.locked())
        self.assertEqual(self.chat.pending_replies, {})
        release[1].set()
        await task
        decision = self.event.get_extra("oncue_decision")
        self.assertEqual(decision["reason"], "updated choice")
        self.assertIn("new information", decision["_chat_context"].render())
        self.assertEqual(decision["_chat_context"].progress, self.chat.incoming_seq)
        self.assertEqual(len(self.chat.pending_replies), 1)
        self.assertEqual(self.plugin._llm_decision.await_count, 2)
        self.assertFalse(self.chat.decision_inflight)

    async def test_extra_silence_replaces_initial_reply_without_reservation(self):
        entered, release = self.pause_decisions(REPLY, SILENT)
        task = self.start(self.plugin.on_message(self.event))
        await self.wait(entered[0])
        await self.receive(self.message("cancel the previous thought", "second"))
        release[0].set()
        await self.wait(entered[1])
        release[1].set()
        await task
        self.assertIsNone(self.event.get_extra("oncue_decision"))
        self.assertFalse(self.event.is_at_or_wake_command)
        self.assertEqual(self.chat.pending_replies, {})
        self.assertFalse(self.chat.decision_inflight)

    async def test_disabled_extra_decision_keeps_initial_snapshot(self):
        self.plugin.config["config_decision"] = {"enable_extra_decision": False}
        entered, release = self.pause_decisions(REPLY)
        task = self.start(self.plugin.on_message(self.event))
        await self.wait(entered[0])
        await self.receive(self.message("later message", "second"))
        release[0].set()
        await task
        self.plugin._llm_decision.assert_awaited_once()
        snapshot = self.event.get_extra("oncue_decision")["_chat_context"]
        self.assertNotIn("later message", snapshot.render())
        self.assertLess(snapshot.progress, self.chat.incoming_seq)

    async def test_disabling_extra_decision_during_initial_request_prevents_followup(self):
        entered, release = self.pause_decisions(REPLY)
        task = self.start(self.plugin.on_message(self.event))
        await self.wait(entered[0])
        await self.receive(self.message("later message", "second"))
        self.plugin.config["enable_extra_decision"] = False
        release[0].set()
        await task
        self.plugin._llm_decision.assert_awaited_once()
        self.assertEqual(self.event.get_extra("oncue_decision")["reason"], "first choice")

    async def test_message_during_extra_request_is_cached_without_third_request(self):
        entered, release = self.pause_decisions(REPLY, UPDATED_REPLY)
        task = self.start(self.plugin.on_message(self.event))
        await self.wait(entered[0])
        await self.receive(self.message("second message", "second"))
        release[0].set()
        await self.wait(entered[1])
        await self.receive(self.message("third message", "third"))
        self.assertEqual(self.plugin._llm_decision.await_count, 2)
        release[1].set()
        await task
        snapshot = self.event.get_extra("oncue_decision")["_chat_context"]
        self.assertIn("second message", snapshot.render())
        self.assertNotIn("third message", snapshot.render())
        self.assertIn("third message", self.plugin._history_text(self.chat))
        self.assertLess(snapshot.progress, self.chat.incoming_seq)
        self.assertEqual(self.plugin._llm_decision.await_count, 2)
        self.assertEqual(len(self.chat.pending_replies), 1)

    async def test_duplicate_empty_self_and_image_status_changes_do_not_add_progress(self):
        image = plugin_module.Image()
        image.url = "https://images.invalid/example.png"
        self.event.get_messages = lambda: [Plain("first message"), image]
        entered, release = self.pause_decisions(REPLY)
        task = self.start(self.plugin.on_message(self.event))
        await self.wait(entered[0])
        initial_progress = self.chat.incoming_seq
        await self.receive(self.message("first message", "first"))
        await self.receive(self.message("", "empty"))
        await self.receive(self.message("own message", "self", sender_id="bot"))
        recorded_image = next(part.image for part in self.chat.messages[0]["parts"] if part.image)
        recorded_image.status = "已转述"
        recorded_image.description = "later description"
        self.assertEqual(self.chat.incoming_seq, initial_progress)
        release[0].set()
        await task
        self.plugin._llm_decision.assert_awaited_once()
        snapshot = self.event.get_extra("oncue_decision")["_chat_context"]
        self.assertNotIn("later description", snapshot.render())

    async def test_distinct_same_text_message_counts_even_at_same_time(self):
        entered, release = self.pause_decisions(REPLY, UPDATED_REPLY)
        task = self.start(self.plugin.on_message(self.event))
        await self.wait(entered[0])
        progress = self.chat.incoming_seq
        await self.receive(self.message("first message", "second"))
        self.assertEqual(self.chat.incoming_seq, progress + 1)
        release[0].set()
        await self.wait(entered[1])
        release[1].set()
        await task
        self.assertEqual(self.plugin._llm_decision.await_count, 2)
        self.assertEqual(self.event.get_extra("oncue_decision")["_chat_context"].render().count("first message"), 2)

    async def test_cache_eviction_during_initial_request_does_not_hide_new_messages(self):
        self.plugin.config["message_cache_count"] = 2
        entered, release = self.pause_decisions(REPLY, UPDATED_REPLY)
        task = self.start(self.plugin.on_message(self.event))
        await self.wait(entered[0])
        for index in range(5):
            await self.receive(self.message(f"new message {index}", f"new-{index}"))
        release[0].set()
        await self.wait(entered[1])
        prompt = self.plugin._llm_decision.call_args.args[1]
        self.assertNotIn("first message", prompt)
        self.assertIn("new message 4", prompt)
        release[1].set()
        await task
        self.assertEqual(self.plugin._llm_decision.await_count, 2)
        self.assertEqual(self.event.get_extra("oncue_decision")["_chat_context"].progress, self.chat.incoming_seq)

    async def test_invalid_initial_result_does_not_retry_because_messages_arrived(self):
        entered, release = self.pause_decisions("not valid JSON")
        task = self.start(self.plugin.on_message(self.event))
        await self.wait(entered[0])
        await self.receive(self.message("later message", "second"))
        release[0].set()
        await task
        self.plugin._llm_decision.assert_awaited_once()
        self.assertIsNone(self.event.get_extra("oncue_decision"))
        self.assertEqual(self.chat.pending_replies, {})

    async def test_failed_extra_result_does_not_fall_back_to_initial_reply(self):
        entered, release = self.pause_decisions(REPLY, "")
        task = self.start(self.plugin.on_message(self.event))
        await self.wait(entered[0])
        await self.receive(self.message("later message", "second"))
        release[0].set()
        await self.wait(entered[1])
        release[1].set()
        await task
        self.assertEqual(self.plugin._llm_decision.await_count, 2)
        self.assertIsNone(self.event.get_extra("oncue_decision"))
        self.assertFalse(self.event.is_at_or_wake_command)
        self.assertEqual(self.chat.pending_replies, {})
        self.assertFalse(self.chat.decision_inflight)

    async def test_cancelled_extra_request_releases_busy_state_without_reply(self):
        entered, release = self.pause_decisions(REPLY, UPDATED_REPLY)
        task = self.start(self.plugin.on_message(self.event))
        await self.wait(entered[0])
        await self.receive(self.message("later message", "second"))
        release[0].set()
        await self.wait(entered[1])
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(self.chat.decision_inflight)
        self.assertFalse(self.chat.lock.locked())
        self.assertEqual(self.chat.pending_replies, {})
        self.assertIsNone(self.event.get_extra("oncue_decision"))

    async def test_exception_building_extra_prompt_releases_busy_state(self):
        self.plugin._decision_card.side_effect = ["character card", RuntimeError("second prompt failed")]
        entered, release = self.pause_decisions(REPLY)
        task = self.start(self.plugin.on_message(self.event))
        await self.wait(entered[0])
        await self.receive(self.message("later message", "second"))
        release[0].set()
        with self.assertRaisesRegex(RuntimeError, "second prompt failed"):
            await task
        self.assertFalse(self.chat.decision_inflight)
        self.assertFalse(self.chat.lock.locked())
        self.assertEqual(self.chat.pending_replies, {})
        self.assertIsNone(self.event.get_extra("oncue_decision"))

    async def test_summon_during_extra_request_invalidates_its_late_reply(self):
        entered, release = self.pause_decisions(REPLY, UPDATED_REPLY)
        task = self.start(self.plugin.on_message(self.event))
        await self.wait(entered[0])
        await self.receive(self.message("later message", "second"))
        release[0].set()
        await self.wait(entered[1])
        summoned = self.message("answer now", "summoned", summoned=True)
        await self.receive(summoned)
        self.assertTrue(summoned.get_extra("oncue_decision")["should_reply"])
        release[1].set()
        await task
        self.assertIsNone(self.event.get_extra("oncue_decision"))
        self.assertEqual(self.chat.pending_replies, {})
        self.assertFalse(self.chat.decision_inflight)

    async def test_disabling_plugin_after_new_message_prevents_extra_request(self):
        entered, release = self.pause_decisions(REPLY)
        task = self.start(self.plugin.on_message(self.event))
        await self.wait(entered[0])
        await self.receive(self.message("later message", "second"))
        self.plugin.config["enable"] = False
        release[0].set()
        await task
        self.plugin._llm_decision.assert_awaited_once()
        self.assertIsNone(self.event.get_extra("oncue_decision"))
        self.assertEqual(self.chat.pending_replies, {})
        self.assertFalse(self.chat.decision_inflight)

    async def test_only_final_decision_retrieves_knowledge_and_reserves_reply(self):
        first = json.dumps({"should_reply": True, "reason": "initial", "need_kb": True, "kb_query": "old query"})
        second = json.dumps({"should_reply": True, "reason": "updated", "need_kb": True, "kb_query": "new query"})
        entered, release = self.pause_decisions(first, second)
        task = self.start(self.plugin.on_message(self.event))
        await self.wait(entered[0])
        await self.receive(self.message("later message", "second"))
        release[0].set()
        await self.wait(entered[1])
        self.plugin._kb_retrieve.assert_not_awaited()
        release[1].set()
        await task
        self.plugin._kb_retrieve.assert_awaited_once_with("new query")
        self.assertEqual(len(self.chat.pending_replies), 1)

    async def test_glance_runs_generation_once_after_extra_decision(self):
        self.chat.last_reply_ts = 0
        await self.receive(self.event)
        _native_pipeline(self.plugin)
        entered, release = self.pause_decisions(REPLY, UPDATED_REPLY)
        task = self.start(self.plugin._glance_due(self.chat_id))
        await self.wait(entered[0])
        await self.receive(self.message("later message", "second"))
        release[0].set()
        await self.wait(entered[1])
        self.plugin.native_generate.assert_not_awaited()
        release[1].set()
        await task
        self.plugin.native_generate.assert_awaited_once()
        self.plugin.context.send_message.assert_awaited_once()
        self.assertEqual(len(self.chat.reply_ts), 1)
        self.assertEqual(self.chat.pending_replies, {})

    async def test_glance_leaves_messages_during_extra_request_for_next_check(self):
        self.chat.last_reply_ts = 0
        await self.receive(self.event)
        _native_pipeline(self.plugin)
        entered, release = self.pause_decisions(SILENT, SILENT)
        task = self.start(self.plugin._glance_due(self.chat_id))
        await self.wait(entered[0])
        await self.receive(self.message("second message", "second"))
        release[0].set()
        await self.wait(entered[1])
        checked_progress = self.chat.incoming_seq
        await self.receive(self.message("third message", "third"))
        release[1].set()
        await task
        self.assertEqual(self.plugin.glance_last_check[self.chat_id], checked_progress)
        self.assertGreater(self.chat.incoming_seq, checked_progress)
        self.now = self.plugin.glance_next_due[self.chat_id]
        await self.plugin._glance_due(self.chat_id)
        self.assertEqual(self.plugin._llm_decision.await_count, 3)
        self.assertIn("third message", self.plugin._llm_decision.call_args.args[1])
        self.assertEqual(self.plugin.glance_last_check[self.chat_id], self.chat.incoming_seq)

    async def test_failed_extra_glance_does_not_advance_initial_progress(self):
        self.chat.last_reply_ts = 0
        await self.receive(self.event)
        _native_pipeline(self.plugin)
        self.plugin.glance_last_check[self.chat_id] = 0
        entered, release = self.pause_decisions(REPLY, "")
        task = self.start(self.plugin._glance_due(self.chat_id))
        await self.wait(entered[0])
        await self.receive(self.message("second message", "second"))
        release[0].set()
        await self.wait(entered[1])
        release[1].set()
        await task
        self.assertEqual(self.plugin.glance_last_check[self.chat_id], 0)
        self.plugin.native_generate.assert_not_awaited()
        self.plugin.context.send_message.assert_not_awaited()

    async def test_older_target_is_resolved_from_current_snapshot(self):
        self.chat.last_reply_ts = 0
        older = self.message("older question", "older", sender_id="bob")
        await self.receive(older)
        self.chat.last_reply_ts = 950.0
        self.plugin._llm_decision.return_value = json.dumps({
            "should_reply": True, "reason": "answer the earlier question", "target_message_id": "older",
        })
        await self.receive(self.event)
        decision = self.event.get_extra("oncue_decision")
        self.assertEqual(decision["_target"].message_id, "older")
        req = self.request()
        await self.plugin.inject_stage_direction(self.event, req)
        self.assertEqual(req.prompt, "first message")
        self.assertIn("older question", _request_text(req))
        self.assertIn('消息ID="older"', _request_text(req))
        self.assertNotIn("older question", req.system_prompt)
        self.assertTrue(req.extra_user_content_parts[-1]._no_save)

    async def test_target_remains_available_when_full_background_is_disabled(self):
        self.chat.last_reply_ts = 0
        await self.receive(self.message("target question", "older", sender_id="bob"))
        await self.receive(self.message("unrelated background", "other", sender_id="carol"))
        self.chat.last_reply_ts = 950.0
        self.plugin.config["enable_context_injection"] = False
        self.plugin._llm_decision.return_value = json.dumps({
            "should_reply": True, "reason": "answer the earlier question", "target_message_id": "older",
        })
        await self.receive(self.event)
        req = self.request()
        await self.plugin.inject_stage_direction(self.event, req)
        content = _request_text(req)
        self.assertIn("target question", content)
        self.assertIn("ID=bob", content)
        self.assertNotIn("unrelated background", content)
        self.assertEqual(req.prompt, "first message")
        self.assertTrue(req.extra_user_content_parts[-1]._no_save)

    async def test_explicit_target_survives_native_history_eviction_with_background_disabled(self):
        self.chat.last_reply_ts = 0
        target_text = "BEGIN-TARGET " + "x" * 200 + " END-TARGET"
        await self.receive(self.message(target_text, "older", sender_id="bob"))
        await self.receive(self.message("unrelated background", "other", sender_id="carol"))
        self.chat.last_reply_ts = 950.0
        self.plugin.config.update(enable_context_injection=False, message_char_limit=80)
        self.plugin._llm_decision.return_value = json.dumps({
            "should_reply": True, "reason": "answer the earlier question", "target_message_id": "older",
        })
        await self.receive(self.event)
        req = self.request()
        req.contexts = [{"role": "user", "message_id": "older", "content": target_text}]
        await self.plugin.inject_stage_direction(self.event, req)
        req.contexts.clear()
        background = "\n".join(part.text for part in req.extra_user_content_parts)
        self.assertIn("BEGIN-TARGET", background)
        self.assertIn("END-TARGET", background)
        self.assertIn("中间省略", background)
        self.assertNotIn(target_text, background)
        self.assertNotIn("unrelated background", background)
        self.assertIn('消息ID="older"', background)
        self.assertEqual(req.prompt, "first message")
        self.assertTrue(req.extra_user_content_parts[-1]._no_save)

    async def test_missing_platform_id_can_be_targeted_by_displayed_internal_id(self):
        self.chat.last_reply_ts = 0
        older = self.message("question without platform id")
        await self.receive(older)
        internal_id = older.get_extra("oncue_record_id")
        self.assertTrue(internal_id.startswith("internal:"))
        self.chat.last_reply_ts = 950.0
        self.plugin._llm_decision.return_value = json.dumps({
            "should_reply": True, "reason": "answer an earlier question", "target_message_id": internal_id,
        })
        await self.receive(self.event)
        self.assertEqual(self.event.get_extra("oncue_decision")["_target"].record_id, internal_id)

    async def test_unknown_target_cannot_authorize_a_reply(self):
        self.plugin._llm_decision.return_value = json.dumps({
            "should_reply": True, "reason": "question", "target_message_id": "invented-id",
        })
        await self.receive(self.event)
        self.assertIsNone(self.event.get_extra("oncue_decision"))
        self.assertEqual(self.chat.pending_replies, {})

    async def test_target_outside_reference_window_cannot_authorize_a_reply(self):
        self.chat.last_reply_ts = 0
        await self.receive(self.message("out of range", "older"))
        self.chat.last_reply_ts = 950.0
        self.plugin.config["context_message_count"] = 1
        self.plugin._llm_decision.return_value = json.dumps({
            "should_reply": True, "reason": "question", "target_message_id": "older",
        })
        await self.receive(self.event)
        self.assertIsNone(self.event.get_extra("oncue_decision"))
        self.assertEqual(self.chat.pending_replies, {})

    async def test_bot_message_cannot_be_selected_as_user_target(self):
        self.plugin._record_reply(self.chat, "earlier bot answer", 900.0)
        bot_id = self.chat.messages[-1]["record_id"]
        self.plugin._llm_decision.return_value = json.dumps({
            "should_reply": True, "reason": "question", "target_message_id": bot_id,
        })
        await self.receive(self.event)
        self.assertIsNone(self.event.get_extra("oncue_decision"))
        self.assertEqual(self.chat.pending_replies, {})

    async def test_empty_target_allows_a_natural_reply(self):
        self.plugin._llm_decision.return_value = json.dumps({
            "should_reply": True, "reason": "join the conversation", "target_message_id": "",
        })
        await self.receive(self.event)
        decision = self.event.get_extra("oncue_decision")
        self.assertTrue(decision["should_reply"])
        self.assertIsNone(decision.get("_target"))

    async def test_native_decoration_conflict_silences_old_target_without_rewriting_event(self):
        self.chat.last_reply_ts = 0
        await self.receive(self.message("earlier question", "older", sender_id="bob"))
        self.chat.last_reply_ts = 950.0
        self.plugin._llm_decision.return_value = json.dumps({
            "should_reply": True, "reason": "answer earlier question", "target_message_id": "older",
        })
        for setting in ("reply_with_mention", "reply_with_quote"):
            with self.subTest(setting=setting):
                self.chat.cooldown_until = 0
                self.plugin.context.get_config = lambda **kwargs: {"platform_settings": {setting: True}}
                current = self.message("current question", f"current-{setting}")
                message_id, sender_id = current.message_obj.message_id, current.get_sender_id()
                with patch.object(plugin_module.logger, "warning") as warning:
                    await self.receive(current)
                    warning.assert_called_once()
                    self.assertIn("回应目标冲突", str(warning.call_args))
                self.assertIsNone(current.get_extra("oncue_decision"))
                self.assertFalse(current.is_at_or_wake_command)
                self.assertEqual(self.chat.pending_replies, {})
                self.assertEqual(current.message_obj.message_id, message_id)
                self.assertEqual(current.get_sender_id(), sender_id)

    async def test_native_decorations_allow_targeting_the_current_message(self):
        self.plugin.context.get_config = lambda **kwargs: {
            "platform_settings": {"reply_with_mention": True, "reply_with_quote": True},
        }
        self.plugin._llm_decision.return_value = json.dumps({
            "should_reply": True, "reason": "answer this message", "target_message_id": "first",
        })
        with patch.object(plugin_module.logger, "warning") as warning:
            await self.receive(self.event)
            warning.assert_not_called()
        decision = self.event.get_extra("oncue_decision")
        self.assertTrue(decision["should_reply"])
        self.assertEqual(decision["_target"].message_id, "first")
        self.assertEqual(self.event.message_obj.message_id, "first")
        self.assertEqual(self.event.get_sender_id(), "alice")

    async def test_glance_older_target_is_not_rejected_by_native_decoration_settings(self):
        self.chat.last_reply_ts = 0
        await self.receive(self.message("earlier question", "older", sender_id="bob"))
        await self.receive(self.event)
        self.plugin.context.get_config = lambda **kwargs: {
            "platform_settings": {"reply_with_mention": True, "reply_with_quote": True},
        }
        _native_pipeline(self.plugin)
        self.plugin._llm_decision.return_value = json.dumps({
            "should_reply": True, "reason": "answer earlier question", "target_message_id": "older",
        })
        with patch.object(plugin_module.logger, "warning") as warning:
            await self.plugin._glance_due(self.chat_id)
            warning.assert_not_called()
        self.plugin.native_generate.assert_awaited_once()
        self.plugin.context.send_message.assert_awaited_once()
        self.assertIn("earlier question", self.plugin.native_generate.call_args.kwargs["prompt"])

    async def test_messages_only_during_final_knowledge_lookup_leave_glance_progress_behind(self):
        self.chat.last_reply_ts = 0
        await self.receive(self.event)
        checked_progress = self.chat.incoming_seq
        _native_pipeline(self.plugin)
        self.plugin._llm_decision.return_value = json.dumps({
            "should_reply": True, "reason": "need knowledge", "need_kb": True, "kb_query": "query",
        })
        entered, release = asyncio.Event(), asyncio.Event()

        async def lookup(*args):
            entered.set()
            await release.wait()
            return "knowledge"

        self.plugin._kb_retrieve.side_effect = lookup
        task = self.start(self.plugin._glance_due(self.chat_id))
        await self.wait(entered)
        self.assertFalse(self.chat.lock.locked())
        await self.receive(self.message("message after the decision finished", "later"))
        release.set()
        await task
        self.plugin._llm_decision.assert_awaited_once()
        self.plugin._kb_retrieve.assert_awaited_once_with("query")
        self.plugin.native_generate.assert_awaited_once()
        self.assertEqual(self.plugin.glance_last_check[self.chat_id], checked_progress)
        self.assertGreater(self.chat.incoming_seq, checked_progress)
        self.assertNotIn("message after the decision finished", self.plugin.native_generate.call_args.kwargs["prompt"])

    async def test_invalid_target_types_cannot_authorize_a_reply(self):
        for index, target in enumerate((True, False, ["first"], {"message_id": "first"})):
            with self.subTest(target=target):
                self.chat.cooldown_until = 0
                self.plugin._llm_decision.return_value = json.dumps({
                    "should_reply": True, "reason": "answer this message", "target_message_id": target,
                })
                current = self.message("current question", f"current-{index}")
                await self.receive(current)
                self.assertIsNone(current.get_extra("oncue_decision"))
                self.assertFalse(current.is_at_or_wake_command)
                self.assertEqual(self.chat.pending_replies, {})

    async def test_hidden_platform_record_id_cannot_replace_displayed_message_id(self):
        self.plugin._llm_decision.return_value = json.dumps({
            "should_reply": True, "reason": "answer this message", "target_message_id": "platform:first",
        })
        with self.assertLogs(plugin_module.logger, level="WARNING"):
            await self.receive(self.event)
        self.assertEqual(self.event.get_extra("oncue_record_id"), "platform:first")
        self.assertIsNone(self.event.get_extra("oncue_decision"))
        self.assertEqual(self.chat.pending_replies, {})

    async def test_disabling_plugin_during_extra_request_invalidates_final_result(self):
        entered, release = self.pause_decisions(REPLY, UPDATED_REPLY)
        task = self.start(self.plugin.on_message(self.event))
        await self.wait(entered[0])
        await self.receive(self.message("new information", "second"))
        release[0].set()
        await self.wait(entered[1])
        self.plugin.config["enable"] = False
        release[1].set()
        await task
        self.assertIsNone(self.event.get_extra("oncue_decision"))
        self.assertEqual(self.chat.pending_replies, {})
        self.assertFalse(self.chat.decision_inflight)
        self.assertFalse(self.chat.lock.locked())

    async def test_disabling_glance_during_extra_request_preserves_unchecked_progress(self):
        self.chat.last_reply_ts = 0
        await self.receive(self.event)
        _native_pipeline(self.plugin)
        entered, release = self.pause_decisions(REPLY, UPDATED_REPLY)
        task = self.start(self.plugin._glance_due(self.chat_id))
        await self.wait(entered[0])
        await self.receive(self.message("new information", "second"))
        release[0].set()
        await self.wait(entered[1])
        self.plugin.config["enable_glance"] = False
        release[1].set()
        await task
        self.assertNotIn(self.chat_id, self.plugin.glance_last_check)
        self.plugin.native_generate.assert_not_awaited()
        self.plugin.context.send_message.assert_not_awaited()
        self.assertEqual(self.chat.pending_replies, {})
        self.assertFalse(self.chat.decision_inflight)
        self.assertFalse(self.chat.glance_inflight)

    async def test_actual_reply_during_extra_request_invalidates_its_late_result(self):
        entered, release = self.pause_decisions(REPLY, UPDATED_REPLY)
        task = self.start(self.plugin.on_message(self.event))
        await self.wait(entered[0])
        await self.receive(self.message("new information", "second"))
        release[0].set()
        await self.wait(entered[1])
        previous = self.message("previous request", "previous")
        previous.set_extra("oncue_decision", {"should_reply": True})
        previous._has_send_oper = True
        previous.result.chain = [Plain("a reply was already sent")]
        await asyncio.wait_for(self.plugin.mark_reply_sent(previous), 1)
        release[1].set()
        await task
        self.assertIsNone(self.event.get_extra("oncue_decision"))
        self.assertEqual(len(self.chat.reply_ts), 1)
        self.assertEqual(self.chat.messages[-1]["text"], "a reply was already sent")
        self.assertEqual(self.chat.pending_replies, {})
        self.assertFalse(self.chat.decision_inflight)
        self.assertFalse(self.chat.lock.locked())

    def test_schema_exposes_enabled_extra_decision_switch(self):
        schema_path = Path(__file__).resolve().parents[1] / "_conf_schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        switch = schema["config_decision"]["items"]["enable_extra_decision"]
        self.assertTrue(switch["default"])
        self.assertEqual(switch["description"], "允许 1 次连续的额外决策请求")


if __name__ == "__main__":
    unittest.main()
