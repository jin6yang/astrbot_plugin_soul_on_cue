"""决策卡唯一来源、缺失时跳过决策，以及配置错误日志。"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from test_timing import _event, _native_pipeline, plugin_module


class DecisionCardTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.now = 1000.0
        clock = patch.object(plugin_module, "time", SimpleNamespace(time=lambda: self.now, monotonic=lambda: self.now))
        clock.start()
        self.addCleanup(clock.stop)
        self.personas = {"decision": {"prompt": "喜欢游戏，倾向回应熟人。"}}
        self.lookup = Mock(side_effect=self.personas.get)
        self.context = SimpleNamespace(
            persona_manager=SimpleNamespace(get_persona_v3_by_id=self.lookup),
            get_current_chat_provider_id=AsyncMock(return_value="model"),
            llm_generate=AsyncMock(return_value=SimpleNamespace(completion_text='{"should_reply": true, "reason": "想接话"}')),
        )
        self.plugin = plugin_module.OnCuePlugin(self.context, {
            "config_character": {"persona_id": "decision"},
            "message_wait_seconds": 0,
        })
        self.event = _event()
        self.chat = self.plugin._chat(self.event.unified_msg_origin)

    async def decide(self, key=None):
        return await self.plugin._decide(self.event, self.chat, "群聊背景", key)

    def assert_no_decision_effects(self):
        self.context.llm_generate.assert_not_awaited()
        self.context.get_current_chat_provider_id.assert_not_awaited()
        self.assertFalse(self.event.is_at_or_wake_command)
        self.assertIsNone(self.event.get_extra("oncue_decision"))
        self.assertEqual(self.chat.pending_replies, {})
        self.assertEqual(self.chat.cooldown_until, 0.0)
        self.assertEqual(self.chat.backoffs, {})
        self.assertFalse(self.chat.decision_inflight)

    async def test_selected_persona_is_used_verbatim_without_an_extra_model_call(self):
        card = "  完整的决策卡正文\n保留段落。  "
        self.personas["decision"]["prompt"] = card
        self.assertTrue(await self.decide())
        self.lookup.assert_called_once_with("decision")
        self.context.llm_generate.assert_awaited_once()
        self.assertIn(card, self.context.llm_generate.call_args.kwargs["prompt"])

    async def test_no_selection_skips_llm_and_logs_actionable_error(self):
        self.plugin.config["config_character"]["persona_id"] = "  "
        with self.assertLogs(plugin_module.logger, level="ERROR") as logs:
            self.assertFalse(await self.decide("dense"))
        self.assertIn("未选择决策卡", logs.output[0])
        self.assertIn("角色设定 → 决策卡", logs.output[0])
        self.lookup.assert_not_called()
        self.assert_no_decision_effects()

    async def test_deleted_persona_skips_llm_and_identifies_the_missing_selection(self):
        self.personas.clear()
        with self.assertLogs(plugin_module.logger, level="ERROR") as logs:
            self.assertFalse(await self.decide("echo:hello"))
        self.assertIn("不存在或已被删除", logs.output[0])
        self.assertIn("decision", logs.output[0])
        self.assert_no_decision_effects()

    async def test_empty_persona_content_skips_llm(self):
        for content in (None, "", " \n\t ", 123):
            with self.subTest(content=content):
                self.personas["decision"]["prompt"] = content
                with patch.object(plugin_module.logger, "error"):
                    self.assertFalse(await self.decide())
                self.assert_no_decision_effects()

    async def test_persona_lookup_failure_is_reported_without_fallback(self):
        self.lookup.side_effect = RuntimeError("persona manager unavailable")
        with self.assertLogs(plugin_module.logger, level="ERROR") as logs:
            self.assertFalse(await self.decide())
        self.assertIn("读取决策卡失败", logs.output[0])
        self.assert_no_decision_effects()

    async def test_same_issue_logs_once_across_multiple_chats(self):
        self.personas.clear()
        other = _event()
        other.unified_msg_origin = "test:GroupMessage:2"
        with patch.object(plugin_module.logger, "error") as error:
            await self.decide()
            await self.decide()
            await self.plugin._decide(other, self.plugin._chat(other.unified_msg_origin), "群聊背景", None)
        error.assert_called_once()
        self.assert_no_decision_effects()

    async def test_changed_issue_is_reported_immediately(self):
        self.personas.clear()
        with patch.object(plugin_module.logger, "error") as error:
            await self.decide()
            self.plugin.config["config_character"]["persona_id"] = "another-deleted-persona"
            await self.decide()
            self.plugin.config["config_character"]["persona_id"] = ""
            await self.decide()
        self.assertEqual(error.call_count, 3)
        self.assert_no_decision_effects()

    async def test_repaired_selection_resumes_decision_without_old_cooldown(self):
        self.personas.clear()
        with patch.object(plugin_module.logger, "error"):
            self.assertFalse(await self.decide("dense"))
        self.personas["decision"] = {"prompt": "修复后的决策卡"}
        with self.assertLogs(plugin_module.logger, level="INFO") as logs:
            self.assertTrue(await self.decide("dense"))
        self.assertTrue(any("决策卡已恢复可用" in line for line in logs.output))
        self.context.llm_generate.assert_awaited_once()
        self.assertIn("修复后的决策卡", self.context.llm_generate.call_args.kwargs["prompt"])

    async def test_persona_edits_and_deletion_are_visible_without_stale_cache(self):
        self.assertEqual(await self.plugin._decision_card(), self.personas["decision"]["prompt"])
        self.personas["decision"]["prompt"] = "编辑后的内容"
        self.assertEqual(await self.plugin._decision_card(), "编辑后的内容")
        self.personas.clear()
        with self.assertLogs(plugin_module.logger, level="ERROR"):
            self.assertIsNone(await self.plugin._decision_card())

    async def test_error_is_reported_again_if_it_recurs_after_recovery(self):
        self.personas.clear()
        with patch.object(plugin_module.logger, "error") as error:
            await self.plugin._decision_card()
            self.personas["decision"] = {"prompt": "可用"}
            await self.plugin._decision_card()
            self.personas.clear()
            await self.plugin._decision_card()
        self.assertEqual(error.call_count, 2)

    async def test_summon_decision_mode_does_not_fall_through_to_direct_reply(self):
        self.personas.clear()
        self.plugin.config["force_reply_when_summoned"] = False
        self.event = _event(summoned=True)
        with self.assertLogs(plugin_module.logger, level="ERROR"):
            await self.plugin.on_message(self.event)
        self.assert_no_decision_effects()
        self.assertFalse(self.event.stopped)

    async def test_direct_summon_does_not_require_a_decision_card(self):
        self.personas.clear()
        self.event = _event(summoned=True)
        with self.assertNoLogs(plugin_module.logger, level="ERROR"):
            await self.plugin.on_message(self.event)
        self.assertTrue(self.event.is_at_or_wake_command)
        self.assertTrue(self.event.get_extra("oncue_decision")["should_reply"])
        self.lookup.assert_not_called()
        self.context.llm_generate.assert_not_awaited()

    async def test_missing_card_finishes_message_wait_without_backoff(self):
        self.personas.clear()
        pending = plugin_module.MessageWait(
            deadline=self.now, last_activity_at=self.now - 5,
            latest_item={}, reply_version=self.chat.reply_version,
        )
        self.chat.message_wait = pending
        self.chat.last_reply_ts = self.now - 1
        with self.assertLogs(plugin_module.logger, level="ERROR"):
            await self.plugin._wait_for_messages(self.event, self.chat, pending)
        self.assertIsNone(self.chat.message_wait)
        self.assert_no_decision_effects()

    async def test_missing_card_does_not_consume_glance_activity_or_start_generation(self):
        self.personas.clear()
        _native_pipeline(self.plugin)
        self.plugin._glance_interval_seconds = lambda: 900
        self.chat.last_activity_ts = 950
        self.plugin.glance_last_check[self.event.unified_msg_origin] = 900
        with self.assertLogs(plugin_module.logger, level="ERROR"):
            await self.plugin._glance_due(self.event.unified_msg_origin)
        self.assertEqual(self.plugin.glance_last_check[self.event.unified_msg_origin], 900)
        self.plugin._run_native_reply.assert_not_awaited()
        self.assertFalse(self.chat.glance_inflight)
        self.assert_no_decision_effects()

    async def test_initialize_reports_missing_card_and_only_loads_glance_state(self):
        self.personas.clear()
        self.plugin._glance_loop = AsyncMock()
        with TemporaryDirectory() as directory:
            # 即使原浓缩缓存损坏，也不会再读取或处理它。
            old_cache = Path(directory) / "persona_cards.json"
            old_cache.write_text("broken JSON", encoding="utf-8")
            star_tools = SimpleNamespace(get_data_dir=lambda name: Path(directory))
            with patch.object(plugin_module, "StarTools", star_tools):
                try:
                    with self.assertLogs(plugin_module.logger, level="ERROR") as logs:
                        await self.plugin.initialize()
                    self.assertEqual(len(logs.output), 1)
                    self.assertIn("不存在或已被删除", logs.output[0])
                    self.assertIsNotNone(self.plugin.glance_task)
                    self.assertEqual(old_cache.read_text(encoding="utf-8"), "broken JSON")
                finally:
                    await self.plugin.terminate()


if __name__ == "__main__":
    unittest.main()
