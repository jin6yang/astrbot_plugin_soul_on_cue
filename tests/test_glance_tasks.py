"""验证原生主动任务只接收本插件临时输入，不把任务总结当作群聊回复。"""

import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, patch

from test_timing import _event, plugin_module


def message_text(message):
    content = message.content
    if isinstance(content, str):
        return content
    return "\n".join(
        part.get("text", "") if isinstance(part, dict) else getattr(part, "text", "")
        for part in content or []
    )


class GlanceTaskContextTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.plugin = plugin_module.OnCuePlugin(
            SimpleNamespace(send_message=AsyncMock(return_value=True)),
            {
                "enable": True, "enable_glance": True,
                "enable_context_injection": True, "enable_speaker_routing": True,
                "message_wait_seconds": 0,
            },
        )
        self.native_event = _event("原生任务说明")
        self.chat_id = self.native_event.unified_msg_origin
        self.chat = self.plugin._chat(self.chat_id)
        for index, body in enumerate(("较早的话题", "刚刚的新消息")):
            event = _event(body)
            event.message_obj.message_id = str(index + 1)
            self.plugin._append_message(self.chat, self.plugin._event_to_item(event, self.chat))
        self.trigger = self.plugin._create_glance_trigger(self.chat)
        if self.trigger is None:
            self.chat.glance_target = {"umo": self.chat_id, "self_id": "bot", "group_id": "1"}
            self.trigger = self.plugin._create_glance_trigger(self.chat)
        self.snapshot = self.plugin._context_snapshot(self.chat, "bot")
        self.decision = {
            "should_reply": True, "speaker": "小星", "mood": "好奇",
            "kb_text": "本轮检索资料", "_backoff_key": "glance",
            "_reply_version": self.chat.reply_version, "_chat_context": self.snapshot,
            "_target": None,
        }
        self.trigger.set_extra("oncue_decision", self.decision)
        self.run = SimpleNamespace(
            trigger=self.trigger, job_id="own-glance-job", native_event=self.native_event,
            injected_contexts=[], started_at=0.0,
        )
        self.plugin.native_tasks = SimpleNamespace(
            owned_run=lambda event: self.run if event is self.native_event else None,
        )
        self.native_event.set_extra("cron_job", {"id": self.run.job_id})

    async def asyncSetUp(self):
        self.assertTrue(await self.plugin._prepare_glance_input(self.trigger))

    @staticmethod
    def context():
        native_system = plugin_module.Message(role="system", content="原生任务系统规则")
        foreign_part = plugin_module.TextPart(text="其它插件的输入")
        native_user = plugin_module.Message(role="user", content=[foreign_part])
        return SimpleNamespace(messages=[native_system, native_user])

    async def test_owned_task_gets_shared_director_and_group_context_without_rewriting_native_input(self):
        context = self.context()
        original = list(context.messages)
        native_part = original[1].content[0]
        cached_rows = list(self.chat.messages)

        await self.plugin.prepare_glance_task(self.native_event, context)

        self.assertIs(context.messages[0], original[0])
        self.assertIs(context.messages[1], original[1])
        self.assertEqual(message_text(original[0]), "原生任务系统规则")
        self.assertIs(original[1].content[0], native_part)
        self.assertEqual(native_part.text, "其它插件的输入")
        added = context.messages[2:]
        text = "\n".join(message_text(message) for message in added)
        for expected in ("小星", "好奇", "本轮检索资料", "较早的话题", "刚刚的新消息"):
            self.assertIn(expected, text)
        self.assertTrue(added)
        self.assertTrue(all(part._no_save for message in added for part in message.content))
        self.assertEqual(list(self.chat.messages), cached_rows)
        self.assertEqual(list(self.chat.reply_ts), [])
        self.plugin.context.send_message.assert_not_awaited()

    async def test_foreign_task_does_not_receive_our_input_or_start_image_work(self):
        context = self.context()
        original = list(context.messages)
        foreign_event = _event("其它任务")
        foreign_event.set_extra("cron_job", {"id": "foreign-job"})
        self.plugin._prepare_reply_context = AsyncMock()

        await self.plugin.prepare_glance_task(foreign_event, context)

        self.assertEqual(context.messages, original)
        self.plugin._prepare_reply_context.assert_not_awaited()

    async def test_repeated_hook_for_the_same_context_does_not_duplicate_input(self):
        context = self.context()
        await self.plugin.prepare_glance_task(self.native_event, context)
        original = list(context.messages)

        await self.plugin.prepare_glance_task(self.native_event, context)

        self.assertEqual(context.messages, original)
        self.assertEqual(sum("较早的话题" in message_text(message) for message in context.messages), 1)

    async def test_native_or_other_plugin_images_are_preserved(self):
        self.plugin.config["image_mode"] = "不转述（仅占位）"
        context = self.context()
        image = plugin_module.ImageURLPart(image_url={"url": "foreign-image"})
        context.messages[1].content.append(image)

        await self.plugin.prepare_glance_task(self.native_event, context)

        self.assertIs(context.messages[1].content[-1], image)
        self.assertEqual(image.image_url.url, "foreign-image")
        self.assertFalse(any(
            isinstance(part, plugin_module.ImageURLPart)
            for message in context.messages[2:]
            for part in message.content
        ))

    async def test_a_new_native_context_can_receive_the_same_fixed_snapshot_once(self):
        first = self.context()
        second = self.context()

        await self.plugin.prepare_glance_task(self.native_event, first)
        await self.plugin.prepare_glance_task(self.native_event, second)

        for context in (first, second):
            self.assertEqual(sum("较早的话题" in message_text(message) for message in context.messages), 1)

    async def test_disabled_or_superseded_task_does_not_prepare_context(self):
        for change in ("disabled", "superseded"):
            with self.subTest(change=change):
                context = self.context()
                original = list(context.messages)
                self.plugin.config["enable_glance"] = change != "disabled"
                self.chat.reply_version = self.decision["_reply_version"] + (change == "superseded")
                self.plugin._prepare_reply_context = AsyncMock()

                await self.plugin.prepare_glance_task(self.native_event, context)

                self.assertEqual(context.messages, original)
                self.plugin._prepare_reply_context.assert_not_awaited()

    async def test_disabled_history_keeps_only_the_explicit_response_target(self):
        self.plugin.config["enable_context_injection"] = False
        self.decision["_target"] = self.snapshot.rows[0]
        self.assertTrue(await self.plugin._prepare_glance_input(self.trigger))
        context = self.context()

        await self.plugin.prepare_glance_task(self.native_event, context)

        text = "\n".join(message_text(message) for message in context.messages[2:])
        self.assertIn("较早的话题", text)
        self.assertNotIn("刚刚的新消息", text)

    async def test_input_preparation_failure_prevents_native_task_creation(self):
        self.plugin._prepare_reply_context = AsyncMock(side_effect=RuntimeError("caption failed"))
        self.plugin.native_tasks.run = AsyncMock()

        with self.assertRaisesRegex(RuntimeError, "caption failed"):
            await self.plugin._run_glance_task(self.trigger)

        self.plugin.native_tasks.run.assert_not_awaited()

    async def test_hook_uses_prepared_snapshot_without_starting_new_image_work(self):
        self.plugin._prepare_reply_context = AsyncMock(side_effect=AssertionError("late preparation"))
        self.plugin._prepare_images = AsyncMock(side_effect=AssertionError("late caption"))
        self.plugin._append_message(self.chat, self.plugin._event_to_item(_event("快照之后的消息"), self.chat))
        context = self.context()

        await self.plugin.prepare_glance_task(self.native_event, context)

        text = "\n".join(message_text(message) for message in context.messages[2:])
        self.assertIn("较早的话题", text)
        self.assertNotIn("快照之后的消息", text)
        self.plugin._prepare_reply_context.assert_not_awaited()
        self.plugin._prepare_images.assert_not_awaited()

    async def test_task_summary_is_not_recorded_as_a_sent_reply(self):
        context = self.context()
        cached_rows = list(self.chat.messages)
        response = SimpleNamespace(role="assistant", completion_text="已向群里发送了回应，这是任务总结")
        budget = plugin_module.CaptionRound(3)
        self.trigger.set_extra("oncue_image_round", budget)

        await self.plugin.close_image_requests(self.native_event, context, response)

        self.assertTrue(budget.closed)
        self.assertIsNone(self.trigger.get_extra("oncue_image_round"))
        self.assertEqual(list(self.chat.messages), cached_rows)
        self.assertEqual(list(self.chat.reply_ts), [])
        self.assertFalse(self.plugin._is_observing(self.chat, 1000.0))
        self.plugin.context.send_message.assert_not_awaited()

    async def test_public_tool_result_does_not_imply_a_structured_delivery_receipt(self):
        cached_rows = list(self.chat.messages)
        tool = SimpleNamespace(name="send_message_to_user")
        args = {"messages": [{"type": "plain", "text": "工具正文"}]}
        result = SimpleNamespace(
            isError=False,
            content=[SimpleNamespace(type="text", text=f"Message sent to session {self.chat_id}")],
        )

        await self.plugin.observe_glance_tool(self.native_event, tool, args, result)

        self.assertEqual(list(self.chat.messages), cached_rows)
        self.assertEqual(list(self.chat.reply_ts), [])
        self.plugin.context.send_message.assert_not_awaited()


class GlancePreflightTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.now = 1000.0
        clock = patch.object(plugin_module, "time", SimpleNamespace(time=lambda: self.now))
        clock.start()
        self.addCleanup(clock.stop)
        self.native_config = {
            "provider_settings": {"enable": True},
            "agent_runner": {"runner_type": "local"},
            "plugin_set": ["*"],
        }
        self.manager = SimpleNamespace(
            add_active_job=AsyncMock(), run_job_now=AsyncMock(), delete_job=AsyncMock(),
        )
        self.plugin = plugin_module.OnCuePlugin(
            SimpleNamespace(
                cron_manager=self.manager,
                get_config=lambda **kwargs: self.native_config,
                send_message=AsyncMock(return_value=True),
            ),
            {
                "enable": True, "enable_glance": True,
                "enable_dense_trigger": False, "enable_echo_trigger": False,
                "message_wait_seconds": 0,
            },
        )
        event = _event("尚未判断的群聊消息")
        self.chat_id = event.unified_msg_origin
        self.chat = self.plugin._chat(self.chat_id)
        self.chat.glance_target = {"umo": self.chat_id, "self_id": "bot", "group_id": "1"}
        self.plugin._append_message(self.chat, self.plugin._event_to_item(event, self.chat))
        self.plugin.glance_last_check[self.chat_id] = 0
        self.plugin._build_prompt = AsyncMock(return_value="decision prompt")
        self.plugin._llm_decision = AsyncMock(return_value='{"should_reply": false}')
        self.plugin.native_tasks.run = AsyncMock()
        self.plugin._glance_interval_seconds = lambda chat=None: 900.0
        self.session_plugin = SimpleNamespace(is_plugin_enabled_for_session=AsyncMock(return_value=True))
        self.session_ai = SimpleNamespace(
            is_session_enabled=AsyncMock(return_value=True),
            is_llm_enabled_for_session=AsyncMock(return_value=True),
        )
        modules = {}
        for name, attribute, value in (
            ("astrbot.core.star.session_plugin_manager", "SessionPluginManager", self.session_plugin),
            ("astrbot.core.star.session_llm_manager", "SessionServiceManager", self.session_ai),
        ):
            module = ModuleType(name)
            setattr(module, attribute, value)
            modules[name] = module
        managers = patch.dict(sys.modules, modules)
        managers.start()
        self.addCleanup(managers.stop)

    async def assert_preflight_skipped(self):
        await self.plugin._glance_due(self.chat_id)
        self.plugin._build_prompt.assert_not_awaited()
        self.plugin._llm_decision.assert_not_awaited()
        self.plugin.native_tasks.run.assert_not_awaited()
        self.manager.add_active_job.assert_not_awaited()
        self.assertEqual(self.plugin.glance_last_check[self.chat_id], 0)
        self.assertEqual(self.chat.incoming_seq, 1)
        self.assertEqual(self.chat.pending_replies, {})
        self.assertEqual(list(self.chat.reply_ts), [])
        self.assertFalse(self.chat.glance_inflight)
        self.assertFalse(self.chat.lock.locked())

    async def test_missing_future_task_api_skips_before_decision(self):
        self.plugin.context.cron_manager = None
        with self.assertLogs(plugin_module.logger, level="ERROR"):
            await self.assert_preflight_skipped()

    async def test_incomplete_future_task_api_skips_before_decision(self):
        self.manager.run_job_now = None
        with self.assertLogs(plugin_module.logger, level="ERROR"):
            await self.assert_preflight_skipped()

    async def test_third_party_agent_skips_before_decision(self):
        self.native_config["agent_runner"]["runner_type"] = "third_party"
        with self.assertLogs(plugin_module.logger, level="ERROR"):
            await self.assert_preflight_skipped()

    async def test_disabled_provider_skips_before_decision(self):
        self.native_config["provider_settings"]["enable"] = False
        await self.assert_preflight_skipped()

    async def test_excluded_plugin_configuration_skips_before_decision(self):
        self.native_config["plugin_set"] = ["another-plugin"]
        await self.assert_preflight_skipped()

    async def test_disabled_session_plugin_skips_before_decision(self):
        self.session_plugin.is_plugin_enabled_for_session.return_value = False
        await self.assert_preflight_skipped()
        self.session_plugin.is_plugin_enabled_for_session.assert_awaited_once_with(
            self.chat_id, "astrbot_plugin_soul_on_cue",
        )
        self.session_ai.is_llm_enabled_for_session.assert_not_awaited()

    async def test_disabled_session_skips_before_decision_even_when_ai_is_enabled(self):
        self.session_ai.is_session_enabled.return_value = False
        await self.assert_preflight_skipped()
        self.session_ai.is_session_enabled.assert_awaited_once_with(self.chat_id)
        self.session_plugin.is_plugin_enabled_for_session.assert_not_awaited()
        self.session_ai.is_llm_enabled_for_session.assert_not_awaited()

    async def test_disabled_session_ai_skips_before_decision(self):
        self.session_ai.is_llm_enabled_for_session.return_value = False
        await self.assert_preflight_skipped()
        self.session_ai.is_llm_enabled_for_session.assert_awaited_once_with(self.chat_id)

    async def test_valid_native_configuration_reaches_shared_decision(self):
        await self.plugin._glance_due(self.chat_id)
        self.plugin._llm_decision.assert_awaited_once()
        self.assertEqual(self.plugin.glance_last_check[self.chat_id], 1)
        self.plugin.native_tasks.run.assert_not_awaited()
        self.manager.add_active_job.assert_not_awaited()

    async def test_point_name_cancels_native_tasks_outside_chat_lock_before_decision(self):
        for direct in (True, False):
            with self.subTest(direct=direct):
                self.plugin.config["force_reply_when_summoned"] = direct
                self.chat.glance_inflight = True
                previous_version = self.chat.reply_version
                order = []

                async def cancel(umo):
                    self.assertEqual(umo, self.chat_id)
                    self.assertFalse(self.chat.lock.locked())
                    order.append("cancel")
                    self.chat.glance_inflight = False

                async def decide(*args, **kwargs):
                    self.assertFalse(self.chat.lock.locked())
                    self.assertEqual(order, ["cancel"])
                    order.append("decide")
                    return False

                self.plugin.native_tasks.cancel_chat = AsyncMock(side_effect=cancel)
                self.plugin._decide = AsyncMock(side_effect=decide)
                event = _event(f"点名 {direct}", summoned=True)

                await self.plugin.on_message(event)

                self.plugin.native_tasks.cancel_chat.assert_awaited_once_with(self.chat_id)
                self.assertEqual(self.chat.reply_version, previous_version + 1)
                if direct:
                    self.plugin._decide.assert_not_awaited()
                    self.assertTrue(event.get_extra("oncue_decision")["should_reply"])
                    self.assertEqual(order, ["cancel"])
                else:
                    self.plugin._decide.assert_awaited_once()
                    self.assertEqual(order, ["cancel", "decide"])
                self.assertEqual(list(self.chat.reply_ts), [])
                self.plugin.context.send_message.assert_not_awaited()

    async def test_disabled_switch_tick_cancels_each_session_without_holding_chat_lock(self):
        second = self.plugin._chat("test:GroupMessage:2")
        for switch in ("enable", "enable_glance"):
            with self.subTest(switch=switch):
                self.plugin.config.update(enable=True, enable_glance=True)
                self.plugin.config[switch] = False

                async def cancel(umo):
                    self.assertFalse(self.plugin.chats[umo].lock.locked())

                self.plugin.native_tasks.cancel_chat = AsyncMock(side_effect=cancel)
                self.plugin._glance_job = AsyncMock()
                self.plugin.glance_next_due[self.chat_id] = self.now - 1

                await self.plugin._glance_tick()

                self.assertEqual(self.plugin.native_tasks.cancel_chat.await_count, 2)
                self.assertEqual(
                    {call.args[0] for call in self.plugin.native_tasks.cancel_chat.await_args_list},
                    {self.chat_id, "test:GroupMessage:2"},
                )
                self.plugin._glance_job.assert_not_awaited()
                self.assertEqual(self.plugin._glance_jobs, {})
                self.plugin._llm_decision.assert_not_awaited()
                self.assertFalse(self.chat.lock.locked())
                self.assertFalse(second.lock.locked())


if __name__ == "__main__":
    unittest.main()
