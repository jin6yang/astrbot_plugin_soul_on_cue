"""临时群聊背景：快照、身份去重、开关和会话配置冲突。"""

import copy
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from test_context import at, reply
from test_timing import Plain, _event, plugin_module


class ContextInjectionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.native_config = {
            "provider_ltm_settings": {
                "group_icl_enable": False,
                "active_reply": {"enable": False},
            },
        }
        self.context = SimpleNamespace(get_config=lambda **kwargs: self.native_config)
        self.plugin = plugin_module.OnCuePlugin(
            self.context,
            {
                "enable_glance": False,
                "enable_echo_trigger": False,
                "enable_dense_trigger": False,
                "message_wait_seconds": 0,
            },
        )
        self.chat_id = _event().unified_msg_origin
        self.chat = self.plugin._chat(self.chat_id)
        self.plugin._decision_card = AsyncMock(return_value="测试角色卡")
        self.plugin._llm_decision = AsyncMock(return_value='{"should_reply": true}')

    def event(self, text="当前原始消息", *, message_id=None, chat_id=None):
        event = _event(text)
        event.message_obj.message_id = message_id
        if chat_id is not None:
            event.unified_msg_origin = chat_id
        return event

    def cache(self, text, *, message_id=None, chat_id=None):
        event = self.event(text, message_id=message_id, chat_id=chat_id)
        chat = self.plugin._chat(event.unified_msg_origin)
        item = self.plugin._event_to_item(event, chat)
        self.assertTrue(self.plugin._append_message(chat, item))
        return event, item

    def approve(self, event=None, chat=None):
        event = event or self.event("[主动发言机会]")
        snapshot = self.plugin._context_snapshot(chat or self.chat)
        event.set_extra("oncue_decision", {"should_reply": True, "_chat_context": snapshot})
        return event, snapshot

    @staticmethod
    def request(prompt="原始正文", *, contexts=None, parts=None):
        return SimpleNamespace(
            prompt=prompt,
            system_prompt="原生人格",
            image_urls=["original-image"],
            contexts=contexts if contexts is not None else [],
            extra_user_content_parts=parts if parts is not None else [],
        )

    @staticmethod
    def extra_text(req):
        return "\n".join(
            part.get("text", "") if isinstance(part, dict) else getattr(part, "text", "")
            for part in req.extra_user_content_parts
        )

    async def test_background_is_temporary_and_preserves_native_request(self):
        self.cache("需要保留的群聊背景", message_id="old")
        event, snapshot = self.approve()
        contexts = [{"role": "user", "content": "原生历史"}]
        foreign = plugin_module.TextPart(text="其他插件的内容")
        parts = [foreign]
        req = self.request(contexts=contexts, parts=parts)
        original_contexts = copy.deepcopy(contexts)
        await self.plugin.inject_stage_direction(event, req)
        self.assertEqual(req.prompt, "原始正文")
        self.assertEqual(req.contexts, original_contexts)
        self.assertEqual(req.image_urls, ["original-image"])
        self.assertIn(snapshot.render(), self.extra_text(req))
        self.assertNotIn(snapshot.render(), req.system_prompt)
        injected = [part for part in req.extra_user_content_parts if part is not foreign]
        self.assertEqual(len(injected), 1)
        self.assertTrue(injected[0]._no_save)
        self.assertIn(foreign, req.extra_user_content_parts)
        self.assertEqual(parts, [foreign], "不应修改其他对象持有的原列表")

    async def test_repeated_hook_does_not_duplicate_background_or_stage_direction(self):
        self.cache("共同快照", message_id="old")
        event, _ = self.approve()
        event.get_extra("oncue_decision")["mood"] = "好奇"
        req = self.request()
        for _ in range(3):
            await self.plugin.inject_stage_direction(event, req)
        self.assertEqual(self.extra_text(req).count("[群聊补充上下文]"), 1)
        self.assertEqual(self.extra_text(req).count("共同快照"), 1)
        self.assertEqual(req.system_prompt.count("[导演指令]"), 1)
        self.assertEqual(req.system_prompt.count("好奇"), 1)

    async def test_new_request_for_same_event_gets_its_own_temporary_background(self):
        self.cache("共同快照", message_id="old")
        event, _ = self.approve()
        first, second = self.request(), self.request()
        await self.plugin.inject_stage_direction(event, first)
        await self.plugin.inject_stage_direction(event, second)
        for req in (first, second):
            self.assertIn("共同快照", self.extra_text(req))
            self.assertEqual(self.extra_text(req).count("[群聊补充上下文]"), 1)

    async def test_disabling_injection_keeps_stage_direction_and_decision_snapshot(self):
        self.plugin.config["config_context"] = {"enable_context_injection": False}
        self.cache("依旧供决策参考", message_id="old")
        event, snapshot = self.approve()
        req = self.request()
        await self.plugin.inject_stage_direction(event, req)
        self.assertIn("依旧供决策参考", snapshot.render())
        self.assertNotIn("依旧供决策参考", self.extra_text(req))
        self.assertIn("[导演指令]", req.system_prompt)
        self.assertEqual(req.prompt, "原始正文")

    async def test_disabling_after_injection_removes_only_owned_background(self):
        self.cache("可关闭的背景", message_id="old")
        event, _ = self.approve()
        foreign = plugin_module.TextPart(text="其他插件的内容")
        req = self.request(parts=[foreign])
        await self.plugin.inject_stage_direction(event, req)
        self.plugin.config["config_context"] = {"enable_context_injection": False}
        await self.plugin.inject_stage_direction(event, req)
        self.assertEqual(req.extra_user_content_parts, [foreign])
        self.assertEqual(req.system_prompt.count("[导演指令]"), 1)

    async def test_current_message_is_not_repeated_as_background(self):
        self.cache("旧消息正文", message_id="old")
        current, _ = self.cache("当前消息唯一正文", message_id="current")
        self.approve(current)
        req = self.request("当前消息唯一正文")
        await self.plugin.inject_stage_direction(current, req)
        self.assertIn("旧消息正文", self.extra_text(req))
        self.assertNotIn("当前消息唯一正文", self.extra_text(req))
        self.assertEqual(req.prompt, "当前消息唯一正文")

    async def test_identical_text_from_another_message_is_preserved(self):
        self.cache("相同文字", message_id="old")
        current, _ = self.cache("相同文字", message_id="current")
        self.approve(current)
        req = self.request("相同文字")
        await self.plugin.inject_stage_direction(current, req)
        self.assertEqual(self.extra_text(req).count("相同文字"), 1)
        self.assertIn('消息ID="old"', self.extra_text(req))

    async def test_empty_native_prompt_does_not_discard_current_message(self):
        current, _ = self.cache("当前消息仍需提供", message_id="current")
        self.approve(current)
        for prompt in (None, "", " \n"):
            with self.subTest(prompt=prompt):
                req = self.request(prompt)
                req.image_urls = []
                await self.plugin.inject_stage_direction(current, req)
                self.assertIn("当前消息仍需提供", self.extra_text(req))
                self.assertEqual(req.prompt, prompt)

    async def test_current_message_keeps_quote_and_mention_identity_without_excerpt(self):
        for sender_id, message_id, text in (
            ("bob", "bob-old", "旧消息的唯一正文"),
            ("carol", "carol-old", "另一个同名成员的消息"),
        ):
            original = self.event(text, message_id=message_id)
            original.get_sender_id = lambda sender=sender_id: sender
            original.message_obj.sender.nickname = "同名"
            self.plugin._append_message(self.chat, self.plugin._event_to_item(original, self.chat))
        current = self.event("当前消息唯一正文", message_id="current")
        current.get_messages = lambda: [reply("bob-old"), at("carol", "同名"), Plain("当前消息唯一正文")]
        self.plugin._append_message(self.chat, self.plugin._event_to_item(current, self.chat))
        self.approve(current)
        native_quote = plugin_module.TextPart(text="<Quoted Message>\n(同名): 旧消息的唯一正文\n</Quoted Message>")
        req = self.request("当前消息唯一正文", parts=[native_quote])
        await self.plugin.inject_stage_direction(current, req)
        background = req.extra_user_content_parts[-1].text
        self.assertIn('[引用 消息ID="bob-old"；作者=成员「同名」（ID=bob）]', background)
        self.assertIn("[提及 成员「同名」（ID=carol）]", background)
        self.assertNotIn("当前消息唯一正文", background)
        self.assertNotIn("摘录=", background)
        self.assertEqual(background.count("旧消息的唯一正文"), 1)
        self.assertIs(req.extra_user_content_parts[0], native_quote)

    async def test_current_message_keeps_position_before_later_waiting_messages(self):
        current, _ = self.cache("首条原始正文", message_id="first")
        self.cache("第二条补充", message_id="second")
        self.cache("第三条补充", message_id="third")
        self.approve(current)
        req = self.request("首条原始正文")
        await self.plugin.inject_stage_direction(current, req)
        background = self.extra_text(req)
        self.assertLess(background.index('消息ID="first"'), background.index('消息ID="second"'))
        self.assertLess(background.index('消息ID="second"'), background.index('消息ID="third"'))
        self.assertNotIn("首条原始正文", background)
        self.assertIn("第二条补充", background)
        self.assertIn("第三条补充", background)

    async def test_direct_summon_keeps_self_identity_in_current_quote(self):
        event = self.event("对你的回复", message_id="current")
        event.is_at_or_wake_command = True
        event.get_messages = lambda: [reply("bot-old", sender_id="bot"), Plain("对你的回复")]
        await self.plugin.on_message(event)
        req = self.request("对你的回复")
        await self.plugin.inject_stage_direction(event, req)
        self.assertIn('[引用 消息ID="bot-old"；作者=Bot（你）]', self.extra_text(req))

    async def test_glance_keeps_latest_members_message(self):
        self.cache("最近群友的实际发言", message_id="latest")
        synthetic = self.event("[主动发言机会]", message_id="oncue-trigger")
        self.approve(synthetic)
        req = self.request("[主动发言机会]")
        await self.plugin.inject_stage_direction(synthetic, req)
        self.assertIn("最近群友的实际发言", self.extra_text(req))

    async def test_explicit_native_message_identity_deduplicates_background(self):
        self.cache("原生历史已含的正文", message_id="old")
        self.cache("仍需补充的正文", message_id="remaining")
        event, _ = self.approve()
        contexts = [{"role": "user", "message_id": "old", "content": "原生历史已含的正文"}]
        req = self.request(contexts=contexts)
        await self.plugin.inject_stage_direction(event, req)
        self.assertNotIn("原生历史已含的正文", self.extra_text(req))
        self.assertIn("仍需补充的正文", self.extra_text(req))
        self.assertEqual(req.contexts, contexts)

    async def test_wrong_role_or_session_cannot_remove_a_message(self):
        self.cache("必须保留的用户消息", message_id="old")
        event, _ = self.approve()
        for native in (
            {"role": "assistant", "message_id": "old", "content": "另一个角色"},
            {"role": "user", "message_id": "old", "unified_msg_origin": "test:GroupMessage:other", "content": "另一个会话"},
        ):
            with self.subTest(native=native):
                req = self.request(contexts=[native])
                await self.plugin.inject_stage_direction(event, req)
                self.assertIn("必须保留的用户消息", self.extra_text(req))

    async def test_text_metadata_and_checkpoints_are_not_message_identity(self):
        self.cache("相同正文也不能猜测身份", message_id="old")
        event, _ = self.approve()
        contexts = [
            {"role": "user", "content": "相同正文也不能猜测身份"},
            {"role": "user", "metadata": {"message_id": "old"}, "content": "其他信息"},
            {"role": "_checkpoint", "content": {"id": "old"}},
        ]
        req = self.request(contexts=contexts)
        await self.plugin.inject_stage_direction(event, req)
        self.assertIn("相同正文也不能猜测身份", self.extra_text(req))

    async def test_snapshot_stays_fixed_when_later_messages_or_image_state_change(self):
        image = plugin_module.Image()
        image.url = "https://images.invalid/one.png"
        original = self.event("最初的文字", message_id="old")
        original.get_messages = lambda: [Plain("最初的文字"), image]
        item = self.plugin._event_to_item(original, self.chat)
        self.plugin._append_message(self.chat, item)
        event, snapshot = self.approve()
        frozen = snapshot.render()
        item["parts"][1].image.description = "后来才完成的描述"
        self.cache("决策之后的新消息", message_id="later")
        req = self.request()
        await self.plugin.inject_stage_direction(event, req)
        self.assertEqual(snapshot.render(), frozen)
        self.assertIn(frozen, self.extra_text(req))
        self.assertNotIn("后来才完成的描述", self.extra_text(req))
        self.assertNotIn("决策之后的新消息", self.extra_text(req))

    async def test_native_context_conflict_warns_once_per_session_and_preserves_config(self):
        self.native_config["provider_ltm_settings"]["group_icl_enable"] = True
        original_config = copy.deepcopy(self.native_config)
        self.cache("背景", message_id="old")
        event, _ = self.approve()
        with patch.object(plugin_module.logger, "warning") as warning:
            await self.plugin.inject_stage_direction(event, self.request())
            await self.plugin.inject_stage_direction(event, self.request())
            self.assertEqual(warning.call_count, 1)
            self.assertIn("群聊消息记录注入上下文", str(warning.call_args))
            self.native_config["provider_ltm_settings"]["group_icl_enable"] = False
            await self.plugin.inject_stage_direction(event, self.request())
            self.native_config["provider_ltm_settings"]["group_icl_enable"] = True
            await self.plugin.inject_stage_direction(event, self.request())
            self.assertEqual(warning.call_count, 2)
        self.assertEqual(self.native_config, original_config)

    async def test_active_reply_and_group_injection_conflicts_are_independent(self):
        self.native_config["provider_ltm_settings"]["group_icl_enable"] = True
        self.cache("背景", message_id="old")
        event, _ = self.approve()
        with patch.object(plugin_module.logger, "warning") as warning:
            await self.plugin.inject_stage_direction(event, self.request())
            self.assertEqual(warning.call_count, 1)
            self.assertIn("群聊消息记录注入上下文", str(warning.call_args))
            self.native_config["provider_ltm_settings"]["active_reply"]["enable"] = True
            await self.plugin.inject_stage_direction(event, self.request())
            self.assertEqual(warning.call_count, 2)
            self.assertIn("主动回复", str(warning.call_args))
            await self.plugin.inject_stage_direction(event, self.request())
            self.assertEqual(warning.call_count, 2)

    async def test_context_injection_conflict_is_not_reported_when_injection_is_disabled(self):
        self.native_config["provider_ltm_settings"]["group_icl_enable"] = True
        self.plugin.config["config_context"] = {"enable_context_injection": False}
        self.cache("背景", message_id="old")
        event, _ = self.approve()
        with patch.object(plugin_module.logger, "warning") as warning:
            await self.plugin.inject_stage_direction(event, self.request())
            warning.assert_not_called()

    async def test_active_reply_conflict_is_reported_even_without_context_injection(self):
        self.native_config["provider_ltm_settings"]["group_icl_enable"] = True
        self.native_config["provider_ltm_settings"]["active_reply"]["enable"] = True
        self.plugin.config["config_context"] = {"enable_context_injection": False}
        event, _ = self.approve()
        with patch.object(plugin_module.logger, "warning") as warning:
            await self.plugin.inject_stage_direction(event, self.request())
            await self.plugin.inject_stage_direction(event, self.request())
            self.assertEqual(warning.call_count, 1)
            self.assertIn("主动回复", str(warning.call_args))
            self.assertIn("参与决策", str(warning.call_args))
            self.assertNotIn("群聊消息记录注入上下文", str(warning.call_args))

    async def test_conflicts_in_separate_sessions_are_not_globally_suppressed(self):
        self.native_config["provider_ltm_settings"]["group_icl_enable"] = True
        self.cache("第一群", message_id="old")
        first, _ = self.approve()
        other_id = "test:GroupMessage:other"
        other_event, _ = self.cache("第二群", message_id="old", chat_id=other_id)
        other, _ = self.approve(other_event, self.plugin._chat(other_id))
        with patch.object(plugin_module.logger, "warning") as warning:
            await self.plugin.inject_stage_direction(first, self.request())
            await self.plugin.inject_stage_direction(other, self.request())
            self.assertEqual(warning.call_count, 2)

    async def test_default_prompt_places_stable_contract_before_card_and_dynamic_context(self):
        prompt = await self.plugin._build_prompt("UNIQUE-HISTORY", "UNIQUE-ENTRY")
        self.assertLess(prompt.index("should_reply"), prompt.index("测试角色卡"))
        self.assertLess(prompt.index("测试角色卡"), prompt.index("UNIQUE-HISTORY"))
        self.assertLess(prompt.index("UNIQUE-HISTORY"), prompt.index("当前时间"))
        self.assertLess(prompt.index("当前时间"), prompt.index("UNIQUE-ENTRY"))

    async def test_custom_template_is_preserved_and_substitutions_do_not_recurse(self):
        template = "自定义:{character_card}|记录:{chat_history}|触发:{entry_context}"
        self.plugin.config["decision_prompt"] = template
        self.plugin._decision_card.return_value = "卡内保留{chat_history}和{entry_context}"
        prompt = await self.plugin._build_prompt("HISTORY", "ENTRY含{character_card}")
        self.assertEqual(
            prompt,
            "自定义:卡内保留{chat_history}和{entry_context}|记录:HISTORY|触发:ENTRY含{character_card}",
        )
        self.assertEqual(self.plugin.config["decision_prompt"], template)

    def test_schema_default_matches_code_and_context_switch_hint(self):
        path = Path(__file__).resolve().parents[1] / "_conf_schema.json"
        schema = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            schema["config_decision"]["items"]["decision_prompt"]["default"],
            plugin_module.DEFAULT_DECISION_PROMPT,
        )
        switch = schema["config_context"]["items"]["enable_context_injection"]
        self.assertTrue(switch["default"])
        self.assertEqual(
            switch["hint"],
            "此为建议项。不可与 AstrBot 的「群聊消息记录注入上下文」同时启用以避免重复注入等问题。",
        )


if __name__ == "__main__":
    unittest.main()
