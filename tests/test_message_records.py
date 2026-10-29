"""结构化群聊记录：身份、容量、图文顺序和显示长度边界。"""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from test_context import at, reply
from test_timing import Plain, _event, plugin_module


class MessageRecordTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.now = 1000.0
        clock = patch.object(plugin_module, "time", SimpleNamespace(time=lambda: self.now))
        clock.start()
        self.addCleanup(clock.stop)
        self.plugin = plugin_module.OnCuePlugin(
            SimpleNamespace(),
            {
                "enable_glance": False,
                "enable_echo_trigger": False,
                "enable_dense_trigger": False,
                "message_wait_seconds": 0,
            },
        )
        self.chat_id = _event().unified_msg_origin
        self.chat = self.plugin._chat(self.chat_id)
        self.plugin._decision_card = AsyncMock(return_value="角色卡")
        self.plugin._llm_decision = AsyncMock(return_value='{"should_reply": false}')

    def event(self, *parts, message_id=None, sender_id="alice", chat_id=None, summoned=False):
        event = _event(summoned=summoned)
        event.get_messages = lambda: list(parts)
        event.get_sender_id = lambda: sender_id
        event.message_obj.message_id = message_id
        if chat_id is not None:
            event.unified_msg_origin = chat_id
        return event

    def cache(self, event, chat=None):
        chat = chat or self.chat
        item = self.plugin._event_to_item(event, chat)
        self.assertTrue(self.plugin._append_message(chat, item))
        return item

    def image(self, source="https://images.invalid/example.png"):
        comp = plugin_module.Image()
        comp.url = source
        return comp

    def test_missing_platform_id_gets_stable_event_identity_without_forging_platform_id(self):
        event = self.event(Plain("同一事件"))
        first = self.plugin._event_to_item(event, self.chat)
        second = self.plugin._event_to_item(event, self.chat)
        self.assertEqual(first["message_id"], "")
        self.assertTrue(first["record_id"])
        self.assertEqual(first["record_id"], second["record_id"])
        other = self.plugin._event_to_item(self.event(Plain("同一事件")), self.chat)
        self.assertNotEqual(first["record_id"], other["record_id"])

    def test_numeric_platform_id_is_preserved_and_repeated_delivery_is_deduplicated(self):
        first = self.cache(self.event(Plain("消息"), message_id=123))
        repeated = self.plugin._event_to_item(self.event(Plain("消息"), message_id="123"), self.chat)
        self.assertEqual(first["message_id"], "123")
        self.assertEqual(first["record_id"], repeated["record_id"])
        self.assertFalse(self.plugin._append_message(self.chat, repeated))
        self.assertEqual(len(self.chat.messages), 1)

    async def test_unknown_platform_ids_are_missing_ids_not_a_shared_identity(self):
        first = self.event(Plain("第一条"), message_id="unknown")
        second = self.event(Plain("第二条"), message_id="unknown")
        await self.plugin.on_message(first)
        await self.plugin.on_message(second)
        self.assertEqual(len(self.chat.messages), 2)
        self.assertTrue(all(m["message_id"] == "" for m in self.chat.messages))
        self.assertNotEqual(self.chat.messages[0]["record_id"], self.chat.messages[1]["record_id"])
        quoted = self.plugin._event_to_item(self.event(reply("unknown")), self.chat)
        self.assertIn("消息ID未知", quoted["text"])
        self.assertIn("原文未知", quoted["text"])

    def test_same_text_from_distinct_messages_and_members_is_not_deduplicated(self):
        for sender, message_id in (("alice", "one"), ("alice", "two"), ("bob", "three")):
            self.cache(self.event(Plain("重复内容"), message_id=message_id, sender_id=sender))
        self.assertEqual(len(self.chat.messages), 3)
        self.assertEqual(len({m["record_id"] for m in self.chat.messages}), 3)

    async def test_duplicate_delivery_does_not_refresh_observation_or_repeat_decision(self):
        self.chat.last_reply_ts = self.now - 1
        event = self.event(Plain("新消息"), message_id="one")
        await self.plugin.on_message(event)
        self.plugin._llm_decision.assert_awaited_once()
        last_activity = self.chat.last_activity_ts
        last_observation = self.chat.last_observation_activity_ts
        cooldown = self.chat.cooldown_until
        self.now += 30
        repeated = self.event(Plain("新消息"), message_id="one", summoned=True)
        await self.plugin.on_message(repeated)
        self.assertEqual(len(self.chat.messages), 1)
        self.assertEqual(self.chat.last_activity_ts, last_activity)
        self.assertEqual(self.chat.last_observation_activity_ts, last_observation)
        self.assertEqual(self.chat.cooldown_until, cooldown)
        self.assertEqual(self.chat.reply_version, 0)
        self.assertIsNone(repeated.get_extra("oncue_decision"))
        self.assertTrue(repeated.stopped)
        self.plugin._llm_decision.assert_awaited_once()

    async def test_equal_platform_ids_in_different_sessions_are_independent(self):
        first = self.event(Plain("本群"), message_id="one")
        second = self.event(Plain("另一群"), message_id="one", chat_id="test:GroupMessage:2")
        await self.plugin.on_message(first)
        await self.plugin.on_message(second)
        self.assertEqual(len(self.chat.messages), 1)
        self.assertEqual(len(self.plugin._chat(second.unified_msg_origin).messages), 1)

    def test_default_cache_keeps_latest_300_messages(self):
        self.assertEqual(self.chat.messages.maxlen, 300)
        for index in range(301):
            self.cache(self.event(Plain(f"消息{index}"), message_id=str(index)))
        self.assertEqual(len(self.chat.messages), 300)
        self.assertEqual(self.chat.messages[0]["message_id"], "1")
        self.assertEqual(self.chat.messages[-1]["message_id"], "300")

    def test_default_reference_window_does_not_inject_all_cached_messages(self):
        for index in range(60):
            self.cache(self.event(Plain(f"entry-{index:03d}"), message_id=str(index)))
        history = self.plugin._history_text(self.chat)
        self.assertEqual(len(self.chat.messages), 60)
        self.assertNotIn("entry-009", history)
        self.assertIn("entry-010", history)
        self.assertIn("entry-059", history)

    def test_nested_cache_and_reference_configuration_have_separate_effects(self):
        self.plugin.config["config_context"] = {
            "message_cache_count": 5,
            "context_message_count": 2,
            "message_char_limit": 10,
        }
        for index in range(7):
            self.cache(self.event(Plain(f"entry-{index:03d}"), message_id=str(index)))
        self.assertEqual(self.chat.messages.maxlen, 5)
        self.assertEqual([m["message_id"] for m in self.chat.messages], ["2", "3", "4", "5", "6"])
        history = self.plugin._history_text(self.chat)
        self.assertNotIn("entry-004", history)
        self.assertIn("entry-005", history)
        self.assertIn("entry-006", history)

    def test_lowering_cache_capacity_keeps_latest_messages(self):
        for index in range(5):
            self.cache(self.event(Plain(str(index)), message_id=str(index)))
        self.plugin.config["config_context"] = {"message_cache_count": 3}
        self.cache(self.event(Plain("5"), message_id="5"))
        self.assertEqual(self.chat.messages.maxlen, 3)
        self.assertEqual([m["message_id"] for m in self.chat.messages], ["3", "4", "5"])

    def test_quote_can_resolve_cached_message_outside_reference_window(self):
        self.plugin.config["config_context"] = {"context_message_count": 1}
        self.cache(self.event(Plain("真实原文"), message_id="old", sender_id="bob"))
        self.cache(self.event(Plain("无关消息"), message_id="middle"))
        item = self.cache(self.event(reply("old"), Plain("回应旧消息"), message_id="new"))
        history = self.plugin._history_text(self.chat)
        self.assertIn("真实原文", history)
        self.assertNotIn("无关消息", history)
        quote = item["parts"][0]
        self.assertEqual(quote.reference_id, "old")
        self.assertEqual(quote.sender_id, "bob")
        self.assertTrue(quote.children)

    def test_internal_record_id_cannot_resolve_as_platform_quote(self):
        original = self.cache(self.event(Plain("不能冒充平台原文")))
        quoted = self.plugin._event_to_item(self.event(reply(original["record_id"])), self.chat)
        self.assertIn("原文未知", quoted["text"])
        self.assertNotIn("不能冒充平台原文", quoted["text"])

    def test_evicted_message_cannot_be_resolved_from_stale_identity(self):
        self.plugin.config["config_context"] = {"message_cache_count": 2}
        for message_id, text in (("old", "已经淘汰的原文"), ("one", "第一条"), ("two", "第二条")):
            self.cache(self.event(Plain(text), message_id=message_id))
        quoted = self.plugin._event_to_item(self.event(reply("old")), self.chat)
        self.assertIn("原文未知", quoted["text"])
        self.assertNotIn("已经淘汰的原文", quoted["text"])

    def test_images_keep_original_order_and_store_sources_without_injecting_them(self):
        first = self.image("https://images.invalid/first.png")
        second = self.image("https://images.invalid/second.png")
        components = [Plain("图前"), first, Plain("图中"), second, Plain("图后")]
        event = self.event(*components, message_id="images")
        item = self.cache(event)
        self.assertEqual([p.kind for p in item["parts"]], ["text", "image", "text", "image", "text"])
        self.assertEqual(item["text"], "图前 [图片，内容未读取] 图中 [图片，内容未读取] 图后")
        refs = [part.image for part in item["parts"] if part.image is not None]
        self.assertEqual([ref.source for ref in refs], [first.url, second.url])
        self.assertNotEqual(refs[0].image_id, refs[1].image_id)
        self.assertTrue(all(ref.size_bytes is None and ref.description == "" for ref in refs))
        history = self.plugin._history_text(self.chat)
        self.assertNotIn("https://", history)
        self.assertFalse(item["is_pure_text"])
        self.assertEqual(event.get_messages(), components)

    def test_nested_quote_images_have_their_own_structured_references(self):
        item = self.cache(self.event(reply("old", chain=[Plain("原图前"), self.image(), Plain("原图后")]), self.image()))
        quoted = item["parts"][0]
        self.assertEqual([p.kind for p in quoted.children], ["text", "image", "text"])
        self.assertNotEqual(quoted.children[1].image.image_id, item["parts"][1].image.image_id)

    def test_quote_preview_does_not_cut_an_embedded_image_marker(self):
        item = self.cache(self.event(reply("old", chain=[Plain("长" * 295), self.image()])))
        self.assertEqual(item["text"].count("["), item["text"].count("]"))
        self.assertNotIn("[图片…", item["text"])

    def test_cache_resolved_quote_does_not_retain_recursive_quote_children(self):
        self.cache(self.event(Plain("最初的内容"), message_id="one"))
        self.cache(self.event(reply("one"), Plain("中间的回应"), message_id="two"))
        item = self.cache(self.event(reply("two"), Plain("最新回应"), message_id="three"))
        quote = item["parts"][0]
        self.assertIn("中间的回应", quote.text)
        self.assertNotIn("最初的内容", quote.text)
        self.assertEqual(quote.children[0].text, "[嵌套引用]")
        self.assertFalse(quote.children[0].children)

    def test_long_message_preserves_first_and_last_250_characters_without_changing_source(self):
        source = "首" * 250 + "中" * 400 + "尾" * 250
        comp = Plain(source)
        item = self.cache(self.event(comp))
        history = self.plugin._history_text(self.chat)
        self.assertIn("首" * 250, history)
        self.assertIn("尾" * 250, history)
        self.assertNotIn("中" * 400, history)
        self.assertIn("省略", history)
        self.assertEqual(item["pure_text"], source)
        self.assertEqual(item["text"], source)
        self.assertEqual(comp.text, source)

    def test_english_is_limited_by_characters_and_later_messages_remain_visible(self):
        self.cache(self.event(Plain("A" * 250 + "M" * 500 + "Z" * 250), message_id="long"))
        self.cache(self.event(Plain("后面成员的新消息"), message_id="later", sender_id="bob"))
        history = self.plugin._history_text(self.chat)
        self.assertIn("A" * 250, history)
        self.assertIn("Z" * 250, history)
        self.assertNotIn("M" * 500, history)
        self.assertIn("后面成员的新消息", history)

    def test_small_limits_do_not_cut_image_and_mention_markers_in_half(self):
        self.plugin.config["config_context"] = {"message_char_limit": 30}
        self.cache(self.event(self.image(), Plain("长" * 100), at("bob", "小王"), self.image()))
        history = self.plugin._history_text(self.chat)
        self.assertIn("省略", history)
        self.assertIn("[图片，内容未读取]", history)
        self.assertEqual(history.count("["), history.count("]"))
        self.assertNotIn("[图片，内容未读[", history)

    def test_multiple_image_descriptions_share_one_message_budget(self):
        item = self.cache(self.event(self.image(), self.image()))
        first, second = (part.image for part in item["parts"])
        first.status = second.status = "已转述"
        first.description = "甲" * 350
        second.description = "乙" * 350
        history = self.plugin._history_text(self.chat)
        self.assertFalse("甲" * 350 in history and "乙" * 350 in history)
        self.assertIn("省略", history)
        self.assertEqual(history.count("["), history.count("]"))
        self.assertEqual(first.description, "甲" * 350)
        self.assertEqual(second.description, "乙" * 350)

    def test_truncation_does_not_turn_distinct_long_messages_into_an_echo(self):
        self.plugin.config.update(enable_echo_trigger=True, echo_threshold=3, echo_message_count=3)
        for sender, middle in (("alice", "甲"), ("bob", "乙"), ("carol", "丙")):
            self.cache(self.event(Plain("首" * 250 + middle * 100 + "尾" * 250), sender_id=sender))
        self.assertIsNone(self.plugin._stat_trigger_locked(self.chat, self.chat.messages[-1], self.now))
        self.plugin._history_text(self.chat)
        self.assertEqual(len({item["pure_text"] for item in self.chat.messages}), 3)

    def test_echo_prompt_is_bounded_without_truncating_matching_key(self):
        self.plugin.config.update(enable_echo_trigger=True, echo_threshold=3, echo_message_count=3)
        source = "首" * 250 + "中" * 400 + "尾" * 250
        for sender in ("alice", "bob", "carol"):
            self.cache(self.event(Plain(source), sender_id=sender))
        context, key = self.plugin._stat_trigger_locked(self.chat, self.chat.messages[-1], self.now)
        self.assertEqual(key, "echo:" + source)
        self.assertNotIn("中" * 400, context)
        self.assertIn("首" * 250, context)
        self.assertIn("尾" * 250, context)

    async def test_bot_reply_preserves_mentions_with_body_but_headers_alone_are_not_a_reply(self):
        event = self.event()
        event._has_send_oper = True
        event.result.chain = [at("bob", "小王"), Plain("这是回复")]
        event.set_extra("oncue_decision", {"should_reply": True})
        await self.plugin.mark_reply_sent(event)
        item = self.chat.messages[-1]
        self.assertEqual([p.kind for p in item["parts"]], ["mention", "text"])
        self.assertEqual(item["parts"][0].reference_id, "bob")
        self.assertIn("这是回复", item["text"])

    def test_wake_name_in_omitted_middle_is_still_detected_from_original_message(self):
        self.plugin.config["wake_names"] = "小助手"
        event = self.event(Plain("首" * 250 + "小助手" + "尾" * 250))
        item = self.cache(event)
        self.assertNotIn("小助手", self.plugin._history_text(self.chat))
        self.assertTrue(self.plugin._is_forced(event, item))

    def test_bot_replies_get_distinct_structured_ids_and_same_display_limit(self):
        source = "首" * 250 + "中" * 300 + "尾" * 250
        self.plugin._record_reply(self.chat, source, self.now)
        self.plugin._record_reply(self.chat, source, self.now)
        first, second = self.chat.messages
        self.assertNotEqual(first["record_id"], second["record_id"])
        self.assertEqual(first["message_id"], "")
        self.assertEqual(first["role"], "assistant")
        self.assertFalse(first["active"])
        self.assertFalse(first["is_pure_text"])
        self.assertEqual(first["pure_text"], "")
        self.assertEqual(first["parts"][0].text, source)
        self.assertIn("省略", self.plugin._history_text(self.chat))
        self.assertEqual(first["text"], source)


if __name__ == "__main__":
    unittest.main()
