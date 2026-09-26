"""验证决策上下文中的成员身份、提及对象和引用关系。"""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from test_timing import Plain, _event, plugin_module


def at(target=None, name=""):
    comp = plugin_module.At()
    comp.qq = target
    comp.name = name
    return comp


def reply(message_id=None, **values):
    comp = plugin_module.Reply()
    fields = dict(id=message_id, sender_id=0, sender_nickname="", chain=[], message_str="", text="")
    fields.update(values)
    for key, value in fields.items():
        setattr(comp, key, value)
    return comp


class ContextTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.now = 1000.0
        clock = patch.object(plugin_module, "time", SimpleNamespace(time=lambda: self.now))
        clock.start()
        self.addCleanup(clock.stop)
        self.plugin = plugin_module.OnCuePlugin(
            SimpleNamespace(
                get_current_chat_provider_id=AsyncMock(return_value="main"),
                llm_generate=AsyncMock(return_value=SimpleNamespace(completion_text="reply")),
            ),
            {"enable_glance": False, "enable_echo_trigger": False, "enable_dense_trigger": False},
        )
        self.chat = self.plugin._chat(_event().unified_msg_origin)
        self.plugin._decision_card = AsyncMock(return_value="角色卡")
        self.plugin._character_card = AsyncMock(return_value="角色卡")
        self.plugin._llm_decision = AsyncMock(return_value='{"should_reply": false}')

    def event(self, *components, message_id="current", sender_id="alice", nickname="Alice"):
        event = _event()
        event.get_messages = lambda: list(components)
        event.get_sender_id = lambda: sender_id
        event.message_obj.message_id = message_id
        event.message_obj.sender.nickname = nickname
        return event

    def cache(self, event):
        item = self.plugin._event_to_item(event, self.chat)
        self.chat.messages.append(item)
        return item

    def test_mentions_distinguish_self_others_same_name_and_everyone(self):
        item = self.cache(self.event(at("bot", "同名"), at("bob", "同名"), at("all"), Plain("你好")))
        self.assertIn("[提及 Bot（你）]", item["text"])
        self.assertIn("[提及 成员「同名」（ID=bob）]", item["text"])
        self.assertIn("[提及全体成员]", item["text"])
        self.assertEqual(item["pure_text"], "你好")
        self.assertFalse(item["is_pure_text"])

    def test_missing_mention_id_does_not_guess_self_from_nickname(self):
        item = self.cache(self.event(at(None, "Bot"), Plain("你好")))
        self.assertIn("成员「Bot」（身份未知）", item["text"])
        self.assertNotIn("Bot（你）", item["text"])

    def test_numeric_identity_is_normalized_for_self_mentions_and_replies(self):
        event = self.event(at(42), reply("m1", sender_id=42, message_str="之前的回复"))
        event.get_self_id = lambda: "42"
        item = self.cache(event)
        self.assertIn("[提及 Bot（你）]", item["text"])
        self.assertIn("作者=Bot（你）", item["text"])

    def test_reply_contains_original_id_author_and_content_without_becoming_current_text(self):
        item = self.cache(self.event(
            reply(123, sender_id=7, sender_nickname="小王", message_str="你今天值班吗？"),
            Plain("明天才值班"),
        ))
        self.assertIn('引用 消息ID="123"', item["text"])
        self.assertIn("作者=成员「小王」（ID=7）", item["text"])
        self.assertIn('摘录="你今天值班吗？"', item["text"])
        self.assertEqual(item["pure_text"], "明天才值班")
        self.assertFalse(item["is_pure_text"])
        self.assertNotIn("[附件]", item["text"])

    def test_reply_chain_preserves_attachments_and_mentions(self):
        item = self.cache(self.event(reply(
            "m1", sender_id="bob", message_str="不完整的纯文本",
            chain=[at("bot"), Plain("看看这张图"), plugin_module.Image(), object()],
        )))
        self.assertIn("[提及 Bot（你）] 看看这张图 [图片] [附件]", item["text"])
        self.assertNotIn("不完整的纯文本", item["text"])
        self.assertEqual(item["pure_text"], "")
        self.assertTrue(item["active"])

    def test_reply_text_fallback_handles_legacy_and_missing_fields(self):
        for comp, expected in (
            (reply("m1", text="旧字段中的引文"), "旧字段中的引文"),
            (reply("m1", chain=None, message_str="文本引文"), "文本引文"),
            (plugin_module.Reply(), "消息ID未知"),
        ):
            with self.subTest(expected=expected):
                item = self.plugin._event_to_item(self.event(comp), self.chat)
                self.assertIn(expected, item["text"])

    async def test_id_only_reply_is_resolved_from_the_same_chat_cache(self):
        original = self.event(Plain("我想换电脑"), message_id=123, sender_id="bob", nickname="小王")
        await self.plugin.on_message(original)
        await self.plugin.on_message(self.event(reply("123"), Plain("预算八千")))
        item = self.chat.messages[-1]
        self.assertIn("作者=成员「小王」（ID=bob）", item["text"])
        self.assertIn('摘录="我想换电脑"', item["text"])
        self.assertEqual(item["pure_text"], "预算八千")
        self.plugin._llm_decision.assert_not_awaited()

    def test_reply_to_old_message_keeps_context_when_history_display_is_shorter(self):
        self.cache(self.event(Plain("我想换电脑"), message_id="m1", sender_id="bob"))
        self.cache(self.event(Plain("其他话题"), message_id="m2"))
        self.cache(self.event(reply("m1"), Plain("预算八千"), message_id="m3"))
        self.plugin.config["context_message_count"] = 1
        history = self.plugin._history_text(self.chat)
        self.assertIn("我想换电脑", history)
        self.assertIn("预算八千", history)
        self.assertNotIn("其他话题", history)

    def test_reply_does_not_resolve_from_another_chat_or_from_neighboring_messages(self):
        other = self.plugin._chat("test:GroupMessage:other")
        other.messages.append(self.plugin._event_to_item(self.event(Plain("另一群的内容"), message_id="m1")))
        self.cache(self.event(Plain("本群相邻消息"), message_id="m2"))
        item = self.cache(self.event(reply("m1"), Plain("收到")))
        self.assertIn("作者=成员（身份未知）", item["text"])
        self.assertIn("原文未知", item["text"])
        self.assertNotIn("另一群的内容", item["text"])
        self.assertNotIn("本群相邻消息", item["text"])

    def test_missing_reply_id_does_not_match_cached_messages_without_ids(self):
        self.cache(self.event(Plain("不能猜测的原文"), message_id=""))
        item = self.cache(self.event(reply(), Plain("收到")))
        self.assertIn("消息ID未知", item["text"])
        self.assertIn("原文未知", item["text"])
        self.assertNotIn("不能猜测的原文", item["text"])

    def test_platform_quote_excerpt_takes_priority_over_cached_full_message(self):
        self.cache(self.event(Plain("原文还有很多其他内容"), message_id="m1", sender_id="bob", nickname="小王"))
        item = self.cache(self.event(reply("m1", message_str="用户只引用的片段")))
        self.assertIn("用户只引用的片段", item["text"])
        self.assertNotIn("原文还有很多其他内容", item["text"])
        self.assertIn("作者=成员「小王」（ID=bob）", item["text"])

    def test_nested_quotes_are_not_recursively_expanded(self):
        nested = reply("nested", message_str="更深一层的内容")
        nested.chain = [nested]  # 损坏的循环引用也不能导致递归。
        item = self.cache(self.event(reply("m1", chain=[nested, Plain("本层内容")]), Plain("新发言")))
        self.assertIn("[嵌套引用] 本层内容", item["text"])
        self.assertNotIn("更深一层的内容", item["text"])

    def test_quote_preview_is_bounded_and_distinguishes_multiline_content(self):
        item = self.cache(self.event(reply("m1", message_str="甲\n乙\t" + "长" * 400)))
        self.assertIn("甲 乙 ", item["text"])
        self.assertIn("…", item["text"])
        self.assertNotIn("长" * 301, item["text"])
        self.assertNotIn("\n", item["text"])

    def test_history_distinguishes_same_names_and_preserves_message_order(self):
        self.cache(self.event(Plain("我想换电脑"), message_id="m1", sender_id="a", nickname="同名"))
        self.cache(self.event(at("b", "同名"), Plain("你今天值班吗"), message_id="m2", sender_id="a", nickname="同名"))
        self.cache(self.event(reply("m1"), Plain("预算八千"), message_id="m3", sender_id="a", nickname="同名"))
        self.cache(self.event(Plain("明天值班"), message_id="m4", sender_id="b", nickname="同名"))
        self.plugin._record_reply(self.chat, "Bot 已发言", self.now)
        history = self.plugin._history_text(self.chat)
        self.assertIn("成员「同名」（ID=a）", history)
        self.assertIn("成员「同名」（ID=b）", history)
        self.assertIn("Bot（你）", history)
        self.assertIn('消息ID="m3"', history)
        self.assertLess(history.index("你今天值班吗"), history.index("预算八千"))
        self.assertLess(history.index("预算八千"), history.index("明天值班"))
        self.assertIn("引用摘录是旧消息，不是当前发言", history)
        self.assertIn("连续消息可能属于不同话题", history)

    def test_empty_history_stays_empty(self):
        self.assertEqual(self.plugin._history_text(self.chat), "")

    def test_quoted_wake_name_and_mention_do_not_become_current_wake_or_echo_text(self):
        self.plugin.config["wake_names"] = "小助手"
        event = self.event(reply("m1", chain=[at("bot"), Plain("小助手，来看看")]), Plain("这句话是对别人说的"))
        item = self.cache(event)
        self.assertFalse(self.plugin._is_forced(event, item))
        self.assertFalse(item["is_pure_text"])
        self.assertEqual(item["pure_text"], "这句话是对别人说的")
        direct = self.event(Plain("小助手，来看看"))
        self.assertTrue(self.plugin._is_forced(direct, self.plugin._event_to_item(direct)))

    async def test_real_decision_prompt_receives_relationships_with_custom_template(self):
        self.plugin.config["decision_prompt"] = "角色：{character_card}\n{chat_history}"
        self.chat.last_reply_ts = self.now - 1
        event = self.event(reply("bot-m1", sender_id="bot", message_str="需要多少预算？"), at("bob", "小王"), Plain("我也想问问你"))
        await self.plugin.on_message(event)
        self.plugin._llm_decision.assert_awaited_once()
        prompt = self.plugin._llm_decision.call_args.args[1]
        self.assertIn("作者=Bot（你）", prompt)
        self.assertIn("[提及 成员「小王」（ID=bob）]", prompt)
        self.assertIn("需要多少预算？", prompt)
        self.assertFalse(event.is_at_or_wake_command)

    async def test_glance_generation_receives_the_same_relationship_context(self):
        self.cache(self.event(reply("m1", sender_id="bob", message_str="值班吗"), Plain("明天")))
        history = self.plugin._history_text(self.chat)
        await self.plugin._glance_generate(history, _event().unified_msg_origin, {})
        prompt = self.plugin.context.llm_generate.call_args.kwargs["prompt"]
        self.assertIn(history, prompt)
        self.assertIn("作者=成员（ID=bob）", prompt)


if __name__ == "__main__":
    unittest.main()
