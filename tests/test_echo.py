"""复读规则回归：消息窗口、不同成员、纯文本筛选和决策入口。"""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from test_timing import Plain, _event, plugin_module


class EchoTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.now = 1000.0
        clock = patch.object(plugin_module, "time", SimpleNamespace(time=lambda: self.now))
        clock.start()
        self.addCleanup(clock.stop)
        self.plugin = plugin_module.OnCuePlugin(
            SimpleNamespace(),
            {"enable_echo_trigger": True, "enable_dense_trigger": False, "enable_glance": False},
        )
        self.chat = self.plugin._chat(_event().unified_msg_origin)
        self.plugin._decision_card = AsyncMock(return_value="测试角色")
        self.plugin._llm_decision = AsyncMock(return_value='{"should_reply": false, "reason": "先听听"}')

    def event(self, sender, text="哈哈", *, components=None):
        event = _event(text)
        event.get_sender_id = lambda: sender
        if components is not None:
            event.get_messages = lambda: components
        return event

    def cache(self, sender, text="哈哈", *, components=None):
        item = self.plugin._event_to_item(self.event(sender, text, components=components))
        self.chat.messages.append(item)
        return item

    def trigger(self):
        return self.plugin._stat_trigger_locked(self.chat, self.chat.messages[-1], self.now)

    async def send(self, sender, text="哈哈", *, components=None):
        event = self.event(sender, text, components=components)
        await self.plugin.on_message(event)
        self.now += 1
        return event

    async def test_three_distinct_members_in_five_messages_get_a_decision_not_a_forced_reply(self):
        # 测试事件的昵称都相同，仍应按成员 ID 识别为三人。
        await self.send("a")
        await self.send("x", "别的话题")
        await self.send("b")
        await self.send("y", "插句话")
        event = await self.send("c")
        self.plugin._llm_decision.assert_awaited_once()
        prompt = self.plugin._llm_decision.call_args.args[1]
        self.assertIn('最近 5 条群聊消息中，有 3 名不同成员重复了同一句纯文本："哈哈"', prompt)
        self.assertIn("请结合角色性格和聊天语境，判断是否想参与", prompt)
        self.assertFalse(event.is_at_or_wake_command)
        self.assertIsNone(event.get_extra("oncue_decision"))
        self.assertEqual(self.chat.pending_replies, {})
        self.assertIn("echo:哈哈", self.chat.backoffs)

    async def test_duplicate_messages_from_one_or_two_members_do_not_meet_three_person_threshold(self):
        for sender in ("a", "a", "a", "a", "b"):
            await self.send(sender)
        self.plugin._llm_decision.assert_not_awaited()
        await self.send("c")
        self.plugin._llm_decision.assert_awaited_once()

    def test_changing_nickname_does_not_create_another_participant(self):
        for nickname in ("Alice", "Bob", "Carol"):
            event = self.event("same-user")
            event.message_obj.sender.nickname = nickname
            self.chat.messages.append(self.plugin._event_to_item(event))
        self.assertIsNone(self.trigger())

    def test_mixed_messages_are_excluded_as_both_current_and_historical_matches(self):
        named_at = plugin_module.At()
        named_at.name = "别人"
        for extra in (plugin_module.Image(), plugin_module.Reply(), plugin_module.At(), named_at, object()):
            with self.subTest(component=type(extra).__name__, name=getattr(extra, "name", "")):
                self.chat.messages.clear()
                self.cache("a")
                self.cache("b")
                item = self.cache("c", components=[Plain("哈哈"), extra])
                self.assertFalse(item["is_pure_text"])
                self.assertIsNone(self.trigger())
                self.chat.messages.clear()
                self.cache("a", components=[Plain("哈哈"), extra])
                self.cache("b", components=[Plain("哈哈"), extra])
                self.cache("c")
                self.assertIsNone(self.trigger())

    def test_empty_messages_cannot_trigger_echo(self):
        self.cache("a")
        self.cache("b")
        for components in ([], [Plain(" \n\t ")], [plugin_module.Image()]):
            with self.subTest(components=components):
                item = self.cache("c", components=components)
                self.assertFalse(item["is_pure_text"])
                self.assertIsNone(self.trigger())

    def test_mixed_messages_occupy_window_positions_before_filtering(self):
        self.cache("a")
        self.cache("b")
        self.cache("x", "别的话题")
        self.cache("y", "插句话")
        self.cache("z", components=[plugin_module.Image()])
        self.cache("c")
        self.assertIsNone(self.trigger())  # 最近五条里只有 b、c 两人复读。

    def test_bot_replies_occupy_positions_but_never_contribute_participants(self):
        self.cache("a")
        self.cache("b")
        self.cache("x", "别的话题")
        self.cache("y", "插句话")
        self.plugin._record_reply(self.chat, "哈哈", self.now)
        self.cache("c")
        self.assertIsNone(self.trigger())

    def test_unrelated_current_message_does_not_retrigger_old_echo(self):
        for sender in ("a", "b", "c"):
            self.cache(sender)
        self.cache("d", "换个话题")
        self.assertIsNone(self.trigger())

    async def test_old_messages_still_count_and_legacy_seconds_setting_is_ignored(self):
        self.plugin.config["echo_window"] = 1
        await self.send("a")
        self.now += 86400
        await self.send("b")
        self.now += 86400
        await self.send("c")
        self.plugin._llm_decision.assert_awaited_once()
        self.assertIn("最近 3 条群聊消息", self.plugin._llm_decision.call_args.args[1])

    def test_nested_configuration_changes_window_and_participant_threshold(self):
        self.plugin.config["config_trigger"] = {"echo_message_count": 4, "echo_threshold": 2}
        self.cache("a")
        self.cache("x", "别的话题")
        self.cache("y", "插句话")
        self.cache("b")
        context, key = self.trigger()
        self.assertIn("最近 4 条群聊消息中，有 2 名不同成员", context)
        self.assertEqual(key, "echo:哈哈")
        self.plugin.config["config_trigger"]["echo_message_count"] = 3
        self.assertIsNone(self.trigger())  # 缩短窗口后 a 已在范围外。

    def test_configured_larger_threshold_requires_that_many_distinct_members(self):
        self.plugin.config.update(echo_message_count=7, echo_threshold=4)
        for sender in ("a", "a", "b", "c"):
            self.cache(sender)
        self.assertIsNone(self.trigger())
        self.cache("d")
        self.assertIn("有 4 名不同成员", self.trigger()[0])

    def test_invalid_low_limits_cannot_allow_single_person_echo(self):
        for window, threshold in ((0, 0), (-5, -1), (1, 1), (5, 1), (2, 9)):
            with self.subTest(window=window, threshold=threshold):
                self.chat.messages.clear()
                self.plugin.config.update(echo_message_count=window, echo_threshold=threshold)
                self.cache("a")
                self.cache("a")
                self.assertIsNone(self.trigger())
                self.cache("b")
                self.assertIsNotNone(self.trigger())

    def test_large_window_and_threshold_are_limited_by_cache_capacity(self):
        self.plugin.config.update(echo_message_count=100, echo_threshold=100)
        for sender in range(49):
            self.cache(str(sender))
        self.assertIsNone(self.trigger())
        self.cache("49")
        self.assertIn("最近 50 条群聊消息中，有 50 名不同成员", self.trigger()[0])

    def test_non_numeric_settings_use_defaults(self):
        self.plugin.config.update(echo_message_count="invalid", echo_threshold=None)
        self.cache("a")
        self.cache("b")
        self.assertIsNone(self.trigger())
        self.cache("c")
        self.assertIsNotNone(self.trigger())

    def test_whitespace_is_normalized_but_case_and_punctuation_remain_significant(self):
        self.cache("a", "  Hello\tworld  ")
        self.cache("b", "Hello\nworld")
        self.cache("c", components=[Plain("Hello"), Plain("world")])
        self.assertEqual(self.trigger()[1], "echo:Hello world")
        self.cache("d", "hello world")
        self.assertIsNone(self.trigger())
        self.cache("e", "Hello world!")
        self.assertIsNone(self.trigger())

    async def test_echo_still_respects_busy_cooldown_backoff_and_reply_capacity(self):
        for gate in ("busy", "cooldown", "backoff", "capacity"):
            with self.subTest(gate=gate):
                self.chat.messages.clear()
                self.chat.decision_inflight = gate == "busy"
                self.chat.cooldown_until = self.now + 30 if gate == "cooldown" else 0
                self.chat.backoffs = {"echo:哈哈": {"until": self.now + 30}} if gate == "backoff" else {}
                self.chat.pending_replies = {"pending": self.now + 30} if gate == "capacity" else {}
                self.plugin.config["max_replies_per_window"] = 1
                self.cache("a")
                self.cache("b")
                event = await self.send("c")
                self.plugin._llm_decision.assert_not_awaited()
                self.assertFalse(event.is_at_or_wake_command)

    async def test_positive_decision_uses_existing_reply_reservation_path(self):
        self.plugin._llm_decision.return_value = '{"should_reply": true, "reason": "想接梗"}'
        for sender in ("a", "b", "c"):
            event = await self.send(sender)
        self.assertTrue(event.is_at_or_wake_command)
        self.assertEqual(event.get_extra("oncue_decision")["_backoff_key"], "echo:哈哈")
        self.assertEqual(len(self.chat.pending_replies), 1)
        self.assertEqual(list(self.chat.reply_ts), [])

    async def test_disabling_echo_stops_decisions(self):
        self.plugin.config["enable_echo_trigger"] = False
        for sender in ("a", "b", "c"):
            await self.send(sender)
        self.plugin._llm_decision.assert_not_awaited()

    def test_dense_trigger_can_still_match_when_echo_does_not(self):
        self.plugin.config.update(enable_dense_trigger=True, dense_message_threshold=3, dense_participant_threshold=2)
        self.cache("a", components=[Plain("哈哈"), plugin_module.Image()])
        self.cache("b", components=[Plain("哈哈"), plugin_module.Image()])
        self.cache("c")
        self.assertEqual(self.trigger()[1], "dense")


if __name__ == "__main__":
    unittest.main()
