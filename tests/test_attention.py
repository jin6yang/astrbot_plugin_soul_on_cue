"""临时关注的固定兜底和过期结果隔离，不依赖 AstrBot。"""

import importlib.util
import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "oncue_attention_tests", Path(__file__).resolve().parents[1] / "attention.py",
)
attention = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = attention
spec.loader.exec_module(attention)
AttentionState = attention.AttentionState


class AttentionTests(unittest.TestCase):
    def setUp(self):
        self.state = AttentionState()

    def apply(self, action, note="关注进展", message_id="message-1", now=1000.0, progress=10):
        return self.state.apply(
            action, note, message_id, now, progress, expected_revision=self.state.revision,
        )

    def test_default_is_inactive_with_no_prompt(self):
        self.assertEqual(self.state.snapshot().render(1000, 10), "")
        self.assertFalse(self.state.expire(999999, 100))
        self.assertEqual(self.state.revision, 0)

    def test_start_records_current_progress_and_time(self):
        self.assertEqual(self.apply("start"), "started")
        self.assertTrue(self.state.active)
        self.assertEqual((self.state.started_at, self.state.start_progress), (1000.0, 10))
        self.assertEqual((self.state.note, self.state.message_id), ("关注进展", "message-1"))
        self.assertEqual(self.state.revision, 1)

    def test_fiftieth_new_message_expires_but_fortyninth_does_not(self):
        self.apply("start")
        self.assertFalse(self.state.expire(1001, 59))
        self.assertTrue(self.state.expire(1001, 60))
        self.assertFalse(self.state.active)
        self.assertIn("50 条", self.state.notice.reason)
        self.assertEqual((self.state.note, self.state.message_id), ("", ""))
        self.assertEqual((self.state.notice.note, self.state.notice.message_id), ("关注进展", "message-1"))

    def test_three_hours_expires_at_exact_boundary(self):
        self.apply("start")
        self.assertFalse(self.state.expire(1000 + attention.ATTENTION_MAX_SECONDS - 0.001, 10))
        self.assertTrue(self.state.expire(1000 + attention.ATTENTION_MAX_SECONDS, 10))
        self.assertIn("3 小时", self.state.notice.reason)

    def test_both_limits_are_reported_once(self):
        self.apply("start")
        self.assertTrue(self.state.expire(11800, 60))
        notice = self.state.notice
        revision = self.state.revision
        self.assertIn("50 条", notice.reason)
        self.assertIn("3 小时", notice.reason)
        self.assertFalse(self.state.expire(20000, 1000))
        self.assertIs(self.state.notice, notice)
        self.assertEqual(self.state.revision, revision)

    def test_unchanged_incoming_progress_does_not_count_bot_or_status_updates(self):
        self.apply("start")
        for tick in range(100):
            self.assertFalse(self.state.expire(1000 + tick, 10))
        self.assertTrue(self.state.active)

    def test_keep_updates_summary_without_resetting_limits(self):
        self.apply("start")
        self.assertEqual(self.apply("keep", note="新进展", message_id="message-2", now=2000, progress=59), "kept")
        self.assertEqual((self.state.note, self.state.message_id), ("新进展", "message-2"))
        self.assertEqual((self.state.started_at, self.state.start_progress), (1000.0, 10))
        self.assertTrue(self.state.expire(2001, 60))

    def test_keep_empty_fields_preserves_summary_and_revision(self):
        self.apply("start")
        revision = self.state.revision
        self.assertEqual(self.apply("keep", note=" ", message_id=" "), "kept")
        self.assertEqual((self.state.note, self.state.message_id), ("关注进展", "message-1"))
        self.assertEqual(self.state.revision, revision)

    def test_repeated_start_behaves_as_keep(self):
        self.apply("start")
        self.assertEqual(self.apply("start", now=2000, progress=59), "kept")
        self.assertEqual((self.state.started_at, self.state.start_progress), (1000.0, 10))
        self.assertTrue(self.state.expire(2000, 60))

    def test_restart_resets_both_limits(self):
        self.apply("start")
        self.assertEqual(self.apply("restart", note="继续关注", now=11799, progress=59), "restarted")
        self.assertEqual((self.state.started_at, self.state.start_progress), (11799, 59))
        self.assertFalse(self.state.expire(11800, 60))
        self.assertTrue(self.state.expire(11800, 109))

    def test_restart_can_start_a_new_inactive_range(self):
        self.assertEqual(self.apply("restart"), "restarted")
        self.assertTrue(self.state.active)

    def test_stop_clears_range_without_creating_exit_notice(self):
        self.apply("start")
        self.assertEqual(self.apply("stop"), "stopped")
        self.assertFalse(self.state.active)
        self.assertEqual((self.state.started_at, self.state.start_progress, self.state.note, self.state.message_id), (0, 0, "", ""))
        self.assertIsNone(self.state.notice)
        self.assertEqual(self.apply("stop"), "none")

    def test_unknown_action_or_missing_summary_does_not_start(self):
        for action in ("unknown", "keep", "stop", "START"):
            with self.subTest(action=action):
                self.assertEqual(self.apply(action), "none")
        for action in ("start", "restart"):
            with self.subTest(action=action):
                self.assertEqual(self.apply(action, note=" \n "), "none")
        self.assertFalse(self.state.active)
        self.assertEqual(self.state.revision, 0)

    def test_empty_restart_does_not_destroy_existing_range(self):
        self.apply("start")
        snapshot = self.state.snapshot()
        self.assertEqual(self.apply("restart", note=""), "none")
        self.assertEqual(self.state.snapshot(), snapshot)

    def test_summary_is_stripped_and_bounded(self):
        self.apply("start", note=" " + "字" * 300 + " ", message_id=" source ")
        self.assertEqual(len(self.state.note), attention.ATTENTION_NOTE_LIMIT)
        self.assertEqual(self.state.message_id, "source")

    def test_stale_result_cannot_undo_forced_exit(self):
        self.apply("start")
        old_revision = self.state.revision
        self.state.expire(1100, 60)
        snapshot = self.state.snapshot()
        for action in ("keep", "start", "restart", "stop"):
            result = self.state.apply(action, "过期结果", "old", 1200, 60, expected_revision=old_revision)
            self.assertEqual(result, "stale")
            self.assertEqual(self.state.snapshot(), snapshot)

    def test_valid_later_result_can_start_again_after_exit(self):
        self.apply("start")
        self.state.expire(1100, 60)
        self.assertEqual(self.apply("start", now=1200, progress=61), "started")
        self.assertEqual((self.state.started_at, self.state.start_progress), (1200, 61))

    def test_notice_is_only_cleared_by_matching_token(self):
        self.apply("start")
        self.state.expire(1100, 60)
        token = self.state.notice.token
        revision = self.state.revision
        self.state.acknowledge(token - 1)
        self.assertIsNotNone(self.state.notice)
        self.state.acknowledge(token)
        self.assertIsNone(self.state.notice)
        self.assertEqual(self.state.revision, revision)

    def test_old_acknowledgement_cannot_clear_new_exit_notice(self):
        self.apply("start")
        self.state.expire(1100, 60)
        old_token = self.state.notice.token
        self.apply("restart", note="另一件事", now=1200, progress=60)
        self.state.expire(1300, 110)
        self.assertNotEqual(self.state.notice.token, old_token)
        self.state.acknowledge(old_token)
        self.assertEqual(self.state.notice.note, "另一件事")

    def test_snapshot_and_nested_notice_are_frozen(self):
        self.apply("start")
        self.state.expire(1100, 60)
        snapshot = self.state.snapshot()
        with self.assertRaises(FrozenInstanceError):
            snapshot.note = "changed"
        with self.assertRaises(FrozenInstanceError):
            snapshot.notice.reason = "changed"
        self.state.acknowledge(snapshot.notice.token)
        self.assertIsNotNone(snapshot.notice)

    def test_rendered_snapshot_does_not_consume_notice(self):
        self.apply("start")
        current = self.state.snapshot().render(1060, 14)
        self.assertIn("关注进展", current)
        self.assertIn("message-1", current)
        self.assertIn("4 条", current)
        self.assertIn("不强制回复", current)
        self.state.expire(1100, 60)
        snapshot = self.state.snapshot()
        first = snapshot.render(1200, 61)
        self.assertIn("50 条", first)
        self.assertIn("恢复普通一瞥", first)
        self.assertIn("不代表话题已经结束", first)
        self.assertIn("不要求自动重新关注", first)
        self.assertEqual(first, snapshot.render(1300, 62))
        self.assertIsNotNone(self.state.notice)

    def test_snapshot_before_changes_stays_unchanged(self):
        self.apply("start")
        snapshot = self.state.snapshot()
        self.apply("keep", note="新事项", message_id="new-source")
        self.assertEqual((snapshot.note, snapshot.message_id), ("关注进展", "message-1"))
        self.assertNotEqual(snapshot.revision, self.state.revision)


if __name__ == "__main__":
    unittest.main()
