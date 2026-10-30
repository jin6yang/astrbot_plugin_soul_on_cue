"""临时关注的一瞥调度、公平领取与卸载收尾；不运行原生回复管道。"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from test_timing import _event, plugin_module


class AttentionSchedulerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.now = 1000.0
        clock = patch.object(plugin_module, "time", SimpleNamespace(time=lambda: self.now))
        clock.start()
        self.addCleanup(clock.stop)
        self.plugin = plugin_module.OnCuePlugin(SimpleNamespace(), {
            "enable": True, "enable_glance": True, "enable_echo_trigger": False,
            "enable_dense_trigger": False, "message_wait_seconds": 0,
        })
        self.plugin._glance_interval_seconds = lambda chat=None: 900.0
        self.started = []
        self.entered = {}
        self.release = {}
        self.tasks = set()

    async def asyncTearDown(self):
        self.tasks.update(self.plugin._glance_jobs.values())
        for task in self.tasks:
            if not task.done():
                task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)

    def block_due(self):
        async def due(chat_id):
            self.started.append(chat_id)
            self.entered[chat_id].set()
            try:
                await self.release[chat_id].wait()
            finally:
                # 与实际 _glance_due 一样，完成后安排下一轮，避免测试桩重复领取旧到期。
                self.plugin.glance_next_due[chat_id] = self.now + 900.0

        self.plugin._glance_due = AsyncMock(side_effect=due)

    def schedule(self, **deadlines):
        for chat_id, due in deadlines.items():
            self.plugin._chat(chat_id)
            self.plugin.glance_next_due[chat_id] = due
            self.entered[chat_id] = asyncio.Event()
            self.release[chat_id] = asyncio.Event()

    async def tick(self):
        await asyncio.wait_for(self.plugin._glance_tick(), 1)
        self.tasks.update(self.plugin._glance_jobs.values())

    async def wait_started(self, *chat_ids):
        await asyncio.wait_for(asyncio.gather(*(self.entered[chat_id].wait() for chat_id in chat_ids)), 1)

    async def test_two_sessions_start_without_waiting_for_the_slow_first_model(self):
        self.block_due()
        self.schedule(slow=700.0, peer=800.0)
        await self.tick()
        await self.wait_started("slow", "peer")
        self.assertEqual(self.started, ["slow", "peer"])
        self.assertFalse(self.release["slow"].is_set())
        self.assertFalse(self.release["peer"].is_set())
        self.assertEqual(set(self.plugin._glance_jobs), {"slow", "peer"})

    async def test_only_two_slots_then_oldest_waiting_session_gets_freed_slot(self):
        self.block_due()
        self.schedule(newest=950.0, middle=800.0, oldest=700.0, waiting=850.0)
        await self.tick()
        await self.wait_started("oldest", "middle")
        self.assertEqual(self.started, ["oldest", "middle"])
        await self.tick()
        self.assertEqual(self.plugin._glance_due.await_count, 2)
        finished = self.plugin._glance_jobs["middle"]
        self.release["middle"].set()
        await asyncio.wait_for(finished, 1)
        self.assertNotIn("middle", self.plugin._glance_jobs)
        await self.tick()
        await self.wait_started("waiting")
        self.assertEqual(self.started, ["oldest", "middle", "waiting"])
        self.assertEqual(set(self.plugin._glance_jobs), {"oldest", "waiting"})
        self.assertFalse(self.release["oldest"].is_set())

    async def test_repeated_ticks_do_not_duplicate_a_running_session(self):
        self.block_due()
        self.schedule(only=700.0, future=1100.0)
        await self.tick()
        await self.wait_started("only")
        original = self.plugin._glance_jobs["only"]
        for _ in range(3):
            await self.tick()
        self.plugin._glance_due.assert_awaited_once_with("only")
        self.assertIs(self.plugin._glance_jobs["only"], original)
        self.assertNotIn("future", self.plugin._glance_jobs)

    async def test_busy_sessions_do_not_prevent_other_due_sessions_from_starting(self):
        self.block_due()
        self.schedule(deciding=600.0, waiting=650.0, glancing=700.0, ready=800.0)
        self.plugin._chat("deciding").decision_inflight = object()
        self.plugin._chat("waiting").message_wait = object()
        self.plugin._chat("glancing").glance_inflight = True
        await self.tick()
        await self.wait_started("ready")
        self.plugin._glance_due.assert_awaited_once_with("ready")
        self.assertEqual(set(self.plugin._glance_jobs), {"ready"})

    async def test_job_exception_logs_reschedules_and_releases_slot(self):
        self.block_due()
        normal = self.plugin._glance_due.side_effect

        async def due(chat_id):
            if chat_id == "broken":
                raise RuntimeError("test model failure")
            await normal(chat_id)

        self.plugin._glance_due.side_effect = due
        self.schedule(broken=700.0, peer=800.0, next=900.0)
        with self.assertLogs(plugin_module.logger, level="ERROR") as logs:
            await self.tick()
            await self.wait_started("peer")
        self.assertIn("test model failure", "\n".join(logs.output))
        self.assertNotIn("broken", self.plugin._glance_jobs)
        self.assertGreater(self.plugin.glance_next_due["broken"], self.now)
        await self.tick()
        await self.wait_started("next")
        self.assertEqual(set(self.plugin._glance_jobs), {"peer", "next"})

    async def test_cancelled_job_releases_slot_for_next_due_session(self):
        self.block_due()
        self.schedule(cancelled=700.0, peer=800.0, next=900.0)
        await self.tick()
        await self.wait_started("cancelled", "peer")
        task = self.plugin._glance_jobs["cancelled"]
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertNotIn("cancelled", self.plugin._glance_jobs)
        await self.tick()
        await self.wait_started("next")
        self.assertEqual(set(self.plugin._glance_jobs), {"peer", "next"})

    async def test_disabled_plugin_or_glance_still_expires_attention_without_starting_jobs(self):
        self.plugin._glance_due = AsyncMock()
        self.now = 12000.0
        for flag in ("enable", "enable_glance"):
            with self.subTest(flag=flag):
                self.plugin.config.update(enable=True, enable_glance=True)
                self.plugin.config[flag] = False
                chat_id = flag + "-disabled"
                chat = self.plugin._chat(chat_id)
                chat.attention.apply("start", "等待进展", "source", 1200.0, 0, expected_revision=0)
                self.plugin.glance_next_due[chat_id] = 1000.0
                await self.tick()
                self.assertFalse(chat.attention.active)
                self.assertIn("3 小时", chat.attention.notice.reason)
                self.assertEqual(self.plugin._glance_jobs, {})
        self.plugin._glance_due.assert_not_awaited()

    async def test_cancel_before_first_execution_releases_the_slot(self):
        self.plugin._glance_due = AsyncMock()
        self.schedule(ready=700.0)
        await self.plugin._glance_tick()
        task = self.plugin._glance_jobs["ready"]
        self.tasks.add(task)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertNotIn("ready", self.plugin._glance_jobs)
        self.plugin._glance_due.assert_not_awaited()

    async def test_terminating_plugin_does_not_start_new_jobs(self):
        self.plugin._glance_due = AsyncMock()
        self.schedule(ready=700.0)
        self.plugin._terminating = True
        await self.tick()
        self.plugin._glance_due.assert_not_awaited()
        self.assertEqual(self.plugin._glance_jobs, {})

    async def test_due_expires_attention_but_does_not_call_model_without_new_messages(self):
        self.now = 12000.0
        chat_id = _event().unified_msg_origin
        chat = self.plugin._chat(chat_id)
        chat.incoming_seq = 10
        self.plugin.glance_last_check[chat_id] = 10
        chat.attention.apply("start", "等待进展", "source", 1200.0, 10, expected_revision=0)
        self.plugin._create_glance_event = lambda chat: _event("[主动发言机会]")
        self.plugin._build_prompt = AsyncMock(return_value="prompt")
        self.plugin._llm_decision = AsyncMock()
        self.plugin._run_native_reply = AsyncMock()
        await self.plugin._glance_due(chat_id)
        self.assertFalse(chat.attention.active)
        self.assertIn("3 小时", chat.attention.notice.reason)
        self.assertEqual(chat.incoming_seq, 10)
        self.assertEqual(self.plugin.glance_last_check[chat_id], 10)
        self.plugin._llm_decision.assert_not_awaited()
        self.plugin._run_native_reply.assert_not_awaited()
        self.assertFalse(chat.glance_inflight)

    async def test_terminate_waits_for_jobs_before_closing_caption_service_and_cache(self):
        order = []
        self.block_due()
        normal = self.plugin._glance_due.side_effect

        async def due(chat_id):
            try:
                await normal(chat_id)
            finally:
                order.append("finished-" + chat_id)

        self.plugin._glance_due.side_effect = due
        self.schedule(first=700.0, second=800.0)
        await self.tick()
        await self.wait_started("first", "second")
        jobs = list(self.plugin._glance_jobs.values())
        self.plugin.image_captions = SimpleNamespace(close=AsyncMock(side_effect=lambda: order.append("captions")))
        self.plugin.image_cache = SimpleNamespace(close=AsyncMock(side_effect=lambda: order.append("cache")))
        await asyncio.wait_for(self.plugin.terminate(), 1)
        self.assertTrue(all(task.cancelled() for task in jobs))
        self.assertEqual(self.plugin._glance_jobs, {})
        self.assertEqual(set(order[:2]), {"finished-first", "finished-second"})
        self.assertEqual(order[2:], ["captions", "cache"])
        self.assertIsNone(self.plugin.image_captions)
        self.assertIsNone(self.plugin.image_cache)

    async def test_terminate_cleans_jobs_cancelled_before_their_coroutine_starts(self):
        self.plugin._glance_due = AsyncMock()
        self.schedule(ready=700.0)
        # 直接调用不加 wait_for，create_task 尚未获得执行机会时立即卸载。
        await self.plugin._glance_tick()
        tasks = list(self.plugin._glance_jobs.values())
        self.tasks.update(tasks)
        await self.plugin.terminate()
        self.assertEqual(self.plugin._glance_jobs, {})
        self.assertTrue(all(task.done() for task in tasks))


if __name__ == "__main__":
    unittest.main()
