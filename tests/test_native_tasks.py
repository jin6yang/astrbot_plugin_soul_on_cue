"""Public native future-task lifecycle, without AstrBot or network access."""

import asyncio
import importlib.util
import logging
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock


def load_bridge():
    path = Path(__file__).resolve().parents[1] / "native_tasks.py"
    spec = importlib.util.spec_from_file_location("oncue_native_task_tests", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bridge = load_bridge()


class NativeManager:
    """Model the public API's cancellation and drain behavior."""

    def __init__(self):
        self.jobs = {}
        self.active = {}
        self.created = []
        self.deleted = []
        self.executed = []
        self.callback = AsyncMock()
        self.creation_started = asyncio.Event()
        self.creation_gate = None
        self.delete_gate = None
        self.fail_delete = False

    async def add_active_job(self, **options):
        self.creation_started.set()
        if self.creation_gate is not None:
            await self.creation_gate.wait()
        job = SimpleNamespace(job_id=f"job-{len(self.created) + 1}")
        self.created.append(options)
        self.jobs[job.job_id] = job
        return job

    async def run_job_now(self, job_id):
        task = asyncio.current_task()
        self.active[job_id] = task
        self.executed.append(job_id)
        try:
            await self.callback(job_id)
        finally:
            self.active.pop(job_id, None)

    async def delete_job(self, job_id):
        self.deleted.append(job_id)
        if self.fail_delete:
            raise OSError("delete failed")
        task = self.active.get(job_id)
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if self.delete_gate is not None:
            await self.delete_gate.wait()
        self.jobs.pop(job_id, None)


def event(job_id="job-1", umo="test:GroupMessage:1"):
    extras = {"cron_job": {"id": job_id}}
    return SimpleNamespace(unified_msg_origin=umo, get_extra=extras.get)


class NativeTaskTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.manager = NativeManager()
        self.context = SimpleNamespace(cron_manager=self.manager)
        self.logger = logging.getLogger("oncue.native.tasks.tests")
        self.service = bridge.NativeGlanceTasks(self.context, self.logger)
        self.trigger = bridge.GlanceTrigger({
            "umo": "test:GroupMessage:1", "self_id": "bot", "group_id": "1",
            "platform_meta": object(), "session_isolated": True,
        })
        self.current = AsyncMock(return_value=True)

    async def asyncTearDown(self):
        await self.service.close()

    async def test_trigger_only_keeps_decision_metadata(self):
        self.assertEqual(self.trigger.unified_msg_origin, "test:GroupMessage:1")
        self.assertEqual(self.trigger.get_self_id(), "bot")
        self.assertEqual(self.trigger.get_group_id(), "1")
        self.assertEqual(self.trigger.get_extra("missing", 9), 9)
        self.trigger.set_extra("decision", {"should_reply": True})
        self.assertTrue(self.trigger.get_extra("decision")["should_reply"])
        self.assertFalse(hasattr(self.trigger, "message_obj"))
        self.assertFalse(hasattr(self.trigger, "platform_meta"))
        self.assertFalse(hasattr(self.trigger, "send"))

    async def test_create_run_and_delete_use_only_public_api(self):
        result = await self.service.run(self.trigger, is_current=self.current)
        self.assertIsNone(result)
        options = self.manager.created[0]
        self.assertFalse(options["enabled"])
        self.assertFalse(options["persistent"])
        self.assertIsNone(options["cron_expression"])
        self.assertEqual(options["payload"]["session"], self.trigger.unified_msg_origin)
        self.assertEqual(set(options["payload"]), {"session", "note"})
        self.assertEqual(options["description"], "群聊一瞥接话")
        self.assertEqual(self.manager.executed, ["job-1"])
        self.assertEqual(self.manager.deleted, ["job-1"])
        self.assertFalse(self.manager.jobs)
        self.assertFalse(self.service._runs)
        self.assertFalse(self.service._tasks)

    async def test_missing_api_is_explicit_and_no_fallback(self):
        self.context.cron_manager = None
        with self.assertRaisesRegex(RuntimeError, "future-task API"):
            await self.service.run(self.trigger, is_current=self.current)
        self.assertFalse(self.manager.created)

    async def test_stale_before_creation_does_not_create_task(self):
        self.current.return_value = False
        await self.service.run(self.trigger, is_current=self.current)
        self.assertFalse(self.manager.created)

    async def test_stale_after_creation_deletes_without_execution(self):
        self.current.side_effect = [True, False]
        await self.service.run(self.trigger, is_current=self.current)
        self.assertEqual(self.manager.deleted, ["job-1"])
        self.assertFalse(self.manager.executed)

    async def test_creation_failure_leaves_no_registry_entry(self):
        self.manager.add_active_job = AsyncMock(side_effect=OSError())
        with self.assertRaises(OSError):
            await self.service.run(self.trigger, is_current=self.current)
        self.assertFalse(self.service._runs)
        self.assertFalse(self.service._tasks)

    async def test_execution_failure_still_deletes(self):
        self.manager.callback.side_effect = RuntimeError("native failure")
        with self.assertRaisesRegex(RuntimeError, "native failure"):
            await self.service.run(self.trigger, is_current=self.current)
        self.assertEqual(self.manager.deleted, ["job-1"])
        self.assertFalse(self.manager.jobs)

    async def test_owned_event_requires_task_id_and_exact_session(self):
        async def callback(job_id):
            self.assertIsNone(self.service.owned_run(event("foreign")))
            self.assertIsNone(self.service.owned_run(event(job_id, "test:GroupMessage:2")))
            native = event(job_id)
            owned = self.service.owned_run(native)
            self.assertIs(owned.trigger, self.trigger)
            self.assertIs(owned.native_event, native)
            self.assertIs(self.service.owned_run(native), owned)
            self.assertIsNone(self.service.owned_run(event(job_id)))
        self.manager.callback.side_effect = callback
        await self.service.run(self.trigger, is_current=self.current)
        self.assertIsNone(self.service.owned_run(event()))

    async def test_unknown_metadata_does_not_claim_an_event(self):
        for value in (None, "job-1", {"id": []}, {"id": 1}, {}):
            with self.subTest(value=value):
                item = SimpleNamespace(get_extra=lambda key: value)
                self.assertIsNone(self.service.owned_run(item))

    async def test_direct_cancellation_drains_and_deletes(self):
        entered = asyncio.Event()
        async def callback(job_id):
            entered.set()
            await asyncio.Event().wait()
        self.manager.callback.side_effect = callback
        task = asyncio.create_task(self.service.run(self.trigger, is_current=self.current))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.manager.deleted, ["job-1"])
        self.assertFalse(self.manager.jobs)
        self.assertFalse(self.service._tasks)

    async def test_cancel_chat_uses_native_delete_without_waiting_cycle(self):
        entered = asyncio.Event()
        async def callback(job_id):
            entered.set()
            await asyncio.Event().wait()
        self.manager.callback.side_effect = callback
        task = asyncio.create_task(self.service.run(self.trigger, is_current=self.current))
        await entered.wait()
        await asyncio.wait_for(self.service.cancel_chat(self.trigger.unified_msg_origin), 1)
        self.assertTrue(task.cancelled())
        self.assertEqual(self.manager.deleted, ["job-1"])
        self.assertFalse(self.manager.jobs)
        self.assertFalse(self.service._runs)

    async def test_cancel_chat_keeps_other_sessions_and_user_jobs(self):
        entered = asyncio.Event()
        release = asyncio.Event()
        async def callback(job_id):
            entered.set()
            await release.wait()
        self.manager.callback.side_effect = callback
        self.manager.jobs["user-job"] = object()
        task = asyncio.create_task(self.service.run(self.trigger, is_current=self.current))
        await entered.wait()
        await self.service.cancel_chat("test:GroupMessage:2")
        self.assertFalse(task.done())
        self.assertFalse(self.manager.deleted)
        release.set()
        await task
        self.assertEqual(set(self.manager.jobs), {"user-job"})

    async def test_cancellation_during_creation_waits_for_id_and_deletes(self):
        self.manager.creation_gate = asyncio.Event()
        task = asyncio.create_task(self.service.run(self.trigger, is_current=self.current))
        await self.manager.creation_started.wait()
        task.cancel()
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        self.manager.creation_gate.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.manager.deleted, ["job-1"])
        self.assertFalse(self.manager.executed)
        self.assertFalse(self.manager.jobs)

    async def test_repeated_cancellation_does_not_leave_created_task(self):
        self.manager.creation_gate = asyncio.Event()
        task = asyncio.create_task(self.service.run(self.trigger, is_current=self.current))
        await self.manager.creation_started.wait()
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        self.manager.creation_gate.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(self.manager.jobs)

    async def test_unload_during_creation_cleans_eventual_job(self):
        self.manager.creation_gate = asyncio.Event()
        task = asyncio.create_task(self.service.run(self.trigger, is_current=self.current))
        await self.manager.creation_started.wait()
        closing = asyncio.create_task(self.service.close())
        await asyncio.sleep(0)
        self.manager.creation_gate.set()
        await asyncio.wait_for(closing, 1)
        self.assertTrue(task.cancelled())
        self.assertFalse(self.manager.jobs)
        await self.service.run(self.trigger, is_current=self.current)
        self.assertEqual(len(self.manager.created), 1)

    async def test_cleanup_failure_is_logged_and_retried_at_close(self):
        self.manager.fail_delete = True
        with self.assertLogs(self.logger, level="WARNING"):
            await self.service.run(self.trigger, is_current=self.current)
        self.assertIn("job-1", self.manager.jobs)
        self.assertIsNone(self.service.owned_run(event()))
        self.manager.fail_delete = False
        await self.service.close()
        self.assertFalse(self.manager.jobs)
        self.assertFalse(self.service._runs)

    async def test_repeated_cancellation_during_cleanup_still_removes_owner(self):
        self.manager.delete_gate = asyncio.Event()
        task = asyncio.create_task(self.service.run(self.trigger, is_current=self.current))
        while not self.manager.deleted:
            await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        self.manager.delete_gate.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(self.manager.jobs)
        self.assertFalse(self.service._tasks)

    async def test_cancelled_deletion_is_retried_at_close(self):
        original = self.manager.delete_job
        self.manager.delete_job = AsyncMock(side_effect=asyncio.CancelledError())
        with self.assertRaises(asyncio.CancelledError):
            await self.service.run(self.trigger, is_current=self.current)
        self.assertFalse(self.service._tasks)
        self.manager.delete_job = original
        await self.service.close()
        self.assertFalse(self.manager.jobs)
        self.assertFalse(self.service._runs)

    async def test_native_completion_is_not_a_delivery_receipt(self):
        self.manager.callback.return_value = SimpleNamespace(
            completion_text="I sent a message", _has_send_oper=True,
        )
        with self.assertLogs(self.logger, level="INFO") as captured:
            result = await self.service.run(self.trigger, is_current=self.current)
        self.assertIsNone(result)
        self.assertTrue(any("投递未确认" in line for line in captured.output))
        self.assertFalse(hasattr(self.trigger, "delivered"))


if __name__ == "__main__":
    unittest.main()
