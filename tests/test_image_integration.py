"""插件入站资源保留、动态引用渲染和缓存卸载契约。"""

import asyncio
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from test_context import reply
from test_timing import Plain, _event, plugin_module


class ImageIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.now = 1000.0
        self.plugin = plugin_module.OnCuePlugin(SimpleNamespace(), {
            "enable_glance": False, "enable_echo_trigger": False,
            "enable_dense_trigger": False, "message_wait_seconds": 0,
        })
        self.cache = plugin_module.SessionImageCache(
            self.root / "images", clock=lambda: self.now,
            limits=self.plugin._image_cache_limits,
        )
        self.plugin.image_cache = self.cache
        self.chat_id = _event().unified_msg_origin
        self.chat = self.plugin._chat(self.chat_id)

    async def asyncTearDown(self):
        await self.plugin.terminate()

    @staticmethod
    def image(*, url="", file="", path=""):
        image = plugin_module.Image()
        image.url, image.file, image.path = url, file, path
        return image

    @staticmethod
    def event(*parts, message_id="current"):
        event = _event()
        event.message_obj.message_id = message_id
        event.get_messages = lambda: list(parts)
        return event

    def source(self, name="platform.png", content=b"image bytes"):
        path = self.root / name
        path.write_bytes(content)
        return path

    async def test_remote_references_do_not_start_downloads_in_this_batch(self):
        self.cache._fetch = AsyncMock(side_effect=AssertionError("No network expected"))
        event = self.event(self.image(url="https://platform.invalid/image.png"))
        await self.plugin.on_message(event)
        self.cache._fetch.assert_not_awaited()
        image = self.chat.messages[-1]["parts"][0].image
        self.assertEqual(image.status, "内容未读取")
        self.assertEqual(list(self.cache.root.iterdir()), [])

    async def test_existing_local_resource_is_copied_without_changing_platform_source(self):
        source = self.source()
        component = self.image(url="https://platform.invalid/image.png", file=str(source))
        event = self.event(component)
        await self.plugin.on_message(event)
        image = self.chat.messages[-1]["parts"][0].image
        self.assertEqual(image.source, component.url)
        self.assertEqual(image.size_bytes, len(b"image bytes"))
        self.assertEqual(component.file, str(source))
        source.unlink()  # Simulate framework-owned temporary resource cleanup.
        async with self.cache.acquire(self.chat_id, image) as retained:
            self.assertIsNotNone(retained)
            self.assertEqual(retained.path.read_bytes(), b"image bytes")
        self.assertEqual(image.status, "内容未读取")

    async def test_file_uri_resource_is_retained_and_original_is_not_deleted(self):
        source = self.source("platform file.png")
        await self.plugin.on_message(self.event(self.image(file=source.as_uri())))
        image = self.chat.messages[-1]["parts"][0].image
        self.assertEqual(image.generation, 1)
        await self.cache.close()
        self.assertEqual(source.read_bytes(), b"image bytes")

    async def test_duplicate_delivery_reuses_canonical_resource_without_new_lifecycle(self):
        source = self.source()
        await self.plugin.on_message(self.event(self.image(file=str(source))))
        first = self.chat.messages[-1]["parts"][0].image
        original = await self.cache.ensure(self.chat_id, first)
        self.now += 300
        duplicate = self.event(self.image(file=str(source)))
        item = self.plugin._event_to_item(duplicate, self.chat)
        self.assertIs(item["parts"][0].image, first)
        await self.plugin.on_message(duplicate)
        again = await self.cache.ensure(self.chat_id, first)
        self.assertEqual(again.path, original.path)
        self.assertEqual(again.generation, original.generation)
        self.assertEqual(self.chat.incoming_seq, 1)

    async def test_explicit_quote_chain_reuses_uniquely_identified_original_image(self):
        source = self.source()
        url = "https://platform.invalid/original.png"
        await self.plugin.on_message(self.event(self.image(url=url, file=str(source)), message_id="original"))
        original = self.chat.messages[-1]["parts"][0].image
        quoted_component = reply("original", chain=[self.image(url=url, file=str(source))])
        await self.plugin.on_message(self.event(quoted_component, Plain("quote"), message_id="quote"))
        quoted = self.chat.messages[-1]["parts"][0].children[0].image
        self.assertIs(quoted, original)
        self.assertEqual(quoted.generation, 1)
        self.assertEqual(len(list(self.cache.root.glob("*.cache"))), 1)

    async def test_local_copy_wait_does_not_hold_chat_lock_or_block_new_messages(self):
        source = self.source()
        entered, release = asyncio.Event(), asyncio.Event()

        async def copy(path, write, max_bytes):
            entered.set()
            await release.wait()
            data = Path(path).read_bytes()
            write(data)
            return len(data)

        self.cache._fetch = copy
        task = asyncio.create_task(self.plugin.on_message(self.event(self.image(file=str(source)))))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            self.assertFalse(self.chat.lock.locked())
            await asyncio.wait_for(self.plugin.on_message(self.event(Plain("another member"), message_id="later")), 1)
            self.assertEqual(self.chat.incoming_seq, 2)
        finally:
            release.set()
            await asyncio.wait_for(task, 1)

    async def test_current_snapshot_and_quoted_image_states_refresh_without_new_activity(self):
        source = self.source(content=b"1234567890")
        await self.plugin.on_message(self.event(self.image(file=str(source)), message_id="original"))
        image = self.chat.messages[-1]["parts"][0].image
        cached = await self.cache.ensure(self.chat_id, image)
        self.assertTrue(self.cache.description_completed(self.chat_id, image, cached.generation, "一张文字图"))
        await self.plugin.on_message(self.event(reply("original"), Plain("quoted"), message_id="quote"))
        before = self.plugin._context_snapshot(self.chat)
        progress = self.chat.incoming_seq
        self.assertIn("10B，已转述，一张文字图", before.render())
        self.now += 2160
        after = self.plugin._context_snapshot(self.chat)
        quoted = self.chat.messages[-1]["parts"][0].render()
        self.assertIn("当前不可用（本地缓存已被清理），一张文字图", quoted)
        self.assertIn("当前不可用（本地缓存已被清理）", after.render())
        self.assertNotIn("当前不可用", before.render())
        self.assertEqual(self.chat.incoming_seq, progress)
        self.assertIn("30 MiB 是插件的大小限制", after.render())
        self.assertIn("本次快照", after.render())

    async def test_cancelling_local_copy_releases_approved_reply_reservation(self):
        source = self.source()
        entered, release = asyncio.Event(), asyncio.Event()

        async def copy(path, write, max_bytes):
            write(b"partial")
            entered.set()
            await release.wait()
            return 7

        self.cache._fetch = copy
        self.chat.last_reply_ts = plugin_module.time.time()
        self.plugin._decision_card = AsyncMock(return_value="a persona")
        self.plugin._llm_decision = AsyncMock(return_value='{"should_reply": true, "reason": "respond"}')
        event = self.event(self.image(file=str(source)))
        task = asyncio.create_task(self.plugin.on_message(event))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            self.assertEqual(len(self.chat.pending_replies), 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(self.chat.pending_replies, {})
            self.assertIsNone(event.get_extra("oncue_decision"))
            self.assertFalse(self.chat.lock.locked())
            self.assertEqual(list(self.cache.root.iterdir()), [])
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_summon_during_local_copy_invalidates_old_reply_only(self):
        await self.assert_later_summon_wins(initial_direct=False)

    async def test_cancel_while_waiting_for_post_copy_lock_releases_only_own_reservation(self):
        source = self.source()
        entered, release = asyncio.Event(), asyncio.Event()

        async def copy(path, write, max_bytes):
            entered.set()
            await release.wait()
            write(b"data")
            return 4

        self.cache._fetch = copy
        self.chat.last_reply_ts = plugin_module.time.time()
        self.plugin._decision_card = AsyncMock(return_value="a persona")
        self.plugin._llm_decision = AsyncMock(return_value='{"should_reply": true, "reason": "respond"}')
        event = self.event(self.image(file=str(source)))
        task = asyncio.create_task(self.plugin.on_message(event))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            async with self.chat.lock:
                self.chat.pending_replies["newer-event"] = plugin_module.time.time() + 300
                release.set()
                for _ in range(30):
                    if not self.cache._jobs:
                        break
                    await asyncio.sleep(0)
                self.assertFalse(self.cache._jobs, "copy should finish before cancellation")
                self.assertFalse(task.done(), "post-copy validation must still wait for the lock")
                task.cancel()
                await asyncio.sleep(0)
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, 1)
            self.assertEqual(set(self.chat.pending_replies), {"newer-event"})
            self.assertIsNone(event.get_extra("oncue_decision"))
            self.assertFalse(self.chat.lock.locked())
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_summon_during_direct_reply_local_copy_invalidates_old_reply_only(self):
        await self.assert_later_summon_wins(initial_direct=True)

    async def assert_later_summon_wins(self, *, initial_direct):
        source = self.source()
        entered, release = asyncio.Event(), asyncio.Event()

        async def copy(path, write, max_bytes):
            entered.set()
            await release.wait()
            write(b"data")
            return 4

        self.cache._fetch = copy
        self.chat.last_reply_ts = plugin_module.time.time()
        self.plugin._decision_card = AsyncMock(return_value="a persona")
        self.plugin._llm_decision = AsyncMock(return_value='{"should_reply": true, "reason": "respond"}')
        event = self.event(self.image(file=str(source)))
        event.is_at_or_wake_command = initial_direct
        task = asyncio.create_task(self.plugin.on_message(event))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            summoned = self.event(Plain("a new summon"), message_id="summon")
            summoned.is_at_or_wake_command = True
            await asyncio.wait_for(self.plugin.on_message(summoned), 1)
            release.set()
            await asyncio.wait_for(task, 1)
            self.assertIsNone(event.get_extra("oncue_decision"))
            self.assertEqual(self.chat.pending_replies, {})
            self.assertTrue(summoned.get_extra("oncue_decision")["should_reply"])
            self.assertTrue(summoned.is_at_or_wake_command)
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_expiry_remains_unavailable_until_explicit_refetch(self):
        source = self.source()
        await self.plugin.on_message(self.event(self.image(file=str(source))))
        image = self.chat.messages[-1]["parts"][0].image
        first_generation = image.generation
        self.now += 7200
        await self.plugin.on_message(self.event(Plain("later"), message_id="later"))
        self.assertTrue(source.exists())
        self.assertEqual(image.unavailable_reason, "本地缓存已被清理")
        self.assertEqual(image.generation, first_generation)
        cached = await self.cache.ensure(self.chat_id, image)
        self.assertGreater(cached.generation, first_generation)
        self.assertEqual(image.unavailable_reason, "")

    async def test_sent_bot_image_keeps_local_copy_without_user_progress(self):
        source = self.source()
        event = self.event(Plain("a request"))
        event.set_extra("oncue_decision", {"should_reply": True})
        event._has_send_oper = True
        event.result.chain = [self.image(file=str(source))]
        await self.plugin.mark_reply_sent(event)
        self.assertEqual(self.chat.incoming_seq, 0)
        self.assertEqual(self.chat.messages[-1]["role"], "assistant")
        image = self.chat.messages[-1]["parts"][0].image
        self.assertEqual(image.generation, 1)
        async with self.cache.acquire(self.chat_id, image) as retained:
            self.assertEqual(retained.path.read_bytes(), b"image bytes")
        await self.plugin.mark_reply_sent(event)
        self.assertEqual(len(self.chat.reply_ts), 1)
        self.assertEqual(len(list(self.cache.root.glob("*.cache"))), 1)

    async def test_cache_settings_read_from_context_category_and_units_are_explicit(self):
        self.plugin.config["config_context"] = {
            "image_max_mb": 12, "image_cache_count": 8,
            "image_cache_mb": 25, "image_cache_minutes": 60,
        }
        limits = self.plugin._image_cache_limits()
        self.assertEqual(limits.max_image_bytes, 12 * 1024**2)
        self.assertEqual(limits.max_count, 8)
        self.assertEqual(limits.max_total_bytes, 25 * 1024**2)
        self.assertEqual(limits.ttl_seconds, 3600)
        schema = json.loads((Path(__file__).resolve().parents[1] / "_conf_schema.json").read_text(encoding="utf-8"))
        items = schema["config_context"]["items"]
        for key, value in (("image_max_mb", 30), ("image_cache_count", 20), ("image_cache_mb", 75), ("image_cache_minutes", 120)):
            self.assertEqual(items[key]["default"], value)

    def test_image_status_fields_keep_order_and_omit_unknown_size_or_description(self):
        cases = (
            (plugin_module.ImageReference("unread"), "[图片，内容未读取]"),
            (
                plugin_module.ImageReference("described", size_bytes=1024, description="一只猫"),
                "[图片，1KiB，已转述，一只猫]",
            ),
            (
                plugin_module.ImageReference("oversized", size_bytes=31 * 1024**2, unavailable_reason="超过插件大小限制"),
                "[图片，31MiB，当前不可用（超过插件大小限制）]",
            ),
            (
                plugin_module.ImageReference("cloud", description="一只猫", unavailable_reason="云端缓存已被清理"),
                "[图片，当前不可用（云端缓存已被清理），一只猫]",
            ),
            (plugin_module.ImageReference("failed", unavailable_reason="获取失败"), "[图片，当前不可用（获取失败）]"),
        )
        for image, expected in cases:
            with self.subTest(image=image.image_id):
                self.assertEqual(image.render(), expected)

    async def test_initialize_and_terminate_manage_cleanup_tasks_and_owned_files(self):
        plugin = plugin_module.OnCuePlugin(SimpleNamespace(), {"enable": False})
        with patch.object(plugin_module, "StarTools", SimpleNamespace(get_data_dir=lambda name: self.root / "plugin")):
            await plugin.initialize()
        try:
            self.assertIsNotNone(plugin.image_cache_task)
            self.assertFalse(plugin.image_cache_task.done())
            source = self.source()
            image = plugin_module.ImageReference("resource", source=str(source))
            retained = await plugin.image_cache.ensure(self.chat_id, image)
            self.assertTrue(retained.path.exists())
        finally:
            await plugin.terminate()
        self.assertIsNone(plugin.image_cache_task)
        self.assertIsNone(plugin.image_cache)
        self.assertFalse(retained.path.exists())
        self.assertTrue(source.exists())


if __name__ == "__main__":
    unittest.main()
