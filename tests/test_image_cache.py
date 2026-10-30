"""独立图片缓存：受限下载、会话容量、租约与生命周期；不访问网络。"""

import asyncio
import gc
import importlib
import sys
import tempfile
import unittest
import weakref
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from test_timing import plugin_module


message_module = ModuleType(plugin_module.__name__ + ".message_context")
message_module.ImageReference = plugin_module.ImageReference
# test_timing 恢复其 AstrBot 桩时会移除临时包；加载真实缓存模块时显式复用
# 同一 ImageReference，避免另一份消息模型掩盖引用共享问题。
with patch.dict(sys.modules, {
    plugin_module.__name__: plugin_module,
    message_module.__name__: message_module,
}):
    cache_module = importlib.import_module(plugin_module.__name__ + ".image_cache")
    resources_module = importlib.import_module(plugin_module.__name__ + ".image_resources")
CacheLimits = cache_module.CacheLimits
SessionImageCache = cache_module.SessionImageCache
ImageFetchError = resources_module.ImageFetchError
ImageReference = plugin_module.ImageReference
MIB = 1024**2


class FetchProbe:
    """大小和完成时机均由测试控制，让缓存自己执行写入上限。"""

    def __init__(self):
        self.sizes = {}
        self.errors = {}
        self.calls = []
        self.prefix_bytes = {}
        self.entered = {}
        self.release = {}
        self.finished = {}
        self.cancelled = set()

    def pause(self, source):
        self.entered[source] = asyncio.Event()
        self.release[source] = asyncio.Event()
        self.finished[source] = asyncio.Event()
        return self.entered[source], self.release[source]

    async def __call__(self, source, write, max_bytes):
        self.calls.append(source)
        try:
            prefix = self.prefix_bytes.get(source, 0)
            if prefix:
                write(b"x" * prefix)
            if source in self.entered:
                self.entered[source].set()
            if source in self.release:
                await self.release[source].wait()
            if source in self.errors:
                raise self.errors[source]
            size = self.sizes.get(source, 4)
            remaining = size - prefix
            while remaining:
                chunk = b"x" * min(remaining, 64 * 1024)
                write(chunk)
                remaining -= len(chunk)
            return size
        except asyncio.CancelledError:
            self.cancelled.add(source)
            raise
        finally:
            if source in self.finished:
                self.finished[source].set()


class ImageCacheTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.now = 1000.0
        temporary = tempfile.TemporaryDirectory(prefix="oncue-image-tests-")
        self.addCleanup(temporary.cleanup)
        self.temp = Path(temporary.name)
        self.root = self.temp / "cache"
        self.fetch = FetchProbe()
        self.caches = []
        self.tasks = []
        self.limits = CacheLimits(max_image_bytes=8, max_count=3, max_total_bytes=20, ttl_seconds=7200)
        self.cache = self.make_cache()

    def make_cache(self, *, root=None, limits=None, fetch=None):
        cache = SessionImageCache(
            root or self.root,
            limits=limits or self.limits,
            fetch=fetch or self.fetch,
            clock=lambda: self.now,
        )
        self.caches.append(cache)
        return cache

    async def asyncTearDown(self):
        for task in self.tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        for cache in self.caches:
            await cache.close()

    def start(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.tasks.append(task)
        return task

    async def wait(self, event):
        await asyncio.wait_for(event.wait(), 1)

    def reference(self, image_id="one", *, size=None, source=None):
        return ImageReference(image_id, source=source or f"fake:{image_id}", size_bytes=size)

    def owned_files(self):
        return list(self.root.glob("oncue_*"))

    def cached_files(self):
        return list(self.root.glob("oncue_*.cache"))

    def test_default_limits_match_confirmed_units(self):
        limits = CacheLimits()
        self.assertEqual(limits.max_image_bytes, 30 * MIB)
        self.assertEqual(limits.max_count, 20)
        self.assertEqual(limits.max_total_bytes, 75 * MIB)
        self.assertEqual(limits.ttl_seconds, 7200)

    async def test_register_returns_same_live_reference_for_message_and_quote(self):
        first = self.reference("same")
        second = self.reference("same")
        canonical = self.cache.register("chat", first)
        self.assertIs(canonical, first)
        self.assertIs(self.cache.register("chat", second), canonical)
        self.assertEqual(self.fetch.calls, [])
        self.assertEqual(self.owned_files(), [])

    async def test_unknown_size_is_measured_from_actual_bytes(self):
        ref = self.reference()
        self.fetch.sizes[ref.source] = 7
        self.assertIsNone(ref.size_bytes)
        cached = await self.cache.ensure("chat", ref)
        self.assertIsNotNone(cached)
        self.assertEqual(cached.size_bytes, 7)
        self.assertEqual(cached.path.stat().st_size, 7)
        self.assertEqual(ref.size_bytes, 7)
        self.assertEqual(ref.generation, cached.generation)
        self.assertEqual(ref.unavailable_reason, "")

    async def test_actual_thirty_mib_is_accepted_and_one_extra_byte_is_rejected(self):
        cache = self.make_cache(limits=CacheLimits())
        allowed = self.reference("at-limit")
        oversized = self.reference("over-limit")
        self.fetch.sizes.update({allowed.source: 30 * MIB, oversized.source: 30 * MIB + 1})
        cached = await cache.ensure("chat", allowed)
        self.assertIsNotNone(cached)
        self.assertEqual(cached.path.stat().st_size, 30 * MIB)
        self.assertIsNone(await cache.ensure("chat", oversized))
        self.assertEqual(oversized.unavailable_reason, "超过插件大小限制")
        self.assertTrue(cached.path.exists())
        self.assertEqual(list(self.root.glob("*.part")), [])
        self.assertEqual(sum(path.stat().st_size for path in self.cached_files()), 30 * MIB)

    async def test_oversize_partial_file_is_removed_and_does_not_consume_capacity(self):
        cache = self.make_cache(limits=CacheLimits(max_image_bytes=8, max_count=1, max_total_bytes=8))
        oversized = self.reference("over-limit")
        self.fetch.sizes[oversized.source] = 9
        self.assertIsNone(await cache.ensure("chat", oversized))
        self.assertEqual(oversized.unavailable_reason, "超过插件大小限制")
        self.assertEqual(self.owned_files(), [])
        allowed = self.reference("allowed")
        self.fetch.sizes[allowed.source] = 8
        cached = await cache.ensure("chat", allowed)
        self.assertIsNotNone(cached)
        self.assertEqual(cached.path.stat().st_size, 8)
        self.assertEqual(len(self.cached_files()), 1)

    async def test_default_twenty_file_limit_is_independent_per_session(self):
        cache = self.make_cache(limits=CacheLimits())
        first = []
        for chat_id in ("chat-one", "chat-two"):
            for index in range(21):
                ref = self.reference(f"{chat_id}-{index}")
                self.fetch.sizes[ref.source] = 1
                cached = await cache.ensure(chat_id, ref)
                self.assertIsNotNone(cached)
                if index == 0:
                    first.append((ref, cached.path))
        for ref, path in first:
            self.assertFalse(path.exists())
            self.assertEqual(ref.unavailable_reason, "本地缓存已被清理")
        self.assertEqual(len(self.cached_files()), 40)

    async def test_total_bytes_are_enforced_separately_for_each_session(self):
        cache = self.make_cache(limits=CacheLimits(max_image_bytes=7, max_count=5, max_total_bytes=7))
        first = []
        for chat_id in ("chat-one", "chat-two"):
            for index, size in enumerate((4, 3)):
                ref = self.reference(f"{chat_id}-{index}")
                self.fetch.sizes[ref.source] = size
                cached = await cache.ensure(chat_id, ref)
                if index == 0:
                    first.append(cached.path)
        self.assertEqual(sum(path.stat().st_size for path in self.cached_files()), 14)
        newer = self.reference("newer")
        self.fetch.sizes[newer.source] = 1
        self.assertIsNotNone(await cache.ensure("chat-one", newer))
        self.assertFalse(first[0].exists())
        self.assertTrue(first[1].exists())
        self.assertEqual(sum(path.stat().st_size for path in self.cached_files()), 11)

    async def test_reading_does_not_refresh_fifo_order(self):
        cache = self.make_cache(limits=CacheLimits(max_image_bytes=8, max_count=2, max_total_bytes=16))
        first, second, third = (self.reference(name) for name in ("first", "second", "third"))
        first_cached = await cache.ensure("chat", first)
        self.now += 1
        second_cached = await cache.ensure("chat", second)
        self.now += 1
        self.assertEqual((await cache.ensure("chat", first)).path, first_cached.path)
        self.now += 1
        third_cached = await cache.ensure("chat", third)
        self.assertFalse(first_cached.path.exists())
        self.assertTrue(second_cached.path.exists())
        self.assertTrue(third_cached.path.exists())
        self.assertEqual(self.fetch.calls.count(first.source), 1)

    async def test_same_image_download_is_shared_by_concurrent_consumers(self):
        first, second = self.reference("same"), self.reference("same")
        entered, release = self.fetch.pause(first.source)
        one = self.start(self.cache.ensure("chat", first))
        await self.wait(entered)
        two = self.start(self.cache.ensure("chat", second))
        await asyncio.sleep(0)
        self.assertEqual(self.fetch.calls, [first.source])
        release.set()
        cached_one, cached_two = await asyncio.gather(one, two)
        self.assertEqual(cached_one.path, cached_two.path)
        self.assertEqual(cached_one.generation, cached_two.generation)
        self.assertEqual(len(self.cached_files()), 1)

    async def test_one_slow_session_does_not_block_other_session_fetches(self):
        first, second = self.reference("first"), self.reference("second")
        entered_one, release_one = self.fetch.pause(first.source)
        entered_two, release_two = self.fetch.pause(second.source)
        one = self.start(self.cache.ensure("chat-one", first))
        await self.wait(entered_one)
        two = self.start(self.cache.ensure("chat-two", second))
        await self.wait(entered_two)
        self.assertFalse(one.done())
        release_two.set()
        self.assertIsNotNone(await two)
        self.assertFalse(one.done())
        release_one.set()
        self.assertIsNotNone(await one)

    async def test_cancelling_one_consumer_keeps_shared_download_for_the_other(self):
        ref = self.reference()
        entered, release = self.fetch.pause(ref.source)
        one = self.start(self.cache.ensure("chat", ref))
        await self.wait(entered)
        two = self.start(self.cache.ensure("chat", ref))
        await asyncio.sleep(0)
        one.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await one
        self.assertNotIn(ref.source, self.fetch.cancelled)
        release.set()
        cached = await two
        self.assertTrue(cached.path.exists())
        self.assertEqual(self.fetch.calls, [ref.source])

    async def test_cancelling_last_consumer_cancels_fetch_and_removes_partial_file(self):
        ref = self.reference()
        self.fetch.prefix_bytes[ref.source] = 3
        entered, _ = self.fetch.pause(ref.source)
        task = self.start(self.cache.ensure("chat", ref))
        await self.wait(entered)
        self.assertEqual(len(list(self.root.glob("*.part"))), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        await self.wait(self.fetch.finished[ref.source])
        self.assertIn(ref.source, self.fetch.cancelled)
        self.assertEqual(self.owned_files(), [])

    async def test_two_hour_expiry_is_exact_and_reads_do_not_extend_it(self):
        ref = self.reference()
        cached = await self.cache.ensure("chat", ref)
        self.now = 8199.999
        self.assertEqual((await self.cache.ensure("chat", ref)).path, cached.path)
        self.cache.cleanup()
        self.assertTrue(cached.path.exists())
        self.now = 8200
        self.cache.cleanup()
        self.assertFalse(cached.path.exists())
        self.assertEqual(ref.unavailable_reason, "本地缓存已被清理")

    async def test_new_description_keeps_only_thirty_percent_of_remaining_ttl_once(self):
        ref = self.reference()
        cached = await self.cache.ensure("chat", ref)
        self.now = 2800
        self.assertTrue(self.cache.description_completed("chat", ref, cached.generation, "a cat"))
        self.assertEqual(ref.description, "a cat")
        self.now = 2900
        self.cache.description_completed("chat", ref, cached.generation, "a cat again")
        self.now = 4419.999
        self.cache.cleanup()
        self.assertTrue(cached.path.exists())
        self.now = 4420
        self.cache.cleanup()
        self.assertFalse(cached.path.exists())

    async def test_redownload_gets_new_generation_full_ttl_and_retains_old_description(self):
        ref = self.reference()
        original = await self.cache.ensure("chat", ref)
        self.assertTrue(self.cache.description_completed("chat", ref, original.generation, "old description"))
        self.now = 10000
        self.cache.cleanup()
        self.assertFalse(original.path.exists())
        downloaded = await self.cache.ensure("chat", ref)
        self.assertGreater(downloaded.generation, original.generation)
        self.assertNotEqual(downloaded.path, original.path)
        self.assertEqual(ref.description, "old description")
        self.assertEqual(ref.unavailable_reason, "")
        self.now = 17199.999
        self.cache.cleanup()
        self.assertTrue(downloaded.path.exists())
        self.now = 17200
        self.cache.cleanup()
        self.assertFalse(downloaded.path.exists())

    async def test_stale_description_callback_cannot_mutate_new_generation(self):
        ref = self.reference()
        original = await self.cache.ensure("chat", ref)
        self.now = 10000
        self.cache.cleanup()
        downloaded = await self.cache.ensure("chat", ref)
        self.assertFalse(self.cache.description_completed("chat", ref, original.generation, "stale result"))
        self.assertEqual(ref.description, "")
        self.assertEqual(ref.generation, downloaded.generation)
        self.now = 17199
        self.cache.cleanup()
        self.assertTrue(downloaded.path.exists())

    async def test_description_completed_at_expiry_cannot_revive_the_cache(self):
        ref = self.reference()
        cached = await self.cache.ensure("chat", ref)
        self.now = 8200
        self.cache.description_completed("chat", ref, cached.generation, "description completed after expiry")
        self.cache.cleanup()
        self.assertFalse(cached.path.exists())
        self.assertEqual(ref.generation, cached.generation)
        self.assertEqual(ref.unavailable_reason, "本地缓存已被清理")

    async def test_expired_lease_protects_file_but_rejects_new_reads_and_still_counts_bytes(self):
        cache = self.make_cache(limits=CacheLimits(max_image_bytes=8, max_count=1, max_total_bytes=8))
        ref = self.reference()
        self.fetch.sizes[ref.source] = 8
        async with cache.acquire("chat", ref) as leased:
            self.assertIsNotNone(leased)
            self.now = 8200
            cache.cleanup()
            self.assertTrue(leased.path.exists())
            self.assertEqual(ref.unavailable_reason, "本地缓存已被清理")
            async with cache.acquire("chat", ref) as fresh:
                self.assertIsNone(fresh)
            self.assertEqual(sum(path.stat().st_size for path in self.owned_files()), 8)
            old_path = leased.path
        self.assertFalse(old_path.exists())
        downloaded = await cache.ensure("chat", ref)
        self.assertIsNotNone(downloaded)
        self.assertGreater(downloaded.generation, leased.generation)

    async def test_overlapping_requests_cannot_exceed_session_total_or_leak_part_files(self):
        cache = self.make_cache(limits=CacheLimits(max_image_bytes=8, max_count=3, max_total_bytes=10))
        first, second = self.reference("first"), self.reference("second")
        self.fetch.sizes.update({first.source: 6, second.source: 6})
        entered_one, release_one = self.fetch.pause(first.source)
        entered_two, release_two = self.fetch.pause(second.source)
        one = self.start(cache.ensure("chat", first))
        two = self.start(cache.ensure("chat", second))
        await self.wait(entered_one)
        release_one.set()
        await self.wait(entered_two)
        release_two.set()
        await asyncio.gather(one, two)
        self.assertEqual(list(self.root.glob("*.part")), [])
        self.assertEqual(len(self.cached_files()), 1)
        self.assertLessEqual(sum(path.stat().st_size for path in self.owned_files()), 10)

    async def test_eviction_and_close_only_delete_owned_copies(self):
        source_path = self.temp / "platform-owned.png"
        source_path.write_bytes(b"data")
        self.root.mkdir(parents=True, exist_ok=True)
        foreign = self.root / "other-plugin.txt"
        foreign.write_text("keep", encoding="utf-8")

        async def copy_source(source, write, max_bytes):
            data = Path(source).read_bytes()
            write(data)
            return len(data)

        cache = self.make_cache(
            limits=CacheLimits(max_image_bytes=8, max_count=1, max_total_bytes=8), fetch=copy_source,
        )
        first = self.reference("first", source=str(source_path))
        second = self.reference("second", source=str(source_path))
        original = await cache.ensure("chat", first)
        self.assertNotEqual(original.path, source_path)
        self.assertIsNotNone(await cache.ensure("chat", second))
        self.assertFalse(original.path.exists())
        await cache.close()
        self.assertEqual(source_path.read_bytes(), b"data")
        self.assertEqual(foreign.read_text(encoding="utf-8"), "keep")
        self.assertEqual(self.owned_files(), [])

    async def test_close_cancels_inflight_fetch_and_removes_owned_files(self):
        existing = await self.cache.ensure("chat", self.reference("existing"))
        ref = self.reference("pending")
        self.fetch.prefix_bytes[ref.source] = 3
        entered, _ = self.fetch.pause(ref.source)
        task = self.start(self.cache.ensure("chat", ref))
        await self.wait(entered)
        self.assertEqual(len(list(self.root.glob("*.part"))), 1)
        await self.cache.close()
        await asyncio.gather(task, return_exceptions=True)
        self.assertIn(ref.source, self.fetch.cancelled)
        self.assertFalse(existing.path.exists())
        self.assertEqual(self.owned_files(), [])

    async def test_generic_fetch_failure_does_not_guess_cloud_cleanup(self):
        ref = self.reference()
        self.fetch.errors[ref.source] = ImageFetchError("获取失败")
        self.assertIsNone(await self.cache.ensure("chat", ref))
        self.assertEqual(ref.unavailable_reason, "获取失败")
        self.assertNotIn("云端", ref.render())
        self.assertEqual(self.owned_files(), [])

    async def test_explicit_cloud_cleanup_reason_is_preserved(self):
        ref = self.reference()
        self.fetch.errors[ref.source] = ImageFetchError("cloudclean")
        self.assertIsNone(await self.cache.ensure("chat", ref))
        self.assertEqual(ref.unavailable_reason, "云端缓存已被清理")
        self.assertEqual(self.owned_files(), [])

    async def test_old_description_survives_failed_redownload(self):
        ref = self.reference()
        cached = await self.cache.ensure("chat", ref)
        self.assertTrue(self.cache.description_completed("chat", ref, cached.generation, "a previous description"))
        self.now = 10000
        self.cache.cleanup()
        self.fetch.errors[ref.source] = ImageFetchError("获取失败")
        self.assertIsNone(await self.cache.ensure("chat", ref))
        self.assertEqual(ref.description, "a previous description")
        self.assertEqual(ref.unavailable_reason, "本地缓存已被清理")

    async def test_dynamic_count_and_total_limits_evict_oldest_caches(self):
        current = [CacheLimits(max_image_bytes=8, max_count=3, max_total_bytes=20)]
        cache = self.make_cache(limits=lambda: current[0])
        downloaded = []
        for name in ("first", "second", "third"):
            ref = self.reference(name)
            self.fetch.sizes[ref.source] = 5
            downloaded.append(await cache.ensure("chat", ref))
        current[0] = CacheLimits(max_image_bytes=8, max_count=1, max_total_bytes=6)
        cache.cleanup()
        self.assertFalse(downloaded[0].path.exists())
        self.assertFalse(downloaded[1].path.exists())
        self.assertTrue(downloaded[2].path.exists())
        self.assertEqual(sum(path.stat().st_size for path in self.cached_files()), 5)

    async def test_lowering_single_image_limit_invalidates_existing_oversized_copy(self):
        current = [CacheLimits(max_image_bytes=8, max_count=3, max_total_bytes=20)]
        cache = self.make_cache(limits=lambda: current[0])
        ref = self.reference()
        self.fetch.sizes[ref.source] = 6
        cached = await cache.ensure("chat", ref)
        current[0] = CacheLimits(max_image_bytes=4, max_count=3, max_total_bytes=20)
        cache.cleanup()
        self.assertFalse(cached.path.exists())
        self.assertIsNone(await cache.ensure("chat", ref))
        self.assertEqual(ref.unavailable_reason, "超过插件大小限制")

    async def test_shrinking_limits_retires_oldest_leased_copy_without_renewing_its_lease(self):
        for limits in (
            CacheLimits(max_image_bytes=8, max_count=1, max_total_bytes=8),
            CacheLimits(max_image_bytes=8, max_count=2, max_total_bytes=4),
        ):
            with self.subTest(max_count=limits.max_count, max_total_bytes=limits.max_total_bytes):
                current = [CacheLimits(max_image_bytes=8, max_count=2, max_total_bytes=8)]
                cache = self.make_cache(limits=lambda: current[0])
                first, second = self.reference("oldest"), self.reference("newest")
                async with cache.acquire("chat", first) as old_lease:
                    async with cache.acquire("chat", second) as new_lease:
                        current[0] = limits
                        cache.cleanup()
                        self.assertTrue(old_lease.path.exists())
                        self.assertTrue(new_lease.path.exists())
                        self.assertEqual(first.unavailable_reason, "本地缓存已被清理")
                        self.assertEqual(second.unavailable_reason, "")
                        async with cache.acquire("chat", first) as renewed:
                            self.assertIsNone(renewed)
                        async with cache.acquire("chat", second) as still_valid:
                            self.assertEqual(still_valid.path, new_lease.path)
                        self.assertEqual(sum(path.stat().st_size for path in self.owned_files()), 8)
                    self.assertTrue(old_lease.path.exists())
                self.assertFalse(old_lease.path.exists())
                self.assertTrue(new_lease.path.exists())
                await cache.close()

    async def test_shared_reference_object_is_isolated_between_sessions_and_caption_callbacks(self):
        cache = self.make_cache(limits=CacheLimits(max_image_bytes=8, max_count=1, max_total_bytes=8))
        shared = self.reference("same-id")
        one = cache.register("chat-one", shared)
        first = await cache.ensure("chat-one", one)
        two = cache.register("chat-two", shared)
        self.assertIsNot(one, two)
        self.assertEqual(two.generation, 0)
        second = await cache.ensure("chat-two", two)
        self.assertNotEqual(first.path, second.path)
        self.assertFalse(cache.description_completed("chat-two", one, first.generation, "wrong session"))
        self.assertFalse(cache.description_completed("chat-one", two, second.generation, "wrong session"))
        self.assertEqual(one.description, "")
        self.assertEqual(two.description, "")
        await cache.ensure("chat-one", self.reference("replacement"))
        self.assertFalse(first.path.exists())
        self.assertTrue(second.path.exists())
        self.assertEqual(one.unavailable_reason, "本地缓存已被清理")
        self.assertEqual(two.unavailable_reason, "")
        newer = await cache.ensure("chat-one", one)
        self.assertGreater(newer.generation, first.generation)
        self.assertEqual(two.generation, second.generation)
        self.assertEqual((await cache.ensure("chat-two", two)).path, second.path)

    async def test_cleanup_releases_session_lock_after_all_refs_jobs_and_files_are_gone(self):
        ref = self.reference()
        await self.cache.ensure("short-lived-chat", ref)
        self.assertIn("short-lived-chat", self.cache._locks)
        lock = weakref.ref(self.cache._locks["short-lived-chat"])
        del ref
        self.now = 8200
        self.cache.cleanup()
        gc.collect()
        self.cache.cleanup()
        self.assertNotIn("short-lived-chat", self.cache._locks)
        self.assertIsNone(lock())
        self.assertEqual(self.owned_files(), [])

    async def test_failed_partial_unlink_is_retried_and_retained_bytes_block_new_downloads(self):
        cache = self.make_cache(limits=CacheLimits(max_image_bytes=4, max_count=1, max_total_bytes=4))
        broken = self.reference("broken")
        self.fetch.prefix_bytes[broken.source] = 3
        self.fetch.errors[broken.source] = ImageFetchError("获取失败")
        original_unlink = Path.unlink
        blocked = [True]
        unlink_attempts = []

        def unlink(path, *args, **kwargs):
            if path.suffix == ".part" and blocked[0]:
                unlink_attempts.append(path)
                raise PermissionError("file temporarily in use")
            return original_unlink(path, *args, **kwargs)

        with patch.object(Path, "unlink", new=unlink):
            self.assertIsNone(await cache.ensure("chat", broken))
            retained = list(self.root.glob("*.part"))
            self.assertEqual(len(retained), 1)
            self.assertEqual(retained[0].stat().st_size, 3)
            replacement = self.reference("replacement")
            self.assertIsNone(await cache.ensure("chat", replacement))
            self.assertEqual(sum(path.stat().st_size for path in self.owned_files()), 3)
            self.assertEqual(len(self.owned_files()), 1)
            self.assertGreaterEqual(len(unlink_attempts), 2)
            blocked[0] = False
            cache.cleanup()
            self.assertEqual(self.owned_files(), [])
            accepted = await cache.ensure("chat", replacement)
            self.assertIsNotNone(accepted)
            self.assertEqual(accepted.path.stat().st_size, 4)

    async def test_registration_does_not_keep_uncached_message_metadata_alive_forever(self):
        references = []
        for index in range(50):
            ref = self.reference(str(index))
            canonical = self.cache.register("chat", ref)
            references.append(weakref.ref(canonical))
        del ref, canonical
        gc.collect()
        self.cache.cleanup()
        self.assertTrue(all(reference() is None for reference in references))
        self.assertEqual(self.owned_files(), [])

    async def test_expired_cache_does_not_keep_discarded_message_metadata_alive_forever(self):
        ref = self.reference()
        weak = weakref.ref(self.cache.register("chat", ref))
        await self.cache.ensure("chat", ref)
        del ref
        self.now = 8200
        self.cache.cleanup()
        gc.collect()
        self.assertIsNone(weak())
        self.assertEqual(self.owned_files(), [])


if __name__ == "__main__":
    unittest.main()
