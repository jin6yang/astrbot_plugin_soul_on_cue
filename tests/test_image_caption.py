"""逐图转述的共享额度、有限并发和需求生命周期；不调用真实模型。"""

import asyncio
import gc
import importlib
import sys
import tempfile
import unittest
import weakref
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, patch

from test_image_cache import CacheLimits, ImageReference, SessionImageCache, cache_module, resources_module
from test_timing import plugin_module


message_module = ModuleType(plugin_module.__name__ + ".message_context")
message_module.__dict__.update(plugin_module.render_message.__globals__)
with patch.dict(sys.modules, {
    plugin_module.__name__: plugin_module,
    message_module.__name__: message_module,
    plugin_module.__name__ + ".image_cache": cache_module,
    plugin_module.__name__ + ".image_resources": resources_module,
}):
    caption_module = importlib.import_module(plugin_module.__name__ + ".image_caption")
CaptionRound = caption_module.CaptionRound
ImageCaptionService = caption_module.ImageCaptionService


class ImageCaptionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="oncue-caption-tests-")
        self.addCleanup(temporary.cleanup)
        self.now = 1000.0
        self.fetches = []

        async def fetch(source, write, max_bytes):
            self.fetches.append(source)
            write(b"image")
            return 5

        self.cache = SessionImageCache(
            Path(temporary.name), limits=CacheLimits(max_image_bytes=100, max_total_bytes=100),
            fetch=fetch, clock=lambda: self.now,
        )
        self.generate = AsyncMock(return_value="visible cat and text")
        self.service = ImageCaptionService(self.cache, generate=self.generate, concurrency=2)
        self.rounds = []

    async def asyncTearDown(self):
        for round in self.rounds:
            round.close()
        await self.service.close()
        await self.cache.close()

    def round(self, **kwargs):
        round = CaptionRound(**kwargs)
        self.rounds.append(round)
        return round

    def ref(self, name="one", description=""):
        return self.cache.register("chat", ImageReference(name, source=f"fake:{name}", description=description))

    async def wait(self, event):
        await asyncio.wait_for(event.wait(), 1)

    def block(self):
        entered, release = {}, {}

        async def generate(chat, ref, cached, max_chars):
            entered.setdefault(ref.image_id, asyncio.Event()).set()
            await release.setdefault(ref.image_id, asyncio.Event()).wait()
            return "description of " + ref.image_id

        self.generate.side_effect = generate
        return entered, release

    async def entered(self, events, name):
        for _ in range(20):
            if name in events:
                await self.wait(events[name])
                return
            await asyncio.sleep(0)
        self.fail(f"caption did not start: {name}")

    async def test_one_round_shares_three_new_requests_across_processing_stages(self):
        round = self.round()
        refs = [self.ref(str(index)) for index in range(4)]
        first = self.service.request("chat", refs[0], round, 100)
        self.assertIs(self.service.request("chat", refs[0], round, 100), first)
        second = self.service.request("chat", refs[1], round, 100)
        third = self.service.request("chat", refs[2], round, 100)
        self.assertIsNone(self.service.request("chat", refs[3], round, 100))
        self.assertEqual(round.used, 3)
        self.assertEqual(round.remaining, 0)
        await asyncio.gather(first, second, third)
        self.assertEqual(self.generate.await_count, 3)

    async def test_existing_description_needs_no_request_or_budget(self):
        round = self.round(limit=0)
        ref = self.ref(description="already known")
        result = self.service.request("chat", ref, round, 100)
        self.assertEqual(await result, "already known")
        self.assertEqual(round.used, 0)
        self.assertEqual(self.fetches, [])
        self.generate.assert_not_awaited()

    async def test_two_rounds_share_job_but_have_separate_cancellable_futures(self):
        entered, release = self.block()
        ref = self.ref()
        first_round, second_round = self.round(), self.round()
        first = self.service.request("chat", ref, first_round, 100)
        await self.entered(entered, "one")
        second = self.service.request("chat", ref, second_round, 100)
        self.assertIsNot(first, second)
        self.assertEqual(first_round.used, 1)
        self.assertEqual(second_round.used, 0)
        first.cancel()
        await asyncio.sleep(0)
        release["one"].set()
        self.assertEqual(await second, "description of one")
        self.generate.assert_awaited_once()
        self.assertEqual(self.fetches, [ref.source])

    async def test_concurrency_limit_blocks_third_job_until_slot_is_free(self):
        entered, release = self.block()
        round = self.round()
        futures = [self.service.request("chat", self.ref(name), round, 100) for name in ("one", "two", "three")]
        await self.entered(entered, "one")
        await self.entered(entered, "two")
        self.assertNotIn("three", entered)
        release["one"].set()
        await self.entered(entered, "three")
        release["two"].set()
        release["three"].set()
        await asyncio.gather(*futures)

    async def test_round_close_cancels_queued_last_demand_without_fetch(self):
        await self.service.close()
        self.service = ImageCaptionService(self.cache, generate=self.generate, concurrency=1)
        entered, release = self.block()
        first_round, second_round = self.round(), self.round()
        first = self.service.request("chat", self.ref("one"), first_round, 100)
        await self.entered(entered, "one")
        second = self.service.request("chat", self.ref("two"), second_round, 100)
        second_round.close()
        self.service.refresh()
        release["one"].set()
        await first
        self.assertTrue(second.cancelled())
        self.assertNotIn("fake:two", self.fetches)
        self.assertEqual(self.generate.await_count, 1)

    async def test_inactive_queued_round_is_discarded_without_model_call(self):
        await self.service.close()
        self.service = ImageCaptionService(self.cache, generate=self.generate, concurrency=1)
        entered, release = self.block()
        first = self.service.request("chat", self.ref("one"), self.round(), 100)
        await self.entered(entered, "one")
        active = [True]
        second = self.service.request("chat", self.ref("two"), self.round(is_active=lambda: active[0]), 100)
        active[0] = False
        self.service.refresh()
        release["one"].set()
        await first
        self.assertTrue(second.cancelled())
        self.assertNotIn("fake:two", self.fetches)

    async def test_unreferenced_queued_round_is_not_kept_alive_by_job(self):
        await self.service.close()
        self.service = ImageCaptionService(self.cache, generate=self.generate, concurrency=1)
        entered, release = self.block()
        first = self.service.request("chat", self.ref("one"), self.round(), 100)
        await self.entered(entered, "one")
        round = CaptionRound()
        weak = weakref.ref(round)
        queued = self.service.request("chat", self.ref("two"), round, 100)
        del round
        gc.collect()
        self.service.refresh()
        self.assertIsNone(weak())
        release["one"].set()
        await first
        self.assertTrue(queued.cancelled())
        self.assertNotIn("fake:two", self.fetches)

    async def test_failed_job_is_not_retried_in_same_round_but_new_round_can_try(self):
        ref, round = self.ref(), self.round()
        self.generate.side_effect = [RuntimeError("model unavailable"), "later description"]
        first = self.service.request("chat", ref, round, 90)
        self.assertEqual(await first, "")
        self.assertIs(self.service.request("chat", ref, round, 90), first)
        self.assertEqual(round.used, 1)
        self.generate.assert_awaited_once()
        retried = self.service.request("chat", ref, self.round(), 90)
        self.assertEqual(await retried, "later description")
        self.assertEqual(self.generate.await_count, 2)

    async def test_dynamic_concurrency_refresh_starts_waiting_job(self):
        await self.service.close()
        concurrency = [1]
        self.service = ImageCaptionService(self.cache, generate=self.generate, concurrency=lambda: concurrency[0])
        entered, release = self.block()
        round = self.round()
        first = self.service.request("chat", self.ref("one"), round, 100)
        await self.entered(entered, "one")
        second = self.service.request("chat", self.ref("two"), round, 100)
        self.assertNotIn("two", entered)
        concurrency[0] = 2
        self.service.refresh()
        await self.entered(entered, "two")
        release["one"].set()
        release["two"].set()
        await asyncio.gather(first, second)

    async def test_running_caption_can_finish_for_cache_after_its_round_closes(self):
        entered, release = self.block()
        ref, round = self.ref(), self.round()
        future = self.service.request("chat", ref, round, 90)
        await self.entered(entered, "one")
        round.close()
        self.assertTrue(future.cancelled())
        release["one"].set()
        for _ in range(20):
            if ref.description:
                break
            await asyncio.sleep(0)
        self.assertEqual(ref.description, "description of one")

    async def test_old_generation_caption_result_does_not_overwrite_redownloaded_image(self):
        entered, release = self.block()
        ref = self.ref()
        future = self.service.request("chat", ref, self.round(), 100)
        await self.entered(entered, "one")
        old_generation = ref.generation
        self.now = 10000
        self.cache.cleanup()
        downloaded = await self.cache.ensure("chat", ref)
        self.assertGreater(downloaded.generation, old_generation)
        release["one"].set()
        await future
        self.assertEqual(ref.description, "")
        self.assertEqual(ref.generation, downloaded.generation)

    async def test_late_caption_does_not_restore_expired_resource(self):
        entered, release = self.block()
        ref = self.ref()
        future = self.service.request("chat", ref, self.round(), 100)
        await self.entered(entered, "one")
        self.now = 8200
        self.cache.cleanup()
        release["one"].set()
        await future
        self.assertEqual(ref.unavailable_reason, "本地缓存已被清理")
        self.assertEqual(list(self.cache.root.glob("*.cache")), [])

    async def test_service_close_cancels_running_and_queued_jobs(self):
        await self.service.close()
        self.service = ImageCaptionService(self.cache, generate=self.generate, concurrency=1)
        entered, _ = self.block()
        round = self.round()
        first = self.service.request("chat", self.ref("one"), round, 100)
        await self.entered(entered, "one")
        second = self.service.request("chat", self.ref("two"), round, 100)
        await self.service.close()
        self.assertTrue(first.cancelled())
        self.assertTrue(second.cancelled())
        self.assertNotIn("fake:two", self.fetches)
        self.assertIsNone(self.service.request("chat", self.ref("three"), round, 100))

    async def test_generator_swallowing_cancel_cannot_commit_after_service_close(self):
        entered = asyncio.Event()

        async def generate(*args):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return "late result after shutdown"

        self.generate.side_effect = generate
        ref = self.ref()
        future = self.service.request("chat", ref, self.round(), 100)
        await self.wait(entered)
        await self.service.close()
        self.assertTrue(future.cancelled())
        self.assertEqual(ref.description, "")


if __name__ == "__main__":
    unittest.main()
