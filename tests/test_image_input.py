"""决策和正文共用的图片输入模式、图文关联和单消息限长。"""

import asyncio
import importlib
import re
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

from test_image_caption import (
    CaptionRound, ImageCaptionService, caption_module, message_module,
)
from test_image_cache import CacheLimits, ImageReference, SessionImageCache, cache_module, resources_module
from test_timing import plugin_module


with patch.dict(sys.modules, {
    plugin_module.__name__: plugin_module,
    message_module.__name__: message_module,
    plugin_module.__name__ + ".image_cache": cache_module,
    plugin_module.__name__ + ".image_resources": resources_module,
    plugin_module.__name__ + ".image_caption": caption_module,
}):
    input_module = importlib.import_module(plugin_module.__name__ + ".image_input")
ImageInputPreparer = input_module.ImageInputPreparer
EncodedImage = input_module.EncodedImage
ContextRow = plugin_module.ContextRow
ContextSnapshot = plugin_module.ContextSnapshot
MessagePart = plugin_module.MessagePart


class ImageInputTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="oncue-image-input-tests-")
        self.addCleanup(temporary.cleanup)
        self.fetches = []

        async def fetch(source, write, max_bytes):
            self.fetches.append(source)
            write(b"image")
            return 5

        self.cache = SessionImageCache(
            Path(temporary.name), limits=CacheLimits(max_image_bytes=100, max_total_bytes=100), fetch=fetch,
        )
        self.generate = AsyncMock(return_value="visible cat and words")
        self.captions = ImageCaptionService(self.cache, generate=self.generate)
        self.encode = AsyncMock(return_value=EncodedImage("data:image/png;base64,eA==", False))
        self.logger = MagicMock()
        self.preparer = ImageInputPreparer(self.cache, self.captions, encode=self.encode, logger=self.logger)
        self.rounds = []

    async def asyncTearDown(self):
        for round in self.rounds:
            round.close()
        await self.captions.close()
        await self.cache.close()

    def round(self, **kwargs):
        round = CaptionRound(**kwargs)
        self.rounds.append(round)
        return round

    def image(self, name="one", **kwargs):
        return self.cache.register("chat", ImageReference(name, source=f"fake:{name}", **kwargs))

    @staticmethod
    def row(name, *parts):
        return ContextRow(
            record_id="platform:" + name, message_id=name, role="user", header="[member / " + name + "]",
            body=plugin_module.render_message(tuple(parts)), parts=tuple(parts),
        )

    async def prepare(self, rows, *, mode="mixed", vision=True, round=None, **kwargs):
        snapshot = rows if isinstance(rows, ContextSnapshot) else ContextSnapshot(tuple(rows), progress=9)
        return await self.preparer.prepare(
            "chat", snapshot, mode=mode, vision=vision, round=round or self.round(), **kwargs,
        )

    async def test_placeholder_uses_true_state_but_hides_old_description_without_io(self):
        ref = self.image(description="private old description", size_bytes=2 * 1024**2)
        ref.unavailable_reason = "本地缓存已被清理"
        row = self.row("current", MessagePart("text", "look"), MessagePart("image", image=ref))
        prepared = await self.prepare([row], mode="placeholder", focus_records=(row.record_id,))
        text = prepared.snapshot.render()
        self.assertIn("[图片，2MiB，当前不可用（本地缓存已被清理）]", text)
        self.assertNotIn("private old description", text)
        self.assertEqual(prepared.image_urls, [])
        self.assertEqual(prepared.image_ids, ())
        self.assertEqual(self.fetches, [])
        self.generate.assert_not_awaited()
        self.encode.assert_not_awaited()
        self.assertEqual(ref.description, "private old description")

    async def test_describe_mode_waits_for_description_and_never_encodes_original(self):
        ref = self.image()
        row = self.row("current", MessagePart("image", image=ref))
        round = self.round()
        prepared = await self.prepare([row], mode="describe", round=round, focus_records=(row.record_id,))
        self.assertIn("，已转述，visible cat and words]", prepared.snapshot.render())
        self.assertEqual(prepared.image_urls, [])
        self.assertEqual(prepared.image_ids, ())
        self.assertEqual(round.used, 1)
        self.generate.assert_awaited_once()
        self.encode.assert_not_awaited()

    async def test_mixed_text_and_unknown_vision_use_descriptions(self):
        for index, vision in enumerate((False, None)):
            with self.subTest(vision=vision):
                ref = self.image(str(index))
                row = self.row(str(index), MessagePart("image", image=ref))
                prepared = await self.prepare([row], vision=vision, focus_records=(row.record_id,))
                self.assertIn("visible cat and words", prepared.snapshot.render())
                self.assertEqual(prepared.image_urls, [])
        self.assertEqual(self.generate.await_count, 2)
        self.encode.assert_not_awaited()

    async def test_visual_mixed_returns_before_background_caption_finishes(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def generate(*args):
            entered.set()
            await release.wait()
            return "late description"

        self.generate.side_effect = generate
        ref = self.image()
        row = self.row("current", MessagePart("image", image=ref))
        prepared = await asyncio.wait_for(self.prepare([row], focus_records=(row.record_id,)), 1)
        await asyncio.wait_for(entered.wait(), 1)
        self.assertEqual(prepared.image_ids, (ref.image_id,))
        self.assertEqual(len(prepared.image_urls), 1)
        self.assertNotIn("late description", prepared.snapshot.render())
        self.assertEqual(ref.description, "")
        release.set()

    async def test_visual_mixed_does_not_attach_unfocused_history_originals(self):
        ref = self.image()
        row = self.row("old", MessagePart("image", image=ref))
        prepared = await self.prepare([row], focus_records=())
        self.assertEqual(prepared.image_urls, [])
        self.assertEqual(prepared.image_ids, ())
        self.encode.assert_not_awaited()

    async def test_focus_and_quoted_images_are_deduplicated_under_direct_limit(self):
        current, quoted, extra = (self.image(name) for name in ("current", "quoted", "extra"))
        old = self.row("old", MessagePart("image", image=quoted))
        focus = self.row(
            "current",
            MessagePart("image", image=current),
            MessagePart("quote", "[quoted]", reference_id="old", children=(MessagePart("image", image=quoted),)),
            MessagePart("image", image=current), MessagePart("image", image=extra),
        )
        prepared = await self.prepare([old, focus], focus_records=(focus.record_id,), max_images=2)
        self.assertEqual(set(prepared.image_ids), {current.image_id, quoted.image_id})
        self.assertEqual(len(prepared.image_ids), 2)
        self.assertEqual(self.encode.await_count, 2)

    async def test_id_only_quote_finds_image_in_referenced_snapshot_row(self):
        ref = self.image()
        old = self.row("old", MessagePart("image", image=ref))
        focus = self.row("current", MessagePart("quote", "[quoted]", reference_id="old"))
        prepared = await self.prepare([old, focus], focus_records=(focus.record_id,))
        self.assertEqual(prepared.image_ids, (ref.image_id,))
        self.encode.assert_awaited_once()

    async def test_native_image_ids_are_skipped_and_new_images_use_remaining_limit_and_offset(self):
        first, second, third = self.image("first"), self.image("second"), self.image("third")
        focus = self.row(
            "current", MessagePart("image", image=first), MessagePart("image", image=second),
            MessagePart("image", image=third),
        )
        prepared = await self.prepare(
            [focus], focus_records=(focus.record_id,), max_images=1, image_offset=1,
            skip_direct_ids={first.image_id},
        )
        self.assertEqual(prepared.image_ids, (second.image_id,))
        self.assertEqual(len(prepared.image_urls), 1)
        self.assertIn("图片2", prepared.snapshot.image_manifest)
        self.assertNotIn("图片3", prepared.snapshot.image_manifest)

    async def test_one_round_budget_is_shared_by_initial_extra_and_body_preparation(self):
        round = self.round(limit=3)
        refs = [self.image(str(index)) for index in range(4)]
        for index in range(4):
            row = self.row(str(index), MessagePart("image", image=refs[index]))
            prepared = await self.prepare([row], mode="describe", round=round)
            if index < 3:
                self.assertIn("visible cat and words", prepared.snapshot.render())
            else:
                self.assertIn("内容未读取", prepared.snapshot.render())
        self.assertEqual(round.used, 3)
        self.assertEqual(self.generate.await_count, 3)
        self.assertEqual(len(self.fetches), 3)

    async def test_existing_description_is_reused_without_new_model_request(self):
        ref = self.image(description="existing objective description")
        row = self.row("old", MessagePart("image", image=ref))
        round = self.round(limit=0)
        prepared = await self.prepare([row], mode="describe", round=round)
        self.assertIn("existing objective description", prepared.snapshot.render())
        self.assertEqual(round.used, 0)
        self.assertEqual(self.fetches, [])
        self.generate.assert_not_awaited()

    async def test_prepared_description_does_not_rewrite_original_snapshot_body(self):
        ref = self.image()
        snapshot = ContextSnapshot((self.row("one", MessagePart("image", image=ref)),), progress=17)
        before = snapshot.render()
        prepared = await self.prepare(snapshot, mode="describe")
        self.assertEqual(snapshot.render(), before)
        self.assertIn("内容未读取", snapshot.render())
        self.assertIn("visible cat and words", prepared.snapshot.render())
        self.assertEqual(prepared.snapshot.progress, 17)

    async def test_switching_from_attached_snapshot_clears_old_manifest_and_attached_state(self):
        ref = self.image()
        row = self.row("one", MessagePart("image", image=ref))
        attached = await self.prepare([row], focus_records=(row.record_id,))
        self.assertTrue(attached.snapshot.image_manifest)
        self.assertIn("已附图", attached.snapshot.render())
        for mode in ("describe", "placeholder"):
            with self.subTest(mode=mode):
                prepared = await self.prepare(attached.snapshot, mode=mode)
                self.assertEqual(prepared.snapshot.image_manifest, "")
                self.assertEqual(prepared.image_urls, [])
                self.assertNotIn("已附图", prepared.snapshot.render())

    async def test_caption_failure_stays_in_describe_mode_without_direct_image_fallback(self):
        self.generate.side_effect = RuntimeError("caption unavailable")
        ref = self.image()
        row = self.row("one", MessagePart("image", image=ref))
        prepared = await self.prepare([row], mode="describe", focus_records=(row.record_id,))
        self.assertIn("内容未读取", prepared.snapshot.render())
        self.assertEqual(prepared.image_urls, [])
        self.assertEqual(ref.unavailable_reason, "")
        self.encode.assert_not_awaited()

    async def test_message_budget_is_shared_by_text_and_all_image_descriptions(self):
        self.generate.return_value = "description " * 80
        first, second = self.image("first"), self.image("second")
        row = self.row(
            "one", MessagePart("text", "HEAD " + "x" * 10),
            MessagePart("image", image=first), MessagePart("image", image=second), MessagePart("text", " TAIL"),
        )
        prepared = await self.prepare([row], mode="describe", message_char_limit=80)
        body = prepared.snapshot.rows[0].body
        without_notice = re.sub(r"\[中间省略；原文 \d+ 字符(?:；含 \d+ 张图片)?\]", "", body)
        self.assertLessEqual(len(without_notice), 80)
        self.assertEqual(self.generate.await_count, 2)
        for call in self.generate.call_args_list:
            self.assertLessEqual(call.args[3], 40)
        self.assertNotIn("description " * 80, body)

    async def test_encoding_failure_logs_without_waiting_for_caption_or_faking_fetch_failure(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def generate(*args):
            entered.set()
            await release.wait()
            return "late caption"

        self.generate.side_effect = generate
        self.encode.side_effect = RuntimeError("encoding unavailable")
        ref = self.image()
        row = self.row("one", MessagePart("image", image=ref))
        prepared = await asyncio.wait_for(self.prepare([row], focus_records=(row.record_id,)), 1)
        self.assertEqual(prepared.image_urls, [])
        self.assertNotIn("late caption", prepared.snapshot.render())
        self.assertEqual(ref.unavailable_reason, "")
        self.logger.warning.assert_called()
        release.set()

    async def test_montage_is_tied_to_attached_image_and_marked_as_partial_frames(self):
        self.encode.return_value = EncodedImage("data:image/png;base64,eA==", True)
        ref = self.image()
        row = self.row("one", MessagePart("image", image=ref))
        prepared = await self.prepare([row], focus_records=(row.record_id,))
        self.assertEqual(prepared.image_ids, (ref.image_id,))
        text = prepared.snapshot.render()
        self.assertIn("动画", text)
        self.assertIn("部分帧", text)

    async def test_inactive_round_does_not_schedule_caption_but_can_attach_focus_image(self):
        round = self.round(is_active=lambda: False)
        ref = self.image()
        row = self.row("one", MessagePart("image", image=ref))
        prepared = await self.prepare([row], round=round, focus_records=(row.record_id,))
        self.assertEqual(prepared.image_ids, (ref.image_id,))
        self.assertEqual(round.used, 0)
        self.generate.assert_not_awaited()

    async def test_description_filling_allowance_keeps_the_image_marker_instead_of_omitting_it(self):
        self.generate.side_effect = lambda chat, ref, cached, max_chars: "x" * max_chars
        for name, size in (("unknown", None), ("known", 5)):
            with self.subTest(size=size):
                ref = self.image(name, size_bytes=size)
                row = self.row(name, MessagePart("image", image=ref))
                prepared = await self.prepare([row], mode="describe", message_char_limit=120)
                body = prepared.snapshot.rows[0].body
                self.assertIn("已转述，xxx", body)
                self.assertNotIn("中间省略", body)
                self.assertLessEqual(len(body), 120)

    async def test_long_existing_description_budget_includes_direct_image_number_label(self):
        ref = self.image(description="x" * 500, size_bytes=5)
        row = self.row("one", MessagePart("image", image=ref))
        prepared = await self.prepare([row], focus_records=(row.record_id,), message_char_limit=120)
        body = prepared.snapshot.rows[0].body
        self.assertEqual(prepared.image_ids, (ref.image_id,))
        self.assertIn("已附图（图片1），xxx", body)
        self.assertNotIn("中间省略", body)
        self.assertLessEqual(len(body), 120)

    async def test_default_encoder_uses_public_preprocessing_and_keeps_borrowed_source(self):
        media = ModuleType("astrbot.core.utils.media_utils")
        directories = []

        async def preview(source, *, max_size, output_dir):
            self.assertEqual(max_size, 333)
            directories.append(output_dir)
            return source, False, False, source

        media.prepare_model_image = AsyncMock(side_effect=preview)
        ref = self.image()
        with patch.dict(sys.modules, {media.__name__: media}):
            async with self.cache.acquire("chat", ref) as cached:
                source = cached.path
                encoded = await input_module.encode_cached_image(cached, max_size=333)
                self.assertIsNotNone(encoded)
                self.assertFalse(encoded.is_montage)
                self.assertEqual(source.read_bytes(), b"image")
                self.assertEqual(media.prepare_model_image.call_args.args, (str(source),))
        media.prepare_model_image.assert_awaited_once()
        self.assertTrue(source.exists())
        self.assertFalse(directories[0].exists())

    async def test_default_encoder_cleans_native_owned_preview_after_capturing_bytes(self):
        media = ModuleType("astrbot.core.utils.media_utils")
        previews = []

        async def preview(source, *, max_size, output_dir):
            output = output_dir / "native-preview.jpg"
            output.write_bytes(b"\xff\xd8preview")
            previews.append(output)
            return str(output), True, True, source

        media.prepare_model_image = AsyncMock(side_effect=preview)
        with patch.dict(sys.modules, {media.__name__: media}):
            async with self.cache.acquire("chat", self.image()) as cached:
                encoded = await input_module.encode_cached_image(cached)
                self.assertTrue(encoded.is_montage)
                self.assertTrue(encoded.data_uri.startswith("data:image/jpeg;base64,"))
                self.assertEqual(cached.path.read_bytes(), b"image")
        self.assertFalse(previews[0].exists())
        self.assertFalse(previews[0].parent.exists())

    async def test_native_refusal_or_failure_is_not_bypassed(self):
        class ImageInputTooLargeError(ValueError):
            pass

        for index, response in enumerate((None, ImageInputTooLargeError("native cap"), RuntimeError("native failure"))):
            with self.subTest(response=response):
                media = ModuleType("astrbot.core.utils.media_utils")
                media.prepare_model_image = AsyncMock(
                    side_effect=response if isinstance(response, Exception) else None,
                    return_value=response,
                )
                with patch.dict(sys.modules, {media.__name__: media}):
                    async with self.cache.acquire("chat", self.image(str(index))) as cached:
                        self.assertIsNone(await input_module.encode_cached_image(cached, logger=self.logger))
                        self.assertEqual(cached.path.read_bytes(), b"image")
                media.prepare_model_image.assert_awaited_once()
        self.assertEqual(self.logger.warning.call_count, 2)
        self.assertIn("ImageInputTooLargeError", self.logger.warning.call_args_list[0].args[0])

    async def test_preview_read_failure_still_cleans_only_owned_output(self):
        media = ModuleType("astrbot.core.utils.media_utils")
        directories = []

        async def preview(source, *, max_size, output_dir):
            directories.append(output_dir)
            (output_dir / "owned-working-file").write_bytes(b"partial")
            return str(output_dir / "missing-preview.png"), False, True, source

        media.prepare_model_image = AsyncMock(side_effect=preview)
        with patch.dict(sys.modules, {media.__name__: media}):
            async with self.cache.acquire("chat", self.image()) as cached:
                self.assertIsNone(await input_module.encode_cached_image(cached, logger=self.logger))
                self.assertEqual(cached.path.read_bytes(), b"image")
        self.assertFalse(directories[0].exists())
        self.logger.warning.assert_called_once()

    async def test_default_encoder_keeps_lease_until_cancelled_public_processing_finishes(self):
        entered, release = threading.Event(), threading.Event()
        paths = []
        previews = []
        media = ModuleType("astrbot.core.utils.media_utils")

        async def preview(source, *, max_size, output_dir):
            def work():
                entered.set()
                if not release.wait(3):
                    raise RuntimeError("test worker was not released")
                self.assertEqual(Path(source).read_bytes(), b"image")
                output = output_dir / "native-preview.png"
                output.write_bytes(b"\x89PNG\r\n\x1a\npreview")
                previews.append(output)
                return str(output), False, True, source

            return await asyncio.to_thread(work)

        media.prepare_model_image = preview
        self.preparer = ImageInputPreparer(self.cache, self.captions, logger=self.logger)
        ref = self.image()
        row = self.row("one", MessagePart("image", image=ref))
        with patch.dict(sys.modules, {media.__name__: media}):
            task = asyncio.create_task(self.prepare([row], round=self.round(limit=0), focus_records=(row.record_id,)))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 1))
                paths = list(self.cache.root.glob("*.cache"))
                self.assertEqual(len(paths), 1)
                task.cancel()
                await asyncio.sleep(0)
                task.cancel()
                await asyncio.sleep(0)
                await self.cache.close()
                self.assertFalse(task.done())
                self.assertTrue(paths[0].exists())
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertFalse(paths[0].exists())
                self.assertFalse(previews[0].exists())
                self.assertFalse(previews[0].parent.exists())
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)


if __name__ == "__main__":
    unittest.main()
