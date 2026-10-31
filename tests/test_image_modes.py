"""图片模式通过现有原生钩子接入，不改原生配置或其它插件内容。"""

import asyncio
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from test_timing import Plain, ImageURLPart, _event, _request_text, plugin_module
from test_image_input import EncodedImage


class ImageModeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.providers = {
            name: SimpleNamespace(provider_config={"id": name, "modalities": modalities}, get_model=lambda: "default")
            for name, modalities in (("decision", ["text"]), ("caption", ["text", "image"]), ("vision", ["text", "image"]))
        }
        self.native = {"provider_settings": {"default_image_caption_provider_id": ""}, "agent_runner": {"config": {"model": {"fallback_provider_ids": []}}}}
        self.context = SimpleNamespace(
            get_provider_by_id=self.providers.get,
            get_current_chat_provider_id=AsyncMock(return_value="decision"),
            get_config=lambda **kwargs: self.native,
            llm_generate=AsyncMock(return_value=SimpleNamespace(completion_text="可见的一只猫")),
        )
        self.plugin = plugin_module.OnCuePlugin(self.context, {
            "enable_glance": False, "enable_echo_trigger": False, "enable_dense_trigger": False,
            "message_wait_seconds": 0, "image_caption_provider": "caption", "analyzer_provider": "decision",
        })
        self.fetches = []

        async def fetch(source, write, max_bytes):
            self.fetches.append(source)
            write(b"image")
            return 5

        self.cache = plugin_module.SessionImageCache(Path(self.directory.name), fetch=fetch)
        self.plugin.image_cache = self.cache
        self.plugin.image_captions = plugin_module.ImageCaptionService(self.cache, self.plugin.image_models.caption)
        self.encode = AsyncMock(return_value=EncodedImage("data:image/png;base64,aW1n"))
        self.plugin.image_input = plugin_module.ImageInputPreparer(self.cache, self.plugin.image_captions, encode=self.encode)
        patcher = patch.dict(plugin_module.ImageModels.caption.__globals__, {"encode_cached_image": self.encode})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.chat_id = _event().unified_msg_origin
        self.chat = self.plugin._chat(self.chat_id)
        self.plugin._decision_card = AsyncMock(return_value="ROLE_CARD_MUST_NOT_CAPTION")
        self.plugin._llm_decision = AsyncMock(return_value='{"should_reply": true, "reason": "interesting"}')
        self.tasks = []

    async def asyncTearDown(self):
        for task in self.tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        await self.plugin.terminate()

    @staticmethod
    def event(message_id="current", *, images=1, summoned=True):
        event = _event("native original text", summoned=summoned)
        event.message_obj.message_id = message_id
        parts = [Plain("native original text")]
        for index in range(images):
            image = plugin_module.Image()
            image.url = f"https://platform.invalid/{message_id}/{index}.png"
            parts.append(image)
        event.get_messages = lambda: parts
        return event

    @staticmethod
    def request(*, images=None, extras=None):
        return SimpleNamespace(
            prompt="native original text", system_prompt="native persona", model=None,
            image_urls=["native-image"] if images is None else images,
            audio_urls=["native-audio"], contexts=[{"role": "tool", "content": "tool history"}],
            extra_user_content_parts=extras if extras is not None else [],
        )

    async def approve(self, event=None):
        event = event or self.event()
        await self.plugin.on_message(event)
        return event

    async def test_placeholder_preserves_native_images_and_foreign_content(self):
        self.plugin.config["image_mode"] = "不转述（仅占位）"
        event = await self.approve()
        ref = self.chat.messages[-1]["parts"][1].image
        ref.description = "OLD_DESCRIPTION_MUST_BE_HIDDEN"
        foreign = ImageURLPart(image_url={"url": "foreign-image"})
        req = self.request(extras=[foreign])
        native_images = req.image_urls
        contexts = list(req.contexts)
        await self.plugin.inject_stage_direction(event, req)
        self.assertIs(req.image_urls, native_images)
        self.assertEqual(req.image_urls, ["native-image"])
        self.assertEqual(req.audio_urls, ["native-audio"])
        self.assertEqual(req.contexts, contexts)
        self.assertIs(req.extra_user_content_parts[0], foreign)
        self.assertEqual(req.prompt, "native original text")
        self.assertNotIn("OLD_DESCRIPTION_MUST_BE_HIDDEN", _request_text(req))
        self.assertIn("[图片，已转述]", _request_text(req))
        self.context.llm_generate.assert_not_awaited()
        self.assertEqual(self.fetches, [])

    async def test_describe_keeps_current_caption_after_original_body_dedup(self):
        self.plugin.config["image_mode"] = "仅转述"
        event = await self.approve()
        req = self.request()
        native_images = req.image_urls
        await self.plugin.inject_stage_direction(event, req)
        self.assertIs(req.image_urls, native_images)
        self.assertEqual(req.image_urls, ["native-image"])
        self.assertIn("可见的一只猫", _request_text(req))
        self.assertEqual(_request_text(req).count("native original text"), 1)
        self.assertTrue(all(part._no_save for part in req.extra_user_content_parts))
        self.context.llm_generate.assert_awaited_once()
        args = self.context.llm_generate.call_args.kwargs
        self.assertEqual(args["chat_provider_id"], "caption")
        self.assertNotIn("ROLE_CARD", args["system_prompt"])
        self.assertIn("长漫画", args["system_prompt"])
        self.assertNotIn("system_prompt", self.plugin._llm_decision.call_args.kwargs if self.plugin._llm_decision.called else {})

    async def test_foreign_provider_request_attachments_are_not_cleared(self):
        self.plugin.config["image_mode"] = "不转述（仅占位）"
        event = self.event()
        event.set_extra("provider_request", object())
        await self.approve(event)
        req = self.request()
        await self.plugin.inject_stage_direction(event, req)
        self.assertEqual(req.image_urls, ["native-image"])

    async def test_both_text_modes_preserve_external_images_and_image_history(self):
        for index, mode in enumerate(("不转述（仅占位）", "仅转述")):
            with self.subTest(mode=mode):
                self.plugin.config["image_mode"] = mode
                event = await self.approve(self.event(f"isolated-{index}"))
                # A stale flag cannot establish ownership of the later request.
                event.set_extra("oncue_native_image_request", True)
                shared_url = f"https://platform.invalid/isolated-{index}/0.png"
                native_images = [shared_url, "other-plugin-image", shared_url]
                foreign = ImageURLPart(image_url={"url": shared_url, "detail": "low"})
                foreign_dict = {"type": "image_url", "image_url": {"url": "emoji-image", "detail": "high"}, "source": "emoji-plugin"}
                extras = [foreign, foreign_dict]
                req = self.request(images=native_images, extras=extras)
                history = [
                    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": shared_url}}]},
                    {"role": "tool", "content": [{"type": "image_url", "image_url": {"url": "tool-image"}}]},
                ]
                req.contexts = history
                original_history = copy.deepcopy(history)
                original_dict = copy.deepcopy(foreign_dict)
                for _ in range(2):
                    await self.plugin.inject_stage_direction(event, req)
                self.assertIs(req.image_urls, native_images)
                self.assertEqual(native_images, [shared_url, "other-plugin-image", shared_url])
                self.assertIs(req.contexts, history)
                self.assertEqual(history, original_history)
                self.assertIs(req.extra_user_content_parts[0], foreign)
                self.assertIs(req.extra_user_content_parts[1], foreign_dict)
                self.assertEqual(foreign.image_url.url, shared_url)
                self.assertEqual(foreign.image_url.detail, "low")
                self.assertEqual(foreign_dict, original_dict)
                self.assertEqual(extras, [foreign, foreign_dict])

    async def test_request_set_after_the_waiting_notification_keeps_all_attachments(self):
        for index, mode in enumerate(("不转述（仅占位）", "仅转述")):
            with self.subTest(mode=mode):
                self.plugin.config["image_mode"] = mode
                event = await self.approve(self.event(f"late-request-{index}"))
                event.set_extra("oncue_native_image_request", True)
                images = ["custom-request-image", "custom-request-image"]
                foreign = ImageURLPart(image_url={"url": "custom-extra-image"})
                req = self.request(images=images, extras=[foreign])
                event.set_extra("provider_request", req)
                await self.plugin.inject_stage_direction(event, req)
                self.assertIs(event.get_extra("provider_request"), req)
                self.assertIs(req.image_urls, images)
                self.assertEqual(images, ["custom-request-image", "custom-request-image"])
                self.assertIs(req.extra_user_content_parts[0], foreign)

    async def test_external_images_added_or_replaced_during_caption_wait_survive(self):
        self.plugin.config["image_mode"] = "仅转述"
        event = await self.approve()
        entered, release = asyncio.Event(), asyncio.Event()

        async def generate(**kwargs):
            entered.set()
            await release.wait()
            return SimpleNamespace(completion_text="插件图片描述")

        self.context.llm_generate.side_effect = generate
        original_list = ["original-native-image"]
        req = self.request(images=original_list)
        task = asyncio.create_task(self.plugin.inject_stage_direction(event, req))
        self.tasks.append(task)
        try:
            await asyncio.wait_for(entered.wait(), 1)
            replacement_images = ["emoji-plugin-replacement"]
            req.image_urls = replacement_images
            foreign = ImageURLPart(image_url={"url": "late-extra-image"})
            foreign_dict = {"type": "image_url", "image_url": {"url": "late-dict-image"}}
            req.extra_user_content_parts.extend([foreign, foreign_dict])
            release.set()
            await asyncio.wait_for(task, 1)
            self.assertIs(req.image_urls, replacement_images)
            self.assertEqual(replacement_images, ["emoji-plugin-replacement"])
            self.assertEqual(original_list, ["original-native-image"])
            self.assertIs(req.extra_user_content_parts[0], foreign)
            self.assertIs(req.extra_user_content_parts[1], foreign_dict)
            self.assertEqual(foreign_dict["image_url"]["url"], "late-dict-image")
            self.assertIn("插件图片描述", _request_text(req))
        finally:
            release.set()

    async def test_emoji_replacement_in_outgoing_result_is_not_modified(self):
        for index, mode in enumerate(("不转述（仅占位）", "仅转述")):
            with self.subTest(mode=mode):
                self.plugin.config["image_mode"] = mode
                event = await self.approve(self.event(f"outgoing-{index}"))
                image = plugin_module.Image()
                image.url = "https://emoji.invalid/replaced-emotion.png"
                body = Plain("表情插件替换后的回复")
                chain = [body, image]
                event.set_result(event.chain_result(chain))
                event._has_send_oper = True
                result = event.get_result()
                await self.plugin.mark_reply_sent(event)
                self.assertIs(event.get_result(), result)
                self.assertIs(result.chain, chain)
                self.assertIs(chain[0], body)
                self.assertIs(chain[1], image)
                self.assertEqual(image.url, "https://emoji.invalid/replaced-emotion.png")
                self.assertEqual(body.text, "表情插件替换后的回复")

    async def test_unknown_reply_replaces_owned_visuals_but_preserves_foreign_parts(self):
        self.plugin.config.update(image_request_count=3, image_caption_budget=0)
        older = self.event("older", summoned=False)
        await self.plugin.on_message(older)
        event = await self.approve(self.event(images=0))
        decision = event.get_extra("oncue_decision")
        decision["_target"] = decision["_chat_context"].target("older")
        self.chat.messages[0]["parts"][1].image.description = "历史图片的已有描述"
        event.set_extra("selected_provider", "vision")
        foreign = ImageURLPart(image_url={"url": "foreign-image"})
        req = self.request(extras=[foreign])
        plugin_module.set_temporary_background(
            req, "old oncue context", plugin_module.TextPart,
            images=["previous-oncue-image"], image_part_type=ImageURLPart,
        )
        for _ in range(2):
            await self.plugin.inject_stage_direction(event, req)
        self.assertEqual(req.image_urls, ["native-image"])
        visuals = [part for part in req.extra_user_content_parts if getattr(part, "type", None) == "image_url"]
        self.assertEqual(len(visuals), 1)
        self.assertIs(visuals[0], foreign)
        self.assertIn("历史图片的已有描述", _request_text(req))
        self.assertNotIn("本次附图关联", _request_text(req))
        self.plugin.config["image_mode"] = "不转述（仅占位）"
        await self.plugin.inject_stage_direction(event, req)
        self.assertEqual([part for part in req.extra_user_content_parts if getattr(part, "type", None) == "image_url"], [foreign])
        self.assertNotIn("已附图", _request_text(req))
        self.assertNotIn("本次附图关联", _request_text(req))

    async def test_unknown_reply_provider_uses_descriptions_without_new_visual_parts(self):
        event = await self.approve()
        req = self.request(images=[])
        await self.plugin.inject_stage_direction(event, req)
        self.assertIn("可见的一只猫", _request_text(req))
        self.assertFalse(getattr(req, "_oncue_image_parts", []))

    async def test_budget_is_shared_by_initial_extra_decision_and_reply(self):
        self.chat.last_reply_ts = plugin_module.time.time() - 1
        entered, release = asyncio.Event(), asyncio.Event()
        calls = 0

        async def decide(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                entered.set()
                await release.wait()
                return '{"should_reply": false}'
            return '{"should_reply": true, "reason": "updated"}'

        self.plugin._llm_decision.side_effect = decide
        event = self.event(images=3, summoned=False)
        task = asyncio.create_task(self.plugin.on_message(event))
        self.tasks.append(task)
        try:
            await asyncio.wait_for(entered.wait(), 1)
        except TimeoutError:
            if task.done():
                task.result()
            raise
        budget = event.get_extra("oncue_image_round")
        self.assertEqual(budget.used, 3)
        await self.plugin.on_message(self.event("later", summoned=False))
        release.set()
        await asyncio.wait_for(task, 1)
        req = self.request()
        await self.plugin.inject_stage_direction(event, req)
        self.assertIs(event.get_extra("oncue_image_round"), budget)
        self.assertEqual(budget.used, 3)
        self.assertEqual(calls, 2)
        self.assertEqual(self.context.llm_generate.await_count, 3)
        await self.plugin.close_image_requests(event, None, None)
        self.assertTrue(budget.closed)

    async def test_missing_or_text_only_caption_model_does_not_download_or_call(self):
        self.plugin.config.update(image_mode="仅转述", image_caption_provider="decision")
        event = await self.approve()
        with self.assertLogs(plugin_module.logger, level="WARNING"):
            await self.plugin.inject_stage_direction(event, self.request())
        self.assertEqual(self.fetches, [])
        self.context.llm_generate.assert_not_awaited()
        self.plugin.config["image_caption_provider"] = ""
        event2 = await self.approve(self.event("next"))
        with self.assertLogs(plugin_module.logger, level="WARNING"):
            await self.plugin.inject_stage_direction(event2, self.request())
        self.assertEqual(self.fetches, [])

    async def test_missing_decision_card_does_not_spend_image_caption_budget(self):
        self.chat.last_reply_ts = plugin_module.time.time() - 1
        self.plugin._decision_card.return_value = None
        event = self.event(summoned=False)
        await self.plugin.on_message(event)
        self.assertEqual(self.fetches, [])
        self.context.llm_generate.assert_not_awaited()
        self.plugin._llm_decision.assert_not_awaited()
        self.assertIsNone(event.get_extra("oncue_image_round"))

    async def test_stale_reply_is_rejected_before_starting_image_work(self):
        self.plugin.config["image_mode"] = "仅转述"
        old = await self.approve()
        latest = await self.approve(self.event("newer-summon", images=0))
        await self.plugin.inject_stage_direction(old, self.request())
        self.assertTrue(old.stopped)
        self.assertIsNone(old.get_extra("oncue_decision"))
        self.assertTrue(latest.get_extra("oncue_decision")["should_reply"])
        self.assertEqual(self.fetches, [])
        self.context.llm_generate.assert_not_awaited()

    async def test_caption_api_error_keeps_image_state_and_reports_provider(self):
        class APIError(Exception):
            status_code = 503

        self.plugin.config["image_mode"] = "仅转述"
        self.context.llm_generate.side_effect = APIError("server busy")
        event = await self.approve()
        images = ["foreign-image-on-error"]
        foreign = ImageURLPart(image_url={"url": "foreign-extra-on-error"})
        req = self.request(images=images, extras=[foreign])
        with self.assertLogs(plugin_module.logger, level="WARNING") as logs:
            await self.plugin.inject_stage_direction(event, req)
        self.assertIn("模型=caption", "\n".join(logs.output))
        self.assertIn("HTTP 503", "\n".join(logs.output))
        ref = self.chat.messages[-1]["parts"][1].image
        self.assertEqual(ref.unavailable_reason, "")
        self.assertEqual(ref.description, "")
        self.assertIn("内容未读取", _request_text(req))
        self.assertFalse(req._oncue_image_parts)
        self.assertIs(req.image_urls, images)
        self.assertEqual(images, ["foreign-image-on-error"])
        self.assertIs(req.extra_user_content_parts[0], foreign)

    async def test_overlong_caption_preserves_ends_and_marks_truncation(self):
        self.plugin.config.update(image_mode="仅转述", message_char_limit=100)
        self.context.llm_generate.return_value = SimpleNamespace(completion_text="START-" + "内容" * 300 + "-END")
        event = await self.approve()
        req = self.request()
        await self.plugin.inject_stage_direction(event, req)
        ref = self.chat.messages[-1]["parts"][1].image
        self.assertTrue(ref.description.startswith("START-"))
        self.assertTrue(ref.description.endswith("-END"))
        self.assertIn("描述已截断", ref.description)
        self.assertIn("描述已截断", _request_text(req))
        self.assertLessEqual(len(ref.description), 100)

    async def test_caption_error_response_is_not_saved_as_description(self):
        self.plugin.config["image_mode"] = "仅转述"
        self.context.llm_generate.return_value = SimpleNamespace(role="err", completion_text="server busy")
        event = await self.approve()
        req = self.request()
        with self.assertLogs(plugin_module.logger, level="WARNING") as logs:
            await self.plugin.inject_stage_direction(event, req)
        self.assertIn("模型返回错误结果", "\n".join(logs.output))
        ref = self.chat.messages[-1]["parts"][1].image
        self.assertEqual(ref.description, "")
        self.assertEqual(ref.unavailable_reason, "")
        self.assertIn("内容未读取", _request_text(req))
        self.assertNotIn("server busy", _request_text(req))

    async def test_native_caption_disabled_has_no_conflict_warning_enabled_has_one(self):
        event = self.event(images=0)
        with patch.object(plugin_module.logger, "warning") as warning:
            self.plugin._check_context_conflict(event)
            warning.assert_not_called()
            self.native["provider_settings"]["default_image_caption_provider_id"] = "native-caption"
            self.plugin._check_context_conflict(event)
            self.plugin._check_context_conflict(event)
            self.assertEqual(warning.call_count, 1)
            self.assertIn("图片转述模型", str(warning.call_args))
        self.assertEqual(self.native["provider_settings"]["default_image_caption_provider_id"], "native-caption")

    async def test_plain_text_or_zero_budget_does_not_warn_about_unused_caption_model(self):
        self.plugin.config["image_caption_provider"] = ""
        event = await self.approve(self.event(images=0))
        with patch.object(plugin_module.logger, "warning") as warning:
            await self.plugin.inject_stage_direction(event, self.request(images=[]))
            warning.assert_not_called()
        self.plugin.config["image_caption_budget"] = 0
        event2 = await self.approve(self.event("picture"))
        with patch.object(plugin_module.logger, "warning") as warning:
            await self.plugin.inject_stage_direction(event2, self.request())
            warning.assert_not_called()
        self.context.llm_generate.assert_not_awaited()

    async def test_provider_selection_and_fallback_config_do_not_control_reply_supplements(self):
        self.plugin.config["image_caption_budget"] = 0
        for index, fallbacks in enumerate(([], ["decision"])):
            with self.subTest(fallbacks=fallbacks):
                self.native["agent_runner"]["config"]["model"]["fallback_provider_ids"] = fallbacks
                event = await self.approve(self.event(f"current-{index}"))
                self.chat.messages[-1]["parts"][1].image.description = "已有描述"
                event.set_extra("selected_provider", "vision")
                req = self.request()
                req.model = "another-model"
                with patch.object(self.context, "get_provider_by_id", side_effect=AssertionError("must not infer reply provider")):
                    await self.plugin.inject_stage_direction(event, req)
                self.assertEqual(req.image_urls, ["native-image"])
                self.assertEqual(req.model, "another-model")
                self.assertFalse(req._oncue_image_parts)
                self.assertIn("已有描述", _request_text(req))

    async def test_provider_object_is_not_an_explicit_native_route(self):
        event = await self.approve()
        event.set_extra("selected_provider", self.providers["vision"])
        req = self.request()
        await self.plugin.inject_stage_direction(event, req)
        self.assertIn("可见的一只猫", _request_text(req))
        self.assertFalse(req._oncue_image_parts)
        self.assertEqual(req.image_urls, ["native-image"])

    async def test_cancelling_reply_image_wait_releases_only_owned_reservation(self):
        self.plugin.config["image_mode"] = "仅转述"
        event = await self.approve()
        decision = event.get_extra("oncue_decision")
        decision["_reservation"] = "owned"
        self.chat.pending_replies.update(owned=9999999999, other=9999999999)
        entered, release = asyncio.Event(), asyncio.Event()

        async def generate(**kwargs):
            entered.set()
            await release.wait()
            return SimpleNamespace(completion_text="caption")

        self.context.llm_generate.side_effect = generate
        images = ["foreign-image-on-cancel"]
        foreign = ImageURLPart(image_url={"url": "foreign-extra-on-cancel"})
        req = self.request(images=images, extras=[foreign])
        task = asyncio.create_task(self.plugin.inject_stage_direction(event, req))
        self.tasks.append(task)
        try:
            await asyncio.wait_for(entered.wait(), 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(set(self.chat.pending_replies), {"other"})
            self.assertIsNone(event.get_extra("oncue_decision"))
            self.assertIsNone(event.get_extra("oncue_image_round"))
            self.assertTrue(event.stopped)
            self.assertIs(req.image_urls, images)
            self.assertEqual(images, ["foreign-image-on-cancel"])
            self.assertIs(req.extra_user_content_parts[0], foreign)
        finally:
            release.set()

    async def test_switch_to_placeholder_while_waiting_does_not_inject_finished_caption(self):
        self.plugin.config["image_mode"] = "仅转述"
        event = await self.approve()
        entered, release = asyncio.Event(), asyncio.Event()

        async def generate(**kwargs):
            entered.set()
            await release.wait()
            return SimpleNamespace(completion_text="CAPTION_SHOULD_BE_HIDDEN")

        self.context.llm_generate.side_effect = generate
        images = ["foreign-image-during-mode-switch"]
        foreign = ImageURLPart(image_url={"url": "foreign-extra-during-mode-switch"})
        req = self.request(images=images, extras=[foreign])
        task = asyncio.create_task(self.plugin.inject_stage_direction(event, req))
        self.tasks.append(task)
        await asyncio.wait_for(entered.wait(), 1)
        self.plugin.config["image_mode"] = "不转述（仅占位）"
        release.set()
        await asyncio.wait_for(task, 1)
        self.assertNotIn("CAPTION_SHOULD_BE_HIDDEN", _request_text(req))
        self.assertIs(req.image_urls, images)
        self.assertEqual(images, ["foreign-image-during-mode-switch"])
        self.assertIs(req.extra_user_content_parts[0], foreign)
        self.assertFalse(req._oncue_image_parts)


if __name__ == "__main__":
    unittest.main()
