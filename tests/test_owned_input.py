"""本插件输入的归属：他插件改写、替换或消费后不得再认领。"""

import copy
import unittest
from types import SimpleNamespace

from test_timing import ImageURLPart, TextPart, plugin_module


FIRST = "data:image/png;base64,Zmlyc3Q="
SECOND = "data:image/png;base64,c2Vjb25k"
REPLACED = "data:image/png;base64,c3RpY2tlcg=="


class OwnedInputTests(unittest.TestCase):
    @staticmethod
    def request(parts=()):
        return SimpleNamespace(
            prompt="original native text",
            image_urls=["native.png"],
            audio_urls=["native.wav"],
            contexts=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "history.png"}}]}],
            extra_user_content_parts=list(parts),
        )

    @staticmethod
    def install(req, text="oncue original", images=(FIRST,)):
        plugin_module.set_temporary_background(
            req, text, TextPart, images=images, image_part_type=ImageURLPart,
        )

    @staticmethod
    def visuals(req):
        return [part for part in req.extra_user_content_parts if getattr(part, "type", None) == "image_url"]

    @staticmethod
    def texts(req):
        return [part.text for part in req.extra_user_content_parts if getattr(part, "type", None) == "text"]

    def test_normal_repeat_replaces_only_unchanged_owned_parts(self):
        foreign = ImageURLPart(image_url={"url": FIRST})
        foreign_text = TextPart(text="external prompt")
        original_parts = [foreign, foreign_text]
        req = self.request(original_parts)
        previous_list = req.extra_user_content_parts
        native_images, history = req.image_urls, req.contexts
        self.install(req)
        self.install(req, "updated context", (SECOND,))
        self.install(req, "latest context", (SECOND,))
        self.assertEqual(previous_list, original_parts)
        self.assertEqual(req.extra_user_content_parts[:2], original_parts)
        self.assertIs(req.extra_user_content_parts[0], foreign)
        self.assertEqual(foreign.image_url.url, FIRST)
        self.assertEqual(self.texts(req), ["external prompt", "latest context"])
        self.assertEqual([part.image_url.url for part in self.visuals(req)], [FIRST, SECOND])
        self.assertIs(req.image_urls, native_images)
        self.assertIs(req.contexts, history)
        self.assertEqual(req.prompt, "original native text")

    def test_same_url_foreign_image_is_never_claimed(self):
        foreign = ImageURLPart(image_url={"url": FIRST}).mark_as_temp()
        req = self.request([foreign])
        self.install(req)
        self.install(req, images=())
        self.assertEqual(self.visuals(req), [foreign])
        self.assertIs(self.visuals(req)[0], foreign)

    def test_inplace_url_replacement_releases_images_permanently(self):
        req = self.request()
        self.install(req)
        replaced = req._oncue_image_parts[0]
        replaced.image_url.url = REPLACED
        self.install(req, "new context", (SECOND,))
        self.assertEqual(self.visuals(req), [replaced])
        self.assertEqual(replaced.image_url.url, REPLACED)
        self.assertEqual(self.texts(req), ["new context"])
        replaced.image_url.url = FIRST
        self.install(req, "last context", (SECOND,))
        self.assertEqual(self.visuals(req), [replaced])
        self.assertEqual(replaced.image_url.url, FIRST)

    def test_nested_url_object_replacement_is_preserved_even_with_equal_values(self):
        req = self.request()
        self.install(req)
        replaced = req._oncue_image_parts[0]
        replacement_url = copy.copy(replaced.image_url)
        replaced.image_url = replacement_url
        self.install(req, images=())
        self.assertEqual(self.visuals(req), [replaced])
        self.assertIs(replaced.image_url, replacement_url)
        self.install(req, images=(SECOND,))
        self.assertEqual(self.visuals(req), [replaced])

    def test_new_image_instance_does_not_inherit_ownership(self):
        req = self.request()
        self.install(req)
        owned = req._oncue_image_parts[0]
        replacement = copy.copy(owned)
        req.extra_user_content_parts = [replacement if part is owned else part for part in req.extra_user_content_parts]
        self.install(req, images=(SECOND,))
        self.assertEqual(self.visuals(req), [replacement])
        self.assertIs(self.visuals(req)[0], replacement)

    def test_removed_image_is_not_resurrected_on_repeat(self):
        req = self.request()
        self.install(req)
        owned = req._oncue_image_parts[0]
        req.extra_user_content_parts.remove(owned)
        self.install(req, images=(SECOND,))
        self.install(req, images=(FIRST, SECOND))
        self.assertEqual(self.visuals(req), [])

    def test_partial_image_replacement_keeps_replacement_and_removes_unchanged_images(self):
        req = self.request()
        self.install(req, images=(FIRST, SECOND))
        changed, unchanged = req._oncue_image_parts
        changed.image_url.url = REPLACED
        self.install(req, images=(FIRST, SECOND))
        self.assertEqual(self.visuals(req), [changed])
        self.assertNotIn(unchanged, req.extra_user_content_parts)

    def test_changed_image_id_releases_ownership(self):
        req = self.request()
        self.install(req)
        changed = req._oncue_image_parts[0]
        changed.image_url.id = "external-identifier"
        self.install(req, images=())
        self.assertEqual(self.visuals(req), [changed])
        self.assertEqual(changed.image_url.id, "external-identifier")

    def test_changed_image_persistence_mark_releases_ownership(self):
        req = self.request()
        self.install(req)
        changed = req._oncue_image_parts[0]
        changed._no_save = False
        self.install(req, images=())
        self.assertEqual(self.visuals(req), [changed])
        self.assertFalse(changed._no_save)

    def test_changed_image_type_is_not_removed(self):
        req = self.request()
        self.install(req)
        changed = req._oncue_image_parts[0]
        changed.type = "text"
        changed.text = "image replaced with external text"
        self.install(req, images=(SECOND,))
        self.assertIn(changed, req.extra_user_content_parts)
        self.assertEqual(changed.text, "image replaced with external text")
        self.assertEqual(self.visuals(req), [])

    def test_inplace_text_rewrite_releases_text_but_not_images(self):
        req = self.request()
        self.install(req)
        changed = req._oncue_context_part
        changed.text = "external rewritten context"
        self.install(req, "must not resurrect", (SECOND,))
        self.assertEqual(self.texts(req), ["external rewritten context"])
        self.assertEqual([part.image_url.url for part in self.visuals(req)], [SECOND])
        changed.text = "oncue original"
        self.install(req, "still must not resurrect", ())
        self.assertEqual(self.texts(req), ["oncue original"])
        self.assertEqual(self.visuals(req), [])

    def test_text_replaced_by_sticker_is_not_recreated(self):
        req = self.request()
        self.install(req)
        owned = req._oncue_context_part
        sticker = ImageURLPart(image_url={"url": REPLACED})
        req.extra_user_content_parts = [sticker if part is owned else part for part in req.extra_user_content_parts]
        self.install(req, "must not resurrect", ())
        self.install(req, "still must not resurrect", ())
        self.assertEqual(req.extra_user_content_parts, [sticker])
        self.assertIs(req.extra_user_content_parts[0], sticker)

    def test_equal_text_new_instance_is_not_claimed(self):
        req = self.request()
        self.install(req)
        owned = req._oncue_context_part
        replacement = copy.copy(owned)
        req.extra_user_content_parts = [replacement if part is owned else part for part in req.extra_user_content_parts]
        self.install(req, "must not replace", ())
        self.assertEqual(req.extra_user_content_parts, [replacement])
        self.assertIs(req.extra_user_content_parts[0], replacement)

    def test_removed_text_is_not_recreated_and_images_can_still_update(self):
        req = self.request()
        self.install(req)
        req.extra_user_content_parts.remove(req._oncue_context_part)
        self.install(req, "must not resurrect", (SECOND,))
        self.assertEqual(self.texts(req), [])
        self.assertEqual([part.image_url.url for part in self.visuals(req)], [SECOND])

    def test_changed_text_persistence_mark_is_preserved(self):
        req = self.request()
        self.install(req)
        changed = req._oncue_context_part
        changed._no_save = False
        self.install(req, "must not replace", ())
        self.assertEqual(req.extra_user_content_parts, [changed])
        self.assertEqual(changed.text, "oncue original")
        self.assertFalse(changed._no_save)


if __name__ == "__main__":
    unittest.main()
