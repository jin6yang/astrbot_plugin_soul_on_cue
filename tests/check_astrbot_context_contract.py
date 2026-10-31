"""Optional persistence and ownership checks against an AstrBot checkout.

Run with Python containing Pydantic 2:
    python -B tests/check_astrbot_context_contract.py <AstrBot-source-directory>

This script uses real AstrBot content models and serialization code, without
starting AstrBot or requiring all of its optional platform dependencies.
"""

import argparse
import ast
import asyncio
import importlib.util
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace


def load_message_models(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("oncue_contract_message_models", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_provider_request(path: Path, message_models: ModuleType):
    # Execute the real class body. Only unrelated module-level imports are
    # omitted; assemble_context and its serialization behavior remain original.
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    request_node = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ProviderRequest"
    )
    source = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            request_node,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(source)
    module = ModuleType("oncue_contract_provider_request")
    module.__dict__.update(
        dataclass=dataclass,
        field=field,
        ContentPart=message_models.ContentPart,
        logger=logging.getLogger(__name__),
        json=json,
    )
    async def resolve_inline_image(url):
        if not url.startswith("data:image/"):
            raise ValueError("This contract check never fetches remote media")
        return SimpleNamespace(to_data_url=lambda: url)
    module.resolve_image_ref_to_base64_data = resolve_inline_image
    sys.modules[module.__name__] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module.ProviderRequest


def load_image_preparation(path: Path, message_models: ModuleType):
    # Execute the actual request-level preparation, including its model_copy.
    # Pixel preparation is an in-memory passthrough here, not a format test.
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "prepare_request_images"
    )
    source = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function,
    ], type_ignores=[])

    async def inline_preview(ref, *, max_size, output_dir):
        assert ref.startswith("data:image/"), "This check never fetches remote media"
        return ref, False, False, ref

    namespace = dict(
        Path=Path, ImageURLPart=message_models.ImageURLPart, TextPart=message_models.TextPart,
        MODEL_IMAGE_MAX_INPUT_BYTES=64 * 1024**2,
        ImageInputTooLargeError=type("ImageInputTooLargeError", (ValueError,), {}),
        normalize_and_dedupe_strings=lambda values: list(dict.fromkeys(values)),
        prepare_model_image=inline_preview, describe_media_ref=lambda ref: ref,
    )
    exec(compile(ast.fix_missing_locations(source), str(path), "exec"), namespace)
    return namespace["prepare_request_images"]


async def check_contract(message_models: ModuleType, provider_request, image_input_path: Path) -> None:
    from test_timing import _event, plugin_module

    # The plugin's AstrBot API is stubbed, but the content model under test is
    # the actual class loaded from the selected source checkout.
    plugin_module.TextPart = message_models.TextPart
    plugin = plugin_module.OnCuePlugin(
        SimpleNamespace(get_config=lambda **kwargs: {"provider_ltm_settings": {"group_icl_enable": False}}),
        {"enable_glance": False},
    )
    event = _event("ORIGINAL_CURRENT_MESSAGE")
    event.message_obj.message_id = "contract-current"
    chat = plugin._chat(event.unified_msg_origin)
    old_event = _event("BACKGROUND_CONTRACT_MARKER")
    old_event.message_obj.message_id = "contract-history"
    old_event.get_sender_id = lambda: "other-member"
    plugin._append_message(chat, plugin._event_to_item(old_event, chat))
    plugin._append_message(chat, plugin._event_to_item(event, chat))
    snapshot = plugin._context_snapshot(chat)
    event.set_extra("oncue_decision", {
        "should_reply": True,
        "_chat_context": snapshot,
        "_target": snapshot.target("contract-history"),
    })
    history = [{"role": "user", "content": "EXISTING_NATIVE_HISTORY"}]
    request = provider_request(
        prompt="ORIGINAL_CURRENT_MESSAGE", system_prompt="ORIGINAL_PERSONA", contexts=history,
    )
    await plugin.inject_stage_direction(event, request)
    await plugin.inject_stage_direction(event, request)

    assert request.prompt == "ORIGINAL_CURRENT_MESSAGE"
    assert request.contexts == history
    background_parts = [
        part for part in request.extra_user_content_parts
        if isinstance(part, message_models.TextPart) and "BACKGROUND_CONTRACT_MARKER" in part.text
    ]
    assert len(background_parts) == 1, "Background must be injected exactly once"
    assert background_parts[0]._no_save, "Background must be marked as temporary"

    # Use the actual image content model through the plugin's owned-part helper.
    inline_image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aL1cAAAAASUVORK5CYII="
    plugin_module.set_temporary_background(
        request, background_parts[0].text, message_models.TextPart,
        images=[inline_image], image_part_type=message_models.ImageURLPart,
    )

    assembled = await request.assemble_context()
    assert "BACKGROUND_CONTRACT_MARKER" in json.dumps(assembled)
    assert "[本次回应目标]" in json.dumps(assembled, ensure_ascii=False)
    assert inline_image in json.dumps(assembled), "Temporary image must reach the model"
    current_message = message_models.Message.model_validate(assembled)
    assert any(
        getattr(part, "_no_save", False) and "BACKGROUND_CONTRACT_MARKER" in getattr(part, "text", "")
        for part in current_message.content
    ), "Temporary marker must survive request assembly and Message validation"

    messages = [message_models.Message.model_validate(item) for item in history]
    messages.extend([current_message, message_models.Message(role="assistant", content="REPLY")])
    saved = message_models.dump_messages_with_checkpoints(messages)
    saved_json = json.dumps(saved)
    assert "BACKGROUND_CONTRACT_MARKER" not in saved_json, "Temporary background leaked into history"
    assert "[本次回应目标]" not in json.dumps(saved, ensure_ascii=False), "Reply target leaked into history"
    assert inline_image not in saved_json, "Temporary image leaked into history"
    assert "ORIGINAL_CURRENT_MESSAGE" in saved_json, "Original user message must remain persisted"
    assert "EXISTING_NATIVE_HISTORY" in saved_json, "Existing history must remain persisted"
    assert saved[-1] == {"role": "assistant", "content": "REPLY"}
    # A plugin may consume or rewrite one of our blocks in place. It must retain
    # control of that block across later calls, including metadata changes.
    replacement_image = "data:image/png;base64,ZW1vamktcGx1Z2lu"
    foreign_image = message_models.ImageURLPart(image_url={"url": inline_image})
    rewritten = provider_request(
        prompt="native prompt", image_urls=[inline_image], contexts=history,
        extra_user_content_parts=[foreign_image],
    )
    plugin_module.set_temporary_background(
        rewritten, "ORIGINAL_OWNED_TEXT", message_models.TextPart,
        images=[inline_image], image_part_type=message_models.ImageURLPart,
    )
    original_text = rewritten._oncue_context_part
    original_image = rewritten._oncue_image_parts[0]
    original_text.text = "EMOJI_PLUGIN_REWRITTEN_TEXT"
    original_image.image_url.url = replacement_image
    original_image.image_url.id = "emoji-plugin"
    original_image._no_save = False
    native_urls = rewritten.image_urls
    for _ in range(2):
        plugin_module.set_temporary_background(
            rewritten, "SHOULD_NOT_REVIVE_OWNED_TEXT", message_models.TextPart,
            images=[inline_image], image_part_type=message_models.ImageURLPart,
        )
    assert rewritten.image_urls is native_urls
    assert rewritten.contexts is history
    assert rewritten.extra_user_content_parts == [foreign_image, original_text, original_image]
    assert original_text.text == "EMOJI_PLUGIN_REWRITTEN_TEXT"
    assert original_image.image_url.url == replacement_image
    assert original_image.image_url.id == "emoji-plugin" and not original_image._no_save
    rewritten_payload = await rewritten.assemble_context()
    assert replacement_image in json.dumps(rewritten_payload)
    assert "SHOULD_NOT_REVIVE_OWNED_TEXT" not in json.dumps(rewritten_payload)

    # AstrBot itself copies image blocks during request preparation. A copied
    # block must not be reclaimed merely because its URL and private flags match.
    copied = provider_request(prompt="native prompt")
    plugin_module.set_temporary_background(
        copied, "OWNED_BACKGROUND", message_models.TextPart,
        images=[inline_image], image_part_type=message_models.ImageURLPart,
    )
    before = copied._oncue_image_parts[0]
    prepare_images = load_image_preparation(image_input_path, message_models)
    await prepare_images(copied, event, max_size=1280, output_dir=Path("unused-inline-preview"), prepared={})
    after = next(part for part in copied.extra_user_content_parts if isinstance(part, message_models.ImageURLPart))
    assert after is not before and after.image_url is not before.image_url
    assert after.image_url.url == inline_image
    for _ in range(2):
        plugin_module.set_temporary_background(
            copied, "UPDATED_BACKGROUND", message_models.TextPart,
            images=[replacement_image], image_part_type=message_models.ImageURLPart,
        )
    image_parts = [part for part in copied.extra_user_content_parts if isinstance(part, message_models.ImageURLPart)]
    assert len(image_parts) == 1 and image_parts[0] is after
    assert replacement_image not in json.dumps(await copied.assemble_context())

    # Replacing a text object with an otherwise identical real model copy is
    # also a transfer of control; metadata copying does not recreate ownership.
    owned_text = copied._oncue_context_part
    text_copy = owned_text.model_copy()
    copied.extra_user_content_parts = [text_copy if part is owned_text else part for part in copied.extra_user_content_parts]
    plugin_module.set_temporary_background(copied, "DO_NOT_REINJECT", message_models.TextPart)
    assert any(part is text_copy for part in copied.extra_user_content_parts)
    assert "DO_NOT_REINJECT" not in json.dumps(await copied.assemble_context())
    print(
        "AstrBot context contract passed: temporary inputs excluded from history; "
        "foreign rewrites and native model_copy results preserved without reinjection."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("astrbot_source", type=Path)
    args = parser.parse_args()
    source = args.astrbot_source.resolve()
    message_path = source / "astrbot/core/agent/message.py"
    request_path = source / "astrbot/core/provider/entities.py"
    if not message_path.is_file() or not request_path.is_file():
        parser.error("The supplied directory does not contain the required AstrBot source files")
    models = load_message_models(message_path)
    request_type = load_provider_request(request_path, models)
    asyncio.run(check_contract(models, request_type, source / "astrbot/core/utils/image_input.py"))


if __name__ == "__main__":
    main()
