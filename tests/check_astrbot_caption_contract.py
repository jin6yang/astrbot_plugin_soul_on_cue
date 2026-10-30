"""Check native captions and provider selection against an AstrBot source.

Runs the real caption branch functions with in-memory provider stubs, without
starting AstrBot, downloading images, or contacting an API.
"""

import argparse
import ast
import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock


class Provider:
    provider_config = {"modalities": ["text"]}

    def __init__(self):
        self.text_chat = AsyncMock(return_value=SimpleNamespace(completion_text="a caption"))


class TextPart:
    def __init__(self, text):
        self.text = text


class Reply:
    sender_nickname = "member"
    message_str = "quoted text"


def load_caption_branches(path):
    names = {
        "_provider_supports_modality", "_decorate_llm_request", "_process_quote_message",
        "_request_img_caption", "_ensure_img_caption", "_select_provider",
    }
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    if {node.name for node in nodes} != names:
        raise ValueError("Required native caption functions are missing")
    namespace = dict(
        Provider=Provider, TextPart=TextPart, Reply=Reply,
        DEFAULT_QUOTED_MESSAGE_SETTINGS=object(), ANIMATION_CAPTION_NOTICE="",
        logger=logging.getLogger(__name__),
        _apply_prompt_prefix=lambda *args: None,
        _ensure_persona_and_skills=AsyncMock(),
        _append_system_reminders=lambda *args: None,
        _apply_workspace_extra_prompt=AsyncMock(),
        extract_quoted_message_text=AsyncMock(return_value="quoted text"),
    )
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes,
    ], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


async def check(source):
    native = load_caption_branches(source / "astrbot/core/astr_main_agent.py")
    provider = Provider()
    event = SimpleNamespace(unified_msg_origin="test:GroupMessage:1", message_obj=SimpleNamespace(message=[Reply()]))
    context = SimpleNamespace(
        get_provider_by_id=lambda provider_id: provider,
        get_using_provider_async=AsyncMock(return_value=provider),
        get_config=lambda **kwargs: {"timezone": "UTC"},
    )
    config = SimpleNamespace(provider_settings={"default_image_caption_provider_id": ""}, timezone="UTC")
    request = SimpleNamespace(conversation=object(), image_urls=["already-downloaded.png"], extra_user_content_parts=[])
    await native["_decorate_llm_request"](event, request, context, config, provider=provider)
    await native["_process_quote_message"](event, request, "", context, image_ref="already-downloaded-quote.png")
    assert provider.text_chat.await_count == 0, "Disabled native captions must not call a model"
    context.get_using_provider_async.assert_not_awaited()
    assert request.image_urls == ["already-downloaded.png"], "Disabling caption must not discard native visuals"
    config.provider_settings["default_image_caption_provider_id"] = "configured-caption-provider"
    await native["_decorate_llm_request"](event, request, context, config, provider=provider)
    assert provider.text_chat.await_count == 1, "Positive control must exercise current-image captioning"
    await native["_process_quote_message"](event, request, "configured-caption-provider", context, image_ref="already-downloaded-quote.png")
    assert provider.text_chat.await_count == 2, "Positive control must exercise quoted-image captioning"

    selected, default = Provider(), Provider()
    extras = {"selected_provider": "explicit-provider-id"}
    route_event = SimpleNamespace(unified_msg_origin=event.unified_msg_origin, get_extra=extras.get)
    route_context = SimpleNamespace(
        get_provider_by_id=Mock(return_value=selected),
        get_using_provider_async=AsyncMock(return_value=default),
    )
    assert await native["_select_provider"](route_event, route_context) is selected
    route_context.get_provider_by_id.assert_called_once_with("explicit-provider-id")
    route_context.get_using_provider_async.assert_not_awaited()

    route_context.get_provider_by_id.reset_mock()
    extras["selected_provider"] = selected
    assert await native["_select_provider"](route_event, route_context) is default
    route_context.get_provider_by_id.assert_not_called()
    route_context.get_using_provider_async.assert_awaited_once_with(umo=event.unified_msg_origin)
    print(
        "AstrBot caption contract passed: disabled=0 calls; enabled current/quote branches each call once; "
        "provider ID selects explicitly, provider object falls back to the session provider."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("astrbot_source", type=Path)
    args = parser.parse_args()
    asyncio.run(check(args.astrbot_source.resolve()))


if __name__ == "__main__":
    main()
