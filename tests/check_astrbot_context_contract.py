"""Optional persistence check against an explicitly supplied AstrBot checkout.

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
    sys.modules[module.__name__] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module.ProviderRequest


async def check_contract(message_models: ModuleType, provider_request) -> None:
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

    assembled = await request.assemble_context()
    assert "BACKGROUND_CONTRACT_MARKER" in json.dumps(assembled)
    assert "[本次回应目标]" in json.dumps(assembled, ensure_ascii=False)
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
    assert "ORIGINAL_CURRENT_MESSAGE" in saved_json, "Original user message must remain persisted"
    assert "EXISTING_NATIVE_HISTORY" in saved_json, "Existing history must remain persisted"
    assert saved[-1] == {"role": "assistant", "content": "REPLY"}
    print("AstrBot context contract passed: visible to the model, excluded from saved history.")


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
    asyncio.run(check_contract(models, request_type))


if __name__ == "__main__":
    main()
