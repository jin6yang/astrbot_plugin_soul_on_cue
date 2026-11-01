"""Check future-task execution against a local AstrBot checkout.

Uses actual native task lifecycle, Agent hooks, runner methods, message models
and summary persistence. Platform, database and provider boundaries stay in
memory. Run with Python containing Pydantic 2; no API calls or files are made.
"""

import argparse
import ast
import asyncio
import copy
import inspect
import json
import logging
import time
import traceback
from contextlib import nullcontext
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from check_astrbot_context_contract import load_message_models, load_provider_request


def native_nodes(path, names, *, owner=None, remove_imports=False):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    body = tree.body
    if owner:
        body = next(node for node in body if isinstance(node, ast.ClassDef) and node.name == owner).body
    found = []
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in names:
            node = copy.deepcopy(node)
            if not isinstance(node, ast.ClassDef):
                node.decorator_list = []
            if remove_imports:
                node.body = [part for part in node.body if not isinstance(part, (ast.Import, ast.ImportFrom))]
            found.append(node)
    assert {node.name for node in found} == set(names), f"Missing native definitions: {names}"
    return found


def execute_nodes(nodes, namespace, filename):
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        *nodes,
    ], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(filename), "exec"), namespace)


class AgentState(Enum):
    IDLE = auto()
    RUNNING = auto()
    DONE = auto()
    ERROR = auto()


class EventType(Enum):
    OnAgentBeginEvent = auto()
    OnLLMResponseEvent = auto()
    OnAgentDoneEvent = auto()
    OnUsingLLMToolEvent = auto()
    OnLLMToolRespondEvent = auto()
    OnCallingFuncToolEvent = auto()


class MessageType(Enum):
    GROUP_MESSAGE = "GroupMessage"
    FRIEND_MESSAGE = "FriendMessage"
    OTHER_MESSAGE = "OtherMessage"


class Session:
    def __init__(self, platform_name, message_type, session_id):
        self.platform_id = platform_name
        self.message_type = message_type
        self.session_id = session_id

    def __str__(self):
        return f"{self.platform_id}:{self.message_type.value}:{self.session_id}"

    @classmethod
    def from_str(cls, value):
        platform, kind, session_id = value.split(":", 2)
        return cls(platform, MessageType(kind), session_id)


class EventBase:
    def __init__(self, message, message_obj, platform_meta, session_id):
        self.message_str = message
        self.message_obj = message_obj
        self.platform_meta = platform_meta
        self._extras = {}
        self.plugins_name = None
        self.trace = SimpleNamespace(record=lambda *args, **kwargs: None)
        self.stopped = False

    @property
    def unified_msg_origin(self):
        return str(self.session)

    def get_extra(self, key, default=None):
        return self._extras.get(key, default)

    def set_extra(self, key, value):
        self._extras[key] = value

    def get_platform_id(self):
        return self.platform_meta.id

    def get_platform_name(self):
        return self.platform_meta.name

    def is_stopped(self):
        return self.stopped


class HookBase:
    @classmethod
    def __class_getitem__(cls, item):
        return cls


class Chain:
    def __init__(self, chain=None, **kwargs):
        self.chain = chain or []

    def message(self, text):
        self.chain = [SimpleNamespace(text=text)]
        return self


class Stats:
    def __init__(self):
        self.time_to_first_token = 0

    def to_dict(self):
        return {}


class ContextManager:
    def __init__(self, config):
        pass

    async def process(self, messages, **kwargs):
        return messages


class MemoryDB:
    def __init__(self):
        self.jobs = {}
        self.statuses = []

    async def create_cron_job(self, **options):
        job = SimpleNamespace(job_id=f"native-{len(self.jobs) + 1}", **options)
        self.jobs[job.job_id] = job
        return job

    async def get_cron_job(self, job_id):
        return self.jobs.get(job_id)

    async def update_cron_job(self, job_id, **changes):
        self.statuses.append(changes.get("status"))
        job = self.jobs.get(job_id)
        if job:
            for key, value in changes.items():
                setattr(job, key, value)
        return job

    async def delete_cron_job(self, job_id):
        self.jobs.pop(job_id, None)


async def check(source):
    # Plugin APIs stay stubbed; the content classes and native execution below
    # come from the selected checkout, rather than recreating their algorithms.
    from test_timing import plugin_module

    models = load_message_models(source / "astrbot/core/agent/message.py")
    Request = load_provider_request(source / "astrbot/core/provider/entities.py", models)
    plugin_module.Message = models.Message
    plugin_module.TextPart = models.TextPart
    logger = logging.getLogger("oncue.native.glance.contract")
    hook_calls = []
    handlers = {}

    class Registry:
        def get_handlers_by_event_type(self, kind, **kwargs):
            hook_calls.append(kind)
            return handlers.get(kind, [])

    namespace = {
        "asyncio": asyncio, "json": json, "time": time,
        "logger": logger, "inspect": inspect, "traceback": traceback,
        "datetime": datetime, "timezone": timezone, "EventType": EventType,
        "AgentState": AgentState, "BaseAgentRunHooks": HookBase,
        "AstrAgentContext": object, "Message": models.Message,
        "TextPart": models.TextPart, "ThinkPart": models.ThinkPart,
        "AgentResponse": SimpleNamespace, "AgentResponseData": SimpleNamespace,
        "MessageChain": Chain, "Json": lambda **kwargs: SimpleNamespace(**kwargs),
        "ContextConfig": SimpleNamespace, "ContextManager": ContextManager,
        "AgentStats": Stats, "bind_checkpoint_messages": models.bind_checkpoint_messages,
        "EstimateTokenCounter": SimpleNamespace,
        "replace": __import__("dataclasses").replace,
        "star_handlers_registry": Registry(),
        "star_map": {"oncue": SimpleNamespace(name="oncue", display_name="OnCue")},
    }
    hook_path = source / "astrbot/core/pipeline/context_utils.py"
    execute_nodes(native_nodes(hook_path, {"call_event_hook"}), namespace, hook_path)
    hooks_path = source / "astrbot/core/astr_agent_hooks.py"
    execute_nodes(native_nodes(hooks_path, {"MainAgentHooks"}), namespace, hooks_path)
    native_hooks = namespace["MainAgentHooks"]()

    runner_path = source / "astrbot/core/agent/runners/tool_loop_agent_runner.py"
    methods = native_nodes(runner_path, {
        "reset", "_assemble_request_context_for_provider", "_complete_with_assistant_response",
        "step", "step_until_done", "get_final_llm_resp",
    }, owner="ToolLoopAgentRunner")

    class RunnerBase:
        @property
        def state(self):
            return self._state

        def _transition_state(self, state):
            self._state = state

        def done(self):
            return self._state in (AgentState.DONE, AgentState.ERROR)

        def _simple_print_message_role(self, *args):
            pass

        def _is_stop_requested(self):
            return False

        def _resolve_unconsumed_follow_ups(self):
            pass

        async def _await_or_stop(self, operation):
            return await operation

        async def _iter_llm_responses_with_fallback(self):
            provider_inputs.append(list(self.run_context.messages))
            if provider_gate is not None:
                provider_started.set()
                await provider_gate.wait()
            yield SimpleNamespace(
                role="assistant", completion_text="NATIVE_FINAL_SUMMARY",
                is_chunk=False, reasoning_content=None, reasoning_signature=None,
                result_chain=None, tools_call_name=[], usage=None,
            )

    namespace["RunnerBase"] = RunnerBase
    execute_nodes([ast.ClassDef(
        name="NativeRunner", bases=[ast.Name(id="RunnerBase", ctx=ast.Load())],
        keywords=[], body=methods, decorator_list=[],
    )], namespace, runner_path)
    provider_inputs = []
    provider_gate = None
    provider_started = asyncio.Event()
    native_events = []
    conv = SimpleNamespace(cid="contract-conversation", history=json.dumps([
        {"role": "user", "content": "EXISTING_HISTORY"},
    ]), token_usage=0)
    context = SimpleNamespace(
        get_config=lambda **kwargs: {"plugin_set": ["*"], "provider_settings": {}},
        conversation_manager=SimpleNamespace(update_conversation=AsyncMock()),
        send_message=AsyncMock(return_value=True),
        get_llm_tool_manager=lambda: SimpleNamespace(
            get_builtin_tool=lambda kind: SimpleNamespace(name="send_message_to_user"),
        ),
    )
    plugin = plugin_module.OnCuePlugin(context, {"enable": True, "enable_glance": True})
    trigger = plugin_module.GlanceTrigger({
        "umo": "test:GroupMessage:1", "self_id": "bot", "group_id": "1",
    })
    chat = plugin._chat(trigger.unified_msg_origin)
    trigger.set_extra("oncue_decision", {"should_reply": True, "_reply_version": chat.reply_version})
    trigger.set_extra("oncue_task_input", ("DIRECTION_MARKER", "CONTEXT_MARKER"))

    def register(kind, callback):
        handlers.setdefault(kind, []).append(SimpleNamespace(
            handler=callback, handler_module_path="oncue", handler_name=callback.__name__,
            handler_full_name=callback.__name__, event_type=kind,
        ))

    register(EventType.OnAgentBeginEvent, plugin.prepare_glance_task)
    register(EventType.OnAgentDoneEvent, plugin.close_image_requests)

    async def build_main_agent(*, event, plugin_context, config, req):
        native_events.append(event)
        runner = namespace["NativeRunner"]()
        await runner.reset(
            provider=SimpleNamespace(provider_config={"id": "contract", "modalities": ["text", "image", "audio"]}),
            request=req,
            run_context=SimpleNamespace(context=SimpleNamespace(event=event), messages=[]),
            tool_executor=None, agent_hooks=native_hooks, streaming=False,
        )
        return SimpleNamespace(agent_runner=runner)

    class ToolSet:
        def add_tool(self, tool):
            pass

    namespace.update(
        ProviderRequest=Request, MainAgentBuildConfig=SimpleNamespace,
        MessageSession=Session, MessageType=MessageType,
        CronMessageEvent=None, ToolSet=ToolSet, SendMessageToUserTool=object,
        _get_session_conv=AsyncMock(return_value=conv), build_main_agent=build_main_agent,
        resolve_context_compression_config=lambda value: {},
        coerce_int_config=lambda value, **kwargs: value,
        log_context=lambda *args: nullcontext(), event_context=lambda *args: nullcontext(),
        trace_execution=lambda *args: nullcontext(),
        AstrMessageEvent=EventBase, AstrBotMessage=SimpleNamespace,
        MessageMember=SimpleNamespace, Plain=lambda text: SimpleNamespace(text=text),
        PlatformMetadata=SimpleNamespace,
        uuid=__import__("uuid"),
    )
    event_path = source / "astrbot/core/cron/events.py"
    execute_nodes(native_nodes(event_path, {"CronMessageEvent"}), namespace, event_path)
    resources = source / "astrbot/core/astr_main_agent_resources.py"
    prompts = ast.parse(resources.read_text(encoding="utf-8")).body
    for node in prompts:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in {
                "PROACTIVE_AGENT_CRON_WOKE_SYSTEM_PROMPT", "PROACTIVE_AGENT_CRON_WOKE_USER_PROMPT",
            } for target in node.targets
        ):
            execute_nodes([node], namespace, resources)
    history_path = source / "astrbot/core/utils/history_saver.py"
    execute_nodes(native_nodes(history_path, {"persist_agent_history"}), namespace, history_path)
    manager_path = source / "astrbot/core/cron/manager.py"
    manager_methods = native_nodes(manager_path, {
        "add_active_job", "delete_job", "run_job_now", "_run_job",
        "_run_active_agent_job", "_woke_main_agent",
    }, owner="CronJobManager", remove_imports=True)
    execute_nodes([ast.ClassDef(name="NativeManager", bases=[], keywords=[], body=manager_methods, decorator_list=[])], namespace, manager_path)
    manager = namespace["NativeManager"]()
    manager.ctx = context
    manager.db = MemoryDB()
    manager._deleting_jobs = {}
    manager._running_tasks = {}
    manager._basic_handlers = {}
    manager._remove_scheduled = lambda job_id: None
    manager._get_next_run_time = lambda job_id: None
    context.cron_manager = manager

    await plugin.native_tasks.run(trigger, is_current=AsyncMock(return_value=True))
    assert not manager.db.jobs, "Temporary native task rows must be deleted"
    assert not manager._running_tasks, "Native running-task registry must unwind"
    assert "completed" in manager.db.statuses
    assert len(provider_inputs) == 1
    model_input = json.dumps([message.model_dump() for message in provider_inputs[0]], ensure_ascii=False)
    assert "DIRECTION_MARKER" in model_input and "CONTEXT_MARKER" in model_input
    assert "autonomous proactive agent" in model_input, "Native task rules must remain intact"
    assert "EXISTING_HISTORY" in model_input
    temporary_parts = [
        part for message in provider_inputs[0] if isinstance(message.content, list)
        for part in message.content if isinstance(part, models.TextPart)
        and ("DIRECTION_MARKER" in part.text or "CONTEXT_MARKER" in part.text)
    ]
    assert len(temporary_parts) == 2 and all(part._no_save for part in temporary_parts)
    assert hook_calls == [EventType.OnAgentBeginEvent, EventType.OnLLMResponseEvent, EventType.OnAgentDoneEvent]
    context.send_message.assert_not_awaited()
    assert not chat.reply_ts and chat.last_reply_ts == 0, "A final task summary is not a delivered reply"
    saved = context.conversation_manager.update_conversation.call_args.kwargs["history"]
    assert saved[-2] == {"role": "user", "content": "Output your last task result below."}
    assert "NATIVE_FINAL_SUMMARY" in saved[-1]["content"]
    assert "DIRECTION_MARKER" not in json.dumps(saved) and "CONTEXT_MARKER" not in json.dumps(saved)
    assert "meme_manager_llm_request_origin" not in native_events[0]._extras

    # Exercise the actual native cancellation/drain behavior as well. The
    # native deleter waits for the executing task, so bridge cleanup must not
    # wait for that deleter from inside the same task's finally block.
    provider_gate = asyncio.Event()
    cancelled = asyncio.create_task(plugin.native_tasks.run(
        trigger, is_current=AsyncMock(return_value=True),
    ))
    await provider_started.wait()
    await asyncio.wait_for(plugin.native_tasks.cancel_chat(trigger.unified_msg_origin), 2)
    assert cancelled.cancelled()
    assert "cancelled" in manager.db.statuses
    assert not manager.db.jobs and not manager._running_tasks
    assert context.conversation_manager.update_conversation.await_count == 1
    await plugin.native_tasks.close()
    print("PASS: native public task lifecycle, real AgentBegin/runner/model visibility, temporary input, summary history and no delivery inference.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("astrbot_source", type=Path)
    arguments = parser.parse_args()
    asyncio.run(check(arguments.astrbot_source))
