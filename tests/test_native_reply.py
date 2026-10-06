"""校验主动事件与 AstrBot 管道的适配契约，不需要安装 AstrBot。"""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, patch

from test_timing import Plain


class EventBase:
    def __init__(self, message, obj, meta, session_id):
        self.message_str = message
        self.message_obj = obj
        self.platform_meta = meta
        self.extras = {}
        self._has_send_oper = False
        self.stopped = False
        self.role = "member"

    @property
    def unified_msg_origin(self):
        return self.session.umo

    def set_extra(self, key, value):
        self.extras[key] = value

    def get_extra(self, key, default=None):
        return self.extras.get(key, default)

    def is_stopped(self):
        return self.stopped

    def stop_event(self):
        self.stopped = True

    async def send(self, message):
        self._has_send_oper = True


def session_from_str(umo):
    platform, kind, session = umo.split(":", 2)
    return SimpleNamespace(umo=umo, platform_id=platform, message_type=kind, session_id=session)


def modules_for(entries):
    modules = {}
    for name, attrs in entries.items():
        modules[name] = ModuleType(name)
        modules[name].__dict__.update(attrs)
    return modules


def load_adapter():
    modules = modules_for({
        "astrbot.api.event": {"AstrMessageEvent": EventBase},
        "astrbot.core.message.components": {"Plain": Plain},
        "astrbot.core.platform.astrbot_message": {"AstrBotMessage": SimpleNamespace, "MessageMember": SimpleNamespace},
        "astrbot.core.platform.message_session": {"MessageSession": SimpleNamespace(from_str=session_from_str)},
    })
    path = Path(__file__).resolve().parents[1] / "native_reply.py"
    spec = importlib.util.spec_from_file_location("oncue_native_tests", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


adapter = load_adapter()


class NativeReplyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.context = SimpleNamespace(send_message=AsyncMock(return_value=True))
        self.target = {
            "umo": "bot-instance:GroupMessage:alice_123",
            "self_id": "bot",
            "group_id": "123",
            "platform_meta": SimpleNamespace(id="bot-instance", name="aiocqhttp", support_streaming_message=True),
            "session_isolated": True,
        }
        self.can_send = AsyncMock(return_value=True)
        self.event = adapter.GlanceEvent(self.context, self.target, self.can_send)

    def test_synthetic_event_preserves_route_without_impersonating_last_member(self):
        self.assertEqual(self.event.unified_msg_origin, self.target["umo"])
        self.assertEqual(self.event.message_obj.group_id, "123")
        self.assertEqual(self.event.platform_meta.name, "aiocqhttp")
        self.assertEqual(self.event.message_obj.sender.user_id, "oncue")
        self.assertEqual(self.event.role, "member")
        self.assertTrue(self.event.get_extra("_session_isolated"))
        self.assertIn("不是某位成员的新消息", self.event.message_str)
        self.assertEqual(self.event.get_extra("activated_handlers"), [])
        self.assertFalse(self.event.get_extra("enable_streaming"))
        self.assertTrue(self.target["platform_meta"].support_streaming_message)

    async def test_successful_delivery_preserves_chain_and_sets_send_marker(self):
        chain = SimpleNamespace(chain=[Plain("正文"), object()])
        await self.event.send(chain)
        self.context.send_message.assert_awaited_once_with(self.event.session, chain)
        self.assertEqual(self.event.delivered, chain.chain)
        self.assertTrue(self.event._has_send_oper)

    async def test_false_delivery_result_is_not_counted_as_success(self):
        self.context.send_message.return_value = False
        with self.assertRaisesRegex(RuntimeError, "发送平台"):
            await self.event.send(SimpleNamespace(chain=[Plain("正文")]))
        self.assertFalse(self.event._has_send_oper)
        self.assertEqual(self.event.delivered, [])
        self.assertTrue(self.event.is_stopped())

    async def test_transport_exception_is_not_counted_as_success(self):
        self.context.send_message.side_effect = RuntimeError("network")
        with self.assertRaisesRegex(RuntimeError, "network"):
            await self.event.send(SimpleNamespace(chain=[Plain("正文")]))
        self.assertFalse(self.event._has_send_oper)
        self.assertEqual(self.event.delivered, [])

    async def test_stale_reply_is_stopped_before_delivery(self):
        self.can_send.return_value = False
        await self.event.send(SimpleNamespace(chain=[Plain("过期正文")]))
        self.context.send_message.assert_not_awaited()
        self.assertTrue(self.event.is_stopped())
        self.assertFalse(self.event._has_send_oper)

    async def test_started_segmented_delivery_finishes_without_rechecking_own_reply_version(self):
        await self.event.send(SimpleNamespace(chain=[Plain("第一段")]))
        self.can_send.return_value = False
        await self.event.send(SimpleNamespace(chain=[Plain("第二段")]))
        self.assertEqual(self.context.send_message.await_count, 2)
        self.assertEqual(len(self.event.delivered), 2)
        self.can_send.assert_awaited_once()

    async def test_empty_delivery_does_not_claim_send_started(self):
        await self.event.send(None)
        await self.event.send(SimpleNamespace(chain=[]))
        self.assertFalse(self.event.delivery_started)
        self.context.send_message.assert_not_awaited()

    def install_pipeline(self, *, plugins=None, session_enabled=True):
        config = {"plugin_set": plugins or ["*"]}
        self.context.get_config = lambda **kwargs: config
        self.context._star_manager = SimpleNamespace(context=self.context)
        self.context.astrbot_config_mgr = SimpleNamespace(get_conf_info=lambda umo: {"id": "session-config"})
        stages = [type(name, (), {})() for name in (
            "WakingCheckStage", "WhitelistCheckStage", "SessionStatusCheckStage", "RateLimitStage",
            "ContentSafetyCheckStage", "PreProcessStage", "ProcessStage", "ResultDecorateStage", "RespondStage",
        )]
        stages[6].agent_sub_stage = SimpleNamespace(prov_wake_prefix="/llm")
        pipeline = SimpleNamespace(stages=stages, initialize=AsyncMock(), execute=AsyncMock())
        ctx_args = []

        def pipeline_context(*args):
            ctx_args.append(args)
            return SimpleNamespace()

        modules = modules_for({
            "astrbot.core.pipeline.context": {"PipelineContext": pipeline_context},
            "astrbot.core.pipeline.scheduler": {"PipelineScheduler": lambda ctx: pipeline},
            "astrbot.core.star.session_plugin_manager": {
                "SessionPluginManager": SimpleNamespace(is_plugin_enabled_for_session=AsyncMock(return_value=session_enabled)),
            },
        })
        patcher = patch.dict(sys.modules, modules)
        patcher.start()
        self.addCleanup(patcher.stop)
        return config, pipeline, ctx_args

    async def test_uses_native_process_decorate_send_and_session_config(self):
        config, pipeline, ctx_args = self.install_pipeline()
        await adapter.run_native_reply(self.context, self.event)
        self.assertIs(ctx_args[0][0], config)
        self.assertIs(ctx_args[0][1], self.context._star_manager)
        self.assertEqual(ctx_args[0][2], "session-config")
        self.assertEqual([type(s).__name__ for s in pipeline.stages], [
            "WhitelistCheckStage", "SessionStatusCheckStage", "RateLimitStage", "ContentSafetyCheckStage",
            "ProcessStage", "ResultDecorateStage", "RespondStage",
        ])
        self.assertEqual(pipeline.stages[-3].agent_sub_stage.prov_wake_prefix, "")
        self.assertFalse(pipeline.stages[-2].reply_with_mention)
        self.assertFalse(pipeline.stages[-2].reply_with_quote)
        pipeline.execute.assert_awaited_once_with(self.event)

    async def test_disabled_plugin_does_not_run_native_reply(self):
        for plugins, session_enabled in ((["other-plugin"], True), (["*"], False)):
            with self.subTest(plugins=plugins, session_enabled=session_enabled):
                _, pipeline, _ = self.install_pipeline(plugins=plugins, session_enabled=session_enabled)
                await adapter.run_native_reply(self.context, self.event)
                pipeline.execute.assert_not_awaited()

    async def test_keeps_enabled_plugin_scope_for_native_hooks_and_tools(self):
        plugins = ["astrbot_plugin_soul_on_cue", "memory"]
        self.install_pipeline(plugins=plugins)
        await adapter.run_native_reply(self.context, self.event)
        self.assertEqual(self.event.plugins_name, plugins)

    async def test_native_error_is_propagated_without_manual_generation_fallback(self):
        _, pipeline, _ = self.install_pipeline()
        pipeline.execute.side_effect = RuntimeError("native failure")
        with self.assertRaisesRegex(RuntimeError, "native failure"):
            await adapter.run_native_reply(self.context, self.event)
        self.context.send_message.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
