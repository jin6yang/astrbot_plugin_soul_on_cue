"""将主动发言接入 AstrBot 原生回复管道，不重放用户消息。"""

from copy import copy
from uuid import uuid4

from astrbot.api.event import AstrMessageEvent
from astrbot.core.message.components import Plain
from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember
from astrbot.core.platform.message_session import MessageSession


class GlanceEvent(AstrMessageEvent):
    def __init__(self, context, target: dict, can_send):
        session = MessageSession.from_str(target["umo"])
        message = "[主动发言机会] 你刚看过群聊，决定自然接话。这是定时触发，不是某位成员的新消息。"
        obj = AstrBotMessage()
        obj.type = session.message_type
        obj.self_id = target["self_id"]
        obj.session_id = session.session_id
        obj.group_id = target["group_id"]
        obj.message_id = "oncue-" + uuid4().hex
        obj.sender = MessageMember(user_id="oncue", nickname="主动发言机会")
        obj.message = [Plain(message)]
        obj.message_str = message
        obj.raw_message = None
        meta = copy(target["platform_meta"])
        # send_by_session 无平台流式事件；交由原生管道按非流式结果处理。
        meta.support_streaming_message = False
        super().__init__(message, obj, meta, session.session_id)
        self.session = session
        self.is_at_or_wake_command = True
        self.is_wake = True
        self.set_extra("activated_handlers", [])
        self.set_extra("enable_streaming", False)
        self.set_extra("_session_isolated", target["session_isolated"])
        self.context_obj = context
        self.can_send = can_send
        self.delivery_started = False
        self.delivered = []

    async def send(self, message):
        if message is None or not message.chain or self.is_stopped():
            return
        if not self.delivery_started:
            if not await self.can_send():
                self.stop_event()
                return
            self.delivery_started = True
        try:
            sent = await self.context_obj.send_message(self.session, message)
        except Exception:
            self.stop_event()
            raise
        if not sent:
            self.stop_event()
            raise RuntimeError("GLANCE 未找到可用的发送平台")
        self.delivered.extend(message.chain)
        await super().send(message)


async def run_native_reply(context, event):
    # 延迟导入，普通消息不依赖这层主动事件适配。
    from astrbot.core.pipeline.context import PipelineContext
    from astrbot.core.pipeline.scheduler import PipelineScheduler
    from astrbot.core.star.session_plugin_manager import SessionPluginManager

    config = context.get_config(umo=event.unified_msg_origin)
    enabled_plugins = config.get("plugin_set", ["*"])
    plugin_name = "astrbot_plugin_soul_on_cue"
    if (
        enabled_plugins != ["*"] and plugin_name not in enabled_plugins
    ) or not await SessionPluginManager.is_plugin_enabled_for_session(
        event.unified_msg_origin, plugin_name
    ):
        return
    event.plugins_name = None if enabled_plugins == ["*"] else list(enabled_plugins)
    manager = context._star_manager
    if manager is None:
        raise RuntimeError("AstrBot 原生回复管道尚未就绪")
    conf_info = context.astrbot_config_mgr.get_conf_info(event.unified_msg_origin)
    scheduler = PipelineScheduler(PipelineContext(config, manager, conf_info["id"]))
    await scheduler.initialize()
    # 无新用户消息：跳过唤醒/命令匹配及入站媒体预处理；保留会话检查、
    # 生成、工具、钩子、结果装饰、发送和原生历史保存。
    scheduler.stages = [
        stage for stage in scheduler.stages
        if type(stage).__name__ not in {"WakingCheckStage", "PreProcessStage"}
    ]
    for stage in scheduler.stages:
        if type(stage).__name__ == "ProcessStage":
            # 主动触发已经获准，无需在合成文本上再次匹配用户唤醒前缀。
            stage.agent_sub_stage.prov_wake_prefix = ""
        if type(stage).__name__ == "ResultDecorateStage":
            # 定时事件没有可引用的消息或可自动 @ 的发起人。
            stage.reply_with_mention = False
            stage.reply_with_quote = False
    await scheduler.execute(event)
