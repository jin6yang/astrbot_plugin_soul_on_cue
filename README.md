<div align=center>
  <img src="logo.png" alt="" width="128">
</div>
<h2 align="center">
Soul On Cue
</h2>

<h2 align="center">
Astrbot Plugin
</h2>
<h2 align="center">
Astrbot 插件 - 应场
</h2>

<div align="center">
💖 赋予 Bot 自由水群的能力！模拟真实的人类网聊思维！使用 Python 编写 😎
</div>
<div align="center">
由 POINTER 用 ❤️ 制作
</div>


<div align="center">

[中文 README](README.md) | [English README](README_en.md)

</div>

> [!NOTE]
> 本插件为自用插件
>
> 本插件即将上架 AstrBot 插件市场
>
> 英文翻译将于之后推出......
>
> 本插件项目因为以上原因，可能无法及时响应和解决 Issues 和 PR

> [!TIP]
> 本项目受到了 [astrbot_plugin_angel_heart](https://github.com/kawayiYokami/astrbot_plugin_angel_heart) 项目的启发并借鉴了部分思路

> [!IMPORTANT]
> 本项目支持与作者的其它 AstrBot 插件项目联动
>
> - [astrbot_plugin_external_knowledgebase](https://github.com/jin6yang/astrbot_plugin_external_knowledgebase)

## 主要功能和特色

- **两种决策状态**：决策层拥有静默 (SILENT) 和观测 (OBSERVING) 两种状态
  - 静默状态为默认状态，观测状态会在 Bot 成功发话后进入
- **一瞥**：每隔 15-35 分钟（默认，可自由调整）Bot 会瞄一眼群内是否有消息，是否有机会插话
- **支持主动参与密集对话和复读**：用户可配置开启或关闭 Bot 主动参与密集对话或复读的能力
  - 静默状态下会持续缓存群消息，每会话独立，最大存储 50 条，只存昵称+文本+时间戳。图片等附件暂时只记占位，如 `[图片]`
- **与其它插件联动**: 支持与作者的其它插件联动

## 安装

前往 AstrBot Web UI - 插件 - 右下角 "+" 号 - 从链接安装，填入本仓库的 URL 即可安装。

## 插件配置说明

> 具体请查看 AstrBot 插件配置中的说明

在 AstrBot Web UI 中打开本插件的设置页面，您可以对如下功能进行微调：

- 🔔 唤醒与观测
- 🧠 决策层基础
- 🤯 角色卡浓缩
- 🕹️ 触发器
- 🎭 角色设定

## 插件安装位置

Windows: `%USERPROFILE%\.astrbot\data\plugins\astrbot_plugin_soul_on_cue`

Linux / macOS / OpenHarmony: 请根据 AstrBot 部署方式查找对应的目录

## Q/A

如果碰到一些奇怪的 Bug, 建议先重启 AstrBot 的后端。

## 感谢

[Dependencies](https://github.com/jin6yang/astrbot_plugin_agent_browser/network/dependencies)

[AstrBot✨](https://github.com/AstrBotDevs/AstrBot)

[astrbot_plugin_angel_heart](https://github.com/kawayiYokami/astrbot_plugin_angel_heart)

## 开发支持

- [AstrBot Repo](https://github.com/AstrBotDevs/AstrBot)
- [AstrBot Plugin Development Docs (Chinese)](https://docs.astrbot.app/dev/star/plugin-new.html)
- [AstrBot Plugin Development Docs (English)](https://docs.astrbot.app/en/dev/star/plugin-new.html)

## 许可证

![](agplv3-155x51.png)