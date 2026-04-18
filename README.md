# Changes: adapt/vertex-feature-20260415-1-sync-20260415

---

## 媒体处理

改动最多的部分。视频和图像处理从原来的边缘路径提升为一等媒体，CLI 和 Gateway 两条链路都做了完整支持。

### 视频输入

**CLI 侧** (`cli.py`)

- 新增视频文件自动识别：拖拽带视频扩展名的文件或使用 `--video` 参数均可触发
- 视频进入处理链后会依次执行：ffmpeg 抽帧 → vision 分析 → 音轨转写，最终拼成结构化上下文再送给模型
- 抽帧与转写失败时有独立回退逻辑，不会拖垮整个查询

**Gateway 侧** (`gateway/run.py`)

- 入站视频消息先走 `enrich_video_context`，补充视频路径上下文
- 再执行抽帧视觉摘要 + 音轨转写，结果注入会话上下文
- 视频 enrichment 与文本文档注入走独立分支，互不干扰

### 视觉增强工具 (`tools/vision_tools.py` / `tools/video_enrichment.py`)

`video_enrichment.py` 是本次新增的核心工具，封装了完整的 ffmpeg 抽帧 + vision 分析流程，输出关键帧视觉摘要。

`vision_tools.py` 这次改动量最大（+470），主要做了：

- **防幻觉清理**：对 vision 输出做后处理，过滤掉模型常见的凭空描述
- **grounded prompt**：改进 vision 分析的 prompt，让输出更贴近帧内实际内容
- **并发控制**：新增信号量限制并发 vision 请求，避免在弱机环境下打爆 API
- **429 回退**：vision 请求遇到限流时有专用回退策略，不影响主流程
- **健康监控**：新增 vision 工具的健康状态追踪，便于排查失败率异常

### 文档与 MIME 处理 (`gateway/platforms/base.py` / `telegram.py` / `slack.py` / `discord.py`)

原来三个平台各自维护了一套文档扩展名解析和文本注入判定逻辑，这次统一收到 `base.py`：

- 扩展了文档类型映射，涵盖更多常见格式
- 新增复合后缀解析（如 `.md.resolved`），修复了 Telegram 侧的一个文档处理 bug
- 新增统一的文本 MIME 判定函数，三个平台改为调用同一套逻辑

---

## 其他改动

**Vertex AI 全链路接入** — CLI 认证层支持按 key 绑定 project/region，runtime 解析与 gateway 会话覆盖路径均已适配。

**凭证池容错** — 遇到 429 直接轮换，不再重试当前 key；全部凭证耗尽时走 exhausted fallback，支持 random / round_robin / least_used / fill_first 四种策略。

**Gateway 启动自修复** — 启动前自动检查 Telegram 依赖并修复；可自动接管残留的 token lock，减少"启动卡死"问题。

---

## 注意事项

- `vision_tools.py` 改动量大，建议重点回归限流回退、并发上限和防幻觉清理的边缘 case
- 大视频或弱机环境下注意 gateway 视频 enrichment 的处理延迟
- 429 轮换策略是行为级变更，多 provider + 多凭证池下建议端到端跑一遍
