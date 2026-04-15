## 相比原版新增支持能力

### 1）Vertex Provider 全链路支持
- 在 provider 和 model 选择流程中注册 Vertex。
- 增加 Vertex 的别名与模型路由兼容处理。
- 扩展认证与运行时解析流程，使 Vertex 能在实际运行中被完整识别和使用。
- 支持按 key 绑定 base_url，可将不同 key 路由到不同的 Vertex 项目和区域。

### 2）凭证池轮换与容错增强
- 首次 429 直接轮换，不再先重试当前 key。
- 全部凭证 exhausted 时仍可选择应急凭证，避免直接无 key。
- exhausted fallback 遵循配置策略：
  - random
  - round_robin
  - least_used
  - fill_first
- 轮换时在有候选的情况下尽量避免原地回到同一 exhausted 凭证。

### 3）会话覆盖场景下的运行时稳定性增强
- 运行时 provider 解析在 pool.select() 无可用项时可走 fallback。
- 会话模型覆盖路径会保留并补齐 provider 对应 credential_pool。
- 结果：会话级模型切换后，凭证轮换能力不再丢失。

### 4）Gateway 媒体处理能力扩展
- 改进多平台网关适配器中的媒体/文档处理路径。
- 增加视频输入的抽帧视觉增强能力。
- 同步完善相关状态与平台行为覆盖。

### 5）测试覆盖补齐
- 新增/更新以下能力的回归测试：
  - 首次 429 立即轮换
  - 全 exhausted fallback 选择
  - 运行时 provider fallback
  - 会话覆盖与 credential_pool 补齐
  - gateway 媒体/文档处理路径

## 主要改动区域
- 核心运行时与故障切换：
  - run_agent.py
  - agent/credential_pool.py
  - hermes_cli/runtime_provider.py
  - gateway/run.py
- Provider/Auth/Model 接线：
  - hermes_cli/auth.py
  - hermes_cli/auth_commands.py
  - hermes_cli/models.py
  - hermes_cli/main.py
- Gateway 平台适配：
  - gateway/platforms/telegram.py
  - gateway/platforms/slack.py
  - gateway/platforms/discord.py
  - gateway/platforms/base.py
- 测试：
  - tests/agent/*
  - tests/gateway/*
  - tests/hermes_cli/*
  - tests/run_agent/*

## 总结
相比原版 main，这个分支重点提供了：
1. 可实用的 Vertex 全链路支持。
2. 更强的凭证池轮换与 exhausted 容错。
3. 更完整的网关媒体能力与回归测试保障。

目标分支：adapt/vertex-feature-20260415-1-sync-20260415
对比基线：main
