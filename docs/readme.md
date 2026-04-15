# Branch Delta From Main

## Purpose
This document highlights what this branch adds compared with upstream main.
Target branch: adapt/vertex-feature-20260415-1-sync-20260415
Baseline: main

## What This Branch Supports Beyond Main

### 1) Vertex Provider End-to-End Support
- Registers Vertex in provider/model selection paths.
- Adds Vertex alias handling in model flow.
- Extends auth flow so Vertex can be configured and resolved in runtime.
- Supports per-key base URL binding, allowing different keys to target different Vertex projects/locations.

### 2) Credential Pool Failover Improvements
- First 429 now rotates credentials immediately.
- When all pool entries are exhausted, runtime can still select an emergency fallback entry.
- Exhausted fallback follows configured pool strategy:
  - random
  - round_robin
  - least_used
  - fill_first
- Rotation avoids staying on the same exhausted entry when alternatives exist.

### 3) Runtime Stability For Vertex Sessions
- Runtime provider resolution now falls back when pool.select() returns no available entry.
- Session model override path keeps and hydrates provider credential_pool.
- Result: model overrides no longer silently disable rotation behavior.

### 4) Gateway Media Handling Expansion
- Improves cross-platform media/document handling in gateway adapters.
- Adds frame-based visual enrichment for video inputs.
- Updates related gateway status and platform behavior coverage.

### 5) Test Coverage Added For New Behavior
- Adds and updates tests for:
  - immediate 429 rotation
  - exhausted-pool fallback selection
  - runtime provider fallback behavior
  - session override + credential_pool hydration
  - gateway media/document handling paths

## Changed Areas At A Glance
- Core runtime and failover:
  - run_agent.py
  - agent/credential_pool.py
  - hermes_cli/runtime_provider.py
  - gateway/run.py
- Provider/auth/model wiring:
  - hermes_cli/auth.py
  - hermes_cli/auth_commands.py
  - hermes_cli/models.py
  - hermes_cli/main.py
- Gateway platform adapters:
  - gateway/platforms/telegram.py
  - gateway/platforms/slack.py
  - gateway/platforms/discord.py
  - gateway/platforms/base.py
- Regression tests:
  - tests/agent/*
  - tests/gateway/*
  - tests/hermes_cli/*
  - tests/run_agent/*

## Quick Reader Summary
Compared with main, this branch mainly delivers three things:
1. Practical Vertex support in real runtime flows.
2. Stronger credential-pool rotation and exhausted fallback behavior.
3. Better gateway media handling with additional regression coverage.
