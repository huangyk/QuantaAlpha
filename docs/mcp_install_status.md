# MCP 工具安装与接入

用户要求「安装代码开发常用的 MCP 和量化开发的工具」。当前状态：**包已全部预取并逐个验证可启动，但 DSH/TUI 的宿主配置文件因 Windows 权限写不进去，尚未生效。**

## 1. 已完成的验证（2026-10-13，逐个真实启动过）

### 代码开发常用 MCP

| 服务器 | 包 | 启动方式 | 验证结果 |
| :--- | :--- | :--- | :--- |
| memory | `@modelcontextprotocol/server-memory` | `npx -y` | ✅ 起在 stdio |
| sequential-thinking | `@modelcontextprotocol/server-sequential-thinking` | `npx -y` | ✅ 起在 stdio |
| context7 | `@upstash/context7-mcp` | `npx -y` | ✅ `--help` 正常，需 API key 才有数据 |
| codegraph | `@astudioplus/codegraph-mcp` | `npx -y` | ✅ Rust 服务端可跑；`~/.codegraph` 不可写→自动降级纯内存 |
| fetch | `mcp-server-fetch` | `uvx --from mcp-server-fetch` | ✅ `--help` 正常 |

### 量化开发 MCP

| 服务器 | 目标 | 验证结果 |
| :--- | :--- | :--- |
| postgres | TimescaleDB `quant_data`（quant/quant123456@127.0.0.1:5432） | ✅ 长驻运行不退出（已连库） |
| redis | `redis://127.0.0.1:6379/0` | ✅ 已安装 56 个依赖并启动 |
| playwright | `@playwright/mcp@latest` | ✅ `--help` 正常 |
| chrome-devtools | `chrome-devtools-mcp@latest` | ✅ 已预取 |

## 2. 配置文件（已写好在仓库里）

- `configs/mcp/mcp.code-dev.json` — 代码开发 5 个服务器
- `configs/mcp/mcp.quant.json` — 量化 4 个服务器
- `configs/mcp/cordis.patch.mcp.example.yml` — DSH profile patch 片段（推荐路径）
- `configs/mcp/README.md` — 环境坑与接入说明

## 3. 卡住的两处（需要你决定）

1. **`C:\Users\lenovo\.deepseek\mcp.json` 写不进去**（`os error 5`）。
   `deepseek mcp add` / `deepseek setup` 均失败。目录 ACL 看起来有
   `lenovo:(I)(F)`，但沙箱令牌仍被拒——这和我们刚开始遇到的
   `D:\qqmt\QuantaAlpha` 是同一类问题（那次靠 acl 脚本修好了）。
2. **`C:\Users\lenovo\.dsh\profiles\desktop\cordis.patch.yml` 未追加**
   MCP 段落。这是 DSH（也就是你现在用的这个 GUI）真正生效的开关；
   desktop profile 开了 HMR，追加后应当热重载。

## 4. 生效方式（二选一）

**A. DSH（推荐，当前会话就是这个）**
把 `configs/mcp/cordis.patch.mcp.example.yml` 里的 `- insert:` 段追加到
`C:\Users\lenovo\.dsh\profiles\desktop\cordis.patch.yml` 末尾，保存等 HMR 重载，
新会话工具名形如 `mcp__memory__create_entities`。

**B. DeepSeek TUI**
把 `configs/mcp/mcp.code-dev.json`（和/或 `mcp.quant.json`）复制成
`C:\Users\lenovo\.deepseek\mcp.json`，然后 `deepseek mcp list` 验证。

## 5. 本机环境坑（已绕过，记录备查）

npm/uv 默认缓存目录在沙箱内不可写，配置里已把缓存指到工作区：
`npm_config_cache`、`UV_CACHE_DIR`、`UV_TOOL_DIR`。
`deepseek mcp add` 的 `--arg -y` 会被 clap 拒绝，必须写 `--arg=-y`。
