# MCP 工具配置（为 DSH / DeepSeek TUI 预置）

本目录保存本项目的 MCP 服务器配置，用于在 DeepSeek Harness（DSH）或
DeepSeek TUI 中启用「代码开发常用 MCP」和「量化开发工具」。

## 文件

| 文件 | 用途 |
| :--- | :--- |
| `mcp.code-dev.json` | 代码开发常用 MCP（memory / sequential-thinking / context7 / codegraph / fetch） |
| `mcp.quant.json` | 量化开发 MCP（postgres 直连 TimescaleDB quant_data、redis、playwright、chrome-devtools） |

## 关键环境差异（本机实测，2026-10-13）

沙箱令牌无法写以下目录，因此所有缓存目录都重定向到工作区内：

| 工具 | 默认缓存位置（不可写） | 替代 |
| :--- | :--- | :--- |
| npm / npx | `%LOCALAPPDATA%\npm-cache` | `npm_config_cache=D:\qqmt\QuantaAlpha\.npm-cache` |
| uv / uvx | `%LOCALAPPDATA%\uv\cache`、`%APPDATA%\uv\tools` | `UV_CACHE_DIR` / `UV_TOOL_DIR` 指向工作区 |
| deepseek TUI MCP 配置 | `C:\Users\lenovo\.deepseek\mcp.json` | 需管理员或解除沙箱后写入 |

`deepseek mcp add` 的 `--arg` 不能传 `-y`（clap 拒绝），需用 `--arg=-y` 形式；
且写入 `~/.deepseek/mcp.json` 会因权限失败，所以本目录提供可直接复制的 JSON。

## 接入 DSH（Web GUI，推荐）

DSH 通过 profile 的 `cordis.patch.yml` 挂载 `@deepseek-ai/dsh-mcp-client`，
无需 `deepseek` TUI 的 `mcp.json`。示例见 `cordis.patch.mcp.example.yml`，
将其内容追加到 `C:\Users\lenovo\.dsh\profiles\desktop\cordis.patch.yml`
（或通过 GUI 的 Plugins 页安装）后，工具以 `mcp__<serverName>__<tool>` 形式出现。

## 接入 DeepSeek TUI

```bash
# 需要写权限（沙箱内会失败）
copy configs\mcp\mcp.code-dev.json C:\Users\lenovo\.deepseek\mcp.json
deepseek mcp list
```
