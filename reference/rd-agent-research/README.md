# 参考：旧项目 D:\qqmt\code 的 RD-Agent 配置与可复用组件

本目录从 `D:\qqmt\code`（分支 `research/rd-agent-factor-mining` + `research/quantaalpha-bridge`）中提取，
供 **QuantaAlpha 框架（本仓库 `D:\qqmt\QuantaAlpha`）** 开发与本地化部署参照。

> 旧项目是 RD-Agent 的「直接使用 + jq_research_system 多因子引擎 + TimescaleDB」主研究工作台；
> QuantaAlpha 是 vendoring 了 **rdagent==0.8.0** 的更高层封装（AlphaAgentLoop / evolution / 因子库）。
> 两者共用同一份 rdagent 0.8.0 与 Qlib 数据，因此旧项目的补丁与数据配置可直接迁移。

## 目录映射

| 子目录 | 内容 | 对本项目（QuantaAlpha）的价值 |
|--------|------|-------------------------------|
| `env/` | `environment-rd-agent.yml`（conda，py3.11，rdagent==0.8.0, pyqlib==0.9.7）、`.env.example`（LiteLLM 后端） | 复现因子挖掘运行环境；`.env` 说明 LLM/Embedding 接法 |
| `config/` | `qlib_profile.json`（research/smoke/research_full 三套数据窗口） | 对应本项目 `configs/backtest.yaml` 的 `data` 段；profile 切换思路可直接复用 |
| `docs/` | `RD_AGENT.md` / `QLIB_DATA_CONTRACT.md` / `QLIB_RESEARCH_WORKFLOW.md` / `rd_agent_iteration_plan.md` / `tools_README.md` | **设计思路与前置要求的核心文档**，必读 |
| `patches/` | 12 个 `tools/` 下的 Windows 补丁（`sitecustomize.py` 自动加载） | **最关键的可复用资产**：让 rdagent 0.8.0 在 Windows 跑通（JSON/日期/路径/IC/UI 等） |
| `data_tools/` | `qlib_refresh.py` / `qlib_regen_h5.py` / `set_qlib_profile.py` / `qlib_data_status.py` / `rd_health_llm_probe.py` | Qlib 数据漏斗（staging→provider→h5→可挖）的本地化脚本 |
| `quantaalpha-bridge/` | 旧仓内 QuantaAlpha 桥接副本的 `experiment.yaml` / `backtest.yaml` / `loop11_sota_factors.yaml` / `BASELINE_FROM_RD_AGENT.md` | **与本框架最贴近的已调优配置**，直接对照本项目 `configs/` |

## 关键版本/机制差异（迁移时务必注意）

1. **Python 版本**：旧 `rd-agent` 环境用 **3.11**；本框架 `pyproject.toml` 要求 **>=3.10**（README 用 3.10）。补丁针对 3.11 验证，3.10 下需复测。
2. **LLM 后端**：旧项目用 `BACKEND=rdagent.oai.backend.LiteLLMAPIBackend`（读 `OPENAI_API_KEY`/`OPENAI_API_BASE`，Embedding 走本机 Ollama `nomic-embed-text`）。本框架 `quantaalpha/llm/` 用自己的 `APIBackend`（OpenAI 兼容）。**因子挖掘若走 rdagent 内部后端，仍需 `.env` 的 LiteLLM 配置**；走本项目 `AlphaAgentLoop` 则走 `CHAT_MODEL`/`REASONING_MODEL`。
3. **数据 profile**：`qlib_profile.json` 的 `research` 指向 `~/.qlib/qlib_data/cn_real_r500`（500 只稳定抽样，2020–2026）。本项目 `run.sh` 默认 symlink 到 `cn_data`；路径需对齐。

## 诚实边界（来自旧项目文档，务必保留）

- `research` profile 的 `$factor` 当前是 **常数 1（非真复权）**；真 adj 未做。
- `cn_real_r500` 是 **500 只抽样，不是全市场**；survivorship / walk-forward 尚未完成。
- 回测 `backtest` 结束日必须 **早于日历最后一天**（现用 `2026-08-17`），否则 Qlib `IndexError` → 收益 NaN。
- 以上情况下，指标仅供研究迭代，**非可投产结论**。

## 旧项目已引用但本提取中缺失的文件（提醒）

- `config/rd_research_prompt.md.example`：文档提到但源目录不存在，无法提取（Research 方向注入模板，可后续自行补）。
- `quantaalpha/factors/loop11_baseline.py`：桥接 README 引用但源文件不存在（基线因子代码缺失，仅有 `loop11_sota_factors.yaml` 清单与说明文档）。

## 建议的下一步

1. 对照 `quantaalpha-bridge/` 与本项目 `configs/`，统一 `experiment.yaml` / `backtest.yaml` 的数据窗口与因子源。
2. 在本项目 Windows 环境下用 `patches/` 的 `sitecustomize.py`（置 `PYTHONPATH=patches`）验证 rdagent 0.8.0 能跑通 `quantaalpha mine`。
3. 把 `qlib_profile.json` 的思路落进本项目配置或 `run.sh` 的 profile 切换。
4. 用 `jq_research_system/jqrs/multi_factor_engine.py`（源仓，未提取）对接本项目 `data/factorlib/all_factors_library.json` 做样本外合成验证（见 `rd_agent_iteration_plan.md` Phase 4）。
