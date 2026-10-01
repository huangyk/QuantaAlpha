# RD-Agent 使用说明（本机 Windows）

> 分支 `research/rd-agent-factor-mining` — 与 **多因子**（`jqrs/multi_factor_engine.py`）配套使用。  
> 项目 README：[../README.md](../README.md)

---

## 1. 快速上手

### 1.1 环境（两套，不要混）

| 环境 | 用途 |
|------|------|
| `.venv312`（Python 3.12） | Web 工作台、pytest、数据管道 |
| Conda `rd-agent`（Python 3.11） | Qlib、RD-Agent、LightGBM |

```powershell
conda env create -f environment-rd-agent.yml   # 首次
.\dev.ps1 setup-core                           # Web 日常环境
```

### 1.2 配置模型（Chat LLM）

**文件：`.env`**（复制自 `.env.example`，不进 Git）

| 变量 | 说明 | 示例 |
|------|------|------|
| `BACKEND` | RD-Agent 后端 | `rdagent.oai.backend.LiteLLMAPIBackend` |
| `CHAT_MODEL` | litellm 模型名 | `openai/agnes-2.5-flash` |
| `OPENAI_API_KEY` | API Key | （你的 Key） |
| `OPENAI_API_BASE` | OpenAI 兼容端点 | `https://apihub.agnes-ai.cn/v1` |

rdagent 0.8.0 的 LiteLLM 后端实际读 **`OPENAI_API_KEY` / `OPENAI_API_BASE`**。

### 1.3 配置 Embedding（不是独立向量库）

**没有 Milvus / Chroma。** RD-Agent 用 embedding 做知识去重，向量在进程内缓存。

**同样在 `.env`：**

| 变量 | 说明 |
|------|------|
| `EMBEDDING_MODEL` | `ollama/nomic-embed-text` |
| `OLLAMA_API_BASE` | `http://127.0.0.1:11434` |

Agnes **无 embedding 接口** → chat 走 Agnes，embedding 走本机 Ollama：

```powershell
ollama pull nomic-embed-text
# 保持 Ollama 服务运行
```

另需：

```ini
MODEL_COSTEER_ENV_TYPE=conda
MLFLOW_ALLOW_FILE_STORE=true
```

### 1.4 启动命令

```powershell
cd D:\qqmt\code

.\dev.ps1 help
.\dev.ps1 rd-health       # 检查模型 + Ollama（含 LLM 探测日志）
.\dev.ps1 rd-health llm   # 怀疑大模型有问题时：打开详细请求/响应日志
.\dev.ps1 qlib-profile    # 查看当前 Qlib 数据 profile（research / smoke）
.\dev.ps1 rd-factor 4     # 因子发掘（完整一轮约 4 步；上限 8）
.\dev.ps1 qlib-info
```

### 1.4.1 正式 Qlib 数据（必读：外接的是谁、各命令干什么）

**一句话：外接的是 Qlib 行情目录；RD-Agent 不读数据库，只读这份目录。**

把数据想成三层「成品」，越往右越接近 RD 能直接用：

```text
① 数据库 TimescaleDB          ← 真源（QMT 同步进来的）
        ↓ 偶尔导出
② staging（CSV 大包）         ← artifacts/cn_real_staging（给人/程序核对用）
        ↓ 「建 / 重建 provider」
③ Qlib provider（.bin 目录）  ← ~/.qlib/.../cn_real_r500（Qlib 官方格式）
        ↓ 「切 profile + 重生 h5」
④ RD 模板（yaml + daily_pv.h5）← RD 写因子、qrun 回测实际用的
```

| 你说的词 | 实际意思 | 什么时候做 |
|----------|----------|------------|
| **建 / 重建 provider** | 把 staging CSV **重新做成** Qlib 的 `calendars/instruments/features/*.bin` 目录。输出到空目录；目录已有内容要先删再建。 | DB/staging 更新了、要换股票数（500→全池）、或 provider 坏了 |
| **切 profile** | 改 `config/qlib_profile.json` 的开关：用 research 还是 smoke 那份目录 | 正式研究用 research；只测链路用 smoke |
| **重生 h5** | 按当前 profile，从 provider **重新生成** RD 写因子用的 `daily_pv_*.h5`，并改模板 yaml 里的路径/日期 | 刚建完 provider，或刚切了 profile |

**现在机器上已经建好 research，一般不用天天重建。** 你日常只要：

```powershell
.\dev.ps1 qlib-status     # 看数据到哪一步
.\dev.ps1 qlib-profile    # 确认是 research
.\dev.ps1 rd-factor 4     # 挖因子（完整一轮约 4 步）
.\dev.ps1 rd-status       # 看 RD 挖到哪一步
.\dev.ps1 rd-ui           # 浏览器看日志（19899）
```

### 数据更新了怎么跑

```powershell
# 常用：DB 已同步、staging 还在 → 一键重建 provider + h5（日志 log\qlib_refresh_*.log）
.\dev.ps1 qlib-refresh

# 连 staging 也要从库重导（慢）
.\dev.ps1 qlib-refresh --with-export --end 2026-08-20
```

等价手动三步（理解用）：

```powershell
Remove-Item -Recurse -Force C:\Users\lenovo\.qlib\qlib_data\cn_real_r500 -ErrorAction SilentlyContinue
.\dev.ps1 qlib-provider artifacts\cn_real_staging C:\Users\lenovo\.qlib\qlib_data\cn_real_r500 --max-instruments 500
.\dev.ps1 qlib-profile research
.\dev.ps1 qlib-regen-h5
.\dev.ps1 qlib-status
```

只有下面情况才需要「重建」整条链：

```powershell
# A. 数据库又更新了 → 先重新导出 staging（耗时长，不常用；也可用 refresh --with-export）
.venv312\Scripts\python.exe -m jq_research_system.qlib_adapter db `
  --output artifacts\cn_real_staging --start 2020-01-01 --end 2026-08-18

# B. 用新 staging 重建 Qlib 目录（接数据的核心一步）
#    注意：输出目录必须不存在或先删掉
Remove-Item -Recurse -Force C:\Users\lenovo\.qlib\qlib_data\cn_real_r500 -ErrorAction SilentlyContinue
.\dev.ps1 qlib-provider artifacts\cn_real_staging C:\Users\lenovo\.qlib\qlib_data\cn_real_r500 --max-instruments 500

# C. 告诉 RD「用这份 research」，并重生 h5
.\dev.ps1 qlib-profile research
.\dev.ps1 qlib-regen-h5
```

| 命令 | 是不是「接数据」 | 说明 |
|------|------------------|------|
| `qlib-provider` | **是** | 只做 Qlib 目录，不训练模型 |
| `qlib-profile` | 否 | 开关：research / smoke |
| `qlib-regen-h5` | 否 | 给 RD 准备 h5 + 改模板 |
| `qlib-baseline` | **不是** | Alpha158+LightGBM 训练，别当接数据用 |
| `rd-factor` | 否 | 在**已接好的**数据上挖因子 |

| Profile | 目录 | 用途 |
|---------|------|------|
| `research`（默认） | `C:\Users\lenovo\.qlib\qlib_data\cn_real_r500` | 正式研究：500 只、2020–2026 |
| `smoke` | `C:\Users\lenovo\.qlib\qlib_data\cn_data_v2` | 链路冒烟：100 只、主要 2025 |

临时覆盖：`$env:RD_QLIB_PROFILE='smoke'`（或 `research`）。

**诚实边界：** `$factor` 目前是常数 1（未做真复权）；500 只是抽样不是全市场。  
**PortAna 注意：** `backtest` 结束日必须 **早于** 日历最后一天（research 现用 `2026-08-17`），否则 Qlib 会 `IndexError` → 收益显示 NaN。基准须在 provider 内（research 用 `SH600000`，指数 `SH000300` 需另导入）。

#### 更新了行情数据，怎么跑？

| 场景 | 命令 |
|------|------|
| **常用：DB 已同步，staging 也要跟着更新** | `.\dev.ps1 qlib-refresh --with-export` |
| **staging 还是新的，只重建 provider+h5** | `.\dev.ps1 qlib-refresh` |
| 手动分步（等同 refresh） | 见下方「手动四步」 |

日志文件：`log\qlib_refresh_YYYYMMDD_HHMMSS.log`

#### 怎么知道数据到哪一步了？

```powershell
.\dev.ps1 qlib-status
```

会打印 `[1/4] staging → [2/4] provider → [3/4] h5 → [4/4] 可跑 rd-factor`，每层 `OK` 或 `缺`。

| 层 | 含义 | 对应命令 |
|----|------|----------|
| 1 staging | CSV 大包在不在 | `qlib_adapter db` / refresh `--with-export` |
| 2 provider | Qlib `.bin` 目录 | `qlib-provider` / refresh |
| 3 h5 | RD 写因子用的表 | `qlib-regen-h5` |
| 4 可挖因子 | 前三层都 OK | `rd-factor`（**不是**接数据） |

`qlib-baseline` 不在这条链里，那是另开的 Alpha158 训练。

#### 自动化 & 观测

| 目的 | 命令 |
|------|------|
| 一键刷新数据 | `.\dev.ps1 qlib-refresh` 或加 `--with-export` |
| 看数据层进度 | `.\dev.ps1 qlib-status` |
| 看挖因子进度 | `.\dev.ps1 rd-status` |
| 看 RD 日志 UI | `.\dev.ps1 rd-ui` → http://127.0.0.1:19899 |
| refresh 文本日志 | `log\qlib_refresh_*.log` |
| RD 会话日志 | `log\<时间戳>\` |

LLM 相关日志：

| 文件 | 内容 |
|------|------|
| `log/rd_health_llm_probe.log` | Chat/Embedding 请求、耗时、回复预览、异常栈（API Key 已脱敏） |
| `log/rd_health_latest.log` | 官方 `health_check` 英文原文；`rd-health llm` 时含 LiteLLM DEBUG |

也可单独跑探测：

```powershell
conda run -n rd-agent --no-capture-output python tools\rd_health_llm_probe.py --debug
```

`rd-health` / `rd-factor` 会自动：检查 Ollama → 设置 `PYTHONPATH=tools` → 在 Conda `rd-agent` 执行。

### 1.5 把自己的策略喂给 Research

复制 `config/rd_research_prompt.md.example` 为 `config/rd_research_prompt.md`，用条目写下因子方向（中英文均可）。  
`.\dev.ps1 rd-factor` 会把它追加进假设生成的 **system**（`hypothesis_specification`）和 **user**（`RAG`）。不改 site-packages。

写成可计算的因子，不要写仓位/止损。之前 LLM 自己跑出 5/10/20 日动量，就是因为没有这份约束。可选环境变量 `RD_RESEARCH_PROMPT` 指向别的文件。素材可从 `jq_research_system/01_核心买法战法.md` 粘要点。

rdagent 0.8 的 Qlib 因子循环**没有**官方「丢一个知识文件夹」的钩子：`RAG` 是硬编码字符串；`APP_TPL` 只能覆盖 site-packages 里的 yaml。`fin_factor_report` 是另一条读研报的回路，不是这条 Research。

---

## 2. 目录说明

```
D:\qqmt\code\
├── dev.ps1                 # 唯一正式入口
├── .env / .env.example     # 模型 + Embedding
├── environment-rd-agent.yml
├── tools/                  # Windows 补丁（自动加载）
├── config/rd_research_prompt.md  # 自定义 Research 方向（gitignore；模板为 .example）
├── artifacts/              # 本地输出，见 artifacts/README.md
├── git_ignore_folder/RD-Agent_workspace/<uuid>/   # 实验产物
│   ├── factor.py / result.h5 / qlib_res.csv / ret.pkl / mlruns/
└── jq_research_system/config/
    ├── db_config.json      # TimescaleDB（Web / 数据管道，与 RD-Agent 无关）
    └── qmt_config.json     # QMT
```

Qlib 数据（由 `config/qlib_profile.json` 切换）：

- research：`C:\Users\lenovo\.qlib\qlib_data\cn_real_r500`
- smoke：`C:\Users\lenovo\.qlib\qlib_data\cn_data_v2`

---

## 3. fin_factor 流程

```
Loop N:
  direct_exp_gen  → LLM 提假设
  coding          → 写 factor.py
  running         → qrun 回测 + read_exp_res
  feedback        → 判 True/False，进入下一轮
```

`--step-n 3` = **一次命令最多 3 个 step**，不是 3 个完整 loop。

---

## 4. Windows 补丁（`tools/`）

通过 `sitecustomize.py` 自动加载，无需手动 import。

| 文件 | 作用 |
|------|------|
| `json_parse_patch.py` | Agnes JSON 解析（空内容、围栏、Python bool） |
| `factor_datetime_patch.py` | 因子 index 字符串 → datetime |
| `windows_qlib_patch.py` | Windows 下 qrun/python 路径、跳过 `/bin/sh` |
| `read_exp_res_fixed.py` | 读 qlib recorder；优先带 portfolio/IC 的结果 |
| `gym_numpy_patch.py` | gym → gymnasium（NumPy2） |
| `safe_ic_patch.py` | 预测近常数时 IC 不再 Windows 硬崩 |
| `research_prompt_patch.py` | 注入 `config/rd_research_prompt.md` 到 Research 假设生成 |
| `qlib_data_profile_patch.py` | 按 `config/qlib_profile.json` 对齐 factor 模板 yaml / `generate.py` |

### read_exp_res 常见问题

| 现象 | 原因 | 修复 |
|------|------|------|
| `No recorders found` | recorder 仍在 RUNNING，无 `end_time` | `read_exp_res_fixed.py` 按 `start_time` 选取 |
| `No result file found` | RD-Agent 要求 workspace 有 `ret.pkl` | portfolio 未就绪时写 stub `ret.pkl` |
| MLflow 对不上 | qrun 写 `./mlruns`，read 读 sqlite | rd-agent 环境 qlib 对齐 file store（见第 5 节） |

---

## 5. 历史问题（2026-07 ~ 2026-08）

| 问题 | 处理 |
|------|------|
| JSON 解析失败 | `json_parse_patch.py` |
| `str - str` datetime | `factor_datetime_patch.py` |
| Windows `/bin/sh`、`select.poll` | `windows_qlib_patch.py` + `MODEL_COSTEER_ENV_TYPE=conda` |
| `qrun` 找不到 | patch 使用 rd-agent 绝对路径 |
| MLflow store 不一致 | qrun 与 read_exp_res 都指向 workspace 下 `./mlruns` |
| `conf_combined_factors.yaml` 用 csi300 | 已由 `qlib_data_profile_patch` 按 `research`/`smoke` 对齐 |

**诚实边界：** research 默认 `cn_real_r500`（500 只稳定抽样，2020–2026）；`$factor=constant_1` 非真复权；全池 / survivorship / walk-forward 未完成前，回测指标仅供研究迭代，非可投产结论。smoke 小样仅用于链路验证。

---

## 6. 排查

```powershell
.\dev.ps1 rd-health

dir git_ignore_folder\RD-Agent_workspace\*\qlib_res.csv

# 手动读某个 workspace 的结果
cd git_ignore_folder\RD-Agent_workspace\<uuid>
$env:PYTHONPATH="D:\qqmt\code\tools"
conda run -n rd-agent python D:\qqmt\code\tools\read_exp_res_fixed.py
```

日志里若出现 `Failed to run ... because [json-parse-patch] applied`，是 patch 的 stdout 打印被误当成错误信息，**不代表 patch 未生效**。

---

## 7. 多因子（本分支第二主线）

代码：`jq_research_system/jqrs/multi_factor_engine.py`

- IC / IR、分组回测、因子相关性、组合打分
- 与 RD-Agent 产出的因子对接，做样本外验证与合成（见 `rd_agent_iteration_plan.md` Phase 4）

---

## 8. 相关文档

| 文档 | 内容 |
|------|------|
| 本文 | RD-Agent 配置、启动、补丁、排错 |
| [RD_AGENT_LEARNING_SUMMARY.md](./RD_AGENT_LEARNING_SUMMARY.md) | 学习笔记、数据架构、真数据方案、进度待办 |
| [README.md](../README.md) | 项目总览 |
