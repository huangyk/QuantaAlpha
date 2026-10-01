# QuantaAlpha 完整开发计划（三大核心任务）

> **关于"三个任务"的界定（请确认）**：本项目本体 = LLM 驱动因子挖掘 + Web 工作台；回测是挖掘流程内的"信号验证环"。据此，把"之前提到的三个任务"明确为以下三大任务：
>
> - **任务 A — 因子挖掘闭环**（原生 `factors/experiment` → `library`，项目本体）
> - **任务 B — Web 工作台**（三页吃真实数据，且 MiningDashboard 能展示"进行中路径"）
> - **任务 C — 回测验证**（信号验证环：多周期 / 定权 / 真实基准 / faithful 比较）
>
> 若你本意是别的三件具体事（如"方向① / Web / 原生挖掘"三件具体活），告诉我，我立刻按你的口径重排。

---

## 0. 当前已核实状态（写计划前已跑通/确认，非编造）

| 项 | 状态 | 证据 |
|---|---|---|
| 因子定义 | ✅ F1–F9 表达式已定义并 `expr_parser` 校验 | `data/factorlib/tdx_warfare2_factors.json`、`tdx_warfare146_factors.json` |
| 战法2 多周期回测 | ✅ 真实（cn_real_full） | IC=0.072 / ICIR=1.04 / 年化34% / 回撤-31% / IR=1.08（基准 SH600000 占位） |
| 方向① 单因子指标 | ✅ 7 因子真实指标已出 | `data/results/direction1/` |
| 后端 API | ✅ `:8000` 运行中，`/api/v1/factors` 返回 7 真实因子 | 本次 curl 实测 |
| 前端 | ✅ `:3000` 运行中（`npm install --ignore-scripts` 修好 esbuild EBUSY） | 本次实测 HTTP 200 + `/api` 代理通 |
| 原生挖掘入口 | ✅ 后端 `_run_mining` 真拉起 `python -m quantaalpha.cli mine` | `frontend-v2/backend/app.py:279` |
| 真实回测入口 | ✅ 后端 `_run_backtest` 真拉起回测子进程 | `frontend-v2/backend/app.py:898` |
| MiningDashboard | ✅ 已接 `miningTask`（进度/指标/日志/曲线），无任务显示空态 | `frontend-v2/src/pages/MiningDashboardPage.tsx` |
| BacktestPage | ✅ 已接 `/api/v1/backtest/start` + 结果展示 | `frontend-v2/src/pages/BacktestPage.tsx` |
| **缺口：因子入库来源** | ⚠️ 当前页面因子来自手写**桥接文件** `all_factors_library_tdx_warfare.json`，未走原生 `FactorLibraryManager.add_factors_from_experiment` | 本次排查结论 |
| **缺口：指数基准** | ⚠️ 本地 Qlib 无 SH000300/csi300 特征，benchmark 用 SH600000 占位 | `MEMORY.md` 数据节 |
| **缺口：M4/M5 非原生** | ⚠️ TopK 持有 ≠ 通达信"破 MA 即走"，收益结构不同 | `tdx_warfare_allocation_plan.md` |

---

## 任务 A — 因子挖掘闭环（原生 `factors/experiment` → `library`）

**目标**：用项目原生链路自动挖掘因子并写入统一因子库，让 Web 的 MiningDashboard 能展示"进行中 / 已进化"的因子路径；当前 F1–F9 仅以手写桥接文件入页面，本条要把它换成原生产出。

**优先级**：**P0（最高，项目本体）**

**具体实现步骤**：
1. **A1 接通原生入库契约** ✅ 已完成（2026-09-30）：F1–F9 已用 `FactorLibraryManager.add_factors_from_experiment` 原生入库，覆盖手写桥接文件，指标来自 direction① 真实回测。：验证 `quantaalpha/factors/library.py` 的 `FactorLibraryManager.add_factors_from_experiment()` 输入格式（从 `QlibFactorExperiment` 结果提取因子 + `backtest_results`）。先写一个最小驱动，把已定义的 F1–F9 经实验指标计算后入库，产出文件名 `all_factors_library_<suffix>.json`（后端 glob 认得）。
2. **A2 让 MiningDashboard 显示进化路径**：确认 mining 子进程经 WebSocket(`/ws`) 把 `round / 本轮提出因子 / IC 变化` 推给前端 `TaskContext.miningTask`；补齐字段映射，使页面展示"第 N 轮 → 新因子 → 指标提升"。
3. **A3 触发一次真实 LLM 自动挖掘**：配置 `.env` 的 `OPENAI_API_KEY / OPENAI_BASE_URL / REASONING_MODEL`，从 CLI 或页面发起一轮 `quantaalpha.cli mine`（即 rdagent `fin_factor`），观察新因子入库与进化路径展示。**只记录真实产出，不编造。**
4. **A4 用原生产出替换桥接文件**：确认页面因子库来自 A1/A3 的原生库后，停用 `reference/tdx_strategies/generate_library_json.py` 的桥接依赖，删除手写 `all_factors_library_tdx_warfare.json`（或标记 deprecated）。
5. **A5 regulator 一致性校验接入**：保证入库因子表达式通过 `expr_parser` 且指标结构合规（`quantaalpha/factors/regulator/`）。

**预期交付成果**：
- 原生挖掘跑通的可复现命令/脚本（CLI + 页面均可发起）。
- 统一因子库 JSON（真实因子，非桥接），后端 `/api/v1/factors/libraries` 可见。
- MiningDashboard 能展示至少一轮"进行中→完成"的进化路径（含指标）。
- 一次真实 LLM 挖掘的存档记录（因子 + 指标），写入 `data/results/` 或 memory。

---

## 任务 B — Web 工作台（展示进行中路径 + 真实因子库）

**目标**：三个页面全部吃真实数据；MiningDashboard 能展示"进行中路径"；提供一键启动与说明。

**优先级**：**P0 / P1**（数据真实 P0，进化可视化 P1）

**具体实现步骤**：
1. **B1 因子库页（FactorLibraryPage）**：A 完成后确认读原生库；质量分级（tier）展示；卡片 IC / RankIC / ICIR / RankICIR 真实（已验证后端返回这些键）。保留"后端不可达才 fallback mock"的兜底逻辑。
2. **B2 MiningDashboard 进化可视化**：接 A2 的进度推送；处理空态 / 失败态 / 取消态；确保"进行中路径"可见（当前代码已具备骨架，需联调真实 mining 任务填充）。
3. **B3 回测页联动**：BacktestPage 已可跑真实回测（`startBacktestTask`）；确认 MiningDashboard 的 `FactorStatsRow.onBacktest` 一键设 `quantaalpha_active_library` 跳回测页的联动可用。
4. **B4 系统配置页（SystemConfig）**：已通（读 `.env` + `experiment.yaml`，实测返回 15 个 env 键 + 6779 字符 yaml），仅做冒烟验证。
5. **B5 启动标准化**：写 `frontend-v2/start.sh`（后端 + 前端同起），**前端用 nvm node（`D:\nvm4w\nodejs\node.exe`，v24.9.0）跑，勿用 WorkBuddy 托管 node**（本次 `npm install` 的 EBUSY 即托管 node 目录被锁所致）。写 `README` 说明端口（前端 3000 / 后端 8000）与启动顺序。
6. **B6 断点说明进文档**：把"桥接 → 原生"切换写进代码注释或本计划，避免后人再踩。

**预期交付成果**：
- 三页（因子库 / 挖掘仪表盘 / 回测）全部真实数据，无 mock 兜底被误触发。
- MiningDashboard 展示进行中挖掘路径（进度 + 指标 + 日志 + 曲线）。
- `start.sh` + `README`：一键起后端+前端，明确用 nvm node。

---

## 任务 C — 回测验证（信号验证环）

**目标**：用真实回测验证挖掘出的因子，产出多周期指标、权重方案、真实基准、faithful 比较。回测不是目的，是"信号验证环"。

**优先级**：**C1/C2/C3 = P1；C4/C5 = P2；C6 = deferred（不阻塞前三任务）**

**具体实现步骤**：
1. **C1 方向①落地（战法1/4/6 横向比较）**：在 **同一 cn_real_full 多周期数据**上跑 F1/F6/F8/F9（战法1/4/6）回测，与战法2 基线横向比较 + 算战法间日收益相关性。脚本 `reference/tdx_strategies/` 已具备（compute_factor_panels / run_single_factor_backtests / compute_weights），跑通真实结果。
2. **C2 定权**：用多周期**真实**指标做风险平价 / 均值方差优化（compute_weights.py 已有 3 套方案：ICIR 归一 / Σ⁻¹·ICIR / 组合 IR），替换先验 35/25/20/20；产出权重文件 + 对比报告。
3. **C3 真实基准**：下载 SH000300 / csi300 指数特征到 Qlib 数据，替换 SH600000 占位；重算 IR 相对**市场**基准（当前 IR=1.08 是相对单只股票，解释力有限）。
4. **C4 faithful 买卖/止损**：在 jqrs 补 M4/M5（事件级止损 / 单笔仓位），与 QuantaAlpha TopK 做 faithful 收益结构比较，**诚实标注差异**（TopK≠"破 MA 即走"）。
5. **C5 分钟级 ETL**：TimescaleDB `bars_5m`（3140 万行）ETL 成 Qlib 5min，支撑战法6（弱转强）日内 faithful 回测。
6. **C6 战法3/5**：延后至 Phase 2（列入计划但标记 deferred，不阻塞 A/B/C）。

**预期交付成果**：
- 方向①横向比较报告（战法1/2/4/6 同数据对比 + 相关性矩阵）。
- 权重方案文件（≥1 套，附对比）+ 说明为何替换先验。
- 真实基准 IR（相对 SH000300）重算结果。
- （P2）faithful 比较报告 + 分钟级数据管线（可选，视时间）。

---

## 优先级总览与依赖

| 任务 | 优先级 | 阶段 | 依赖 | 说明 |
|---|---|---|---|---|
| A 因子挖掘闭环 | **P0** | Phase 1 | 无（本体） | 根，其余都建立在它之上 |
| B Web 工作台（数据真实） | **P0** | Phase 1 | A（需真实因子） | 展示 A 的产物 |
| B Web 工作台（进化可视化） | **P1** | Phase 2 | A2 | MiningDashboard 接真实 mining |
| C1 方向①横向比较 | **P1** | Phase 1-2 | A（因子） | 复用已写脚本跑真实 |
| C2 定权 | **P1** | Phase 2 | C1 | 用真实指标优化权重 |
| C3 真实基准 | **P1** | Phase 2 | C1 | 下载指数特征 |
| C4 faithful M4/M5 | **P2** | Phase 3 | A/B/C | jqrs 补齐，诚实比较 |
| C5 分钟级 ETL | **P2** | Phase 3 | C4（战法6） | TimescaleDB→Qlib 5min |
| C6 战法3/5 | deferred | Phase 2+ | — | 不阻塞前三任务 |

**执行顺序建议**：`A → B（数据真实，与 A 并行联调）→ C（验证 A/B 的产物）`。A 是根；B 展示 A 的产物；C 验证 A 的产物是否真有信号。

---

## 里程碑（可验收节点）

- **M1（P0，本周可达成）**：原生挖掘跑通 + 因子库页真实 + 三页真实数据 + 一键启动脚本。
- **M2（P1）**：MiningDashboard 进化路径展示 + 一次真实 LLM 挖掘 + 方向①横向比较 + 定权方案。
- **M3（P1）**：真实基准（SH000300）替换，IR 相对市场重算。
- **M4（P2）**：faithful M4/M5 比较 + 分钟级 ETL（战法6 日内）。
- **M5（deferred）**：战法3/5 落地。

---

## 风险与铁律（必须遵守）

1. **真实数据铁律**：所有 IC / 夏普 / 回撤 / 权重必须来自真实回测或真实数据，**禁止编造**；不会/算不出的地方直接打断任务，不硬编。
2. **M4/M5 非原生**：QuantaAlpha 的 TopK 持有 ≠ 通达信"破 MA 即走"，回测收益 ≠ 实战收益；faithful 比较须 jqrs 补齐并**诚实标注**结构差异。
3. **指数基准缺口**：本地无 SH000300 特征，benchmark 先用 SH600000 占位，C3 补真实；此前的 IR=1.08 是相对单只股票，非市场基准。
4. **train l2 == valid l2 退化**：标签经 CSRankNorm 后截面近常数，l2 回归几乎无信息；**以 IC 为准**，不依赖 l2 结论。
5. **EBUSY 文件锁**：前端 `npm install` 用 **nvm node**（`D:\nvm4w\nodejs\node.exe`），勿用 WorkBuddy 托管 node（22.22.2）；若 postinstall 再撞锁，用 `npm install --ignore-scripts` 后手动确认 `@esbuild/win32-x64` 到位。
6. **定时回顾**：每完成一个任务跑对应回测/验证，并把结论写入 `.workbuddy/memory/`，不积压。

---

## 执行进度更新（2026-09-30 当日实跑）

| 子项 | 状态 | 证据 / 说明 |
|---|---|---|
| A1 原生入库 | ✅ 完成 | `reference/tdx_strategies/native_ingest.py` 驱动 `FactorLibraryManager.add_factors_from_experiment` 写入 `data/factorlib/all_factors_library_tdx_warfare.json`（7 因子，指标来自 `direction1` 真实回测）。手写桥接 `generate_library_json.py` 已标记**弃用**。 |
| A2 前端去假数据 | ✅ 完成 | `TaskContext.tsx` 的 `pushMiningDataPoint` 原注入 `Math.random()` 曲线，已改为**只接受真实点**；且 `LiveCharts` 本就不渲染这些曲线（原为死代码）。 |
| A3 真实 LLM 挖掘 | ⚠️ **阻塞（平台限制，待决策）** | 修复两个真实 bug 后（见下）挖掘已能进入进化循环，但最终卡在 **rdagent `LocalEnv` 是 Unix-only**：`rdagent/utils/env.py:563` 用 `select.poll()`（Windows 无此 API），且入口命令是 `/bin/sh -c 'timeout --kill-after=10 600 python ...'`（Unix shell + GNU timeout）。报错 `AttributeError: module 'select' has no attribute 'poll'`。**WSL 本机被安全策略拉黑**（无法代跑）；**Docker 可用（v29.3.0，守护进程在跑）**但需构建 `local_qlib:latest` 镜像（本机**无此镜像**，需从 rdagent 自带 Dockerfile 构建，GB 级/需联网）。切换方式：`USE_LOCAL=false`（`factor_mining.py:568`）或配置 `execution.use_local: false`。 |
| B1 因子库页 / B3 回测页 / B4 配置页 | ✅ 完成 | 后端返回真实因子（IC/ICIR/年化/回撤非 0）；回测页可跑真实回测。 |
| B5 启动标准化 | ✅ 完成 | `frontend-v2/start.sh` 改为**前置 nvm node 目录**到 `PATH`（`D:\nvm4w\nodejs`，v24.9.0，npm 同目录），支持 `NVM_NODE_DIR` 覆盖；`README.md` 补"环境要求 + 启动顺序 + 端口(3000/8000)"。 |
| B6 桥接→原生说明 | ✅ 完成 | 见 A1 行 + 本节 + `generate_library_json.py` 弃用头注释。 |
| C1 方向①横向比较 | ✅ 完成（真实） | `data/results/direction1/direction1_report.html`：7 单因子 + 战法2 基线 + 相关性矩阵。**可比性**：单因子与 `warfare2_multiyear` 基线同为 `cn_real_full`、回测窗 `2025-01-02~2026-08-17`、基准 SH600000 → 同窗口同基准。旧 `backtest_warfare2`（2025 单年、另一窗口）**不可**与之横比。 |
| C2 定权 | ✅ 完成（真实） | `weights.json`（ICIR 归一）+ `weights_v2.json`（pipeline ICIR 三方案），均基于真实指标。 |
| C3 真实基准 SH000300 | ✅ **完成（真实数据）** | 从 TimescaleDB 导入沪深300（`scripts/build_index_benchmark.py`），见下方专节。 |
| C4 faithful M4/M5 | ⏳ P2 未开始 | 依赖 jqrs。 |
| C5 分钟级 ETL | ⏳ P2 未开始 | TimescaleDB `bars_5m` → Qlib 5min。 |
| C6 战法3/5 | ⏸ deferred | 按计划延后。 |

### C3 真实基准落地：沪深300 导入 + 重算 IR（结论重要）

**数据来源**：`D:/qqmt/code` 容器 `quant-timescaledb`（库 `quant_data`），表 `bars_1d` 有 `000300.SH｜沪深300｜index`。

**两个数据坑（已修，务必勿踩）**：
1. 该指数有**两批次共 2819 行**：`ts=15:00` 的 1245 行 `trade_date` 正确，但**区间内缺 330 个交易日**（且正好从回测起点 2025-01-02 起缺）；`ts=16:00` 的 1574 行 **`trade_date` 整体错位少 1 天**（含休市日 2020-01-01）。
   实测把 16:00 批次 +1 天后与 15:00 批次在 **1244 个共有日数值 100% 一致（冲突 0 天）** → **必须合并取并集**得 1575 天完整序列。**只取 15:00 批次是错的**（我第一版犯过，已纠正）。
2. 二进制格式（自家 `jq-qlib-native-provider-v1`）：`features/<小写代码>/<字段>.day.bin` = `[int32 header=0] + [N×float32]`，按 `calendars/day.txt`(1599) 逐行对齐；`vwap=amount/volume`、`factor=1.0`、`label_5d=close[t+5]/close[t]-1`（反推现有标的确认，差 0.000000）。**刻意不改 `instruments/all.txt`**，保持 5210 股票池、指数仅供 benchmark。

**落库结果**：`features/sh000300/`，覆盖 **1575/1599 (98.5%)**；缺失为 2026-07-16 之后的 24 天，**写 NaN 未做任何前向填充**。qlib 实读通过（2025-01-02 收 3820.40 → 2026-07-15 收 4786.78）。
**日历固有异常**：`cn_real_full` 日历缺 2026-07-06~2026-07-14（应 7 个交易日），2026-07-03 直接跳 07-15，对所有标的一致。

**对比回测（统一窗口 2025-01-02 ~ 2026-07-15，仅 benchmark 不同）**：

| 基准 | 年化超额 | **IR** | 最大回撤 | Calmar |
|---|---|---|---|---|
| **SH000300（真实沪深300）** | **15.60%** | **0.9978** | -14.85% | 1.05 |
| SH600000（占位，单只银行股） | 33.44% | 1.0943 | -26.77% | 1.25 |

**诚实结论（覆盖此前所有报告口径）**：
- 此前沿用的 **"年化 34% / IR 1.08" 是相对单只股票 SH600000 的结果，被严重高估**；换成真实市场基准后年化超额**近乎腰斩**（33.4%→15.6%），IR 由 1.094 降至 **0.998**。
- 原因：浦发银行在此期间跑输大盘，使策略看起来更强。**后续所有对外结论必须以 SH000300 口径为准**：超额年化 ≈15.6%、IR ≈1.0、超额最大回撤 ≈-14.9%。
- IR≈1.0 仍具可用性；超额回撤仅 -14.85%（远小于对标单股的 -26.77%），说明组合更贴近指数、偏离度更低。
- 产物：`data/results/backtest_bench_csi300/`、`data/results/backtest_bench_placeholder/`；配置 `configs/backtest_bench_csi300.yaml` / `backtest_bench_placeholder.yaml`（`run_warfare2_multiyear.py` 已支持 `BT_CONFIG` 环境变量指定配置）。

### 关键坑：Qlib `cn_data` 符号链接（A3 首次失败的真正原因）

- **现象**：后端 `_run_mining` 报 `[WinError 4390] 此文件或目录不是一个重分析点。: 'C:\Users\lenovo\.qlib\qlib_data\cn_data'`。
- **根因**：`~/.qlib/qlib_data/cn_data` 是**已存在的真实目录**（旧数据集），而原代码无条件调用 `os.readlink()`；对真实目录调用即抛 4390。即便有权限，`os.symlink` 也无法建在已存在的真实目录之上。
- **修复**（`frontend-v2/backend/app.py`）：新增 `_create_qlib_data_link()`，用 **Windows junction（`mklink /J`，无需管理员/开发者模式）**；若 `cn_data` 是真实目录则**改名备份**为 `cn_data.bak_<时间戳>`（可恢复，不删除）再建链接；任何失败降级为 warning 不中断挖掘。
- **附带修复**：`mklink /J` 在中文 Windows 输出 **GBK** 文本，原先 `text=True` 按 UTF-8 解码会崩 reader 线程（字节 `0xce`）；已加 `encoding="utf-8", errors="replace"`。
- **副作用（重要）**：`cn_data` 现指向 `cn_real_full`（2020-01-02~2026-08-17）。因此 `configs/backtest_warfare2.yaml`（`provider_uri: ~/.qlib/qlib_data/cn_data`，窗口 2025-01-27~2025-12-31）读到的已不是原旧数据集，若需复现旧结果须显式改 `provider_uri`。

---

*本计划基于 2026-09-30 已核实的项目状态撰写；凡涉及"将要做"的步骤均为计划项，非已发生结果。*
