# QuantaAlpha 项目交接说明（HANDOFF）

> **用途**：把项目的长期记忆 + 数据资产 + 结论 + 操作，集中成一份可移交文档。
> 新人/接手人**先读本文件**，再按「文档索引」深入。
> 快照截止：**2026-10-13**。活跃长期记忆在 `.workbuddy/memory/MEMORY.md`（随迭代更新）；本文件是交接时点快照。
> 数据源铁律：**仅 TimescaleDB `bars_1d`/`bars_5m` 真实价，禁默认 akshare**（用户硬规则，QMT 取不到须先告知用户）。

---

## 0. 一句话现状

本项目反复验证后，**结论是：股票单资产的横截面选股/择时 alpha 基本全军覆没；唯一稳健存活的是「跨资产 ETF 轮动」（beta，年化~10%/夏普~1.03/回撤−9.4%）**。
股票侧唯一的新增量是**低吸战法 F4（回踩MA20）**——全 A 事件驱动回测显示**真实但温和、regime-dependent 的择时边（+22.8bp/笔，须带 ≤8% 止损）**，可作入场触发器/小仓位试，**样本外(2025–26)因数据缺口未验证**。

---

## 1. 用户真实目标（评判一切战法的标尺，2026-10-10 明确）

- 用户**不是要发财、也不是追市场 alpha**；核心诉求 = 「找到一个**能稳定赚钱**的策略」。
- 期望标尺：一个月 10% 就「很了不起」（用户自认高标杆）；**底线是「只要能赚」**——持续正收益 + 回撤可控，比跑赢指数更重要。
- 诚实告知：本项目已验证方法年化约 **10%**（低回撤）；**月化 10% 复利≈214%/年，任何回测都未出现**，属不现实标杆。把「稳定小赚、回撤可控」当可达目标。
- ⚠ 不因用户想「赚」就放松铁律：任何策略仍须真实回测 + 防过拟合三重测（walk-forward / 参数成簇 / 成本敏感），不得编造、不得用 akshare 替代 QMT。

---

## 2. 数据资产清单（实测，2026-10-13）

### 2.1 TimescaleDB `quant_data`（127.0.0.1:5432 / quant / quant123456，键 `database` 非 `dbname`）
| 表 | 行数 | 说明 |
|---|---|---|
| **bars_1d** | 12,364,650 | 日线主表，2011–2026 / 5364 只；含 open/high/low/close/volume/amount/turnover_rate/float_shares/capital/total_shares |
| bars_5m | 41,442,358 | 5 分钟线（2026 起为主） |
| bars_1m | 49,792,132 | 1 分钟线（已决定跳过挖因子，保留） |
| asset_universe_daily | 8,277,583 | 时点股票池（按 list/delist date，无幸存者偏差） |
| corporate_action_factor | 6,646,207 | 复权因子 |
| bar_missing | 44,357 | 缺失记录 |
| asset_master | 5,256 | 标的主表（含 is_st 当前标记） |
| daily_feature_snapshot | 21,766 | 日特征快照 |
| intraday_research_feature | 513,069 | 日内研究特征 |
| jq_backtest_*（run/trade/order/daily_equity/metric/log） | 194/95961/95984/182629/186/1138537 | jqrs 回测产物（另一项目 jq_research_system） |
| **fundamentals_daily** | **0** | ⚠️ 无 PE/PB/市值基本面（数据缺口） |
| **asset_status_history** | **0** | ⚠️ 无 ST 历史（仅当前标记，回测近似） |
| sector_membership | ~36 万 | 行业成分（可用，来自早前记录） |

> **数据完整性陷阱（关键）**：`bars_1d` 名义 2011–2026，但**实际可交易历史 = 2020–2026**——2011–2019 过稀/断裂（事件驱动 T+1 逻辑零成交），2025–2026 回填未完（稀疏）。故任何「全 A 16 年」结论须先查数据密度；当前股票因子结论有效期 **2020–2024**。

### 2.2 qlib 数据集
- `cn_real_full`：2020-01~2026-08，全 A ~5210 只，日频 8 字段（**无市值/PE/行业**）。基准 SH000300 已补齐。
- QLIB_DATA_DIR 见 `docs/environment_setup.md`。

### 2.3 关键结果/数据文件（可复现查证）
| 文件 | 内容 |
|---|---|
| `tools/_dixi_fullpanel_out.json` + `tools/_dixi_fullpanel_run.log` | **低吸全 A 事件驱动回测**（F4/F5/F6/F7 + 随机对照 + 分年） |
| `tools/_lowvol_true_neutral_out.json` | 低波动真中性化 IC/组合 |
| `tools/_neutral_test_out.json` | 低波动市值中性（旧） |
| `tools/_tdx_feilong_out.json` / `_tdx_feilong_chip_out.json` | 飞龙验证 |
| `data/results/eval_1d.csv` | 19 个日线因子全 A 面板 IC/夏普（0/19 正） |
| `data/results/candidate_factors_*.csv` | 候选因子网格/分位/逐年 |
| `data/results/dixi_path_results.csv` / `dixi_daily_verdict.csv` | 低吸路径回测 |
| `data/results/blend_frontier.csv` / `equity_curve_m*.csv` | ETF 混合前沿/净值曲线 |
| `data/results/current_holdings.csv` | ETF 战法当前持仓 |
| `data/results/_all_a_panel.parquet` / `_feilong_panel_v2.parquet` | 全 A / 飞龙面板缓存 |
> `data/` 被 .gitignore（数据不入库，落盘备查）。`tools/_*.json` 结果文件**未提交**（本地产物，交接时需附带或重跑）。

---

## 3. 环境与运行

- **主力环境 `quantaalpha`（Python 3.11）**：挖因子(rdagent 0.8.0)+回测+后端全在这。含 qlib 0.9.7 / lightgbm 4.7.0 / torch 2.5.1 / xgboost 2.1.4 / psycopg2 2.9.13 / scipy / numpy 1.26.4。路径 `D:\Users\lenovo\miniconda3\envs\quantaalpha\python.exe`。
- `qmt312` = QMT 312 客户端 API 环境（xtdata 取数），numpy=2.4 **不可与 quantaalpha 混用**。
- **沙箱注意**：xtdata 取数接口（get_local_data/get_market_data）会 segfault，本沙箱仅 connect()/download 可用；**取数须在用户 QMT 机或 datahub 侧完成**。DB(127.0.0.1:5432) 沙箱**可连**。
- 跑回测示例：`conda activate quantaalpha && python tools/dixi_fullpanel_backtest.py --start 2011-01-04 --end 2026-09-30`
- Web 工作台：后端 `quantaalpha` 跑 `frontend-v2/backend/app.py`→:8000；前端 `cd frontend-v2 && npm run dev`→:3000（nvm node v24.9.0）。健康接口 `/api/health`。UI 约定见 `docs/ui_audit_20261009.md`（深灰+蓝紫、左侧 fixed 侧栏、改深色须整套替换 `.glass`）。
- 本机 QMT：`D:\中航证券QMT实盘-交易端\`（内置 Python 3.6.8，只认 GBK；策略只能手动粘贴，见下）。

---

## 4. 战法/策略总状态（详见 `docs/warfares_catalog.md`）

图例：✅ 可用 / ⚠️ 条件可用 / ❌ 证伪。

| 类别 | 战法/策略 | 状态 | 关键证据 |
|---|---|---|---|
| 通达信 | **战法2 回踩MA20** | ✅ 真实择时边 | +22.8bp/笔(t+14.6), 比随机+19bp, 须≤8%止损, regime-dependent |
| 通达信 | **战法5 龙回头** | ✅ 真实边(稀有) | F7 +19.5bp/笔(t+2.1), 950笔/年 |
| 通达信 | 战法1 突破 / 战法3 首阴 / 战法4 黄金坑 / 战法6 弱转强 | ❌ 证伪 | IC≈0 / 比随机差 / 无择时 / IC≈0 |
| 跨资产 | **ETF 跨资产轮动(absmom)** | ✅ **唯一主策略** | 底盘满仓 ~10%/1.03/−9.4%；60%档 ~5.5%/−4.8% |
| 股票 | 低波动异象 | ⚠️ 真信号不可变现 | IC t+7.35 但多空/单多全负；仅防御过滤器 |
| 股票 | 个股趋势 / 市场择时 / ML择时 / 飞龙 | ❌ 全线证伪 | 见 `docs/stock_verdict.md` |

**跨资产 ETF 轮动（七星高照）要点**：34 只干净跨资产池、2019–2026、万1；池等权买入持有 +9.3%/夏普0.65/回撤−22.2%（同口径最优）；**网上 32% 无法复现**；LB=25 是过拟合；+9.3% 是窗口 beta，勿加杠杆。部署 = QMT 手动粘贴 `docs/absmom_qmt_paste.txt`（用户明确：只更新那一个 txt，勿动 QMT 目录/部署脚本）。

---

## 5. 验证铁律（写代码/回测前必查，反复踩坑）

1. **信号-收益对齐**：T日收盘出分→收益从 T+1 起（买T+1、卖T+1+h）；`close.shift(-h)`=未来函数。IC 高得离谱先查对齐。
2. **基准同口径**：同池等权；沪深300 只能作环境参照（管中小盘池方向会反）。
3. **判 alpha 看分位组单调性 + 市值中性**，不只看 top-K；分位用 `rank(ascending=False)`。
4. **低吸/买回调必须带 ≤8% 止损**（否则统计上自杀）；加了止损期望才稳健正。
5. 剔次新股(<250日)、剔涨停买不进(open/昨收≥1.098)；停牌 NaN 先 `ffill(limit=5)` 再算指标，收益/成交用原始 OHLC。
6. 超额用**日超额复利**；成本含市场冲击（Almgren）；成交量单位=手（`成交额=close×volume×100`，优先用 amount）。
7. **防过拟合三重测**：walk-forward 隔离 / 参数成簇 / 成本敏感；在 train 内选参，禁 test 期网格回挑。
8. 池构建 `.index[:N]` 前必须已 `sort_values`（字母序=随机池，曾致结论翻转）。

---

## 6. 长期记忆全文（镜像自 `.workbuddy/memory/MEMORY.md`）

### 定位与数据
LLM 驱动 Alpha 因子挖掘（rdagent 0.8.0）。架构：TimescaleDB 底座 → ETL → qlib(离线研究) → 策略 → QMT/Ptrade(仅实盘增量)。（数据细节见 §2）

### QMT datadir 直读（破局：不用 akshare、绕开 xtquant segfault）
本机 QMT `D:/中航证券QMT实盘-交易端/datadir`（API 崩只因客户端没跑，本地文件可直接读）。
- `{SH,SZ}/86400/<6位>.DAT` = 8B头 + 64B记录(16×int32 LE)：`[0]Unix秒 [1..4]OHLC(×1000) [6]量(手) [8]额(元)`；300=5分钟。
- 脚本 `tools/backfill_etf_qmt_local.py --clean-legacy`。坑：`bars_1d` 唯一键 `(asset_code,ts)` 非 trade_date，ts 须北京午夜；datadir 未复权→份额拆分假暴跌，用收盘判据+整数比例吸附修；旧 qmt ETF 数据损坏(159915/159949/159922 缺2025)→踢出池。

### 网上 QMT 策略参考（`docs/qmt_strategy_references.md`）
首推 `guoyaohua/etf-adaptive-rotation-qmt`：逆波动率权重 + 波动率上限10% + 四周错峰 A/B/C/D sleeve + 相关性去重 + ATR止损 + 前缀不变性测试；自报 CAGR 5.83%/回撤 6.46%/夏普 1.004（诚实）。**其低回撤来自组合层/风控层(L4)而非信号层**。判据：任何策略先问 ①滚动窗口最差值 ②2×/3×成本压力 ③防未来数据验证，给不出就不信。

### 关键踩坑（血泪，务必保留）
- **19 强因子实为窄样本产物**：5min CoSTEER 在窄样本 `|t|≥2 且三段同号` 门槛挖出，放到全 A 面板 `eval_1d.csv` **0/19 正 RankIC、0/19 正夏普**。判定因子一律以全 A 面板为准。
- **低波动 decile 方向标错**：旧 `factor_neutral_test.py` 用 `pd.qcut(sig_neu,10)`，decile0=最高波动，却 `long=dec0` → 实为做多高波动/做空低波动（小盘投机溢价），误标"低波动多空 +27.3%"。正确方向多空为 **−29.7%**。低波动 = 真实 rank 信号但均值收益不可变现（厚尾背离），仅防御用。
- **低吸旧终审"≈0"是窄样本+基准带止损的产物**：liquid500(415只,2021-2026)+基准也走止损/止盈(剥离机制收益) → 只剩≈+4bp/日择时边,小样本 t<2 被否。全 A 事件驱动修订为"真实择时边"。
- **QMT 回测"只买不卖"**：回测无 account → holdings 恒空 → 恒全额买入 → 资金烧穿。任何"按差额调仓"策略，回测/无账号场景必须自建持仓台账。
- **QMT `is_last_bar()` 回测恒 False**：会把回测全拦掉；须区分 `do_back_test`。月末判定用"跨月即上月末"，勿依赖下一根 bar。
- **编码**：给机器读的文件按机器要求编码（QMT 要 GBK），给人看的用 UTF-8+BOM，分两份交付，别混。

### 优化方向（破局）
补行业分类+市值+涨跌停/ST 做真中性化；换成熟成分股池；优先做执行层降换手。任何新因子须先显著跑赢同池等权。

---

## 7. 文档索引（`docs/`，按用途分组）

**核心结论/成绩**
- `warfares_catalog.md` — **战法总册**（长期记忆入口，信号定义+状态+证据）
- `factor_scorecard.md` — 因子成绩表（19/10因子、传统波动族、低波动§8、战法§4）
- `stock_verdict.md` — 股票三路线(趋势/择时/ML)全线证伪
- `dixi_fullpanel_verdict.md` — 低吸全 A 终审（F4 真实择时边）
- `dixi_final_verdict.md` — 低吸旧终审（liquid500，已被上修订）
- `lowvol_true_neutral_verdict.md` — 低波动真中性化终审（含方向勘误）
- `feilong_v2_verdict.md` / `tdx_feilong_validation.md` / `tdx_feilong_chip_validation.md` / `feilong_market_gate.md` — 飞龙系列（证伪）
- `five_min_realcost_verdict.md` — 5min 因子真成本终审
- `qixing_etf_verdict.md` / `etf_portfolio_verdict.md` — ETF 轮动/组合终审
- `candidate_factors_verdict.md` — 短期反转+量能退潮终审
- `verification_2011_2026.md` / `leakage_analysis.md` / `pool_experiment_verdict.md` / `warfare2_truth_after_datafix.md` — 验证/对齐/池实验
- `roadmap_stock_weekly.md` — 股票周频路线图

**策略与部署**
- `strategy_absmom.md` — ETF 轮动策略文档（含 60% 低回撤档）
- `strategy_card.html` / `trading_plan*.html` / `blend_frontier.html` / `ml_timing_report.html` / `etf_vs_stock_backtest.html` / `backtest_report.html` — 各类报告
- `absmom_qmt_paste.txt` — **QMT 粘贴代码（唯一交付形态）**
- `qmt_deploy_guide.md` / `ops_checklist.md` — 部署/实盘清单
- `qmt_strategy_references.md` — 网上 QMT 策略参考

**环境/架构/流程**
- `environment_setup.md` — 环境（Python 3.11 主力）
- `data_architecture.md` / `PROJECT_STRUCTURE.md` — 数据/项目结构
- `development_plan.md` / `optimization_roadmap.md` / `experiment_guide.md` / `experiment_hyperparameters.md` — 计划/实验
- `user_guide.md` / `daily_trade_plan_system.md` / `ui_audit_20261009.md` / `final_summary_20261008.md` — 指南/UI/总结

---

## 8. 待办 / 数据缺口

- [ ] **补齐 2025–2026 `bars_1d` 回填** → 重跑 `dixi_fullpanel_backtest.py` 验证 F4/F7 最近样本外（当前 OOS 未验证，结论有效期 2020–2024）
- [ ] 补行业分类+市值+涨跌停/ST 历史 → 做真中性化、真停牌/涨跌停模拟
- [ ] F10 板块强需外部板块资金数据（L2 主力流缺失）
- [ ] 若落地 F4：做「每日 F4 前 N 只」小组合验证容量/集中度
- [ ] 用户 QMT 机跑 `absmom_qmt_paste.txt` 回测（沙箱起不了 QMT），按铁律重跑确认
- [ ] `tools/_*.json` 结果文件未入库，交接时附带或让接手人重跑脚本复现

---

## 9. 怎么复现关键结论（命令）

```bash
conda activate quantaalpha   # 或 D:\Users\lenovo\miniconda3\envs\quantaalpha\python.exe
# 低吸全 A 事件驱动（F4/F5/F6/F7 + 随机对照 + 分年）
python tools/dixi_fullpanel_backtest.py --start 2011-01-04 --end 2026-09-30
# 低波动真中性化
python tools/factor_lowvol_true_neutral.py
# ETF 混合前沿/杠杆/vol-targeting/bootstrap
python -m tools.index_lab <cmd>
# 19 因子全 A 面板评估（旧，结论 0/19 正）
#   见 data/results/eval_1d.csv
```

---

*免责声明：所有信号均须实盘前经历史回测验证，本文仅作交接/研究归档，不构成投资建议。*
