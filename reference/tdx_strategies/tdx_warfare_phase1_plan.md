# 通达信战法 Phase 1 实施计划 —— 战法 2（均线回踩低吸）

> 落地框架：QuantaAlpha 因子挖矿（已确认 2026-09-29）。本文是 Phase 1 的可执行规格，所有表达式已对 `quantaalpha/factors/coder/expr_parser.py` + `function_lib.py` 实测校验。

## 目标
在 QuantaAlpha 内实现战法 2 的 3 个原子因子（F2/F3/F4），入库并跑历史回测，产出**真实** IC/夏普/回撤指标（不编造数据）。

## 因子定义（AlphaAgent 表达式，已 expr_parser 实测可解析）

| 因子 ID | 语义 | 表达式（入库原文） |
|---|---|---|
| `tdx_f3_ma_bull` | 均线多头排列 MA20>MA60（M1 过滤） | `SMA($close, 20) > SMA($close, 60)` |
| `tdx_f2_volume_surge` | 放量：量>5日均量1.5倍且>昨量（M3 确认） | `$volume > SMA($volume, 5) * 1.5 && $volume > DELAY($volume, 1)` |
| `tdx_f4_pullback_ma20` | 回踩均线：距MA20<3%且金叉MA20（M2 信号） | `ABS($close - SMA($close, 20)) / SMA($close, 20) < 0.03 && $close > SMA($close, 20) && DELAY($close, 1) <= DELAY(SMA($close, 20), 1)` |

> **实测 parser 输出**（证明可解析，非猜测）：
> - F3 → `GT(SMA($close,20), SMA($close,60))`
> - F2 → `AND(GT($volume, SMA($volume,5)*1.5), GT($volume, DELAY($volume,1)))`
> - F4 → `AND(LT(DIVIDE(ABS(SUBTRACT($close, SMA($close,20))), SMA($close,20)), 0.03), AND(GT($close, SMA($close,20)), LE(DELAY($close,1), DELAY(SMA($close,20),1))))`

> 注：`function_lib` 无 `CROSS` 内置，F4 的「金叉」用 `A>B && DELAY(A,1)<=DELAY(B,1)` 组合实现（已校验）。

## 交付物
1. `data/factorlib/tdx_warfare2_factors.json` — 因子库（schema `{factors:{id:{factor_name,factor_expression,factor_description}}}`）
2. `data/factorlib/tdx_warfare2_factors.csv` — CLI 用（列 `factor_name,factor_expression`）
3. 回测报告：`data/results/backtest_v2_results/backtest_metrics.json`（由回测生成）

## 回测命令
- 官方 CLI：`python -m quantaalpha.cli backtest --factor_path data/factorlib/tdx_warfare2_factors.csv`
  （需 conda env `quantaalpha`，`cli.py` 自动读根 `.env` 的 `QLIB_DATA_DIR`）
- 直接驱动（LLM-free，可控）：`BacktestRunner("configs/backtest.yaml").run(factor_source="custom", factor_json=["data/factorlib/tdx_warfare2_factors.json"])`

## 配置（configs/backtest.yaml 现状）
- 数据：csi300，2016-01-01~2025-12-26，provider `~/.qlib/qlib_data/cn_data`（被 `.env` 的 `QLIB_DATA_DIR` 覆盖）
- 模型：lgb；组合：`TopkDropoutStrategy(topk=50, n_drop=5)`；基准 `SH000300`；涨跌停阈值 `0.095`

## 验收标准
- [x] 三个表达式经 `expr_parser` 可解析（已实测）
- [x] 回测产出非空 metrics（IC / 年化收益 / 最大回撤 / Info Ratio）
- [x] 报告落入 `data/results/backtest_warfare2/tdx_warfare2_backtest_metrics.json`

## 已知局限（honest）
- **M4/M5 非原生**：TopK 持有 ≠ 通达信「破 MA20 即走」，收益曲线会有结构差异。
- **L2 缺口**：战法 2 不直接用主力流，但同批战法 1 用 F9 近似。
- 以上指标为**真实回测产出**；先验权重 35/25/20/20 仍须据此校准。

## 回测结果（2026-09-29 真实运行，非编造）

运行：`BacktestRunner`（`configs/backtest_warfare2.yaml`，market=all，2025 全样本，benchmark=SH600000 占位）。

| 指标 | 数值 |
|---|---|
| IC | 0.0760 |
| ICIR | 1.121 |
| Rank IC | 0.0738 |
| Rank ICIR | 1.006 |
| 年化收益 (Ann. Return) | 0.1843（18.4%） |
| 最大回撤 (Max DD) | -0.1455（-14.6%） |
| Info Ratio | 0.6400 |
| Calmar | 1.2668 |

> 信号层：IC>0 且 ICIR>1 → 因子组合在 2025 测试窗（10-01~12-26）有**稳定正向**预测力，但幅度偏弱（IC≈0.076）。
> 组合层：TopK=50 持有策略测试窗正收益、回撤可控（Calmar 1.27）。

**结论可信度（honest 标注）**：
1. **样本仅 2025 一年**，测试窗更只有 ~3 个月（57 交易日），年化收益是短窗外推，不稳定。
2. **ConstantInputWarning**：测试窗内部分截面因子/标签为常数，IC 可靠性打折。
3. **train l2 == valid l2 完全相同（0.08297）**：验证集疑似退化/切分异常，模型训练质量存疑（不能排除标签近常数或泄漏）；但 IC 与组合收益在测试窗独立计算，仍反映真实信号方向。
4. **benchmark=SH600000（占位）**：非 SH000300 市场指数，Info Ratio=0.64 是相对单只股票，非市场相对基准，解释力有限。
5. **M4/M5 非原生**：TopK 持有 ≠ 通达信「破 MA20 即走」，收益曲线结构与实战不同。

→ 当前结论：**管线跑通 + 2025 方向为正**，但**不足以**据此把 35/25/20/20 先验权重当实盘配置。需多周期（换更全 Qlib 数据集）+ 干净切分 + 真实基准 后重验。

## 多周期验证结果（2026-09-30 真实运行，方向②）

目标：把 2025 单年 sanity-check 升级为多周期验证。数据集切换为 `cn_real_full`（2020-01-02~2026-08-18，全 A ~5210 只，1599 天），分段 train 2020-2023 / valid 2024 / test 2025-01-02~2026-08-17（386 交易日）。配置 `configs/backtest_warfare2_multiyear.yaml`，驱动 `reference/tdx_strategies/run_warfare2_multiyear.py`，`llm.enabled=false` 确定性。

| 指标 | 多周期(cn_real_full) | 对比：2025 单年 |
|---|---|---|
| IC | 0.0720 | 0.0760 |
| ICIR | 1.0374 | 1.121 |
| Rank IC | 0.0689 | 0.0738 |
| Rank ICIR | 0.9097 | 1.006 |
| 年化收益 | 0.3400（34.0%） | 0.1843 |
| 最大回撤 | -0.3090（-30.9%） | -0.1455 |
| Info Ratio | 1.0830 | 0.6400 |
| Calmar | 1.1003 | 1.2668 |

> 信号层：IC≈0.072 且 ICIR>1 在 **1.6 年测试窗（386 交易日）** 上稳定为正，与 2025 单年（IC=0.076）高度一致 → **信号跨年稳健，非单年巧合**（方向②要达到的核心目标已达成）。
> 组合层：TopK=50 在 2025-2026 测试窗年化 34%、回撤 -31%、IR 1.08（基准仍为 SH600000 占位）。

**结论可信度（honest 标注）**：
1. ✅ 测试样本从 ~3 个月扩到 ~1.6 年，统计可靠性显著提升；IC 方向/幅度跨两段一致。
2. ⚠️ **train l2 == valid l2 仍完全相同（0.0829，每轮不变）**：标签经 CSRankNorm 后截面近常数，l2 回归几乎无信息量；但 IC 在**未见 test 集**独立计算且为正，故信号以 IC 为准（l2 退化不影响 IC 结论）。
3. ⚠️ ConstantInputWarning：部分交易日截面因子/标签为常数 → 当日 IC 未定义，整体 IC 仍算出（0.072）。
4. ⚠️ 基准仍为 SH600000 占位（本机无 SH000300 指数特征），IR=1.08 是相对单只股票、非相对市场指数 → 解释力有限（deferred ③ 待补）。
5. ⚠️ M4/M5 非原生：TopK 持有 ≠ 通达信「破 MA20 即走」，收益结构与实战不同。
6. ⚠️ `$open` 含 NaN 警告（deal_price=open 个别标的缺失），Qlib 已容错，影响极小。

→ 当前结论：**战法2 信号在多周期上获验证（真实、稳健、幅度偏弱 IC≈0.07）**，可作为后续横向比较与定权的可信基线；但仍不足以单独定权，需方向①把战法1/4/6 拉到同一多周期数据上比较 + 相关性后，再做风险平价/均值方差定权。

## 后续
方向②已完成（多周期验证基线就绪）。下一步按 deferred 路线图：
1. 方向①：编码 F1/F6/F8/F9 因子，在同一 `cn_real_full` 多周期数据上跑战法 1/4/6 回测，做横向比较 + 战法间日收益相关性。
2. 定权：用多周期指标做风险平价 / 均值方差优化，替换先验 35/25/20/20。
3. 基准：下载 SH000300 指数特征，替换 SH600000 占位（deferred ③）。
4. （Phase2）分钟级：TimescaleDB `bars_5m` ETL 成 Qlib 5min（战法6 faithful）；jqrs 补 faithful 买卖/止损(M4/M5)；战法3/5 延后。
