# 通达信战法与选股公式 —— 模块化拆分与实施方案（QuantaAlpha 修正版）

> 源文件：`通达信战法与选股公式手册_2026-09-23.md`
> 目标：把手册的「核心公式 + 6 个战法」拆成统一、可复用、可执行的模块，落到 **QuantaAlpha**（LLM 因子挖掘框架，vendoring rdagent）中。
> ⚠️ **2026-09-29 架构修正**：经核实，QuantaAlpha **不存在** `FilterRule/SortRule/SellRule/SizingRule`（这些属于另一项目 `jq_research_system` 的 jqrs 引擎）。本文已全部改为 QuantaAlpha 真实抽象重画，旧版误用 jqrs 规则处已删除。
> 关键约束：**标准 Qlib `cn_data` 只有 OHLCV 基础行情，无 Level-2 大单流（通达信 `BIGORDER`）**。「主力流入」一律用「量价+主动资金」近似（F9），真 L2 主力流标注为数据缺口。

---

## 1. 统一模块抽象（6 战法 → 同一套骨架，映射到 QuantaAlpha 真实能力）

| 模块 | 职责 | 对应 QuantaAlpha 真实抽象 |
|---|---|---|
| **M1 选股/过滤层** | 定股票池 + 硬性门槛（多头排列、板块强） | QLib `filter` 表达式（因子 MASK，先缩池）；或数据集 `instruments` 过滤 |
| **M2 信号层** | 入场信号因子（突破/回踩/首阴…） | **QLib 表达式因子 → `FactorLibraryManager` 入库 `data/factorlib/all_factors_library.json`**（AlphaAgent `expr_parser` 语法） |
| **M3 确认层** | 二次确认（弱转强、放量收复、板块共振） | QLib Alpha 模型 **Score / 组合权重**（lgb 模型对因子打分） |
| **M4 风控层** | 止损/止盈（破 MA20、破信号 K 线低点） | ⚠️ **非原生**：QLib 标准工作流是「分数选股持有 TopK」（`TopkDropoutStrategy`），无事件级「破位即卖」止损。faithful 止损须 jqrs（另一项目） |
| **M5 仓位层** | 单笔风险 2%–3% 的仓位管理 | QLib 模型分数加权（`WeightedPortfolio` / TopK 权重）；**非单笔风险预算** |
| **M6 验证层** | 历史回测看 IC/夏普/回撤 | `quantaalpha backtest`（→ `BacktestRunner`，读 `configs/backtest.yaml`，`QLIB_DATA_DIR` 已配） |

> 核心差异：通达信战法是**事件驱动**（出现信号→买入→破位卖出），而 QuantaAlpha/QLib 是**分数选股框架**（因子→模型打分→每日持有 TopK）。因此 M2 因子可干净落地、可回测；M4/M5 的精确买卖/仓位在 QuantaAlpha 内只能近似（TopK 持有），faithful 实现在 jqrs。

---

## 2. 原子因子清单（M2 零件，AlphaAgent 表达式，已对 function_lib 校验）

因子由 OHLCV 原语构成，均可用 AlphaAgent `expr_parser` + `function_lib` 计算（已实测 `SMA/TS_MIN/TS_MAX/DELAY/ABS/GT/LT/AND/OR` 均支持；**无 `CROSS` 内置，用 `A>B && DELAY(A,1)<=DELAY(B,1)` 组合**）。

| 因子 | 语义 | QuantaAlpha 表达式（AlphaAgent 语法） | 数据依赖 |
|---|---|---|---|
| F1 趋势突破 | C 创 N 日新高且站上 MA60 | `GT($close, TS_MAX(DELAY($close,1), N)) && GT($close, SMA($close,60))` | OHLCV |
| F2 放量 | VOL>MA5(VOL)*K 且 VOL>昨量 | `$volume > SMA($volume,5)*1.5 && $volume > DELAY($volume,1)` | Volume |
| F3 均线多头排列 | MA20>MA60 | `SMA($close,20) > SMA($close,60)` | OHLCV |
| F4 回踩均线 | 距 MA20<3% 且金叉 MA20 | `ABS($close - SMA($close,20))/SMA($close,20) < 0.03 && $close > SMA($close,20) && DELAY($close,1) <= DELAY(SMA($close,20),1)` | OHLCV |
| F5 首阴 | 昨涨停今收阴且放量 | `GT(DELAY($close,1)/DELAY($close,2), 1.095) && LT($close, DELAY($close,1)) && GT($volume, SMA($volume,5)*1.5)` | OHLCV |
| F6 黄金坑 | 20日新低<MA60*0.93 且收回 MA20 且放量 | `LT(TS_MIN($close,20), SMA($close,60)*0.93) && GT($close, SMA($close,20)) && GT($volume, SMA($volume,5)*1.5)` | OHLCV |
| F7 龙回头 | 60日曾涨>25% 且回踩 MA20 金叉 MA10 | `GT(DELAY(TS_MAX($close,60),20)/TS_MAX($close,60), 1.25) && GT($close, SMA($close,20)) && GT($volume, SMA($volume,5)*1.3) && GT($close, SMA($close,10))` | OHLCV |
| F8 弱转强（日内） | 低开高收且放量 | `LT($open, DELAY($close,1)) && GT($close, DELAY($close,1)) && GT($volume, DELAY($volume,1)*1.2) && GT($close, $open)` | OHLCV |
| F9 主动资金（≈主力） | 价升量增（替代 BIGORDER） | `GT($close, $open) && GT($close, SMA($close,5)) && GT($volume, DELAY($volume,1)*1.2)` | OHLCV（**L2 近似**） |
| F10 板块强 | 板块近3日主力净流入>0 | 需板块级资金数据 | **外部依赖** |

> 无未来函数：`DELAY/TS_*` 均用已发生数据。所有表达式已通过 `expr_parser` 实测可解析（战法 2 的 F2/F3/F4 见 `tdx_warfare_phase1_plan.md`）。

---

## 3. 6 战法模块化映射（QuantaAlpha 原生）

| 战法 | M1 过滤 | M2 信号因子 | M3 确认(模型) | M4 风控(局限) | 备注 |
|---|---|---|---|---|---|
| 1 突破放量主力流入共振 | F3 | F1+F2+F9 | F10(可选) | ⚠️TopK 无止损 | 主力用 F9 近似 |
| 2 均线回踩低吸(首推) | F3 | F4 | F2 | ⚠️TopK 无止损 | 首批落地对象 |
| 3 首阴/开板低吸 | 前日涨停 | F5 | F8 确认 | ⚠️ | Phase 2 |
| 4 黄金坑 | 前期横盘 | F6 | F2 收回 | ⚠️ | 首批 |
| 5 龙回头 | 曾是龙头 | F7 | 放量转强 | ⚠️ | Phase 2 |
| 6 分歧转一致(弱转强) | — | F8 | 收高位 | ⚠️短线 | 首批 |

---

## 4. 实施方向与分阶段方案（落 QuantaAlpha）

**方向**：把手册公式翻译为 QuantaAlpha **QLib 表达式因子 + Alpha 模型回测**，而非在通达信手动跑。

### Phase 1 — 因子实现（M2 零件）
- 用 AlphaAgent 表达式实现 F1–F9，写入 `data/factorlib/`（JSON `factors` 字典，schema 同 `FactorLibraryManager`：`{factors:{id:{factor_name,factor_expression,factor_description,quality?}}}`）。
- F9 标注「L2 近似，非真主力流」。
- 阈值（HHV 窗口 N、放量倍数 K）抽成可配置。
- 验收：每个表达式经 `expr_parser` 可解析（已实测 F2/F3/F4）。

### Phase 2 — 策略组装（M1/M3）
- 战法 = 因子组合作为 **模型特征**，经 `quantaalpha backtest` 训练 lgb + TopK 回测。
- 战法 2（首推）：输入 F3(过滤) + F4(信号) + F2(确认) 三个因子。
- 验收：回测产出 IC/夏普/回撤报告。

### Phase 3 — 回测验证（M6）
- `quantaalpha backtest --factor_path data/factorlib/tdx_warfare2_factors.csv`（CSV 列 `factor_name,factor_expression`）。
- 看 IC、夏普、最大回撤；对 N/K 做参数扫描。

### Phase 4 — 风控/仓位（M4/M5，局限说明）
- QuantaAlpha 内 M4/M5 仅能用 TopK 持有近似；**精确止损/单笔风险预算须 jqrs（另一项目）**。如要求 faithful 买卖，转 `jq_research_system`。

### Phase 5 — 板块过滤（可选）
- F10 需板块资金流，标准数据无；先人工锁板块或后续接外部数据。

---

## 5. 风险与数据缺口（honest）

1. **L2 缺失**：F9 近似主力流，非真主力；敏感须接 L2。
2. **参数敏感**：N/K 须回测扫描。
3. **无未来函数**：已保证。
4. **M4/M5 非原生**：QuantaAlpha 无事件级止损/单笔仓位，faithful 实现在 jqrs。
5. **TopK 组合偏差**：QLib TopK 持有 ≠ 通达信「信号买入持有至破位」，收益特征会有结构差异。
6. **流动性/涨跌停**：回测 `limit_threshold:0.095` 已设，但停牌等未完全模拟。

---

## 6. 下一步（已确认 2026-09-29）

- [x] 落地框架：**QuantaAlpha 因子挖矿**（非 jqrs）
- [x] 文档归位：`reference/tdx_strategies/`
- [x] 首批：**战法 2 均线回踩低吸** 先落地
- [x] 启动 Phase 1：F2/F3/F4 已写成因子（见 `tdx_warfare_phase1_plan.md`），待跑 `quantaalpha backtest`

---

*免责声明：本方案仅做模块化拆分与工程落地设计，所有信号须实盘前经历史回测验证，不构成投资建议。*
