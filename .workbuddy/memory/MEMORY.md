# QuantaAlpha 项目长期记忆

> 详细内容都在 `docs/`：`research_sop.md`（研究规范）、`warfares_catalog.md`（战法总册）、
> `portfolio_construction_verdict.md`（B方向终审）、`a_share_warfare_landscape.md`（生态/主线）。
> **本文件只留"不查文档就会踩坑"的硬规则。**

## 定位与数据
LLM 驱动 Alpha 挖掘（rdagent 0.8.0）。回测必须真实数据、禁编造；**QMT 取不到的数据须告知用户，绝不默认改用 akshare**（用户硬规则）。
- TimescaleDB `quant_data`（127.0.0.1:5432，user=quant/pw=quant123456，**键名 `dbname` 不是 `database`**）：`bars_1d` 1231万行/2011~2026/5225只；`bars_5m` 3140万行。
- **主力环境 `quantaalpha`（Py3.11）**：qlib 0.9.7 + lightgbm + torch + xgboost + psycopg2 + scipy + numpy 1.26.4。另 `qmt312` 是 QMT 客户端 API 环境，numpy=2.4 **不可混用**。
- 指数数据在 `bars_1d` 的 `asset_type='index'`（000300.SH/000905.SH/000852.SH/000016.SH/000001.SH/399006.SZ 等 11 个）。

## ★ 用户真实目标（评判任何战法的标尺）
不是追 alpha，是「**找到一个能稳定赚钱的策略**」。月10% 不现实（复利≈214%/年，任何回测都没出现）；**底线是"只要能赚"**= 持续正收益 + 回撤可控。⚠ 不因想赚就放松验证铁律。
用户已明确 **不做 ETF**（反复说过）。股票侧目前只能做到"减伤+相对跑赢"，未满足该目标。

## ★ 验证铁律（写代码前必查）
0. **三口径基准**（等权/中位数/同规模档）。只看等权=被妖股绑架。`float_shares`/`turnover_rate` 在 2025-02~2026-03 整段 NULL → 该区间规模用 `amount`。
0b. **禁止用「每日同期排名」分档**（未来函数）→ 一律滞后 60 日均值排名。
1. T日收盘出信号 → 收益从 **T+1** 起。IC 高得离谱先查对齐。
2. 基准与选股池**同口径**；跨池基准凭空造超额。
3. 判 alpha 看**分位组单调性**；分位 `rank(ascending=False)`。
4. 停牌 NaN：算指标前 `ffill(limit=5)`，收益/成交用原始 OHLC。
5. 剔次新（上市<250日）；剔涨停（open/昨收≥1.098）。
6. 超额用**日超额复利**。
7. 成本含 Almgren 冲击（0.1×σ342bp×√participation）。股票成本=佣金1bp+印花5bp(卖)+冲击。
8. `成交额=close×volume×100`（volume 单位=手）；优先用 `amount`。
9. 宇宙构建前必须 `sort_values`。
10. train 内选参，禁 test 期网格回挑。
11. 低吸必带 **≤8% 止损**（F4：带止损 +36.53bp vs 无止损 −68.15bp）。
12. **基准必须与策略同口径**：每日再平衡策略 ↔ 每日再平衡等权基准；且「逐笔超额」须与「组合超额」交叉验证。
13. **`bars_1d` 读前必做 `(asset_code,trade_date)` 去重**（曾 42.8万组重复，使 F4 毛超额从 +13.00bp 变 −7.29bp；已清洗，备份 `bars_1d_dup15_bak_20261014`）。
14. **未复权修正**：单日 |收益|>21% 物理不可能 → 除权跳变，用当日横截面中位数填充。
15. **上市天数用全表 `min(trade_date)` 算**，勿用 `px.notna().cumsum()`（只加载部分区间时会把老股票误判次新）。
16. **本环境 cvxpy（OSQP/SCS/CLARABEL）全部静默硬崩进程** → QP 用自写 numpy FISTA+Dykstra 投影（已过 KKT 验证，0.73s/n=500）。见 `tools/portfolio_risk.py`。
17. 框架用 **Qlib 0.9.7**（表达式引擎 + qlib.backtest）；Windows 须 `if __name__=="__main__"` + `QLIB_KERNALS=1`。TopkDropoutStrategy 不支持止损，三口径基准须自写。
18. **L4 执行层通用规律**：T+1 开盘价已接近全天最优买点；全天 VWAP / 长 TWAP 是负优化；限价单(≈VWAP−20bp) 比开盘再省 ~8bp。

## ★ 总裁定：股票侧无可持续 alpha（勿重复投入）
F1-F10 + 10 横截面 + 7 门控/时序 + 通达信6战法 + 斐波那契 + **组合构建族（B方向）** 全部证伪。
共性：**正 IC ≠ 可提取 alpha**；IC 与"小盘程度"单调相关 = 风格不是 alpha；成本翻倍即归零。
**低波动异象是唯一真信号**（市值中性 IC=0.0332/t=6.17），但只能多空配对变现，A股做空受限。

## ★ B 方向（组合构建）终审 2026-10-14 —— `docs/portfolio_construction_verdict.md`
宇宙=总市值 top-N（2020-2024），Ledoit-Wolf 收缩协方差，月末定权次月执行，含成本。
- **2021-2024（对齐窗口）**：top300 ERC **−1.34%/年、回撤 −33.5%** vs 等权 −6.04%/−41.0% vs 沪深300 −6.07%/−45.6%
  → **+4.7pp/年、回撤小 12pp、逐年跑赢沪深300**。成本仅 59bp/5年，非成本伪影。
- **2025-2026（陈旧宇宙近似）→ 反转**：R300 ERC **−6.70%** vs 等权 **+6.74%**；逆波动两个 N 都跑输等权。
  原因是风格轮动：2025-26 奖励高波中小盘（中证500 +15.2%、创业板指 +19.7%），惩罚低波大盘（上证50 +3.7%）。
- **裁定**：ERC/InvVol = **会轮动的风格倾斜，不是 alpha**。⚠️ 有条件可用（须有 regime 判别器），否则不建议实盘。
- **最大局限**：总市值字段 2025 后失效 → 2025-26 只能用「2024 年底定盘的陈旧宇宙」近似；要真正回答须补**指数成分股名单**（需从 QMT 拉）。
- 附带发现：2025-26 最笨的等权大市值一篮子 +8.94%/年、回撤仅 −13.1%，跑赢所有风险优化版（但主要是 beta，且跑输中证500）。

## ★ 战法总册 / SOP / 主线（入口）
- `docs/warfares_catalog.md` = 所有战法/因子/策略的精确定义+验证状态+证据。**迭代先查此册。**
- `docs/research_sop.md` = 十步流水线 + 11 条偏差预检 + 7 验证关卡。新战法统一走此 SOP，任一关不过不出结论。
- `docs/a_share_warfare_landscape.md` = A股战法生态 + 后续三条主线（资产配置 / 执行层L4 / 风险组合层）。
- ETF 侧唯一活口（用户不做）：TSMOM+vt=5% 真实年化 +4.58%、回撤 −9.3%。vt×λ 全网格 Sharpe 0.56~0.64 = 高原无尖峰 → TSMOM 才是收益来源，vt/λ 只是风控旋钮。

## ★ QMT datadir 直读（不用 akshare、绕开 xtquant segfault）
`D:/中航证券QMT实盘-交易端/datadir`，`{SH,SZ}/86400/<6位>.DAT` = 8B头 + 64B记录(16×int32 LE)：
`[0]Unix秒(交易日00:00北京) [1..4]OHLC(×1000) [6]量(手) [8]额(元)`；300=5分钟。
坑：① 取 `.date()` 必须 `tz=+08`，否则全部早1天；② `bars_1d` 唯一键是 `(asset_code,ts)`，按 (code,trade_date) 先删再插；③ 未复权 → 份额拆分假暴跌，用收盘判据+整数比例吸附修。
全A日线更新：`cd /d/qqmt/quant_datahub && python -m datahub daily --end YYYYMMDD`（qmt312 环境；`import pandas` 必须先于 `import xtdata`）。

## Web 工作台
后端 conda `quantaalpha` 跑 `frontend-v2/backend/app.py` → :8000；前端 `cd frontend-v2 && npm run dev` → :3000。健康接口 `/api/health`。
UI：深灰+蓝紫，`index.html` 的 `<html class="dark">` 启用深色；改深色时 `.glass`/`.glass-strong` 须整套替换。空状态用 `components/ui/EmptyState.tsx`。
