# QuantaAlpha 项目长期记忆

## 项目定位
LLM 驱动 Alpha 因子挖掘框架（vendoring rdagent==0.8.0）。因子 `quantaalpha/factors/`，回测 `quantaalpha/backtest/runner.py`（QLib）。铁律：回测必须基于真实数据，禁编造。

## 数据
- qlib `cn_real_full`：2020-01-02~2026-08-18，全A~5210只，日频8字段（无市值/PE）。基准 SH000300 已补齐。
- TimescaleDB `quant-timescaledb`：`127.0.0.1:5432/quant_data/quant/quant123456`（配置键`database`非`dbname`）。`bars_1d` 1231万行/2011~2026/5364只，含 amount/turnover_rate/float_shares（比qlib全）。xtquant 仅 `qmt312` 环境；`import pandas` 须先于 `import xtdata`，否则 segfault。
- 架构：TimescaleDB底座 → ETL → qlib(离线研究) → 策略代码(与下单解耦) → QMT/Ptrade(仅实盘增量)。miniqmt收紧不影响研究。

## ★ 验证铁律（反复踩坑，最高优先级，写代码前必查）
1. 信号-收益对齐：T日收盘出分→收益从T+1起（买T+1卖T+1+h）。`close.shift(-h)/close`=未来函数。IC高得离谱先查对齐。
2. 基准须与选股池同口径（同池等权/中证全指）；沪深300只作环境参照。跨池基准凭空造超额。
3. 判alpha看分位组单调性，不只看top-K；合成RankIC为正时D1(最高分)-D10必正（须`rank(ascending=False)`，默认升序label0=最低）。
4. 成本必含市场冲击（Almgren `0.1×σ日342bp×√participation`）；超额用日超额复利非年化相减。
5. 剔除次新股（上市<250日污染基准）；模型在train内选参，禁test期回挑。
6. 成交量单位=「手」：`成交额=close×volume×100`；优先用`amount`字段（曾低估100倍→participation高估100倍、冲击高估10倍）。
7. 停牌NaN放大：算均线前先`ffill(limit=5)`；rolling(min_periods=N)停牌1天使后N-1天全NaN。
8. 池构建：`.index[:N]`前必须确认已按目标指标`sort_values`（字母序会变随机池，曾致结论翻转）。
9. 数据完整性：建仓/回测加`len(ok)>0.5*len(codes)`，跳过残缺日（如2020-05-05全市场仅120只更新）。

## ★ 战法裁定（2026-10-08，终审，勿重复投入）
**所有横截面选股 alpha 在真实成本+正确对齐下全部证伪**：飞龙/战法2/5min低波动族/短期反转+量能退潮/换手率因子/8种权重方案/阈值再平衡——无一通过（正IC≠可提取alpha，分组不单调/成本吃掉）。
**唯一存活：低换手池等权持有（pool_hold）**，但本质是beta/风格策略非alpha。
- 修正池构建bug后（13年长周期年化）：流动性Top500 +2.02% / 中后段(1001-3000) **+2.89%最优** / 全部成熟 +2.58%；**2021后全部转负**（中后段 −1.41%，2025起 −0.10%）。
- 旧「liquid500 +19.85%」是**幸存者偏差**（池按2026流动性选前500），**不可作预期**，只作上限参考。
- 低流动性长期跑赢是真实规律；容量经volume单位修正后可承载数亿。
- 执行细则：买09:30集合竞价（比VWAP便宜12bp），避开10:00/13:05；拆单全天；不择时/不调仓/不用大盘闸门（管中小盘池方向反）。
- 代码 `quantaalpha/strategy/pool_hold.py`；战法卡 `docs/strategy_card.html`；验证 `tools/strategy_final.py`；下单清单 `tools/make_order_list.py`。

## ★ 换手率数据（akshare→QMT）
- miniQMT不提供turnoverRate/股本；库内2020-2024有(ETL入)，2025-2026空（datahub qmt.py漏取字段）。
- 统一面板 `data/cache/turnover_unified_2020_2026.parquet`（DB优先+akshare补缺口÷100，`volume×100/float_shares`反算）。
- 回填 `tools/backfill_turnover_qmt.py`（沙箱xtdata取数segfault，须在QMT正常机器跑）；面板 `tools/build_turnover_panel.py --no-akshare`。

## Web 启动
`frontend-v2/backend/app.py`(FastAPI:8000, quantaalpha环境) + `node_modules/vite/bin/vite.js`(Vite:3000, nvm node)；`npm-cli.js`路径不存在，直跑vite.js。

## 优化方向（破局，非日线选股）
补行业分类+市值+涨跌停/ST做真中性化；执行层降换股率（杠杆在L4不在L2）；换成熟成分股池。当前无验证通过的alpha，任何新因子须先显著跑赢同池等权。
