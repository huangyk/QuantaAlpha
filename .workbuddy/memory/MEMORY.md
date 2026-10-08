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

## Web 工作台启动（用户提供正：必须用 npm run dev）
- 后端：conda `quantaalpha` 跑 `frontend-v2/backend/app.py` → :8000
- 前端：**`cd frontend-v2 && PATH="/d/nvm4w/nodejs:$PATH" npm run dev`** → :3000（nvm node v24.9.0/npm 11.6.0）
  ⚠ 不要直跑 `node_modules/vite/bin/vite.js`，也不要用托管 node——用户环境不是这样
- vite proxy 已配 `/api`、`/ws` → :8000；健康接口是 `/api/health`（**没有 /api/experiments**）

## ★ 网上 QMT 策略调研（2026-10-08，见 docs/qmt_strategy_references.md）
- 首推 **github.com/guoyaohua/etf-adaptive-rotation-qmt**：逆波动率权重+波动率上限10%
  +**四周错峰A/B/C/D sleeve**+相关性去重+ATR止损+前缀不变性测试。
  自报 CAGR 5.83%/回撤6.46%/夏普1.004（诚实，非30%）。
  → **其低回撤主要来自组合层/风控层(L4)，不是信号层**，与本项目结论一致。
- 看任何策略先过三条：滚动窗口最差值、2×/3×成本压力、防未来数据验证。
  本站已亲手证伪"聚宽25日动量×R²年化30%"（真实4.6~10.1%，LB=25无优势）。

## ★★★ QMT datadir 直读（2026-10-08，破局：绕开 segfault、不用 akshare）
QMT 实盘终端装在本机 `D:/中航证券QMT实盘-交易端/datadir`（xtquant segfault 只因客户端没跑，
本地文件可直接读 → 最权威 QMT 数据，不违反"禁用akshare"硬规则）。
- `{SH,SZ}/86400/<6位>.DAT` = **8B头 + 64B记录(16×int32小端)**：
  `[0]Unix时间戳 [1..4]OHLC(×1000) [6]成交量(手) [8]成交额(元)`；86400=日线，300=5分钟。
- 脚本 `tools/backfill_etf_qmt_local.py --clean-legacy`（含拆份修正+清旧数据）。
- **坑1**：`bars_1d` 唯一键是 `(asset_code, ts)` 非 trade_date，ts 须 **北京时间午夜**，
  否则同日插两行。按 (asset_code, trade_date) 先删再插。
- **坑2**：datadir 是**未复权**，ETF 份额拆分造成假暴跌(−87%/−73%/−51%/+176%)。
  修法：收盘/昨收超 ±35%/+50% → 吸附整数比例(**用收盘判据，开盘判据会漏判**)。修8处。
- **坑3**：旧 `qmt` ETF 数据损坏(159915/159949/159922 缺整段2025；159919 +159%) → 踢出池。

## ★ 七星高照跨资产 ETF 轮动（2026-10-08 终审）— 见 docs/qixing_etf_verdict.md
34只干净跨资产池、2019-2026、万1佣金。**网上32%无法复现。**
- 池等权买入持有 = **+9.3%/夏普0.65/回撤−22.2%/换手1次**（同口径基准，最优）
- 最好轮动 月频top2+门槛 = +10.0%/夏普0.43/回撤−25.4% → 仅微超但风险调整更差
- 日频top1/top3 仅 +7.0~7.1%、夏普0.23~0.35；LB扫描15~35呈噪声带 → **LB=25是过拟合**
- **换手=最大成本**：日频1474次→+7.0%，月频91次→+9.3%，零换手→夏普最优。再证杠杆在L4。
- 落地：跨资产ETF等权分散 + 月频再平衡 + 换仓门槛 + 5步框架择时/止损止盈（②选股仍不可用）。
  ⚠ +9.3% 是窗口 beta（纳指/黄金强势），**非可持续 alpha，勿加杠杆**。

## 优化方向（破局，非日线选股）
补行业分类+市值+涨跌停/ST做真中性化；执行层降换股率（杠杆在L4不在L2）；换成熟成分股池。当前无验证通过的alpha，任何新因子须先显著跑赢同池等权。
