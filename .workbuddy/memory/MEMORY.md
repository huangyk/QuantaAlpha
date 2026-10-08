# QuantaAlpha 项目长期记忆

## 定位与数据
LLM 驱动 Alpha 因子挖掘（rdagent 0.8.0）。回测必须真实数据，禁编造；**QMT 取不到的数据须告知用户，绝不默认用 akshare**（用户硬规则）。
- qlib `cn_real_full`：2020-01~2026-08，全A~5210只，日频8字段（**无市值/PE/行业**）。基准 SH000300 已补齐。
- TimescaleDB `quant_data`（127.0.0.1:5432/quant/quant123456，配置键 `database` 非 dbname）：`bars_1d` 1231万行/2011~2026/5364只，含 amount/turnover_rate/float_shares（比 qlib 全）；`bars_5m` 3140万行（2026起）。
- 架构：TimescaleDB底座 → ETL → qlib(离线研究) → 策略 → QMT/Ptrade(仅实盘增量)。miniqmt 收紧不影响研究。
- 环境：conda `quantaalpha`（含 qlib 0.9.7）；psycopg2 需 conda base。

## ★ 验证铁律（写代码前必查，反复踩坑）
1. **信号-收益对齐**：T日收盘出分→收益从 T+1 起（买T+1、卖T+1+h）。`close.shift(-h)/close`=未来函数。**IC 高得离谱先查对齐，不是庆祝**；写完回测必须**手算一天**验证价格取的是哪天。
2. **基准须与选股池同口径**（同池等权）；沪深300只能作环境参照——跨池基准凭空造超额；**沪深300 管中小盘池方向会反**。
3. **判 alpha 看分位组单调性**，不只看 top-K（普涨窗口随机抽 K 只也正）。分位须 `rank(ascending=False)`，默认升序 label0=最低分（踩过两次把符号说反）。自检：合成 RankIC 为正时 D1−D10 必为正。
4. **停牌 NaN 放大**：`rolling(min_periods=N)` 停牌 1 天使其后 N−1 天全 NaN → 算指标前 `ffill(limit=5)`，但收益/成交一律用**原始** OHLC。
5. 剔次新股（上市<250日）；剔除涨停买不进（open/昨收≥1.098）。
6. 超额用**日超额复利**，不能用「策略年化 − 基准年化」。
7. 成本必含市场冲击（Almgren `0.1×σ日342bp×√participation`）。
8. 成交量单位=「手」：`成交额=close×volume×100`；优先用 `amount`（曾低估100倍→冲击高估10倍）。
9. 池构建 `.index[:N]` 前必须确认已 `sort_values`（字母序=随机池，曾致结论翻转）。
10. 在 train 内选参，禁 test 期网格回挑；网格最优必须报 t 并标「探索性」。

## ★ 裁定：所有横截面选股 alpha 已证伪（勿重复投入）
飞龙 / 战法2 / 5min低波动族 / 短期反转+量能退潮 / 换手率 / 8种权重 / 阈值再平衡 / 6种分池——无一通过。
共性：**正 IC ≠ 可提取 alpha**（分组不单调 + 换手成本吃掉）。因子天花板：毛 alpha ~9%/年 < 换手成本。
IC 与「小盘程度」单调正相关 → 那是**风格/流动性因子**不是个股 alpha（加流动性约束就消失）。

**★ 低吸（战法2）专项终审 2026-10-09（见 docs/dixi_final_verdict.md，脚本 tools/dixi_deep_analysis.py）**
- 剔次新+剔涨停买不进后：**2025 年不反向**（f4 IC +0.0220/t+2.08），真正失效只有 2026（t+0.24）
  → 更正 10-07 warfare2 报告。
- **风格择时假想证伪**：breadth/trend/vol 三分组 9 格 f4 IC **全为正**，无符号翻转。
  "弱市低吸/强市追强"是数据污染假象 → **路线A 作废**。
- **真正死因 = 无止损**：f4前20% 日超额 **−46.4bp/日(≈−117%/年) t=−6.19**；加 ≤8% 止损回到 ≈+5bp/日。
- 但日度聚合后全部 |t|<2、测试期(25-26)全归零/转负 → **终审不可用**。
- **唯一可落地结论**：任何低吸/买回调票**必须带 ≤8% 止损**（否则统计上自杀），
  但加了止损后期望≈0，不是盈利策略。f4 最多当**排除**用的弱点过滤器，不能当选股器。
**唯一存活 = 低换手池等权持有（beta）**：13年长周期年化 +2.02~2.89%；2021 后全部转负。旧「liquid500 +19.85%」是**幸存者偏差**，不可作预期。
→ 杠杆在 **L4 执行层（降换手/降冲击）**，不在 L2 因子层。
交付：`quantaalpha/strategy/pool_hold.py`、`tools/strategy_final.py`、`tools/make_order_list.py`、`docs/strategy_card.html`。

## ★ QMT datadir 直读（破局：不用 akshare、绕开 xtquant segfault）
本机 QMT `D:/中航证券QMT实盘-交易端/datadir`（API 崩只因客户端没跑，本地文件可直接读）。
- `{SH,SZ}/86400/<6位>.DAT` = 8B头 + 64B记录(16×int32 LE)：`[0]Unix秒 [1..4]OHLC(×1000) [6]量(手) [8]额(元)`；300=5分钟。
- 脚本 `tools/backfill_etf_qmt_local.py --clean-legacy`。
- 坑1：`bars_1d` 唯一键是 `(asset_code,ts)` 非 trade_date，ts 须**北京时间午夜**；按 (code, trade_date) 先删再插。
- 坑2：datadir 未复权 → ETF 份额拆分假暴跌（−87%/−73%/−51%/+176%）。用**收盘**判据（开盘判据漏判）+整数比例吸附，修 8 处。
- 坑3：旧 qmt ETF 数据损坏（159915/159949/159922 缺整段2025）→ 踢出池。

## ★ 跨资产 ETF 轮动（七星高照，2026-10-08 终审）
34 只干净跨资产池、2019-2026、万1。**网上 32% 无法复现。**
- 池等权买入持有 = **+9.3%/夏普0.65/回撤−22.2%/换手1次**（同口径最优）
- 最好轮动 月频top2+门槛 +10.0%/夏普0.43 → 仅微超但风险调整更差；日频 top1/3 仅 +7.0%
- LB 扫描 15~35 呈噪声带 → **LB=25 是过拟合**
- +9.3% 是窗口 beta（纳指/黄金强），**非可持续 alpha，勿加杠杆**

## ★ 网上 QMT 策略参考（docs/qmt_strategy_references.md）
首推 `guoyaohua/etf-adaptive-rotation-qmt`：逆波动率权重 + 波动率上限10% + **四周错峰 A/B/C/D sleeve** + 相关性去重 + ATR止损 + 前缀不变性测试；自报 CAGR 5.83%/回撤 6.46%/夏普 1.004（诚实，非30%）。**其低回撤来自组合层/风控层(L4)而非信号层**，与本项目结论一致。
判据：任何策略先问 ①滚动窗口最差值 ②2×/3×成本压力 ③防未来数据验证，给不出就不信。

## 优化方向（破局）
补行业分类+市值+涨跌停/ST 做真中性化；换成熟成分股池；优先做执行层降换手。任何新因子须先显著跑赢同池等权。

## Web 工作台
后端 conda `quantaalpha` 跑 `frontend-v2/backend/app.py` → :8000
前端 **`cd frontend-v2 && PATH="/d/nvm4w/nodejs:$PATH" npm run dev`** → :3000（nvm node v24.9.0）
⚠ 勿直跑 vite.js、勿用托管 node。健康接口 `/api/health`（无 `/api/experiments`）。

### UI 实现约定（2026-10-09 核对后确立，报告 `docs/ui_audit_20261009.md`）
- **主题**：深灰+蓝紫。tailwind 配了 `darkMode:'class'`，靠 `index.html` 的 `<html class="dark">` 启用；
  浅/深两套 CSS 变量并存。**改深色时 `.glass`/`.glass-strong` 必须整套替换**，否则白玻璃浮在深灰底上。
- **导航**：左侧 `<aside fixed left-0 w-[72px] lg:w-64>`（窄屏只显示图标）。
  ⚠ 任何 `fixed bottom-0 left-0 right-0` 的元素（如 ChatInput）都要加 `ml-[72px] lg:ml-64` 让开侧栏。
- **空状态统一用** `components/ui/EmptyState.tsx`（此前有两套不一致实现）。
- **JSX： `return (` 后面不能直接跟 `{/* 注释 */}`** → 会被解析成对象字面量而报错，注释放 return 前。
