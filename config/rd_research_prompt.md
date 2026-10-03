# 本轮挖掘追加约束（2026-10-04 在线修正，热生效）

## 表达式写法硬规则（违反即作废）

1. **"显式取负"只能出现 1 个乘性 `-1`**。
   禁止写 `-1 * A * -1 * B` —— 两个 `-1` 相乘等于 `+1`，**等于根本没取负**。
   （今天已产出 3 条这种表达式，全部作废：`-1 * TS_STD($return,20) * -1 * TS_CORR($return,$volume,20)` 等。）
   要取负请写 `-1 * (A * B)` 或 `RANK(-1 * A * B)`，保证整个式子里 `-1` 只出现一次。
2. 滚动窗口 ≥ 5，推荐 10 / 20 / 60；**禁止窗口 1~3**的短周期结构（换手高、实盘被成本吃穿）。
3. `DELAY(` 不存在，用 `Ref(`；禁止三元 `?:`；`COUNT()` 实参里禁止 `&&`（改写成 `A*B` 或 `A+B-A*B`）。
4. 字段只能写小写：`$open $high $low $close $volume $vwap`；不要写 `$Return`（qlib 无此字段，会整列 NaN）。

## 信号方向（上一轮复验的实测结论，不是猜测）

- **波动 / 振幅 / 尾部风险类信号在 A 股是负向的**：高波动、大振幅、近期涨幅大的股票未来收益更低。
  复验实据：`TS_ZSCORE($high-$low,20) * TS_PCTCHANGE($close,3)` → RankIC **−0.0268**；
  `TS_CORR($close,$volume,10) * TS_ZSCORE($close,10)` → **−0.0229**；三段 train/valid/test 全部同号。
  → 构造时必须取负，才能让因子与未来收益同向。
- **不要重复已被证伪的 4 类**：缩量收缩 `$volume < 0.5*TS_MEAN($volume,N)`、
  下轨回撤 `$close < k*TS_MAX($close,N)`、波动分歧 `TS_STD(return,20)-TS_STD(return,60)`、
  孤峰 `TS_SKEW*TS_STD`。这 4 类实测独立 RankIC 都在 ±0.01 内。
- **新维度**：隔夜缺口 `$open/Ref($close,1)-1`；上下影不对称
  `(Max($open,$close)-$high)/($high-$low+1e-8) - ($low-Min($open,$close))/($high-$low+1e-8)`。

## 验收

单因子独立 RankIC 绝对值 ≥ 0.03，且 train / valid / test 三段不能互相反号。
不接受"只有多因子合成才有信号" —— 上一轮平台报的 0.06 就是 LightGBM 合成的，
15 条因子独立复算没有一条超过 ±0.03。
