# How to read files.
For example, if you want to read `filename.h5`
```Python
import pandas as pd
df = pd.read_hdf("filename.h5", key="data")
```
NOTE: **key is always "data" for all hdf5 files **.

# Here is a short description about the data

| Filename       | Description                                                      |
| -------------- | -----------------------------------------------------------------|
| "daily_pv.h5"  | **5-MINUTE** price and volume data (intraday bars, not daily).   |

# !! 数据是 5 分钟 K 线，不是日线 !!

`daily_pv.h5` 的每一行是 (datetime, instrument) 上的一根 **5 分钟 K 线**：
- datetime 精确到分钟，形如 `2026-01-05 09:35:00`，日内取值 `09:35, 09:40, ..., 11:30, 13:05, ..., 15:00`
- **每个交易日 48 根**（A 股 4 小时 × 12）
- 区间 2026-01-05 ~ 2026-09-30（约 190 个交易日、8640 根）
- 标的池约 500 只（liquid500，按成交额取 Top500）

## 时间窗口换算（写因子时必须按这个换算）

| 你想表达的周期 | 需要的 bar 根数 |
| -------------- | --------------- |
| 30 分钟        | 6               |
| 1 小时         | 12              |
| 半天           | 24              |
| **1 个交易日** | **48**          |
| 2 天           | 96              |
| 1 周           | 240             |

**常见错误**：直接照抄日频因子的窗口（TS_MEAN($close, 5)、TS_STD($return, 20)）。
在 5 分钟数据上，TS_MEAN(x, 20) 是 100 分钟均线，不是 20 日均线 —— 语义完全不同。
请把日频直觉下的"N 天"翻译成"N × 48 根"。

## 变量

$open: open price of the 5-minute bar.
$close: close price of the 5-minute bar.
$high: high price of the 5-minute bar.
$low: low price of the 5-minute bar.
$volume: volume traded during the 5-minute bar.
$vwap: volume-weighted average price of the 5-minute bar.
$factor: adjustment factor — **恒为 1.0**（入库是不复权价），不要用它在日间做复权调整。
$return: return of this 5-minute bar vs the previous bar of the **same stock**.

## 横截面 vs 时序

- `RANK(x)` / `ZSCORE(x)` / `MEAN(x)` 等不带 `TS_` 前缀的算子 = **横截面**，
  在每一根 bar 上对所有股票同时计算（同一时刻的截面）。
- `TS_MEAN(x, n)` 等 `TS_` 前缀算子 = **时序**，在单只股票内沿时间滚动 n 根 bar。

## 分钟级特有的、值得挖掘的方向

日频数据看不到、只有分钟数据能算的信号：
1. **日内已实现波动** —— 用 5min 收益算 TS_STD，比日频用 high/low 近似精确得多。
2. **路径效率** —— |收盘-开盘| / 区间内 |逐根收益| 之和，衡量价格是"直走"还是"来回震荡"。
3. **VWAP 偏离** —— ($close - $vwap)/$vwap 或其日内累计，衡量成交重心相对价格的位置。
4. **成交量日内分布** —— 单根 bar 成交量占当日总量的比例，识别开盘/尾盘集中成交。
5. **日内收益偏度 / 峰度** —— TS_SKEW($return, 48)，捕捉"急拉慢跌"一类的形态。
6. **高低点出现位置** —— 当日最高/最低价出现在第几根 bar，早盘冲高与尾盘冲高含义不同。

## 注意

- 5 分钟数据噪声远大于日频，短窗口（< 12 根）的因子容易过拟合，
  建议窗口 ≥ 24 根（半天）起步。
- 同一因子在相邻 bar 上的取值高度相关（相邻样本共享 47/48 的价格路径），
  样本不独立，显著性指标会被高估。
