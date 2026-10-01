# QlibDataAdapter v1 数据契约

## 目标

第一版只做一件事：把 `bars_1d` 和可选因子转换成可审计的 Qlib staging 数据，不在核心环境导入 Qlib，也不执行回测。

输出目录：

```text
artifacts/<experiment_id>/
  features.csv
  labels.csv
  calendar.csv
  calendars/day.txt
  instruments/stocks.txt
  manifest.json
```

这是 `qlib-staging-v1`，不是 Qlib 原生 binary provider。先固定字段、日期、股票代码和标签语义，再做 native provider，避免一次改太多层。

## 输入

日线输入至少包含：

```text
trade_date 或 datetime/ts
asset_code
close
```

建议字段：

```text
open, high, low, close, volume, amount, turnover_rate
```

因子可以：

1. 直接作为输入 CSV 的列；
2. 使用 DB 模式从 `m_factor_value` 加载，并通过 `--factor chip_skew` 指定。

## 特征和标签隔离

`features.csv` 只允许非未来字段。以下字段会被拒绝：

```text
ret_5d
label_5d
future_*
max_up_5d
```

标签单独写入 `labels.csv`，例如：

```text
label_5d = close[t+5] / close[t] - 1
```

最后 `label_horizon` 个交易日没有标签是正常的，不能向前填充。

## 命令

从 CSV：

```powershell
.venv312\Scripts\python.exe -m jq_research_system.qlib_adapter csv `
  --input path/to/bars.csv `
  --output artifacts/20260722_alpha158_baseline `
  --start 2018-01-01 `
  --end 2026-06-30 `
  --factor chip_skew
```

从 TimescaleDB：

```powershell
.venv312\Scripts\python.exe -m jq_research_system.qlib_adapter db `
  --output artifacts/20260722_alpha158_baseline `
  --start 2018-01-01 `
  --end 2026-06-30 `
  --factor chip_skew `
  --factor winner_pct
```

## staging → native provider（接正式 Qlib 数据）

**外接主体是 Qlib**，不是给 RD-Agent 单独接库。标准漏斗：

```text
TimescaleDB → qlib_adapter(staging) → qlib-provider(native bins) → qlib.init / qrun / RD
```

```powershell
# 只建 provider（不要用 qlib-baseline；那是 Alpha158+LGB 训练）
.\dev.ps1 qlib-provider artifacts\cn_real_staging C:\Users\lenovo\.qlib\qlib_data\cn_real_r500 --max-instruments 500

# 切换 RD/Qlib 研究 profile，并重生因子模板 h5
.\dev.ps1 qlib-profile research
.\dev.ps1 qlib-regen-h5
```

Profile 配置：`config/qlib_profile.json`（`smoke` = 小样链路；`research` = 正式研究）。  
环境变量 `RD_QLIB_PROFILE=smoke|research` 可临时覆盖。

首版 research provider 写入 `$factor` 常数 `1.0`（`factor_policy: constant_1`），**不是真复权**；真 adj 后续迭代。

## 版本一的边界

当前不做：

- 自动改表结构；
- 自动下载 QMT 数据；
- 自动运行模型；
- 自动把结果写入 Web 或模拟盘；
- 真复权 `$factor` / 全日池 / survivorship 修正（见 RD 文档诚实边界）。

下一步：用 native provider 做因子分析 / RD fin_factor；需要训练基线时再用 `.\dev.ps1 qlib-baseline`。
