# Qlib 与 RD-Agent 研究工作流

## 定位

Qlib 和 RD-Agent 属于离线研究层，不属于 Web 页面。

- RD-Agent：提出因子或模型方案、生成实验代码、迭代研究。
- Qlib：统一数据集、训练/验证/测试切分、模型训练、信号分析、组合回测和实验记录。
- Web：只消费经过人工确认的信号、计划和模拟盘结果。

## 当前环境

```text
Conda 环境: rd-agent
Python: 3.11.15
pyqlib: 0.9.7
rdagent: 0.8.0
```

```powershell
.\dev.ps1 qlib-info
.\dev.ps1 rd-health
```

## 标准研究漏斗

### 1. 数据基线

固定股票池和交易日历，检查停牌、涨跌停、复权、上市退市和未来数据泄漏，并记录数据截止日期。

### 2. 因子分析

检查覆盖率、缺失率、极值比例、Rank IC、ICIR、行业稳定性和市场状态稳定性。

### 3. 模型样本外验证

使用 `DatasetH` 和严格时间切分：`train < valid < test`。标准化、缺失值处理和特征选择只能在训练区间拟合。

### 4. 组合验证

统一接口：

```text
features -> model -> pred_score -> TopkDropoutStrategy -> portfolio analysis
```

必须明确 TopK、换仓数量、交易价格、涨跌停限制、成本、最低佣金和基准指数。

### 5. 人工晋级

只有在不同年份、行业和市场状态中都较稳定，参数扰动后不立即失效，交易成本后仍有空间，并且有失败条件时，才进入观察或模拟盘。

## RD-Agent 成本控制

```powershell
# 默认一步
.\dev.ps1 rd-factor

# 最多连续三步
.\dev.ps1 rd-factor 3
```

每步结束后人工检查：假设变化、未来数据、重复因子、区间依赖、是否值得继续。

## 实验目录

```text
experiments/<type>/<experiment_id>/
  experiment.yaml
  README.md
```

大文件放到 `artifacts/<experiment_id>/` 或 `mlruns/`，不提交 Git。

## 推荐 Qlib 组件

- `DataHandler` / `DatasetH`
- `LGBModel`
- `R` / Recorder
- `SignalRecord`
- `SigAnaRecord`
- `PortAnaRecord`
- `TopkDropoutStrategy`
- `qrun`

## 当前状态与下一步

2026-08-20 已落地标准接数漏斗（**外接主体是 Qlib**）：

```powershell
.\dev.ps1 qlib-provider artifacts\cn_real_staging C:\Users\lenovo\.qlib\qlib_data\cn_real_r500 --max-instruments 500
.\dev.ps1 qlib-profile research
.\dev.ps1 qlib-regen-h5
```

- research provider：`cn_real_r500`（2020–2026，500 只稳定抽样，`$factor=constant_1`）
- smoke：`cn_data_v2`（链路小样，可 `qlib-profile smoke` 切回）
- `qlib-baseline` = Alpha158+LGB **训练基线**，不是接数据命令

2026-07-23 曾跑通 staging → 实验内 provider → Alpha158 → LightGBM：

```powershell
.\dev.ps1 qlib-baseline `
  artifacts\20260722_qlib_data_baseline_v1 `
  artifacts\20260723_alpha158_lightgbm_baseline `
  --max-instruments 500
```

下一步优先：

1. 在 **research** profile 上解读 RD IC（勿用 smoke）；
2. 补真复权 `$factor` 与（可选）全池 provider；
3. walk-forward / 成本约束下的组合验证；
4. 门禁通过后再扩大 RD 挖因子轮次。

不要直接让 RD-Agent 修改 Web、交易计划或模拟盘主线。

官方参考：

- https://github.com/microsoft/qlib/blob/main/docs/component/workflow.md
- https://github.com/microsoft/qlib/blob/main/docs/component/recorder.md
- https://github.com/microsoft/qlib/blob/main/docs/start/initialization.md
