# tools/ 脚本索引与约定

> 本目录是**研究脚本工作区**（159 个 .py）。研究规范见 `docs/research_sop.md`，
> 交付流程见 `docs/warfare_intake_sop.md`，战法状态见 `docs/warfares_catalog.md`。

---

## 命名约定（请遵守）

| 前缀 | 含义 | 是否入库 |
|:---|:---|:---|
| `xxx.py` | **正式脚本**：可复现的研究/数据脚本，有结论价值 | ✅ 提交 |
| `_xxx.py` | **诊断/一次性脚本**：排障用，结论已沉淀到文档 | ✅ 提交（保留下划线标记） |
| `_xxx_out.json` | 正式脚本的结果快照（小体积，供他人核验） | ✅ 提交 |
| `*_patch.py` | rdagent/依赖库的**运行时补丁**（不改上游） | ✅ 提交 |
| `*.log` / `*.dmp` | 运行日志、崩溃转储 | ❌ 已 gitignore |

**新脚本落位规则**：
1. 一个战法的验证脚本 → `validate_<战法名>.py` 或 `<战法名>_backtest.py`
2. 诊断/探针 → 加 `_` 前缀
3. 结果快照 → `<脚本名>_out.json`，并在脚本里用 `QA_OUT_DIR` 控制输出目录

---

## 一、数据管道

| 脚本 | 作用 |
|:---|:---|
| `backfill_etf_qmt_local.py` | QMT datadir → TimescaleDB（ETF 池，**含日期口径修复**） |
| `download_qmt_etf_recent.py` | xtdata 拉 ETF 日线到 datadir（qmt312 环境跑） |
| `backfill_turnover_qmt.py` | 回填 `turnover_rate` / `float_shares` 缺口 |
| `backfill_etf_qmt.py` | 早期 ETF 回填（已被 `_local` 版取代） |
| `gen_pv_5min.py` | 生成 5min 价量 HDF5 |
| `qlib_refresh.py` / `qlib_regen_h5.py` / `qlib_data_status.py` | Qlib 数据刷新/状态 |

> **全 A 日线/分钟采集不在本目录**，在独立项目 `D:/qqmt/quant_datahub`：
> `python -m datahub daily|minute|gap|dump-qlib`（qmt312 环境）。

## 二、框架迁移（Qlib，2026-10-14 起）

| 脚本 | 作用 |
|:---|:---|
| `qlib_f4_pilot.py` | **Qlib 表达式引擎试点**：F4 与手写版对齐验证（差 0.8%） |

## 三、执行层 L4

| 脚本 | 作用 |
|:---|:---|
| `exec_layer_e0.py` | 前置关：信号毛超额是否仍为正（决定 L4 值不值得做） |
| `exec_layer_e1_buy.py` | 买入执行方式对比（C0 开盘 / C1 VWAP / C2 TWAP / C3 缺口过滤 / C4 限价） |
| `turnover_cost_curve.py` / `etf_exec_cost.py` | 换手-成本曲线 |

## 四、战法验证（核心研究）

| 脚本 | 作用 |
|:---|:---|
| `fibonacci_test.py` | 斐波那契三兄弟（回撤位/时间周期/均线周期）检验 |
| `why_stock_tdx_fails.py` | 战法失效四层归因（信号密度/成本/风格/行业） |
| `validate_tdx_feilong.py` / `validate_tdx_feilong_chip.py` | 飞龙 / 战法2（涨停回调） |
| `dixi_fullpanel_backtest.py` / `dixi_deep_analysis.py` | 低吸全 A 事件驱动 |
| `w3c_limitup_*.py` / `w3c_executability_check.py` / `w3c_final_verify.py` | 涨停战法 |
| `warfare2_*.py` | 战法2 信号/评分/OOS |
| `yelan_warfare_backtest.py` / `run_feilong_plan.py` | 其它战法 |
| `verify_feilong*.py`（6 个） | 飞龙多轮验证 |
| `stock_gate_signals.py` / `midhigh_universe_signals.py` | 门控/中高成交额宇宙信号 |

## 五、基准与口径复核（方法论）

| 脚本 | 作用 |
|:---|:---|
| `rebuild_bench_rerun.py` | 用不同基准重跑 F4/F5/F6/F7（IS + OOS） |
| `f4_bench_compare.py` | F4 多基准对比（等权/中位数/同档/买入持有） |
| `is_2020_2024_tribench.py` / `oos_2025_tribench.py` | 长 IS / OOS 三口径复核 |
| `stable_bucket_rerun.py` | 滞后 60 日分档重跑（杀每日分档未来函数） |
| `industry_neutral_rescan.py` | 行业中性化重扫（新增「同行业」口径） |
| `realistic_backtest.py` / `backtest_reality_check.py` | 现实性检查 |

## 六、ETF 研究（17 个）

`etf_pool_sensitivity_v1~v4.py`（池敏感性）、`etf_tsmom_*.py`（时间序列动量）、
`etf_vt_lambda_grid.py`（波动率目标×逆波动网格）、`etf_tribench_check.py`（三口径）、
`etf_recalib_check.py`（底盘复现）、`etf_variant_check.py`、`etf_dualmom_check.py`、
`etf_pool38_check.py`、`etf_backtest.py`、`etf_timeseries_signals.py`、`etf_calibration_check.py`、`etf_algo_align_check.py`

## 七、因子

`factor_neutral_test.py`（中性化 RankIC）、`factor_lowvol_true_neutral.py`、
`build_traditional_factors.py` / `build_validated_factors.py` / `build_panel_2011.py`、
`eval_factors.py`、`rank_single_factors.py`、`candidate_factors_check.py`、`chip_distribution.py`（筹码还原）

## 八、组合 / 风险 / 策略

| 脚本 | 作用 |
|:---|:---|
| `portfolio_risk.py` / `portfolio_risk_yearly.py` | 组合构建（ERC/InvVol/MinVar + 成本） |
| `build_ensemble.py` / `weight_scheme_test.py` | 集成与权重方案 |
| `rebalance_frequency_study.py` / `strategy_bh_vs_rebalance.py` | 再平衡频率 |
| `strategy_final.py` / `strategy_long_history.py` / `pool_experiment.py` | 策略主线 |
| `make_order_list.py` / `make_trading_plan.py` / `trading_checklist.py` | **下单清单 / 交易计划（实盘用）** |
| `make_backtest_report.py` / `make_etf_report.py` | 报告生成 |

## 九、诊断脚本（`_` 前缀，19 个）

| 脚本 | 诊断对象 |
|:---|:---|
| `_benchmark_sanity.py` | 等权 vs 中位数基准扭曲 |
| `_decile_anomaly_diag.py` | 每日分档未来函数 |
| `_db_float_shares_gap.py` / `_db_missing_months_diag.py` | `float_shares` 14 个月空窗 |
| `_db_coverage.py` / `_db_monthly_audit.py` / `_db_density_check.py` / `_db_field_audit.py` / `_db_null_probe.py` / `_db_days_per_month.py` / `_db_etf_check.py` | 库表覆盖/字段/空值审计 |
| `_signal_sparsity_diag.py` | 信号稀疏度 |
| `_hang_probe.py` / `_nan_probe.py` / `_probe_lowvol_sign.py` / `_probe_rankweight.py` | 探针 |
| `_validate_eval.py` / `_norm_selftest.py` / `_verify_menu_numbers.py` | 自检 |

## 十、环境补丁（`*_patch.py`）

rdagent / qlib / 依赖库的运行时补丁，**不改上游源码**：
`windows_qlib_patch.py`、`gym_numpy_patch.py`、`factor_datetime_patch.py`、`factor_empty_patch.py`、
`json_parse_patch.py`、`parquet_read_patch.py`、`safe_ic_patch.py`、`chat_retry_patch.py`、
`cpu_throttle_patch.py`、`feedback_metrics_patch.py`、`final_decision_patch.py`、
`research_prompt_patch.py`、`ui_metrics_patch.py`、`qlib_data_profile_patch.py`、`sitecustomize.py`

---

## 使用注意

1. **跑任何读 `bars_1d` 的脚本前**，先确认 `(asset_code, trade_date)` 唯一
   （2026-10-14 曾因 42.8 万组重复行导致信号全错，详见 `docs/warfare_intake_sop.md` §9）。
2. **规模分档一律用滞后 60 日 `amount`**，禁用当日排名（未来函数）。
3. **低吸类必带 ≤8% 止损**对照。
4. 输出统一走 `QA_OUT_DIR` 环境变量（默认临时目录），避免污染仓库。
