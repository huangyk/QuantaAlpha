# RD-Agent Windows 补丁

`dev.ps1` 执行 `rd-health` / `rd-factor` / `qlib` 时设置 `PYTHONPATH=tools`，`sitecustomize.py` 自动加载：

| 文件 | 作用 |
|------|------|
| `sitecustomize.py` | 入口 |
| `json_parse_patch.py` | Agnes JSON 解析 |
| `factor_datetime_patch.py` | 因子 datetime index |
| `windows_qlib_patch.py` | Windows qrun/python 路径 |
| `read_exp_res_fixed.py` | 读 qlib 回测结果 |
| `gym_numpy_patch.py` | gym→gymnasium，避免 NumPy2 崩溃 |
| `safe_ic_patch.py` | 预测近似常数时 IC 不再硬崩 |
| `final_decision_patch.py` | Agnes 缺 final_decision / output_format_* 时软失败继续 |
| `factor_empty_patch.py` | 因子 MultiIndex/concat 崩溃 → FactorEmptyError，跳过本轮 |
| `ui_metrics_patch.py` | Streamlit UI 缺 IC/收益指标时显示 NaN/可用指标，不 KeyError；`rdagent ui` 需 `PYTHONPATH=tools` |
| `research_prompt_patch.py` | 把 `config/rd_research_prompt.md` 注入 Research 假设生成 |

说明见 [docs/RD_AGENT.md](../docs/RD_AGENT.md)。
