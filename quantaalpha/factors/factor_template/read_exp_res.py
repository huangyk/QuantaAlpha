"""Summarise the latest qlib recorder into qlib_res.csv + ret.pkl.

这是覆盖 rdagent 默认版本的健壮版（放在 quantaalpha/factors/factor_template/ 下，
由 QlibFBWorkspace.inject_code_from_folder 注入到每个 workspace）。

为什么必须改：
  原版最后一行无条件执行
      ret_data_frame = latest_recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
      ret_data_frame.to_pickle("ret.pkl")
  一旦 workflow 里没挂 PortAnaRecord（5min 数据集就是这种情况：qlib 的组合回测
  要把行情 resample 到 1min，而数据集只有 5min，会抛
  "can't find a freq from Freq(5min) that can resample to 1min!"），
  这行就会抛异常 —— 于是 ret.pkl 根本不生成。
  而 rdagent 的 QlibFBWorkspace.execute() 是这么写的：
      if not (workspace/"ret.pkl").exists():
          logger.error("No result file found."); return None, log
  → 返回 None → 上层 _extract_metrics 拿不到东西 → **IC / RankIC 明明算出来了，
    pipeline 里却是 RankIC=None，feedback 的 LLM 只会看到"因子失败了"。**

  所以这里：metrics 照常落 qlib_res.csv（IC / Rank IC 都在里面），
  ret.pkl 拿不到就写一个占位（它只用于画净值图，不影响 IC 类指标）。
"""

import pickle
from pathlib import Path

import pandas as pd
import qlib
from qlib.workflow import R

qlib.init()

experiments = R.list_experiments()

experiment_name = None
latest_recorder = None
for experiment in experiments:
    recorders = R.list_recorders(experiment_name=experiment)
    for recorder_id in recorders:
        if recorder_id is None:
            continue
        recorder = R.get_recorder(recorder_id=recorder_id, experiment_name=experiment)
        try:
            end_time = recorder.info["end_time"]
        except Exception as exc:
            print(f"Error: {exc}")
            continue
        if end_time is None:
            print(f"Warning: Recorder {recorder_id} has no valid end time")
            continue
        if latest_recorder is None or end_time > latest_recorder.info["end_time"]:
            experiment_name = experiment
            latest_recorder = recorder

if latest_recorder is None:
    print("No recorders found")
else:
    print(f"Latest recorder: {latest_recorder}")

    metrics = pd.Series(latest_recorder.list_metrics())
    output_path = Path(__file__).resolve().parent / "qlib_res.csv"
    metrics.to_csv(output_path)
    print(f"Output has been saved to {output_path}")

    # 组合回测结果可能不存在（未挂 PortAnaRecord 时），不能因此让整个脚本崩掉
    here = Path(__file__).resolve().parent
    try:
        ret_df = latest_recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
        ret_df.to_pickle(here / "ret.pkl")
        print("ret.pkl written from portfolio_analysis")
    except Exception as exc:
        print(f"WARN: portfolio_analysis unavailable ({exc}); writing placeholder ret.pkl")
        # 占位：rdagent 的 execute() 只检查文件是否存在并 read_pickle 画图，
        # 空 DataFrame 不影响 qlib_res.csv 里的 IC / RankIC。
        pd.DataFrame(columns=["return", "bench", "cost"]).to_pickle(here / "ret.pkl")
        with open(here / "ret.pkl.placeholder", "w", encoding="utf-8") as fh:
            fh.write("portfolio_analysis unavailable; IC/RankIC metrics are still valid.\n")
            fh.write(f"reason: {exc}\n")
