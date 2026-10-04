"""Summarise the latest qlib recorder into qlib_res.csv + ret.pkl.

这是覆盖 rdagent 默认版本的健壮版（放在 quantaalpha/factors/factor_template/ 下，
由 QlibFBWorkspace.inject_code_from_folder 注入到每个 workspace）。

改了什么、为什么
----------------
1. 【原版会崩】原版最后一行无条件执行
       ret_data_frame = latest_recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
       ret_data_frame.to_pickle("ret.pkl")
   一旦 workflow 里没挂 PortAnaRecord（5min 数据集就是这种情况：qlib 的组合回测
   要把行情 resample 到 1min，而数据集只有 5min，会抛
   "can't find a freq from Freq(5min) that can resample to 1min!"），
   这行就抛异常 —— ret.pkl 不生成，rdagent 的 execute() 直接返回 None，
   IC / RankIC 明明算出来了却传不回 pipeline。
   → 这里改成 try/except，拿不到就写占位。

2. 【占位不能滥发 —— 2026-10-04 的血泪教训】
   只写占位会让「qrun 压根没跑成功、读到上一次的陈旧 recorder」这种情况
   也伪装成成功：实测 17 条轨迹读到同一个 recorder，RankIC 全部等于
   0.04130751359431582（16 位小数一致），而因子表达式明明各不相同。
   这比直接报错危险得多 —— 假结果进了轨迹池，feedback 的 LLM 会被误导。
   → 现在加两道校验，任一不过就**不写 qlib_res.csv**，让 execute() 诚实返回 None：
     a) 新鲜度：recorder 的 start_time 必须晚于 combined_factors_df.parquet 的 mtime。
        （runner.py 每次 execute 前都会重写这个 parquet，所以它是可靠的"本轮开始"锚点；
         而 workspace 是按内容 hash 复用的，里面残留着上一轮的 recorder。）
     b) 有效性：pred.pkl 必须存在、非空、行数 > 0。

3. 【日志克制】只在关键节点输出一行，不刷屏。
"""

import pickle
from pathlib import Path

import pandas as pd
import qlib
from qlib.workflow import R

here = Path(__file__).resolve().parent

qlib.init()

# ---- 新鲜度锚点：本轮 factor 写入时间 ------------------------------------
anchor_path = here / "combined_factors_df.parquet"
anchor_ms = int(anchor_path.stat().st_mtime * 1000) if anchor_path.exists() else 0


def _ms(info, key):
    """recorder.info 里的时间有两种形态，都得认：
      - 毫秒时间戳（int/float）
      - **字符串** '2026-10-04 05:32:11'（本机 qlib 版本就是这种）

    2026-10-04 踩的坑：只按 int 处理，字符串 int() 抛 ValueError 被吞成 0，
    于是所有 recorder 都被判为「no valid end_time」而跳过 —— 9 条轨迹全部
    RankIC=None（successful=0），而指标其实已经算出来了（日志里能看到
    Rank IC=0.0454）。**校验写错比不校验更糟：它会把真结果变成失败。**
    """
    v = info.get(key)
    if v is None:
        return 0
    if isinstance(v, (int, float)):
        return int(v)
    try:
        return int(pd.to_datetime(str(v)).timestamp() * 1000)
    except Exception:
        return 0


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
            info = recorder.info
        except Exception as exc:
            print(f"WARN: cannot read recorder {recorder_id}: {exc}")
            continue
        end_time = _ms(info, "end_time")
        if end_time == 0:
            print(f"WARN: recorder {recorder_id} has no valid end_time; skipped")
            continue
        # 新鲜度：必须在本轮因子表写入之后才开始
        if anchor_ms and _ms(info, "start_time") and _ms(info, "start_time") < anchor_ms:
            print(f"WARN: recorder {recorder_id} is stale (start < factor mtime); skipped")
            continue
        if latest_recorder is None or end_time > _ms(latest_recorder.info, "end_time"):
            experiment_name = experiment
            latest_recorder = recorder

if latest_recorder is None:
    # 不写 qlib_res.csv、不写 ret.pkl —— 让上层看到"这次没结果"，而不是拿到旧结果
    print(f"ERROR: no fresh recorder found (anchor={anchor_ms}). qlib_res.csv NOT written.")
    raise SystemExit(1)

print(f"Latest recorder: {latest_recorder}")

# ---- 有效性：预测必须真的存在且非空 --------------------------------------
try:
    pred = latest_recorder.load_object("pred.pkl")
except Exception as exc:
    print(f"ERROR: pred.pkl unreadable ({exc}). qlib_res.csv NOT written.")
    raise SystemExit(1)

if pred is None or len(pred) == 0:
    print("ERROR: pred.pkl is empty. qlib_res.csv NOT written.")
    raise SystemExit(1)

metrics = pd.Series(latest_recorder.list_metrics())
if metrics.empty:
    print("ERROR: recorder has no metrics. qlib_res.csv NOT written.")
    raise SystemExit(1)

output_path = here / "qlib_res.csv"
metrics.to_csv(output_path)
rank_ic = metrics.get("Rank IC")
print(f"qlib_res.csv written: n_pred={len(pred)} RankIC={rank_ic}")

# ---- ret.pkl：组合回测可能不存在，占位即可（只用于画净值图） -------------
try:
    ret_df = latest_recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    ret_df.to_pickle(here / "ret.pkl")
    print("ret.pkl written from portfolio_analysis")
except Exception as exc:
    pd.DataFrame(columns=["return", "bench", "cost"]).to_pickle(here / "ret.pkl")
    with open(here / "ret.pkl.placeholder", "w", encoding="utf-8") as fh:
        fh.write("portfolio_analysis unavailable; IC/RankIC metrics are still valid.\n")
        fh.write(f"reason: {exc}\n")
    print(f"ret.pkl placeholder written ({type(exc).__name__})")
