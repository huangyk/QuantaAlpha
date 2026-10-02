"""演化因子的时序样本外（OOS）复验器。

背景 / 为什么要做这个
----------------------
AlphaAgent 演化挖掘在「全样本」上跑回测并按 RankIC 选 best，这个「选 best」的动作
本身就吃进了未来信息（selection bias）：一条轨迹在 2020-2026 全段 RankIC=0.064，
并不代表它在训练段之后仍有效。真正能拿来说话的是**时序分段**表现：

    train (2020~2023) -> valid (2024) -> test (2025~2026 纯样本外)

判读规则（避免自欺）：
  - test 段 RankIC 与 train 段同量级 (=0.6~1.0x) -> 信号真实可用；
  - test 段 RankIC 明显衰减 (<0.5x)             -> 主要是选择偏差 + 过拟合，不能上；
  - 单因子 RankIC 高但 TopK 组合扣费后年化为负 -> 换手太贵，无交易价值。

本脚本只做**评估**，不参与任何挖矿/选因子，因此不存在把 test 信息泄漏回因子的问题。

用法
----
    python tools/oos_validate.py --pool log/latest_trajectory_pool.json --k 5
    python tools/oos_validate.py --pool log/latest_trajectory_pool.json --k 3 \
        --train-start 2020-01-01 --train-end 2023-12-31 \
        --valid-start 2024-01-01 --valid-end 2024-12-31 \
        --test-start 2025-01-01 --test-end 2026-08-18 \
        --topk 50 --horizon 1 --cost-bps 5 --out data/results/oos_validate_topk.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("oos_validate")

# qlib 表达式里 $return 的标准定义（与 quantaalpha/backtest/factor_loader.py 一致）
RET_FIELD = "($close/Ref($close,1)-1)"
DEFAULT_QLIB_DIR = "C:/Users/lenovo/.qlib/qlib_data/cn_real_full"

# rdagent(TS_*) 算子 -> qlib 原生算子。
# qlib 已注册算子只有驼峰这 52 个（Abs/Mean/Std/Max/Min/Sum/Rank/Quantile/Ref/Corr/Sign...），
# 没有 TsRank / Delay / TsMean 这些 rdagent 别名，不转换会报 "operator not registered"。
# DELAY -> Ref 尤其关键：qlib 里根本没有 Delay。
OP_CONVERSIONS = {
    "DELAY": "Ref",
    "TS_DECAY_LINEAR": "WMA",
    "TS_ARGMAX": "IdxMax",
    "TS_ARGMIN": "IdxMin",
    # 纯函数运算符：qlib 注册名同样是驼峰，全大写写法需一并归一
    "ABS": "Abs",
    "MIN": "Min",
    "MAX": "Max",
    "SIGN": "Sign",
    # 其余 TS_XXX 走通用规则（TS_CORR->Corr、TS_RANK->Rank ...），见 _ts_to_camel
    "$return": RET_FIELD,
}

_CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def _ts_to_camel(name: str) -> str:
    """TS_CORR -> Corr：qlib 注册名一律驼峰，去掉 TS_ 前缀后整体首字母大写。

    注意必须用 capitalize() 而不是仅首字母大写：TS_MEAN 的尾部是 "MEAN"，
    只改首字母仍会得到 "MEAN"，qlib 里注册的是 "Mean"。
    """
    tail = name[3:] if name.upper().startswith("TS_") else name
    return tail.capitalize()


# 实参要能吃一层嵌套括号，否则 TS_ZSCORE(ABS($return), 10) 这类写法会被漏掉，
# 落到后面的通用归一里变成 qlib 未注册的 Zscore 而被跳过（算不出来，虽安全但白瞎一条因子）。
_ARG = r"((?:[^()]|\([^()]*\))+?)"
_ZSCORE_RE = re.compile(rf"\bTS_ZSCORE\s*\(\s*{_ARG}\s*,\s*(\d+)\s*\)", re.I)
_PCTCHG_RE = re.compile(rf"\bTS_PCTCHANGE\s*\(\s*{_ARG}\s*,\s*(\d+)\s*\)", re.I)


def _expand_ts_funcs(expr: str, max_round: int = 3) -> str:
    """把没有单一对应算子、只能用基础算子组合还原的 TS_* 展开成 qlib 原生表达式。

    目前两类：
      TS_ZSCORE(x, N)    -> (x - Mean(x, N)) / (Std(x, N) + 1e-8)
      TS_PCTCHANGE(x, N) -> x / Ref(x, N) - 1

    多轮执行以支持嵌套（如 TS_ZSCORE(ABS($return), 10)）：正则的实参模式
    [^,()]+ 只吃不含括号的表达式，第一轮替换掉最外层后才轮到内层。
    """
    result = expr
    for _ in range(max_round):
        new = _ZSCORE_RE.sub(
            lambda m: f"(({m.group(1)} - Mean({m.group(1)}, {m.group(2)}))"
                      f" / (Std({m.group(1)}, {m.group(2)}) + 1e-8))",
            result,
        )
        new = _PCTCHG_RE.sub(
            lambda m: f"({m.group(1)} / Ref({m.group(1)}, {m.group(2)}) - 1)", new
        )
        if new == result:
            break
        result = new
    return result


def normalize_expr(expr: str) -> str:
    """把平台简写表达式还原成 qlib 原生表达式（口径与挖掘时一致）。"""
    result = _expand_ts_funcs(expr)
    for old, new in OP_CONVERSIONS.items():
        result = result.replace(old, new)
    # 通用 TS_* 归一：逐个函数调用位置替换，避免误伤表达式里的普通大写串
    result = _CALL_RE.sub(lambda m: f"{_ts_to_camel(m.group(1))}{m.group(0)[len(m.group(1)):]}", result)
    return result


def unsupported_operators(expr: str, known: set[str]) -> list[str]:
    """找出表达式里 qlib 引擎不认识的算子名（用于提前跳过，避免整批 D.features 崩掉）。"""
    ops = {m for m in _CALL_RE.findall(expr)} - {"Feature", "PFeature"}
    return sorted(o for o in ops if o not in known)


# ---------------------------------------------------------------- factor loading
def is_qlib_compatible(expr: str, known: set[str]) -> bool:
    """能否用 qlib 表达式引擎复算：排除三元 ? : 以及 qlib 未注册的算子。

    这两类因子平台实际是走 pandas 因子计算器（见 backtest/factor_calculator.py）算的，
    本脚本用 qlib 引擎，因此跳过并如实记录，不做假结果。
    """
    if "?" in expr or ":" in expr:
        return False
    return not unsupported_operators(expr, known)


def load_top_factors(pool_path: str, k: int, known: set[str]) -> tuple[list[dict], int]:
    """从轨迹池里按 RankIC 取前 k 条轨迹，返回其全部因子（按表达式去重）。"""
    with open(pool_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    per_traj = []
    for tid, t in (data.get("trajectories") or {}).items():
        metrics = t.get("backtest_metrics") or {}
        ric = metrics.get("RankIC")
        if ric is None:
            continue
        per_traj.append((float(ric), tid, t.get("phase"), t.get("factors") or []))

    per_traj.sort(key=lambda x: -x[0])

    out, seen, skipped = [], set(), 0
    for ric, tid, phase, factors in per_traj[:k]:
        for f in factors:
            expr = (f.get("expression") or "").strip()
            if not expr or expr in seen:
                continue
            # 必须先归一化再检查算子，否则 TS_MEAN/ABS/DELAY 这些 rdagent 别名全会被误判成未注册
            norm_expr = normalize_expr(expr)
            if not is_qlib_compatible(norm_expr, known):
                skipped += 1
                logger.warning(
                    "跳过不可复算因子 [%s]：%s",
                    f.get("name"), unsupported_operators(norm_expr, known) or ["ternary ? :"],
                )
                continue
            seen.add(expr)
            out.append(
                {
                    "trajectory_id": tid,
                    "phase": phase,
                    "traj_rank_ic": ric,
                    "name": f.get("name"),
                    "expression": expr,
                }
            )
    return out, skipped


# ---------------------------------------------------------------- metric helpers
def find_levels(df: pd.DataFrame) -> tuple[int, int]:
    """定位 (datetime 层, instrument 层)。

    qlib 的 MultiIndex 层序不保证（可能是 (datetime, instrument) 也可能是 (instrument, datetime)），
    写死 level=0 会把「按个股分组」当成「按日分组」，成交量少时表现为每天只有 1~2 行、RankIC 完全失真。
    因此按 index 名 datetime 定位，退化时取唯一值个数最少的那层（交易日数 << 标的数）。
    """
    names = list(df.index.names)
    dt_level = None
    if "datetime" in names:
        dt_level = names.index("datetime")
    else:
        for i in range(len(names)):
            if df.index.get_level_values(i).nunique() < 1000:
                dt_level = i
                break
    if dt_level is None:
        dt_level = 0
    ins_level = (dt_level + 1) % len(names) if len(names) > 1 else -1
    return dt_level, ins_level


def calc_daily_ic(df: pd.DataFrame, factor_col: str, fwd_col: str, min_cnt: int = 30, dt_level: int = 0) -> pd.Series:
    """按日截面相关（spearman = RankIC；pearson = IC），返回daily 序列。"""
    sub = df[[factor_col, fwd_col]].replace([np.inf, -np.inf], np.nan).dropna()
    res = {}
    for day, g in sub.groupby(level=dt_level):
        if len(g) < min_cnt:
            continue
        c = g.corr(method="spearman")
        val = c.iloc[0, 1] if factor_col != fwd_col else c
        if isinstance(val, pd.DataFrame):
            val = val.iloc[0, 1]
        res[day] = val
    out = pd.Series(res).dropna()
    # qlib 给的日期可能是 str 也可能是 Timestamp，统一成 Timestamp 才能跟配置段边界比较
    out.index = pd.to_datetime(out.index, errors="coerce")
    return out.dropna()


def icir(ic: pd.Series, ann: int = 252) -> float:
    """ICIR = 均值 / 标准差 * sqrt(年化交易日数)。"""
    if ic.empty or ic.std() == 0:
        return float("nan")
    return float(ic.mean() / ic.std() * np.sqrt(ann))


def topk_equity_curve(
    df: pd.DataFrame, factor_col: str, fwd_col: str, topk: int, cost_bps: float,
    dt_level: int = 0, ins_level: int = 1,
) -> pd.DataFrame:
    """每日截面取因子值最高 topk 只等权持有一期，按日换手扣费。

    返回 (ret, turnover, cost) 日频序列。
    """
    rows = []
    prev_hold: set[str] | None = None
    for day, g in df[[factor_col, fwd_col]].replace([np.inf, -np.inf], np.nan).dropna().groupby(level=dt_level):
        g = g.sort_values(factor_col, ascending=False).head(topk)
        hold = set(g.index.get_level_values(ins_level))
        fwd_ret = float(g[fwd_col].mean())
        if prev_hold is None:
            turnover = 1.0
        else:
            turnover = 1.0 - len(hold & prev_hold) / max(len(hold), 1)
        cost = turnover * cost_bps / 1e4
        rows.append({"datetime": day, "ret": fwd_ret, "turnover": turnover, "cost": cost})
        prev_hold = hold

    out = pd.DataFrame(rows).set_index("datetime")
    out.index = pd.to_datetime(out.index, errors="coerce")
    out["net_ret"] = out["ret"] - out["cost"]
    return out.dropna(subset=["net_ret"])


def perf_stats(equity: pd.Series, periods_per_year: int = 252) -> dict:
    """净收益序列 -> 年化 / 波动 / 夏普 / 最大回撤。"""
    if equity.empty:
        return {"annualized": float("nan"), "vol": float("nan"), "sharpe": float("nan"), "max_drawdown": float("nan")}
    curve = (1 + equity).cumprod()
    ann = float((1 + equity).mean() ** periods_per_year - 1)
    vol = float(equity.std() * np.sqrt(periods_per_year))
    sharpe = float(equity.mean() / equity.std() * np.sqrt(periods_per_year)) if equity.std() else float("nan")
    mdd = float((curve / curve.cummax() - 1).min())
    return {"annualized": ann, "vol": vol, "sharpe": sharpe, "max_drawdown": mdd}


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="演化因子时序样本外复验")
    ap.add_argument("--pool", required=True, help="轨迹池 json（如 log/latest_trajectory_pool.json）")
    ap.add_argument("--k", type=int, default=3, help="取 RankIC 最高的前 k 条轨迹")
    ap.add_argument("--qlib-dir", default=os.environ.get("QLIB_DATA_DIR", DEFAULT_QLIB_DIR))
    ap.add_argument("--market", default="all", help="qlib instruments")
    ap.add_argument("--train-start", default="2020-01-01")
    ap.add_argument("--train-end", default="2023-12-31")
    ap.add_argument("--valid-start", default="2024-01-01")
    ap.add_argument("--valid-end", default="2024-12-31")
    ap.add_argument("--test-start", default="2025-01-01")
    ap.add_argument("--test-end", default="2026-08-18")
    ap.add_argument("--horizon", type=int, default=1, help="前瞻期（交易日），1=次日收益")
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--cost-bps", type=float, default=5.0, help="单边成本（基点），A股双边常见 5~10")
    ap.add_argument("--out", default="data/results/oos_validate.csv")
    args = ap.parse_args()

    import qlib
    from qlib.data import D
    from qlib.data.ops import Operators

    qlib.init(provider_uri=args.qlib_dir, region="cn")
    known_ops = set(Operators._ops.keys())
    logger.info("qlib 已注册算子 %d 个", len(known_ops))

    # ---- 一次性算全时段因子值 ----
    start = args.train_start
    end = args.test_end

    # 注意：本项目 qlib 被 patch 过，list_instruments 不直接收 market 参数，
    # 必须先 D.instruments(market) 拿到 instruments 对象再传（见 backtest/runner.py:591）。
    codes = D.list_instruments(D.instruments(args.market), start_time=start, end_time=end, as_list=True)
    logger.info("instruments=%d, provider=%s", len(codes), args.qlib_dir)

    factors, skipped = load_top_factors(args.pool, args.k, known_ops)
    if not factors:
        logger.error("池里没有带 RankIC 的轨迹，或字段格式不匹配")
        return 2
    logger.info(
        "载入 %d 个待复验因子（来自 RankIC top-%d 轨迹）%s",
        len(factors), args.k,
        f"，跳过 {skipped} 个含三元算子的因子" if skipped else "",
    )
    exprs = [normalize_expr(f["expression"]) for f in factors]
    names = [f"F{i}_{f['name']}" for i, f in enumerate(factors)]

    # 逐个因子单独试算：qlib 对算子元数要求严格（如 Rank 必须给 N），
    # 混在一次 D.features 里任何一个算子炸掉都会拖垮整批，这里逐个隔离、失败即跳过并如实记录。
    logger.info("计算因子截面（%d 表达式）...", len(exprs))
    series_list, ok_names = [], []
    for name, expr in zip(names, exprs):
        try:
            s = D.features(codes, [expr], start_time=start, end_time=end, freq="day").iloc[:, 0]
            s.name = name
            series_list.append(s)
            ok_names.append(name)
        except Exception as exc:  # noqa: BLE001 - 复算失败要如实记录而不是中断整批
            logger.warning("因子 %s 复算失败（跳过）：%s", name, exc)
    if not series_list:
        logger.error("没有任何因子能用 qlib 引擎复算，终止")
        return 3
    f_df = pd.concat(series_list, axis=1)
    names = ok_names
    logger.info("factor shape=%s（成功复算 %d/%d）", f_df.shape, len(ok_names), len(exprs))

    # ---- 前瞻收益 ----
    # 必须与平台口径一致：configs/backtest.yaml 的 dataset.label 就是
    #   Ref($close, -2) / Ref($close, -1) - 1
    # 语义是「t 日收盘出信号、t+1 才成交」，所以 t 日信号要对应 t+1 -> t+2 的收益。
    # 若退化成 Ref($close,-h)/$close-1（让 t 日信号直接赚 t->t+1 的钱），
    # 实测会把 RankIC 抬高约 60%（同一因子 0.0174 vs 0.0281），且对短周期因子
    # 会放大到符号反转，复验结论完全不可比。
    fwd_expr = f"Ref($close, -{args.horizon + 1}) / Ref($close, -{args.horizon}) - 1"
    fwd = D.features(codes, [fwd_expr], start_time=start, end_time=end, freq="day").iloc[:, 0]
    fwd.name = "fwd"
    logger.info("fwd shape=%s", fwd.shape)

    df = f_df.join(fwd, how="inner")
    dt_level, ins_level = find_levels(df)
    logger.info(
        "joined shape=%s；index levels=%s -> datetime层=%d, instrument层=%d",
        df.shape, [str(x) for x in df.index.names], dt_level, ins_level,
    )

    # ---- 分段统计 ----
    segments = [
        ("train", args.train_start, args.train_end),
        ("valid", args.valid_start, args.valid_end),
        ("test", args.test_start, args.test_end),
    ]

    summary = []
    per_day_ric = {}
    for name in names:
        ric_all = calc_daily_ic(df, name, "fwd", dt_level=dt_level)
        row = {"factor": name, "traj_rank_ic": None}
        logger.info(
            "[diag] %s: n_days=%d mean=%.5f std=%.5f", name, len(ric_all), ric_all.mean(), ric_all.std()
        )
        row["ric_all"] = float(ric_all.mean()) if not ric_all.empty else float("nan")
        row["icir_all"] = icir(ric_all)
        per_day_ric[name] = ric_all
        for seg, s, e in segments:
            # index 是 datetime，配置的段边界是字符串，必须统一成 Timestamp 再切
            ric = ric_all[(ric_all.index >= pd.Timestamp(s)) & (ric_all.index <= pd.Timestamp(e))]
            row[f"ric_{seg}"] = float(ric.mean()) if not ric.empty else float("nan")
            row[f"icir_{seg}"] = icir(ric)
        summary.append(row)
        # 关联回轨迹 RankIC
        idx = int(name.split("_")[0][1:])
        summary[-1]["traj_rank_ic"] = factors[idx]["traj_rank_ic"]
        summary[-1]["phase"] = factors[idx]["phase"]
        summary[-1]["expression"] = factors[idx]["expression"]

    summ_df = pd.DataFrame(summary)

    # ---- TopK 组合 + 扣费敏感性 ----
    top_name = summ_df.sort_values("ric_all", ascending=False)["factor"].iloc[0]
    logger.info("组合基准因子: %s", top_name)
    # gross 必须是「零成本」那一档的年化。
    # 注意 perf_stats 吃的是 net_ret（已扣费），同一序列无论在哪个 bps 下都已经是净收益，
    # 若把 st["annualized"] 同时赋给 gross_ann，毛/净两列会恒等、扣费前后的毛收益对比彻底失真。
    gross_eq = topk_equity_curve(df, top_name, "fwd", args.topk, 0.0, dt_level, ins_level)

    def _seg_stats(eq: pd.DataFrame, seg: str, s: str, e: str) -> dict:
        q = eq[(eq.index >= pd.Timestamp(s)) & (eq.index <= pd.Timestamp(e))]
        return {
            "series": q["net_ret"],
            "days": len(q),
            "turnover": float(q["turnover"].mean()) if not q.empty else float("nan"),
        }

    gross_cache = {seg: _seg_stats(gross_eq, seg, s, e) for seg, s, e in segments}
    sens_rows = []
    for bps in [0.0, 3.0, 5.0, 10.0, 20.0]:
        eq = topk_equity_curve(df, top_name, "fwd", args.topk, bps, dt_level, ins_level)
        for seg, s, e in segments:
            q = _seg_stats(eq, seg, s, e)
            st = perf_stats(q["series"])
            g = gross_cache[seg]
            sens_rows.append(
                {
                    "cost_bps": bps,
                    "segment": seg,
                    "days": q["days"],
                    "avg_turnover": q["turnover"],
                    "gross_ann": perf_stats(g["series"])["annualized"],
                    "net_ann": st["annualized"],
                    "net_minus_gross": st["annualized"] - perf_stats(g["series"])["annualized"],
                    "sharpe": st["sharpe"],
                    "max_drawdown": st["max_drawdown"],
                }
            )
    sens_df = pd.DataFrame(sens_rows)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    summ_df.to_csv(args.out, index=False, encoding="utf-8-sig")
    sens_path = str(Path(args.out).with_name(Path(args.out).stem + "_cost.csv"))
    sens_df.to_csv(sens_path, index=False, encoding="utf-8-sig")

    # ---- 打印 ----
    print("\n" + "=" * 96)
    print(f"样本外复验（horizon={args.horizon}, topk={args.topk}, cost={args.cost_bps}bps单边）")
    print("=" * 96)
    show = summ_df[["factor", "phase", "traj_rank_ic", "ric_all", "icir_all", "ric_train", "ric_valid", "ric_test"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n衰减比 test/train:")
    for _, r in summ_df.iterrows():
        if r["ric_train"] and not pd.isna(r["ric_train"]) and r["ric_train"] != 0:
            print(f"  {r['factor']:<45} {r['ric_test']/r['ric_train']:.2f}x")
    print("\n扣费敏感性（基准因子 %s）：" % top_name)
    order = [0.0, 3.0, 5.0, 10.0, 20.0]
    print("毛收益（零成本口径）:")
    print(
        sens_df.pivot_table(index="cost_bps", columns="segment", values="gross_ann")
        .reindex(order)
        .to_string(float_format=lambda v: f"{v:.4f}")
    )
    print("\n净收益:")
    print(
        sens_df.pivot_table(index="cost_bps", columns="segment", values="net_ann")
        .reindex(order)
        .to_string(float_format=lambda v: f"{v:.4f}")
    )
    print("\n成本吃掉的比例 (1 - net/gross):")
    ratio = pd.DataFrame(
        {
            seg: 1 - sens_df[(sens_df.segment == seg)].set_index("cost_bps")["net_ann"]
            / sens_df[(sens_df.segment == seg)].set_index("cost_bps")["gross_ann"]
            for seg in ("train", "valid", "test")
        }
    ).reindex(order)
    print(ratio.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\n结果已写：{args.out}\n           {sens_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
