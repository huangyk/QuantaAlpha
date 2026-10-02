"""战法原子因子（F1-F10）严谨评估器。

直接计算每个因子的 IC / RankIC / ICIR（按时间截面相关），并做简单的 TopK 多头回测，
在 train/valid/test 三段分别报告，避免过拟合错觉。

用法：
  python tools/eval_factors.py \
      --provider D:/qqmt/qlib_data/cn_5m_full --freq 5min \
      --market liquid500 --start 2026-01-05 --end 2026-09-30 \
      --horizon 12 --topk 50 --out D:/qqmt/QuantaAlpha/data/results/eval_5m.csv

因子在 pandas 中从原始 OHLCV 计算（不依赖 qlib 表达式引擎，避免 SMA/TS_MAX 兼容问题）。
F10 板块强依赖外部板块资金流，本环境无，标记 N/A 跳过（诚实）。
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_factors")

FIELDS = ["$open", "$high", "$low", "$close", "$volume", "$vwap"]


# ---------------- 因子定义（输入单只股票的 OHLCV DataFrame，返回 0/1 或连续值 Series） ----------------
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n // 2).mean()


def ts_min(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n // 2).min()


def ts_max(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n // 2).max()


def _F1(c, v, N=20):
    # 趋势突破：收盘价创 N 日新高且站上 MA60
    return (c > ts_max(c.shift(1), N)) & (c > sma(c, 60))


def _F2(c, v, N=5, K=1.5):
    # 放量：VOL>MA5(VOL)*K 且 VOL>昨量
    return (v > sma(v, N) * K) & (v > v.shift(1))


def _F3(c, v):
    # 均线多头：MA20>MA60
    return sma(c, 20) > sma(c, 60)


def _F4(c, v):
    # 回踩均线：距 MA20<3% 且金叉 MA20
    ma20 = sma(c, 20)
    return (np.abs(c - ma20) / ma20 < 0.03) & (c > ma20) & (c.shift(1) <= ma20.shift(1))


def _F5(c, v):
    # 首阴：昨涨停今收阴且放量
    yin = (c.shift(1) / c.shift(2) > 1.095) & (c < c.shift(1)) & (v > sma(v, 5) * 1.5)
    return yin


def _F6(c, v):
    # 黄金坑：20日新低<MA60*0.93 且收回 MA20 且放量
    return (ts_min(c, 20) < sma(c, 60) * 0.93) & (c > sma(c, 20)) & (v > sma(v, 5) * 1.5)


def _F7(c, v):
    # 龙回头：60日曾涨>25% 且回踩 MA20 金叉 MA10 且放量
    mx60 = ts_max(c, 60)
    pullback = mx60.shift(20) / mx60 > 1.25
    return pullback & (c > sma(c, 20)) & (v > sma(v, 5) * 1.3) & (c > sma(c, 10))


# 注意：pandas DataFrame 列访问；下面用正确的列名重写所有因子
def build_factors(df: pd.DataFrame, N=20) -> pd.DataFrame:
    """df: 单只股票，index=datetime，列含 open/high/low/close/volume/vwap。返回各因子 0/1 列。"""
    o, h, l, c, vc, vw = df["open"], df["high"], df["low"], df["close"], df["volume"], df["vwap"]
    out = pd.DataFrame(index=df.index)
    out["F1"] = _F1(c, vc, N).astype(float)
    out["F2"] = _F2(c, vc).astype(float)
    out["F3"] = _F3(c, vc).astype(float)
    out["F4"] = _F4(c, vc).astype(float)
    out["F5"] = _F5(c, vc).astype(float)
    out["F6"] = _F6(c, vc).astype(float)
    out["F7"] = _F7(c, vc).astype(float)
    # F8 弱转强：低开(open<昨收) 高收(close>昨收) 放量(vol>昨量*1.2) 且 close>open
    out["F8"] = ((o < c.shift(1)) & (c > c.shift(1)) & (vc > vc.shift(1) * 1.2) & (c > o)).astype(float)
    # F9 主动资金(≈主力)：价升量增
    out["F9"] = ((c > o) & (c > sma(c, 5)) & (vc > vc.shift(1) * 1.2)).astype(float)
    # F10 板块强：外部依赖，跳过（N/A）
    return out


FACTOR_NAMES = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"]


# ---------------- 连续打分因子（C1-C9，对应同一套战法逻辑，但用连续值而非 0/1） ----------------
# 布尔因子截面方差太小、IC 弱；连续因子是 alpha 挖掘的标准输入，能给出真实 IC 信号。
def build_factors_continuous(df: pd.DataFrame, N=20) -> pd.DataFrame:
    o, h, l, c, vc, vw = df["open"], df["high"], df["low"], df["close"], df["volume"], df["vwap"]
    ma20, ma60 = sma(c, 20), sma(c, 60)
    out = pd.DataFrame(index=df.index)
    # C1 突破幅度：收盘价 / N日最高 - 1（越大越强）
    out["C1"] = c / ts_max(c.shift(1), N) - 1.0
    # C2 量比：logVOL - log(MA_N(VOL))
    out["C2"] = (np.log1p(vc) - np.log1p(sma(vc, N))).astype(float)
    # C3 均线多头偏离：(MA20-MA60)/MA60
    out["C3"] = (ma20 - ma60) / ma60
    # C4 回踩强度：收盘价/MA20 - 1（贴均线为正，远离为负），限幅
    out["C4"] = (c / ma20 - 1.0).clip(-0.1, 0.1)
    # C5 均值回复：当日收益取负（捕捉首阴/回调），仅当近期偏强
    ret = c / c.shift(1) - 1.0
    out["C5"] = -ret
    # C6 黄金坑：距20日最低涨幅取负（越在坑里越负，捕捉坑）
    out["C6"] = -(c / ts_min(c, 20) - 1.0)
    # C7 龙回头回撤：收盘价/60日最高 - 1（负=回撤中，配合前期涨幅）
    out["C7"] = c / ts_max(c, 60) - 1.0
    # C8 弱转强：阳线强度 × 量比
    out["C8"] = ((c - o) / o) * (vc / sma(vc, 5))
    # C9 主力近似：阳线强度 × 站上5MA
    out["C9"] = ((c - o) / o) * (c > sma(c, 5)).astype(float)
    return out.replace([np.inf, -np.inf], np.nan)


CONT_FACTOR_NAMES = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]


# ---------------- IC 计算 ----------------
def compute_ic(factor_long: pd.DataFrame, fwdret_long: pd.DataFrame):
    """factor_long / fwdret_long: MultiIndex(datetime, instrument) 单列。
    返回 (IC_series, RankIC_series) 按 datetime 截面的相关。"""
    df = pd.concat([factor_long.rename("f"), fwdret_long.rename("r")], axis=1).dropna()
    ic_list, ric_list, dates = [], [], []
    for dt, g in df.groupby(level=0):
        if len(g) < 30:
            continue
        f = g["f"].values
        r = g["r"].values
        if np.std(f) < 1e-12 or np.std(r) < 1e-12:
            continue
        ic = np.corrcoef(f, r)[0, 1]
        ric = pd.Series(f).corr(pd.Series(r), method="spearman")
        if np.isfinite(ic) and np.isfinite(ric):
            ic_list.append(ic)
            ric_list.append(ric)
            dates.append(dt)
    return pd.Series(ic_list, index=dates), pd.Series(ric_list, index=dates)


def summarize(ic: pd.Series, ric: pd.Series, seg_mask=None):
    if seg_mask is None:
        seg_mask = pd.Series(True, index=ic.index)
    s = ic[seg_mask]
    rs = ric[seg_mask]
    if len(s) == 0:
        return dict(n=0, IC=0.0, ICIR=0.0, RankIC=0.0, RankICIR=0.0)
    return dict(
        n=int(len(s)),
        IC=float(s.mean()),
        ICIR=float(s.mean() / s.std()) if s.std() > 0 else 0.0,
        RankIC=float(rs.mean()),
        RankICIR=float(rs.mean() / rs.std()) if rs.std() > 0 else 0.0,
    )


# ---------------- TopK 回测（市场中性多空：多 top-K - 空 bottom-K，抵消 5min 微观结构漂移） ----------------
def topk_backtest(factor_long: pd.DataFrame, fwdret_long: pd.DataFrame, topk: int, ann_scaler: int):
    df = pd.concat([factor_long.rename("f"), fwdret_long.rename("r")], axis=1).dropna()
    ls_by_date, long_by_date = {}, {}
    for dt, g in df.groupby(level=0):
        g = g.sort_values("f", ascending=False)
        n = len(g)
        k = min(topk, n // 2)
        if k < 1:
            continue
        long_r = g.head(k)["r"].mean()
        short_r = g.tail(k)["r"].mean()
        ls_by_date[dt] = float(long_r - short_r)   # 多空（市场中性，漂移抵消）
        long_by_date[dt] = float(long_r)           # 纯多头（仅参考，5min 含负漂移）
    ls = pd.Series(ls_by_date).sort_index()
    if len(ls) == 0:
        return dict(ann_ret=0.0, mdd=0.0, sharpe=0.0, long_ann=0.0)
    nav = (1 + ls).cumprod()
    ann_ret = float(nav.iloc[-1] ** (ann_scaler / len(ls)) - 1)
    peak = nav.cummax()
    mdd = float((nav / peak - 1).min())
    vol = ls.std()
    sharpe = float(ls.mean() / vol * np.sqrt(ann_scaler)) if vol > 0 else 0.0
    tr = pd.Series(long_by_date).sort_index()
    long_nav = (1 + tr).cumprod()
    long_ann = float(long_nav.iloc[-1] ** (ann_scaler / len(tr)) - 1) if len(tr) else 0.0
    return dict(ann_ret=ann_ret, mdd=mdd, sharpe=sharpe, long_ann=long_ann)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True)
    ap.add_argument("--freq", default="day")
    ap.add_argument("--market", default="all")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--horizon", type=int, default=5, help="持有周期（bar 数）：日频默认5(1周)，5min默认12(1小时)")
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--N", type=int, default=20, help="F1 突破窗口")
    ap.add_argument("--mode", default="bool", choices=["bool", "cont", "both"],
                    help="因子形态：bool=0/1战法信号; cont=连续打分(C1-C9); both=两者都跑")
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--segments", default="train:[2020-01-01,2022-12-31];valid:[2023-01-01,2023-12-31];test:[2024-01-01,2025-12-31]")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.mode == "cont":
        active = CONT_FACTOR_NAMES
        build_active = build_factors_continuous
    elif args.mode == "both":
        active = FACTOR_NAMES + CONT_FACTOR_NAMES
        def build_active(g, N):
            return pd.concat([build_factors(g, N), build_factors_continuous(g, N)], axis=1)
    else:
        active = FACTOR_NAMES
        build_active = build_factors

    import qlib
    from qlib.data import D
    qlib.init(provider_uri=args.provider, region="cn")

    # 直接从 instruments 文件读取标的（避免 qlib D.instruments 的 freq 默认 day 问题）
    provider = os.path.expanduser(args.provider)
    inst_file = Path(provider) / "instruments" / f"{args.market}.txt"
    if not inst_file.exists():
        inst_file = Path(provider) / "instruments" / "all.txt"
    instrs = []
    if inst_file.exists():
        with open(inst_file) as f:
            for line in f:
                parts = line.split()
                if parts:
                    instrs.append(parts[0])
    instrs = list(dict.fromkeys(instrs))
    logger.info("标的数量: %d (from %s)", len(instrs), inst_file)

    ann_scaler = {"day": 252, "5min": 252 * 48, "1min": 252 * 240}[args.freq]
    bar_total = 0
    fac_cols = {f: [] for f in active}
    fwd_cols = []

    for i in range(0, len(instrs), args.batch):
        batch = instrs[i:i + args.batch]
        raw = D.features(batch, FIELDS, start_time=args.start, end_time=args.end, freq=args.freq)
        if raw is None or len(raw) == 0:
            continue
        raw = raw.reset_index()
        # reset_index 后前两列即 multiindex 层级名 (instrument, datetime)，无需按位置改；只去 $ 前缀
        ren = {c: c.lstrip("$") for c in raw.columns if c.startswith("$")}
        raw = raw.rename(columns=ren)
        for code, g in raw.groupby("instrument"):
            g = g.sort_values("datetime").set_index("datetime")
            if len(g) < 60:
                continue
            facs = build_active(g, N=args.N)
            fwd = g["close"].shift(-args.horizon) / g["close"] - 1.0
            idx = facs.index.intersection(fwd.dropna().index)
            if len(idx) < 30:
                continue
            facs = facs.loc[idx]
            facs["instrument"] = code
            facs = facs.reset_index().set_index(["datetime", "instrument"])
            fr = fwd.loc[idx].to_frame("r")
            fr["instrument"] = code
            fr = fr.reset_index().set_index(["datetime", "instrument"])["r"]
            for f in active:
                fac_cols[f].append(facs[f])
            fwd_cols.append(fr)
        bar_total += 1
        logger.info("批次 %d/%d 完成", i // args.batch + 1, (len(instrs) + args.batch - 1) // args.batch)

    factor_long = {f: pd.concat(v) for f, v in fac_cols.items()}
    fwdret_long = pd.concat(fwd_cols)
    logger.info("因子样本: %d 行", len(fwdret_long))

    # 解析 walk-forward 段
    segs = {}
    for part in args.segments.split(";"):
        name, rng = part.split(":")
        a, b = rng.strip("[]").split(",")
        segs[name.strip()] = (a.strip(), b.strip())

    rows = []
    for f in active:
        ic, ric = compute_ic(factor_long[f], fwdret_long)
        row = {"factor": f, "IC_all": summarize(ic, ric)["IC"],
               "ICIR_all": summarize(ic, ric)["ICIR"],
               "RankIC_all": summarize(ic, ric)["RankIC"],
               "RankICIR_all": summarize(ic, ric)["RankICIR"]}
        for sname, (a, b) in segs.items():
            m = (ic.index >= a) & (ic.index <= b)
            s = summarize(ic, ric, m)
            row[f"IC_{sname}"] = s["IC"]
            row[f"RankIC_{sname}"] = s["RankIC"]
        bt = topk_backtest(factor_long[f], fwdret_long, args.topk, ann_scaler)
        row["topk_ann_ret"] = bt["ann_ret"]
        row["topk_mdd"] = bt["mdd"]
        row["topk_sharpe"] = bt["sharpe"]
        row["topk_long_ann"] = bt["long_ann"]
        rows.append(row)
        logger.info("%s: IC=%.4f RankIC=%.4f ICIR=%.3f | TopK ann=%.3f mdd=%.3f",
                   f, row["IC_all"], row["RankIC_all"], row["ICIR_all"], bt["ann_ret"], bt["mdd"])

    res = pd.DataFrame(rows).sort_values("RankIC_all", ascending=False)
    cols = ["factor", "IC_all", "ICIR_all", "RankIC_all", "RankICIR_all",
            "IC_train", "IC_valid", "IC_test", "RankIC_train", "RankIC_valid", "RankIC_test",
            "topk_ann_ret", "topk_mdd", "topk_sharpe", "topk_long_ann"]
    cols = [c for c in cols if c in res.columns]
    print(res[cols].to_string(index=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        res[cols].to_csv(args.out, index=False)
        logger.info("报告已写: %s", args.out)
    # JSON 也存一份（含全部字段）
    if args.out:
        jp = Path(args.out).with_suffix(".json")
        res.to_json(jp, orient="records", indent=2, force_ascii=False)
        logger.info("JSON 已写: %s", jp)


if __name__ == "__main__":
    main()
