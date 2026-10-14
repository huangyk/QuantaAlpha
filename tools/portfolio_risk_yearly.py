"""分年度 regime 分解：ERC vs 等权 vs 沪深300（N=300/500, 2020-2024）
==================================================================
SOP 关卡：不能只报"全期 + OOS"两个数——要按年拆开看赢在哪些年、输在哪些年，
否则"全期跑赢"可能只是某一年牛市贡献的。本脚本复用 portfolio_risk.py 的
回测引擎（同一宇宙/同一成本/同一协方差），只加按年收益分解。
"""
from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import portfolio_risk as pr  # noqa: E402


def yearly(nav: pd.Series) -> dict:
    out = {}
    for y, g in nav.groupby(nav.index.year):
        if len(g) < 30:
            continue
        out[int(y)] = round(float(g.iloc[-1] / g.iloc[0] - 1), 4)
    return out


def main():
    pr.START, pr.END = "2020-01-01", "2024-12-31"
    rows = []
    for N in (300, 500):
        pr.UNIVERSE_N = N
        for m in ("ew", "invvol", "erc"):
            nav, turn, cost_bp = pr.backtest(method=m, cost_on=True)
            rows.append(dict(universe=N, method=m, yearly=yearly(nav),
                             turnover=round(turn, 1), cost_bp=round(cost_bp, 1)))
            print(f"  N{N} {m:7s} " + "  ".join(f"{y}:{v*100:+6.2f}%" for y, v in yearly(nav).items()), flush=True)

    # 指数逐年（外部同口径参照）
    import psycopg2
    conn = psycopg2.connect(**pr.DB)
    df = pd.read_sql("select asset_code, trade_date, close from bars_1d "
                     "where asset_type='index' and asset_code in ('000300.SH','000905.SH') "
                     "order by asset_code, trade_date", conn)
    conn.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.drop_duplicates(subset=["asset_code", "trade_date"], keep="last")
    px = df.pivot(index="trade_date", columns="asset_code", values="close").sort_index()
    for c in px.columns:
        s = px[c].dropna()
        w = s[(s.index >= "2020-01-01") & (s.index <= "2024-12-31")]
        yl = yearly(w)
        print(f"  {c:10s} " + "  ".join(f"{y}:{v*100:+6.2f}%" for y, v in yl.items()), flush=True)
        rows.append(dict(universe=None, method=c, yearly=yl, turnover=None, cost_bp=None))

    out = os.path.join(os.environ.get("QA_OUT_DIR", os.path.dirname(os.path.abspath(__file__))),
                       "portfolio_risk_yearly.json")
    import json
    json.dump(rows, open(out, "w"), indent=1, ensure_ascii=False)

    # 去掉冷启动伪影的同年窗口对比：策略因 COV_WIN=120 天预热，2020 年基本空仓
    # （2020 沪深300 +25.5%，对策略不公平）→ 只用 2021-2024 四年对齐重算
    print("\n===== 对齐窗口 2021-01-01~2024-12-31（剔除 2020 冷启动伪影）=====", flush=True)
    matched = {}
    for N in (300, 500):
        pr.UNIVERSE_N = N
        for m in ("ew", "invvol", "erc"):
            nav, _, _ = pr.backtest(method=m, cost_on=True)
            w = nav[(nav.index >= "2021-01-01") & (nav.index <= "2024-12-31")]
            p = pr.perf(w)
            matched[f"N{N}_{m}"] = p
            print(f"  N{N} {m:7s} {p['ann']*100:+6.2f}%/S{p['sharpe']:.2f}/回撤{p['mdd']*100:+6.1f}%/波动{p['vol']*100:.1f}%", flush=True)
    for c in px.columns:
        s = px[c].dropna()
        w = s[(s.index >= "2021-01-01") & (s.index <= "2024-12-31")]
        p = pr.perf(w / w.iloc[0])
        matched[c] = p
        print(f"  {c:10s} {p['ann']*100:+6.2f}%/S{p['sharpe']:.2f}/回撤{p['mdd']*100:+6.1f}%/波动{p['vol']*100:.1f}%", flush=True)
    json.dump(dict(yearly=rows, matched_2021_2024=matched),
              open(out, "w"), indent=1, ensure_ascii=False)
    print(f"\n-> {out}", flush=True)


if __name__ == "__main__":
    main()
