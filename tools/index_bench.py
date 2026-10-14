"""指数基准（2020-2024 / OOS 2023-2024）→ 合并进 portfolio_risk_out.json
=====================================================================
为什么需要它：portfolio_risk.py 的宇宙是"总市值 top-N"，其朴素对照是同池等权（EW），
但 EW 本身在亏钱 regime 里也是负的 → 单看 EW 无法判断"ERC -1.08%"到底好不好。
必须再摆一个**外部同口径参照**：沪深300 / 中证500 / 中证1000 / 上证50 / 创业板指
（bars_1d 的 asset_type='index' 表，11 个指数，2011~2026）。
→ 这样能回答：风险构建是"跑赢指数"还是"只是少亏"。

口径与 portfolio_risk.py 完全一致：close.pct_change()、252 日年化、MDD=NAV/cummax-1、
指数为价格指数（不含股息，与组合口径一致——组合也没算股息）。
注意：指数表同样有 (code,date) 重复行 → 必须 drop_duplicates（本项目已知数据坑）。
"""
from __future__ import annotations
import json
import os
import tempfile

import numpy as np
import pandas as pd
import psycopg2

DB = dict(host="127.0.0.1", port=5432, dbname="quant_data", user="quant", password="quant123456")
CODES = ["000300.SH", "000905.SH", "000852.SH", "000016.SH", "000001.SH", "399006.SZ"]
NAME = {"000300.SH": "沪深300", "000905.SH": "中证500", "000852.SH": "中证1000",
        "000016.SH": "上证50", "000001.SH": "上证指数", "399006.SZ": "创业板指"}
WINDOWS = {"full": ("2020-01-01", "2024-12-31"), "oos": ("2023-01-01", "2024-12-31")}


def perf(nav: pd.Series) -> dict:
    nav = nav / nav.iloc[0]                      # ★归一化，否则 total=点位不是涨幅
    r = nav.pct_change().dropna()
    if len(r) < 2:
        return {}
    tot = float(nav.iloc[-1])
    yrs = len(nav) / 252
    ann = tot ** (1 / yrs) - 1 if tot > 0 else np.nan
    vol = float(r.std() * np.sqrt(252))
    return dict(total=round(tot, 4), ann=round(float(ann), 4),
                sharpe=round(float(ann / vol), 3) if vol > 0 else None,
                vol=round(vol, 4), mdd=round(float((nav / nav.cummax() - 1).min()), 4),
                n_days=int(len(nav)))


def main():
    conn = psycopg2.connect(**DB)
    df = pd.read_sql("select asset_code, trade_date, close from bars_1d "
                     "where asset_type='index' order by asset_code, trade_date", conn)
    conn.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.drop_duplicates(subset=["asset_code", "trade_date"], keep="last")
    px = df.pivot(index="trade_date", columns="asset_code", values="close").sort_index()
    px = px.loc[:, ~px.columns.duplicated()]

    out = {}
    for c in CODES:
        if c not in px.columns:
            continue
        s = px[c].dropna()
        rec = {"name": NAME[c]}
        for k, (a, b) in WINDOWS.items():
            w = s[(s.index >= a) & (s.index <= b)]
            rec[k] = perf(w) if len(w) > 30 else {}
        out[c] = rec
        f, o = rec["full"], rec.get("oos", {})
        print(f"  {NAME[c]:6s} 全期 {f.get('ann',0)*100:+6.2f}%/S{f.get('sharpe')}/回撤{f.get('mdd',0)*100:+6.1f}%"
              f" || OOS(23-24) {o.get('ann',0)*100:+6.2f}%/S{o.get('sharpe')}/回撤{o.get('mdd',0)*100:+6.1f}%"
              f" || {f.get('n_days')}日", flush=True)

    out_dir = os.environ.get("QA_OUT_DIR", tempfile.gettempdir())
    path = os.path.join(out_dir, "portfolio_risk_out.json")
    if os.path.exists(path):
        d = json.load(open(path))
    else:
        d = {}
    d["benchmarks_index"] = out
    json.dump(d, open(path, "w"), indent=2, ensure_ascii=False)
    print(f"\n合并进 -> {path}", flush=True)


if __name__ == "__main__":
    main()
