"""E0 前置关：2026 年各信号「毛超额」是否仍为正？
============================================

执行层 L4 的前提假设：**信号毛 alpha 还在，只是被成本吃掉**。
若 2026 年毛超额已 ≤0，则问题不在执行层、而在信号本身，L4 无从施救。

本脚本在 **2026-01~10**（唯一有 5m 数据的窗口）测：
  - 各信号 T+1 开盘买入、最低价止损-8% / 最高价止盈+10% / 最长 20 日
  - 报 0% / 0.25% / 0.5% / 1% 四档往返成本下的净超额
  - 基准 = 同期全池等权（与策略同口径，见 MEMORY 铁律 12）

若某信号在 0.25% 下为正、1% 下转负 → **正是 L4 的目标猎物**（成本敏感）。
"""
import json
import os
import tempfile
import numpy as np
import pandas as pd
import psycopg2

DB = dict(host="127.0.0.1", port=5432, dbname="quant_data", user="quant", password="quant123456")
LOAD_START, SIM_START, END = "2018-01-01", "2026-01-05", "2026-10-13"
COSTS = [0.0, 0.0025, 0.005, 0.01]
STOP, TP, MAXHOLD = -0.08, 0.10, 20


def main():
    c = psycopg2.connect(**DB)
    c.autocommit = True
    cur = c.cursor()
    cur.execute("set max_parallel_workers_per_gather=0")
    cur.execute(
        """select b.asset_code, extract(epoch from b.trade_date)::bigint as ts,
                  b.open, b.high, b.low, b.close, b.volume, b.amount
           from bars_1d b
           where b.trade_date between %s and %s and b.asset_type='stock' and b.close>0
           order by b.asset_code, b.trade_date""", (LOAD_START, END))
    rows = cur.fetchall()
    c.close()
    df = pd.DataFrame(rows, columns=["code", "ts", "open", "high", "low", "close", "volume", "amount"])
    df["date"] = pd.to_datetime(df["ts"], unit="s")
    g = df.groupby("code")
    df["n"] = g.cumcount() + 1
    df["ret"] = g["close"].transform(lambda s: s / s.shift(1) - 1)
    for w in (10, 20, 60):
        df[f"sma{w}"] = g["close"].transform(lambda s, w=w: s.rolling(w, min_periods=w).mean())
    df["svol5"] = g["volume"].transform(lambda s: s.rolling(5, min_periods=5).mean())
    df["tsmin20"] = g["close"].transform(lambda s: s.rolling(20, min_periods=20).min())
    for col in ["sma10", "sma20", "sma60", "tsmin20"]:
        df[col] = df.groupby("code")[col].transform(lambda s: s.ffill(limit=5))
    p1, p2, c_ = df["close"].shift(1), df["close"].shift(2), df["close"]
    prev_sma20 = df.groupby("code")["sma20"].shift(1)
    df["F4"] = ((c_ - df["sma20"]).abs() / df["sma20"] < 0.03) & (c_ > df["sma20"]) & (p1 <= prev_sma20)
    df["F6"] = (df["tsmin20"] < df["sma60"] * 0.93) & (c_ > df["sma20"]) & (df["volume"] > df["svol5"] * 1.5)
    df["F5"] = (p1 / p2 > 1.095) & (c_ < p1) & (df["volume"] > df["svol5"] * 1.5)
    df["S2"] = (c_ / df["sma20"] - 1 > 0.05) & (df["volume"] > df["svol5"] * 1.5)

    p = df[(df["n"] >= 250) & (df["open"] > 0) & (df["date"] >= pd.Timestamp(SIM_START))].copy()
    print(f"样本 {p['date'].min().date()} ~ {p['date'].max().date()}  {len(p):,} 行", flush=True)

    # 同期全池等权基准（与策略同口径的"每日再平衡等权"）
    ew = p.groupby("date")["ret"].mean()
    nav_ew = {ts.date(): v for ts, v in (1 + ew).cumprod().items()}

    out = {}
    for col in ["F4", "F6", "F5", "S2"]:
        trades = []
        for code, gg in p.groupby("code"):
            gg = gg.sort_values("date")
            ao, ah, al, ac = gg["open"].values, gg["high"].values, gg["low"].values, gg["close"].values
            sig, ad = gg[col].values, gg["date"].values
            n = len(gg)
            for i in range(n):
                if i + 2 > n - 1 or not sig[i] or not np.isfinite(ao[i + 1]) or ao[i + 1] <= 0:
                    continue
                buy = ao[i + 1]
                ej, xp = min(i + 1 + MAXHOLD, n - 1), ac[min(i + 1 + MAXHOLD, n - 1)]
                for j in range(i + 2, min(i + 1 + MAXHOLD, n - 1) + 1):
                    if not np.isfinite(ac[j]) or ac[j] <= 0:
                        continue
                    if al[j] / buy - 1 <= STOP:
                        xp, ej = buy * (1 + STOP), j
                        break
                    if ah[j] / buy - 1 >= TP:
                        xp, ej = buy * (1 + TP), j
                        break
                d0, d1 = pd.Timestamp(ad[i + 1]).date(), pd.Timestamp(ad[ej]).date()
                if d0 not in nav_ew or d1 not in nav_ew or nav_ew[d0] <= 0:
                    continue
                trades.append(dict(entry=pd.Timestamp(ad[i + 1]),
                                   gross=xp / buy - 1,
                                   bench=nav_ew[d1] / nav_ew[d0] - 1))
        tr = pd.DataFrame(trades)
        print(f"\n=== {col}  {len(tr):,} 笔（2026） ===", flush=True)
        out[col] = {}
        line = "    成本档   净超额bp    t    胜率   年化净超额"
        print(line)
        for cst in COSTS:
            ex = (tr["gross"] - tr["bench"] - cst).values
            m = ex.mean()
            t = m / (ex.std() / np.sqrt(len(ex))) if ex.std() > 0 else 0
            ann = m * (242 / tr["entry"].dt.year.value_counts() if False else 1)
            print(f"    {cst*100:>5.2f}%  {m*1e4:>8.2f}  {t:>+6.2f}  {(ex>0).mean():>6.1%}")
            out[col][f"c{cst}"] = dict(n=len(ex), mean_bp=float(m * 1e4), t=float(t), win=float((ex > 0).mean()))
        print(f"    毛超额 bp: {(tr['gross']-tr['bench']).mean()*1e4:+.2f}  (t={ (tr['gross']-tr['bench']).mean()/((tr['gross']-tr['bench']).std()/np.sqrt(len(tr))) :+.2f})")

    out_dir = os.environ.get("QA_OUT_DIR", tempfile.gettempdir())
    path = os.path.join(out_dir, "exec_layer_e0_out.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n完成 -> {path}")


if __name__ == "__main__":
    main()
