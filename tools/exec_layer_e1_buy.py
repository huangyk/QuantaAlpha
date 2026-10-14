"""E1 买入执行层对比（执行层 L4 · 核心实验，v2 同子样本口径）
=================================================================

设计见 docs/execution_layer_L4_plan.md §4 E1。信号层样本 2026-01-05~2026-10-13（5m 唯一可用窗口）。

执行组（只改买入成交价；止损/止盈仍用日线高低价，与 E0 一致 → Δ 纯粹是成交价）：
  C0  T+1 开盘价（基线）
  C1  T+1 全天 5m VWAP
  C2a/b/c T+1 开盘后 30/60/120 分钟 TWAP
  C3  缺口过滤（T+1 开盘涨幅>3% 放弃）+ C1
  C4  限价单 = C1-20bp（当日 5m 最高价≥委托价才成交，成交价=委托价；否则作废）

两种口径（v2 修正 C3 的样本选择效应）：
  口径A 同子样本：只用「7 组全部有有效成交价」的 entry → 纯粹比成交方式（公平）
  口径B 各Groups 全样本：含 C3 的过滤效应 → 交易层面（看得见降换手收益）
单位注意：bars_5m.volume 单位=手 → VWAP = Σamount / (Σvolume × 100)
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
GAP_LIMIT, LIMIT_OFF = 0.03, 0.002
SIGS = ["F4", "S2"]
GROUPS = [("C0 开盘", "vwap_open"), ("C1 全天VWAP", "vwap_all"), ("C2a TWAP30m", "vwap_30"),
          ("C2b TWAP60m", "vwap_60"), ("C2c TWAP120m", "vwap_120"),
          ("C3 缺口过滤+VWAP", "vwap_gapf"), ("C4 限价-20bp", "vwap_lim")]


def main():
    c = psycopg2.connect(**DB)
    c.autocommit = True
    cur = c.cursor()
    cur.execute("set max_parallel_workers_per_gather=0")

    print("[0] 5m↔日线一致性校验（抽 20 只）...", flush=True)
    cur.execute("""select b.asset_code, b.trade_date, b.amount from bars_1d b
                   where b.trade_date between '2026-09-01' and '2026-10-13'
                     and b.asset_type='stock' and b.close>0 order by random() limit 20""")
    ok, worst = 0, 0.0
    for code, d, damt in cur.fetchall():
        cur.execute("select sum(amount) from bars_5m where asset_code=%s and trade_date=%s", (code, d))
        r = cur.fetchone()[0]
        if r is None or not damt:
            continue
        diff = abs(r / damt - 1)
        worst = max(worst, diff)
        ok += diff < 0.005
    print(f"     通过 {ok}/20，最大相对偏差 {worst:.3%}", flush=True)
    if worst > 0.02:
        print("     ⚠ 偏差偏大，先查 5m 质量")

    print("[1] 日线 ...", flush=True)
    cur.execute(
        """select b.asset_code, extract(epoch from b.trade_date)::bigint as ts,
                  b.open, b.high, b.low, b.close, b.volume
           from bars_1d b
           where b.trade_date between %s and %s and b.asset_type='stock' and b.close>0
           order by b.asset_code, b.trade_date""", (LOAD_START, END))
    rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["code", "ts", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["ts"], unit="s")
    g = df.groupby("code")
    df["listed_n"] = g.cumcount() + 1
    df["prev_close"] = g["close"].shift(1)
    df["buy_open"] = g["open"].shift(-1)
    for w in (20, 60):
        df[f"sma{w}"] = g["close"].transform(lambda s, w=w: s.rolling(w, min_periods=w).mean())
    df["svol5"] = g["volume"].transform(lambda s: s.rolling(5, min_periods=5).mean())
    for col in ["sma20", "sma60"]:
        df[col] = df.groupby("code")[col].transform(lambda s: s.ffill(limit=5))
    cc, p1 = df["close"], df["prev_close"]
    prev_sma20 = df.groupby("code")["sma20"].shift(1)
    df["F4"] = ((cc - df["sma20"]).abs() / df["sma20"] < 0.03) & (cc > df["sma20"]) & (p1 <= prev_sma20)
    df["S2"] = (cc / df["sma20"] - 1 > 0.05) & (df["volume"] > df["svol5"] * 1.5)

    p = df[(df["listed_n"] >= 250) & (df["open"] > 0) & (df["date"] >= pd.Timestamp(SIM_START))].copy()
    p["dret"] = p.groupby("code")["close"].pct_change()
    nav_ew = {ts.date().isoformat(): v for ts, v in (1 + p.groupby("date")["dret"].mean()).cumprod().items()}
    print(f"     信号层 {len(p):,} 行  {p['date'].min().date()} ~ {p['date'].max().date()}", flush=True)

    print("[2] 5m 聚合 ...", flush=True)
    cur.execute(
        """select asset_code, trade_date,
                  sum(amount)/nullif(sum(volume)*100,0) as vwap_all,
                  sum(amount) filter (where ts::time <= '10:00')/nullif(sum(volume) filter (where ts::time <= '10:00')*100,0) as vwap_30,
                  sum(amount) filter (where ts::time <= '10:30')/nullif(sum(volume) filter (where ts::time <= '10:30')*100,0) as vwap_60,
                  sum(amount) filter (where ts::time <= '11:30')/nullif(sum(volume) filter (where ts::time <= '11:30')*100,0) as vwap_120,
                  max(high) as hi5, count(*) as nbars
           from bars_5m where trade_date between %s and %s group by 1,2""", (SIM_START, END))
    agg = {(r[0], r[1].isoformat()): dict(vwap_all=r[2], vwap_30=r[3], vwap_60=r[4],
                                          vwap_120=r[5], hi5=r[6], nbars=r[7]) for r in cur.fetchall()}
    cur.close(); c.close()
    print(f"     5m 聚合 {len(agg):,} 条", flush=True)

    out = {}
    for sig in SIGS:
        recs = []   # (entry_key, gn, buy, gross, bench)
        for code, gg in df[df["date"] >= pd.Timestamp(SIM_START)].groupby("code", sort=False):
            gg = gg.sort_values("date")
            if (gg["listed_n"] < 250).all():
                continue
            ao, ah, al, ac = gg["open"].values, gg["high"].values, gg["low"].values, gg["close"].values
            ad, bo, pc, sg = gg["date"].values, gg["buy_open"].values, gg["prev_close"].values, gg[sig].values
            n = len(gg)
            for i in range(n - 1):
                if not sg[i] or not np.isfinite(bo[i]) or bo[i] <= 0 or bo[i] / pc[i] >= 1.098:
                    continue
                key = (code, pd.Timestamp(ad[i + 1]).date().isoformat())
                a5 = agg.get(key)
                if not a5 or a5["nbars"] < 40 or not a5["vwap_all"] or a5["vwap_all"] <= 0:
                    continue
                gap = bo[i] / pc[i] - 1
                lim = a5["vwap_all"] * (1 - LIMIT_OFF)
                pxs = {"vwap_open": bo[i], "vwap_all": a5["vwap_all"], "vwap_30": a5["vwap_30"],
                       "vwap_60": a5["vwap_60"], "vwap_120": a5["vwap_120"],
                       "vwap_gapf": a5["vwap_all"] if gap <= GAP_LIMIT else np.nan,
                       "vwap_lim": lim if (a5["hi5"] and a5["hi5"] >= lim) else np.nan}
                d0 = key[1]
                if d0 not in nav_ew or nav_ew[d0] <= 0:
                    continue
                last = min(i + 1 + MAXHOLD, n - 1)
                for gn, col in GROUPS:
                    buy = pxs.get(col)
                    if buy is None or not np.isfinite(buy) or buy <= 0:
                        continue
                    ej, xp = last, ac[last]
                    for j in range(i + 2, last + 1):
                        if not np.isfinite(ac[j]) or ac[j] <= 0:
                            continue
                        if al[j] / buy - 1 <= STOP:
                            xp, ej = buy * (1 + STOP), j
                            break
                        if ah[j] / buy - 1 >= TP:
                            xp, ej = buy * (1 + TP), j
                            break
                    d1 = pd.Timestamp(ad[ej]).date().isoformat()
                    if d1 not in nav_ew or nav_ew[d1] <= 0:
                        continue
                    recs.append((key, gn, float(buy), float(xp / buy - 1), float(nav_ew[d1] / nav_ew[d0] - 1)))

        A = pd.DataFrame(recs, columns=["key", "gn", "buy", "gross", "bench"])
        print(f"\n{'='*72}\n=== {sig}  记录 {len(A):,} ===", flush=True)

        # 口径 A：同子样本（7 组全有效的 key）
        cnt = A.groupby("key")["gn"].nunique()
        common = set(cnt[cnt == len(GROUPS)].index)
        Ac = A[A["key"].isin(common)]
        base_buy = Ac[Ac["gn"] == "C0 开盘"].set_index("key")["buy"]
        print(f"\n  口径A 同子样本（{len(common):,} 笔，7 组均有效）—— 纯粹比成交方式")
        print(f"    {'执行组':<20s}{'均买入价/开盘':>13s}{'毛超额bp':>10s}{'成交价改善':>11s}{'净超额@25bp':>12s}")
        out[sig] = {}
        resA = {}
        for gn, _ in GROUPS:
            s = Ac[Ac["gn"] == gn].set_index("key")
            s = s.reindex(base_buy.index)
            bm, gm = s["buy"].mean(), (s["gross"] - s["bench"]).mean()
            impr = (base_buy / s["buy"] - 1) * 1e4
            net25 = ((s["gross"] - s["bench"] - 0.0025).values).mean()
            print(f"    {gn:<20s}{bm/base_buy.mean()-1:>12.2%}{gm*1e4:>10.1f}"
                  f"{impr.mean():>10.1f}bp{net25*1e4:>11.1f}")
            resA[gn] = dict(buy_ratio=float(bm / base_buy.mean()),
                            gross_bp=float(gm * 1e4), price_impr_bp=float(impr.mean()),
                            net25_bp=float(net25 * 1e4), n=int(len(s)))
        out[sig]["A_same_subset"] = resA

        # 口径 B：各组全样本（含 C3 过滤效应）
        print(f"\n  口径B 各组全样本（含 C3 过滤 → 看总账）")
        print(f"    {'执行组':<20s}{'笔数':>8s}{'毛超额bp':>10s}{'净@0.25%':>10s}{'净@0.5%':>10s}{'净@1%':>10s}")
        resB = {}
        for gn, _ in GROUPS:
            s = A[A["gn"] == gn]
            if s.empty:
                continue
            gex = (s["gross"] - s["bench"]).values
            line = f"    {gn:<20s}{len(s):>8,}{gex.mean()*1e4:>10.1f}"
            for cst in COSTS[1:]:
                ex = gex - cst
                t = ex.mean() / (ex.std() / np.sqrt(len(ex))) if ex.std() > 0 else 0
                line += f"{ex.mean()*1e4:>9.1f} "
                resB.setdefault(gn, {})[f"c{cst}"] = dict(n=len(ex), mean_bp=float(ex.mean() * 1e4),
                                                          t=float(t), win=float((ex > 0).mean()))
            print(line)
        out[sig]["B_full"] = resB

    out_dir = os.environ.get("QA_OUT_DIR", tempfile.gettempdir())
    path = os.path.join(out_dir, "exec_layer_e1_out.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n完成 -> {path}")


if __name__ == "__main__":
    main()
