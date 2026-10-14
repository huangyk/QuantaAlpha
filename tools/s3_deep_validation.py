"""S3 缩量回踩MA10 深度验证（本轮唯一在同档口径为正的候选）
=========================================================

中高成交额宇宙扫描结果：8 个信号里只有 **S3_shrink_pull** 在
「同 amount 档」口径下 IS +191.15bp(t=12.53) / OOS +56.00bp(t=4.81) 都为正。
其余 7 个同档口径全负。

但 S3 必须过全部严格检验才能算"找到战法"：
  1. 定义复核（精确信号，无未来函数）
  2. 成本敏感性 0/0.25%/0.5%/1%
  3. 止损敏感性：无止损 / 止损 5%/8%/10%/15%
  4. 持有期敏感性：10/20/30 日
  5. 止盈敏感性：5%/10%/20%/无
  6. 规模分档拆解（看是否只集中在某一档）
  7. 逐年稳定性
  8. 参数成簇（缩量阈值 0.5/0.7/0.9，MA 周期 5/10/20）
  9. 宇宙敏感性：decile>=5 / decile>=7 / 全A
  10. 信号频率与容量

铁律：train 内不选参。本次扫描已用 2020-2024 做 IS、2025-2026 做 OOS，
敏感性分析只用来说明"不是尖峰"，不作为选参依据。
"""
import os
import tempfile
import json
import numpy as np
import pandas as pd
import psycopg2

DB = dict(host="127.0.0.1", port=5432, dbname="quant_data", user="quant", password="quant123456")
LOAD_START = "2018-01-01"
SIM_START = "2020-01-02"
END = "2026-09-30"


def load_panel(start, end):
    c = psycopg2.connect(**DB)
    cur = c.cursor()
    cur.execute(
        """
        select b.asset_code, b.trade_date, b.open, b.high, b.low, b.close,
               b.volume, b.amount, coalesce(m.is_st, false) as is_st
        from bars_1d b
        left join asset_master m on m.asset_code = b.asset_code
        where b.trade_date between %s and %s
          and b.asset_type is distinct from 'etf' and b.close is not null
        order by b.asset_code, b.trade_date
        """,
        (start, end),
    )
    rows = cur.fetchall()
    cur.close()
    c.close()
    df = pd.DataFrame(rows, columns=["code", "date", "open", "high", "low", "close", "volume", "amount", "is_st"])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["code", "date"]).reset_index(drop=True)


def build_benchmarks(panel):
    g = panel.groupby("code")
    panel["dret"] = g["close"].transform(lambda s: s / s.shift(1) - 1)
    valid_b = panel["listed_n"] >= 250
    d = panel.loc[valid_b & panel["dret"].notna() & (panel["dret"].abs() < 0.5),
                  ["date", "dret", "size_bucket"]].copy()
    return {
        "broad": (1 + d.groupby("date")["dret"].mean()).cumprod(),
        "median": (1 + d.groupby("date")["dret"].median()).cumprod(),
        "decile": (1 + d.groupby(["date", "size_bucket"])["dret"].mean().unstack("size_bucket")).cumprod(),
    }


def add_s3(panel, ma_win=10, shrink=0.7, vol_win=20):
    """S3 缩量回踩：今日收>MA10、今日量<20日均量*shrink、昨日收<昨日MA10。"""
    g = panel.groupby("code")
    c = panel["close"]
    panel["sma"] = g["close"].transform(lambda s: s.rolling(ma_win, min_periods=ma_win).mean())
    panel["sv"] = g["volume"].transform(lambda s: s.rolling(vol_win, min_periods=vol_win).mean())
    prev_sma = g["sma"].shift(1)
    prev_close = g["close"].shift(1)
    panel["S3"] = (c > panel["sma"]) & (panel["volume"] < panel["sv"] * shrink) & (prev_close < prev_sma)
    return panel


def simulate(panel, benches, signal_col="S3", cost=0.0025,
             stop=None, take=None, max_hold=20, only_after=None, only_before=None):
    nav_d = {b: {ts.date(): v for ts, v in benches["decile"][b].items()} for b in benches["decile"].columns}
    nav_m = {ts.date(): v for ts, v in benches["median"].items()}
    trades = []
    for code, g in panel.groupby("code"):
        g = g.sort_values("date")
        ao = g["open"].values
        ah = g["high"].values
        al = g["low"].values
        ac = g["close"].values
        sig = g[signal_col].values
        ad = g["date"].values
        ab = g["size_bucket"].values
        n = len(g)
        for i in range(n):
            if i + 2 > n - 1:
                break
            if not sig[i] or not np.isfinite(ao[i + 1]) or ao[i + 1] <= 0:
                continue
            buy = ao[i + 1]
            b = ab[i + 1]
            last_j = min(i + 1 + max_hold, n - 1)
            exited = False
            ej = last_j
            exit_price = ac[last_j]
            reason = "timeout"
            for j in range(i + 2, last_j + 1):
                if not np.isfinite(ac[j]) or ac[j] <= 0:
                    continue
                hi = ah[j] / buy - 1
                lo = al[j] / buy - 1
                if stop is not None and lo <= stop:
                    exit_price = buy * (1 + stop)
                    ej = j
                    exited = True
                    reason = "stop"
                    break
                if take is not None and hi >= take:
                    exit_price = buy * (1 + take)
                    ej = j
                    exited = True
                    reason = "tp"
                    break
            if exit_price is None or not np.isfinite(exit_price):
                continue
            ret = exit_price / buy - 1 - cost
            d0 = pd.Timestamp(ad[i + 1]).date()
            d1 = pd.Timestamp(ad[ej]).date()
            entry = pd.Timestamp(ad[i + 1])
            if only_after is not None and entry < only_after:
                continue
            if only_before is not None and entry >= only_before:
                continue
            exd = exm = np.nan
            dm = nav_d.get(b)
            if dm is not None and d0 in dm and d1 in dm and dm[d0] > 0:
                exd = ret - (dm[d1] / dm[d0] - 1)
            if d0 in nav_m and d1 in nav_m and nav_m[d0] > 0:
                exm = ret - (nav_m[d1] / nav_m[d0] - 1)
            trades.append(dict(entry=entry, decile=int(b), ex_decile=exd, ex_median=exm, ret=ret, reason=reason))
    return pd.DataFrame(trades)


def st(df, col="ex_decile"):
    v = df[col].dropna().values
    if len(v) == 0:
        return "n=0"
    m = v.mean()
    t = m / (v.std() / np.sqrt(len(v))) if v.std() > 0 else 0.0
    return f"n={len(v):>7,} {m*1e4:+8.2f}bp t={t:+7.2f} 胜率{(v>0).mean():.1%}"


def main():
    print("[1] 加载 + 中高成交额宇宙 ...")
    df = load_panel(LOAD_START, END)
    grp = df.groupby("code")
    df["listed_n"] = grp.cumcount() + 1
    df["prev_close"] = grp["close"].shift(1)
    df = df[(df["listed_n"] >= 250) & (~df["is_st"]) & (df["open"] > 0)].copy()
    df = df[df["date"] >= pd.Timestamp(SIM_START)].copy()
    df["size_bucket"] = df.groupby("date")["amount"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop"))
    panel = df[df["size_bucket"] >= 5].copy()
    print(f"     {len(panel):,} 行 / {panel['code'].nunique():,} 只")
    benches = build_benchmarks(panel)
    panel = add_s3(panel)
    print(f"     S3 信号天数: {int(panel['S3'].sum()):,}")

    IS = pd.Timestamp("2025-01-01")
    out = {}

    print("\n[2] 基准配置 IS/OOS（成本0.25%, 止损-8%, 止盈+10%, 持有20日）...")
    tr_is = simulate(panel, benches, only_before=IS)
    tr_oos = simulate(panel, benches, only_after=IS)
    print(f"     IS  2020-2024: [同档] {st(tr_is)} | [中位] {st(tr_is,'ex_median')}")
    print(f"     OOS 2025-2026: [同档] {st(tr_oos)} | [中位] {st(tr_oos,'ex_median')}")
    out["base"] = {"is": float(tr_is["ex_decile"].mean())*1e4, "oos": float(tr_oos["ex_decile"].mean())*1e4,
                   "is_n": int(len(tr_is)), "oos_n": int(len(tr_oos))}

    print("\n[3] 成本敏感性（同档口径, IS / OOS）...")
    for cost in [0.0, 0.0025, 0.005, 0.01]:
        a = simulate(panel, benches, cost=cost, only_before=IS)
        b = simulate(panel, benches, cost=cost, only_after=IS)
        print(f"     成本{cost*100:5.2f}%: IS {a['ex_decile'].mean()*1e4:+8.2f}bp | OOS {b['ex_decile'].mean()*1e4:+8.2f}bp")
        out[f"cost_{cost}"] = {"is": float(a["ex_decile"].mean())*1e4, "oos": float(b["ex_decile"].mean())*1e4}

    print("\n[4] 止损敏感性（成本0.25%）...")
    for stop in [None, -0.05, -0.08, -0.10, -0.15]:
        a = simulate(panel, benches, stop=stop, only_before=IS)
        b = simulate(panel, benches, stop=stop, only_after=IS)
        sa = "无止损" if stop is None else f"{stop*100:.0f}%"
        print(f"     止损{sa:>6s}: IS {a['ex_decile'].mean()*1e4:+8.2f}bp | OOS {b['ex_decile'].mean()*1e4:+8.2f}bp")
        out[f"stop_{stop}"] = {"is": float(a["ex_decile"].mean())*1e4, "oos": float(b["ex_decile"].mean())*1e4}

    print("\n[5] 持有期敏感性 ...")
    for mh in [10, 20, 30]:
        a = simulate(panel, benches, max_hold=mh, only_before=IS)
        b = simulate(panel, benches, max_hold=mh, only_after=IS)
        print(f"     持有{mh:2d}日: IS {a['ex_decile'].mean()*1e4:+8.2f}bp | OOS {b['ex_decile'].mean()*1e4:+8.2f}bp")
        out[f"hold_{mh}"] = {"is": float(a["ex_decile"].mean())*1e4, "oos": float(b["ex_decile"].mean())*1e4}

    print("\n[6] 止盈敏感性 ...")
    for take in [None, 0.05, 0.10, 0.20]:
        a = simulate(panel, benches, take=take, only_before=IS)
        b = simulate(panel, benches, take=take, only_after=IS)
        sa = "无止盈" if take is None else f"+{take*100:.0f}%"
        print(f"     止盈{sa:>6s}: IS {a['ex_decile'].mean()*1e4:+8.2f}bp | OOS {b['ex_decile'].mean()*1e4:+8.2f}bp")
        out[f"take_{take}"] = {"is": float(a["ex_decile"].mean())*1e4, "oos": float(b["ex_decile"].mean())*1e4}

    print("\n[7] 规模分档（同档口径, 成本0.25%, 止损-8%）...")
    for d in range(5, 10):
        sub_i = tr_is[tr_is["decile"] == d]
        sub_o = tr_oos[tr_oos["decile"] == d]
        print(f"     decile{d}: IS {st(sub_i) if not sub_i.empty else 'n=0'} | "
              f"OOS {st(sub_o) if not sub_o.empty else 'n=0'}")

    print("\n[8] 逐年（同档口径）...")
    for y in range(2020, 2027):
        a = tr_is[tr_is["entry"].dt.year == y]
        b = tr_oos[tr_oos["entry"].dt.year == y]
        s = a if not a.empty else b
        if s.empty:
            continue
        print(f"     {y}: n={len(s):>6,} {s['ex_decile'].mean()*1e4:+8.2f}bp "
              f"t={(s['ex_decile'].mean()/(s['ex_decile'].std()/np.sqrt(len(s))) if s['ex_decile'].std()>0 else 0):+6.2f} "
              f"胜率{(s['ex_decile']>0).mean():.1%}")

    print("\n[9] 参数成簇（缩量阈值 × MA 周期，同档口径 IS/OOS）...")
    for shrink in [0.5, 0.7, 0.9]:
        for ma in [5, 10, 20]:
            p2 = add_s3(panel.copy(), ma_win=ma, shrink=shrink)
            a = simulate(p2, benches, only_before=IS)
            b = simulate(p2, benches, only_after=IS)
            print(f"     shrink{shrink} MA{ma:2d}: IS {a['ex_decile'].mean()*1e4:+8.2f}bp | "
                  f"OOS {b['ex_decile'].mean()*1e4:+8.2f}bp")

    print("\n[10] 宇宙敏感性 ...")
    for lo in [0, 5, 7]:
        sub = df[df["size_bucket"] >= lo].copy()
        bb = build_benchmarks(sub)
        sub = add_s3(sub)
        a = simulate(sub, bb, only_before=IS)
        b = simulate(sub, bb, only_after=IS)
        print(f"     decile>={lo}: 信号{int(sub['S3'].sum()):>7,} | IS {a['ex_decile'].mean()*1e4:+8.2f}bp | "
              f"OOS {b['ex_decile'].mean()*1e4:+8.2f}bp")

    print("\n[11] 信号频率/容量（中高成交额宇宙）...")
    print(f"     信号日 {tr_is['entry'].nunique() + tr_oos['entry'].nunique()} 个，"
          f"日均信号数 {(panel['S3'].sum()/panel['date'].nunique()):.1f} 只")
    print(f"     IS 笔数 {len(tr_is):,}，OOS 笔数 {len(tr_oos):,}")

    out_dir = os.environ.get("QA_OUT_DIR", tempfile.gettempdir())
    path = os.path.join(out_dir, "s3_deep_out.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n完成 -> {path}")


if __name__ == "__main__":
    main()
