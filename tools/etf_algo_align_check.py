"""复现 absmom_qmt_model.py 的确切算法，定位 +10.0% 与 +8.54% 的差距
==================================================================

上一轮发现：报告的 +10.0%/1.03/−9.4% 与我的复现 +8.54%/0.86/−11.1% 有差距，
且 vol_floor 从 0.05 提到 0.10~0.15 能补齐。

但 `quantaalpha/strategy/absmom_qmt_model.py` 的 INV_VOL_FLOOR 确实是 0.05。
所以差异必然在别处。本脚本**逐行对齐**该文件的算法：

  1. **收益率算法**：`np.diff(c)/c[:-1]` 是**简单日收益**（不是 pct_change 的对数，
     但注意 pct_change 也是简单收益）——关键差别在**取数窗口**。
  2. **sigma 算法**：`np.std(rets[-VOL_WINDOW:])` —— **没有减均值**（np.std 默认 ddof=0），
     而 pandas 的 `.std()` 默认 **ddof=1**（样本标准差）。
     这是系统性偏差：np.std 比分母 N，pandas 分母 N-1 → pandas 值更大约 sqrt(N/(N-1))。
     N=60 时比值 sqrt(60/59)=1.0085，只差 0.85%，不是主因。
  3. **SMA 算法**：`np.mean(closes[-SMA_W:])` —— 只取最后 200 个，**不要求连续 200 日**。
     若某 ETF 上市不足 200 日但有数据，`len(closes) < SMA_W` 会被 skip。
  4. **attribute_history 的取数**：QMT 返回的是**前复权**价还是原始价？
     我用的是 bars_1d（QMT datadir 回填，**未复权**，已修份额拆分）。
     这是**最可能的差异来源**：QMT attribute_history 默认前复权。

本脚本做三件事：
  A. 完全按 absmom_qmt_model.py 的 numpy 算法实现（np.std ddof=0、diff、mean），
     在 bars_1d 上跑，看是否逼近 +10.0%
  B. 对比 pandas 口径（ddof=1）与 numpy 口径（ddof=0）的差异
  C. 列出 QMT attribute_history 复权假设的影响方向
"""
import os
import tempfile
import json
import numpy as np
import pandas as pd
import psycopg2

DB = dict(host="127.0.0.1", port=5432, dbname="quant_data", user="quant", password="quant123456")
START = "2019-01-02"
END = "2026-09-30"

# absmom_qmt_model.py 的 POOL（38 只）
POOL = [
    "159915.SZ", "159949.SZ", "159967.SZ", "159981.SZ", "159985.SZ", "161226.SH",
    "501018.SH", "510030.SH", "510300.SH", "510500.SH", "510900.SH", "511010.SH",
    "511090.SH", "511220.SH", "511380.SH", "511880.SH", "512100.SH", "512400.SH",
    "512500.SH", "512890.SH", "513030.SH", "513050.SH", "513080.SH", "513090.SH",
    "513100.SH", "513130.SH", "513180.SH", "513330.SH", "513500.SH", "513520.SH",
    "513650.SH", "513660.SH", "513730.SH", "513980.SH", "518880.SH", "563300.SH",
    "588000.SH", "588080.SH",
]
SMA_W = 200
VOL_WINDOW = 60
INV_VOL_FLOOR = 0.05
COMM = 0.0001


def load_etf(codes, start, end):
    c = psycopg2.connect(**DB)
    cur = c.cursor()
    cur.execute(
        """
        select asset_code, trade_date, close
        from bars_1d
        where trade_date between %s and %s and asset_type = 'etf' and asset_code = any(%s)
        order by asset_code, trade_date
        """,
        (start, end, list(codes)),
    )
    rows = cur.fetchall()
    cur.close()
    c.close()
    df = pd.DataFrame(rows, columns=["code", "date", "close"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def perf(nav):
    r = pd.Series(nav).pct_change(fill_method=None).dropna()
    if len(r) < 2:
        return {}
    total = float(nav[-1])
    years = len(nav) / 252
    ann = total ** (1 / years) - 1 if total > 0 else np.nan
    vol = float(r.std() * np.sqrt(252))
    sharpe = ann / vol if vol > 0 else np.nan
    s = pd.Series(nav)
    dd = float((s / s.cummax() - 1).min())
    return dict(total=total, ann=float(ann), vol=vol, sharpe=float(sharpe), mdd=dd)


def run_numpy_style(closes_by_code, dates, ddof=0, vol_floor=INV_VOL_FLOOR, comm=COMM):
    """完全按 absmom_qmt_model.py 的算法：np.diff / np.std(ddof) / np.mean。"""
    nav = 1.0
    navs = []
    cur_w = {}
    n_rebal = 0
    turnover = 0.0
    date_idx = {d: i for i, d in enumerate(dates)}
    # 预计算每只的 sma / vol 时间序列
    sma_s, vol_s = {}, {}
    for code, arr in closes_by_code.items():
        c = np.asarray(arr, dtype=float)
        if len(c) < SMA_W:
            sma_s[code] = None
            vol_s[code] = None
            continue
        sma = np.full(len(c), np.nan)
        for i in range(SMA_W - 1, len(c)):
            sma[i] = np.mean(c[i - SMA_W + 1:i + 1])
        rets = np.diff(c) / c[:-1]
        vol = np.full(len(c), np.nan)
        for i in range(VOL_WINDOW, len(c)):
            vol[i] = float(np.std(rets[i - VOL_WINDOW:i], ddof=ddof) * np.sqrt(252))
        sma_s[code] = sma
        vol_s[code] = vol
    for i, d in enumerate(dates):
        is_month_start = (i == 0) or (dates[i - 1].to_period("M") != d.to_period("M"))
        if is_month_start and i > 0:
            held, inv = [], {}
            for code in closes_by_code:
                sma = sma_s[code]
                vol = vol_s[code]
                if sma is None or i >= len(sma):
                    continue
                if not np.isfinite(sma[i - 1]) or not np.isfinite(vol[i - 1]):
                    continue
                if closes_by_code[code][i - 1] > sma[i - 1]:
                    held.append(code)
                    inv[code] = 1.0 / max(vol[i - 1], vol_floor)
            if not held:
                new_w = {}
            else:
                tot = sum(inv.values())
                new_w = {x: inv[x] / tot for x in held}
            codes = set(new_w) | set(cur_w)
            t = sum(abs(new_w.get(x, 0.0) - cur_w.get(x, 0.0)) for x in codes)
            turnover += t
            nav *= (1 - comm * t)
            cur_w = new_w
            n_rebal += 1
        day_ret = 0.0
        for code, wt in cur_w.items():
            arr = closes_by_code.get(code)
            if arr is None or i >= len(arr) or i == 0:
                continue
            prev = arr[i - 1]
            cur = arr[i]
            if prev > 0 and np.isfinite(prev) and np.isfinite(cur):
                day_ret += wt * (cur / prev - 1)
        nav *= (1 + day_ret)
        navs.append(nav)
    return pd.Series(navs, index=dates), n_rebal, turnover


def run_pandas_style(px, ret, vol_floor=INV_VOL_FLOOR, comm=COMM):
    """我上一轮的口径：pandas rolling + std(ddof=1)。"""
    dates = px.index
    months = dates.to_period("M")
    sma = px.rolling(SMA_W, min_periods=SMA_W).mean()
    vol = ret.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std() * np.sqrt(252)
    nav = 1.0
    navs = []
    cur_w = {}
    for i, d in enumerate(dates):
        is_month_start = (i == 0) or (months[i] != months[i - 1])
        if is_month_start and i > 0:
            row_sma = sma.iloc[i - 1]
            row_vol = vol.iloc[i - 1]
            last = px.iloc[i - 1]
            held = [c for c in px.columns
                    if pd.notna(row_sma[c]) and pd.notna(row_vol[c]) and last[c] > row_sma[c]]
            if not held:
                new_w = {}
            else:
                inv = {c: 1.0 / max(row_vol[c], vol_floor) for c in held}
                tot = sum(inv.values())
                new_w = {c: inv[c] / tot for c in held}
            codes = set(new_w) | set(cur_w)
            t = sum(abs(new_w.get(x, 0.0) - cur_w.get(x, 0.0)) for x in codes)
            nav *= (1 - comm * t)
            cur_w = new_w
        day_ret = 0.0
        r = ret.iloc[i]
        for code, wt in cur_w.items():
            if code in r.index and pd.notna(r[code]):
                day_ret += wt * r[code]
        nav *= (1 + day_ret)
        navs.append(nav)
    return pd.Series(navs, index=dates)


def main():
    print("[1] 加载 POOL 38 只 ...")
    df = load_etf(set(POOL), START, END)
    px = df.pivot(index="date", columns="code", values="close").sort_index()
    ret = px.pct_change(fill_method=None)
    print(f"     {px.shape[0]} 日 x {px.shape[1]} 只（库里有数据的）")

    print("\n[2] A. 完全按 absmom_qmt_model.py 的 numpy 算法（ddof=0, floor=0.05）...")
    closes_by_code = {c: px[c].dropna().values for c in px.columns}
    # numpy 版需要每只代码自己的日期索引（因为 dropna 会错位）→ 用完整索引+NaN
    closes_full = {c: px[c].values for c in px.columns}
    nav_np, nr, to = run_numpy_style(closes_full, list(px.index), ddof=0, vol_floor=0.05)
    p = perf(list(nav_np.values))
    print(f"     numpy口径(ddof=0): 年化 {p['ann']*100:+.2f}%  夏普 {p['sharpe']:.2f}  "
          f"回撤 {p['mdd']*100:+.1f}%  调仓{nr} 换手{to:.1f}")

    print("\n[3] B. pandas 口径（ddof=1, floor=0.05，我上轮版本）...")
    nav_pd = run_pandas_style(px, ret, vol_floor=0.05)
    p2 = perf(list(nav_pd.values))
    print(f"     pandas口径(ddof=1): 年化 {p2['ann']*100:+.2f}%  夏普 {p2['sharpe']:.2f}  "
          f"回撤 {p2['mdd']*100:+.1f}%")

    print("\n[4] C. ddof 与 vol_floor 的交叉扫描 ...")
    for ddof in [0, 1]:
        for vf in [0.05, 0.10, 0.15]:
            if ddof == 0:
                nav_x, _, _ = run_numpy_style(closes_full, list(px.index), ddof=0, vol_floor=vf)
            else:
                nav_x = run_pandas_style(px, ret, vol_floor=vf)
            px_ = perf(list(nav_x.values))
            print(f"     ddof={ddof} floor={vf:.2f}: {px_['ann']*100:+6.2f}%/S{px_['sharpe']:.2f}/"
                  f"回撤{px_['mdd']*100:+.1f}%")

    print("\n[5] D. 波动率算法差异量化 ...")
    c = px["518880.SH"].dropna().values
    r = np.diff(c) / c[:-1]
    for n in [20, 60]:
        s0 = np.std(r[-n:], ddof=0)
        s1 = np.std(r[-n:], ddof=1)
        print(f"     N={n}: np.std(ddof=0)={s0*np.sqrt(252):.4f}  ddof=1={s1*np.sqrt(252):.4f}  "
              f"比值 {s1/s0:.4f}")

    print("\n[6] E. 信号日选择差异（用 i-1 还是 i）...")
    # absmom_qmt_model.py: handlebar 里用 closes[-1]（= 信号日收盘）
    # 我上轮用 month_last[prev_m]（= 上月末收盘），等价于 i-1
    # 测试：改用"当月首个交易日的上一交易日" vs "上月末"——已在 C/D 证明相同
    print("     （已由 etf_variant_check.py 证明 sig_lag 两种取值结果相同）")

    print("\n[7] F. 与报告数字对照 ...")
    print(f"     报告: +10.0% / 1.03 / -9.4%")
    print(f"     numpy ddof=0 floor=0.05: {p['ann']*100:+.2f}% / {p['sharpe']:.2f} / {p['mdd']*100:+.1f}%")
    print(f"     pandas ddof=1 floor=0.05: {p2['ann']*100:+.2f}% / {p2['sharpe']:.2f} / {p2['mdd']*100:+.1f}%")

    out_dir = os.environ.get("QA_OUT_DIR", tempfile.gettempdir())
    path = os.path.join(out_dir, "etf_algo_align_out.json")
    res = {"numpy_ddof0_floor005": p, "pandas_ddof1_floor005": p2}
    with open(path, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\n完成 -> {path}")


if __name__ == "__main__":
    main()
