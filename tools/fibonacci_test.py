"""斐波那契（Fibonacci）在 A股 的有效性检验
============================================

背景（2026-10-14，用户"学习下斐契那波数列在股票中，其实飞龙战法就是那个"）：
飞龙线 = `SQRT(EMA89*COST(89))`，其中 **EMA89 正是斐波那契数**（1,2,3,5,8,13,21,34,55,89,144…）。
本项目此前已证伪飞龙因子（IC≈0、-11.44%/年 vs 同池等权）。

斐波那契在技术分析里有 3 类核心应用，本脚本对其中两类做**全 A 面板统计检验**：

  **A. 回撤位（黄金分割 38.2% / 61.8%）是否真有支撑/阻力**
     检验方法：算每只股票相对「过去 60 日高低区间」的回撤深度 dd
       dd=0 → 在区间最高点；dd=1 → 在区间最低点
     然后看「未来 h 日收益」随 dd 的变化曲线。
     若 FIB 位真有效，应在 dd≈0.382 和 dd≈0.618 处出现**凸起/尖峰**（相对相邻区间）；
     若曲线单调，则只是动量/反转效应，与斐波那契无关。

  **B. 时间周期（5/8/13/21/34/55/89 日）是否是转折点**
     计算「距过去 60 日最高点已过去的交易日数」days_since_high，
     看未来 h 日收益随 days_since_high 的曲线是否在 FIB 日出现尖峰。

  关键混杂控制：
   - dd 与 days_since_high 都会与近因动量相关 → 我们**不宣称"预测力"**，
     只回答"相对相邻区间是否有**额外**效应"。
   - 剔次新(<250日)/ST；用 close 计算，不复权（ dividends 影响个股内部排序但对截面检验中性）。

数据：TimescaleDB bars_1d 全A，2018-2026（暖机 2018，2019-01 起出统计）。
"""
import json
import os
import tempfile
import numpy as np
import pandas as pd
import psycopg2

DB = dict(host="127.0.0.1", port=5432, dbname="quant_data", user="quant", password="quant123456")
LOAD_START = "2018-01-01"
STAT_START = "2019-01-02"
END = "2026-10-13"
WIN = 60          # 区间窗口
HORIZONS = [5, 10, 20]
FIB_RETRACE = [0.382, 0.618]
FIB_TIME = [5, 8, 13, 21, 34, 55, 89]


def load_panel():
    """只取 close + epoch 秒（避免 to_datetime 逐行解析 1200 万 date 的慢路径）。"""
    c = psycopg2.connect(**DB)
    c.autocommit = True
    cur = c.cursor()
    cur.execute("set max_parallel_workers_per_gather=0")
    cur.execute(
        """select b.asset_code,
                  extract(epoch from b.trade_date)::bigint as ts,
                  b.close
           from bars_1d b
           where b.trade_date between %s and %s and b.asset_type='stock' and b.close>0
           order by b.asset_code, b.trade_date""", (LOAD_START, END))
    rows = cur.fetchall()
    c.close()
    df = pd.DataFrame(rows, columns=["code", "ts", "close"])
    df["date"] = pd.to_datetime(df["ts"], unit="s")
    return df


def _dsh_vec(a, win):
    """距过去 win 日（不含当日）最高点已过去的交易日数。0 天 = 最高价在昨日。
    a: 1-D ndarray 价格序列 -> 返回同长数组，前 win 个为 nan。
    用 sliding_window_view 向量化，避免逐行 apply。"""
    n = len(a)
    out = np.full(n, np.nan)
    if n <= win:
        return out
    sw = np.lib.stride_tricks.sliding_window_view(a, win)   # (n-win+1, win)，第 i 行 = a[i:i+win]
    sw_prev = sw[:-1]                                        # 对应 shift(1) 的窗口
    k = (win - 1) - np.argmax(sw_prev[:, ::-1], axis=1)        # 0=最高价在窗口末尾(=昨日)
    out[win:] = k + 1                                          # 天数从 1 起
    return out


def build(df):
    g = df.groupby("code")
    df["listed_n"] = g.cumcount() + 1
    # 过去 WIN 日区间（不含当日）
    df["hi_prev"] = g["close"].transform(lambda s: s.shift(1).rolling(WIN, min_periods=WIN).max())
    df["lo_prev"] = g["close"].transform(lambda s: s.shift(1).rolling(WIN, min_periods=WIN).min())
    rng = df["hi_prev"] - df["lo_prev"]
    df["dd"] = (df["hi_prev"] - df["close"]) / rng.where(rng > 0)
    # 距过去 WIN 日最高点的天数（向量化）
    df["dsh"] = np.concatenate([
        _dsh_vec(gv["close"].to_numpy(dtype=float), WIN) for _, gv in df.groupby("code", sort=False)
    ])
    for h in HORIZONS:
        df[f"fwd{h}"] = g["close"].transform(lambda s, h=h: s.shift(-h) / s - 1)
    return df


def curve_by_bin(df, col, bin_edges, horizon, label, fib_points=None):
    """固定边界分箱（dd 用 0~1，避免极值把等分箱拉宽导致 FIB 落进同一箱）。"""
    d = df[[col, f"fwd{horizon}"]].dropna().copy()
    if d.empty:
        return None
    d["bin"] = pd.cut(d[col], bins=bin_edges, labels=False)
    agg = d.groupby("bin")[f"fwd{horizon}"].agg(["mean", "median", "size"])
    agg = agg[agg["size"] >= 100]
    agg["mean_bp"] = agg["mean"] * 1e4
    agg["median_bp"] = agg["median"] * 1e4
    agg["lo"] = np.asarray(bin_edges, dtype=float)[agg.index.values.astype(int)]
    agg["hi"] = np.asarray(bin_edges, dtype=float)[agg.index.values.astype(int) + 1]
    return agg


def peak_test(agg, fib_points, col_bp="mean_bp"):
    """在 FIB 点附近找局部凸起：比较 FIB 箱与其相邻箱。"""
    out = []
    for fp in fib_points:
        if agg is None or agg.empty:
            continue
        mid = (fp - (0.5 / len(agg)))  # 兜底
        # 找中心最接近 fp 的 bin
        centers = (agg["lo"] + agg["hi"]) / 2
        k = int(np.argmin(np.abs(centers - fp)))
        if k <= 0 or k >= len(agg) - 1:
            continue
        fib_v = agg[col_bp].iloc[k]
        left = agg[col_bp].iloc[k - 1]
        right = agg[col_bp].iloc[k + 1]
        neigh = np.nanmean([left, right])
        out.append(dict(fib=float(fp), center=float(centers.iloc[k]),
                        fib_bp=float(fib_v), neighbor_bp=float(neigh),
                        diff_bp=float(fib_v - neigh)))
    return out


def main():
    print("[1] 加载面板 ...", flush=True)
    df = load_panel()
    df = build(df)
    p = df[(df["listed_n"] >= 250) & (df["date"] >= pd.Timestamp(STAT_START))].copy()
    print(f"     有效样本 {len(p):,}  {p['date'].min().date()} ~ {p['date'].max().date()}")

    out = {"n_stocks": int(p['code'].nunique()), "n_days": int(p['date'].nunique()),
           "dd": {}, "time": {}}

    # ---------- 检验 A：回撤位 ----------
    DD_BINS = np.round(np.arange(0.0, 1.0001, 0.04), 4)   # 0~1 共 25 箱
    for h in HORIZONS:
        p["dd_c"] = p["dd"].clip(0.0, 1.0)
        agg = curve_by_bin(p, "dd_c", DD_BINS, h, "dd")
        print(f"\n=== A. 回撤深度 dd → 未来{h}日收益（{len(DD_BINS)-1} 固定箱, 0~1）===")
        if agg is None or agg.empty:
            print("     样本不足")
            continue
        print(f"     {'箱区间(回撤深度)':>18s} {'样本':>8s} {'均值bp':>9s} {'中位bp':>9s}")
        step = max(1, len(agg) // 20)
        for i in range(0, len(agg), step):
            r = agg.iloc[i]
            print(f"     {r['lo']:.3f}~{r['hi']:.3f} {int(r['size']):>8,} {r['mean_bp']:>9.1f} {r['median_bp']:>9.1f}")
        pk = peak_test(agg, FIB_RETRACE)
        for r in pk:
            print(f"     ** FIB {r['fib']:.3f} 中心 {r['center']:.3f}: {r['fib_bp']:+.1f}bp vs 相邻 {r['neighbor_bp']:+.1f}bp → 差 {r['diff_bp']:+.1f}bp")
            out["dd"].setdefault(f"h{h}", []).append(r)

    # ---------- 检验 B：时间周期 ----------
    NB_T = 60
    for h in [10]:
        d = p[["dsh", f"fwd{h}"]].dropna().copy()
        d = d[(d["dsh"] >= 1) & (d["dsh"] <= 120)]
        d["bin"] = pd.cut(d["dsh"], bins=NB_T, labels=False)
        agg = d.groupby("bin")[f"fwd{h}"].agg(["mean", "size"])
        agg = agg[agg["size"] >= 100]
        agg["mean_bp"] = agg["mean"] * 1e4
        edges = pd.cut(d["dsh"], bins=NB_T, retbins=True)[1]
        agg["lo"] = edges[agg.index.values]
        agg["hi"] = edges[agg.index.values + 1]
        cen = (agg["lo"] + agg["hi"]) / 2
        print(f"\n=== B. 距60日高点天数 dsh → 未来{h}日收益（1~120交易日，{NB_T}箱）===")
        print(f"     {'天数区间':>14s} {'样本':>8s} {'均值bp':>9s}")
        step = max(1, len(agg) // 24)
        for i in range(0, len(agg), step):
            r = agg.iloc[i]
            print(f"     {r['lo']:.0f}~{r['hi']:.0f} {int(r['size']):>8,} {r['mean_bp']:>9.1f}")
        # FIB 日尖峰检验
        print("     ** FIB 时间日检验：")
        for t in FIB_TIME:
            if t < 1 or t > 120:
                continue
            k = int(np.argmin(np.abs(cen - t)))
            if k <= 0 or k >= len(agg) - 1:
                continue
            fv = agg["mean_bp"].iloc[k]
            nv = np.nanmean([agg["mean_bp"].iloc[k - 1], agg["mean_bp"].iloc[k + 1]])
            print(f"       t={t:<3d}(箱~{cen.iloc[k]:.1f}) {fv:>+8.1f}bp vs 相邻 {nv:>+8.1f}bp → 差 {fv-nv:>+7.1f}bp")
            out["time"].setdefault(f"h{h}", []).append(dict(t=t, fib_bp=float(fv),
                                    neighbor_bp=float(nv), diff_bp=float(fv - nv)))

    out_dir = os.environ.get("QA_OUT_DIR", tempfile.gettempdir())
    path = os.path.join(out_dir, "fibonacci_test_out.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n完成 -> {path}")


if __name__ == "__main__":
    main()
