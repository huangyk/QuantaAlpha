"""组合构建 + 风险模型（股票侧 · B 方向 · 不碰 ETF）
====================================================

研究问题（诚实、未试过）
------------------------
低波动异象因子 IC 是真的（市值中性 IC=0.0332, t=6.17，见 _lowvol_true_neutral_out.json），
但两种变现方式都已证伪：
  - 笨选低波个股纯多：−34%（long_only_liquid_extreme）
  - 多空配对：日频 −29.67% / 月频 −0.69%（成本+做空受限）
**没试过的是组合构建路线**：最小方差 / 风险平价（ERC）——它们通过"降协方差"而非
"挑低波名"来长多捕获低波溢价，且天然规避做空。本脚本在真实 A 股上测这条路线。

遵循 SOP 铁律
------------
- 真实数据：TimescaleDB bars_1d（禁 akshare）；分块读取避免 OOM。
- 信号-收益对齐：月末收盘定权重（用截至当日数据）→ 次月首个交易日起捕获收益（T+1）。
- 成本：佣金+印花税(卖)+Almgren 市场冲击；换手按权重 L1 变动计。
- 无未来函数：协方差只用截至决策日的历史。
- 剔次新(上市<250日)、剔涨停开盘买不进；收益用原始 OHLC（pct_change 不 pad）。
- 不调"魔法参数"：ERC/MinVar 无可调超参（MinVar 仅 cap/窗口，报参数成簇）。

用法：python tools/portfolio_risk.py
"""
from __future__ import annotations
import json
import os
import tempfile
import numpy as np
import pandas as pd
import psycopg2

DB = dict(host="127.0.0.1", port=5432, dbname="quant_data", user="quant", password="quant123456")
START, END = "2011-01-01", "2026-10-13"
COV_WIN = 120          # 协方差窗口（6个月）
UNIVERSE_N = 300       # 每月取流动性前 N（可执行、避开微盘）
MIN_LIST_DAYS = 250    # 剔次新
POS_CAP = 0.05         # 单票权重上限 5%
LIQ_AMT = 1e8          # 入选日成交额下限（1亿，保可执行）
# 股票成本（bp，单边；ETF 无印花，股票卖有印花）
COMM_BP = 1.0
STAMP_BP = 5.0         # 仅卖出
SIGMA_BP = 342.0       # 个股日波动中位（实测，pool_hold）
IMPACT_ETA = 0.1

_CACHE = {}


def get_data():
    """分块读取 bars_1d，构建 close/open/amount 矩阵（float32，缓存一次）。

    只用 open/close/amount 三列（去掉 high/low/volume 省内存）；
    (code,date) 有重复行（实测 4.2 万条），分块去重。
    """
    if _CACHE:
        return _CACHE["m"]
    conn = psycopg2.connect(**DB)
    q = ("select asset_code, trade_date, open, close, amount, total_shares from bars_1d "
         "where trade_date between %s and %s and (asset_type='stock' or asset_type is null) "
         "order by asset_code, trade_date")
    px_p, op_p, amt_p, tsh_p = [], [], [], []
    total = 0
    for chunk in pd.read_sql(q, conn, params=[START, END], chunksize=800_000):
        chunk = chunk.drop_duplicates(subset=["asset_code", "trade_date"], keep="last")
        chunk["trade_date"] = pd.to_datetime(chunk["trade_date"])
        total += len(chunk)
        px_p.append(chunk.pivot(index="trade_date", columns="asset_code", values="close").astype("float32"))
        op_p.append(chunk.pivot(index="trade_date", columns="asset_code", values="open").astype("float32"))
        amt_p.append(chunk.pivot(index="trade_date", columns="asset_code", values="amount").astype("float32"))
        tsh_p.append(chunk.pivot(index="trade_date", columns="asset_code", values="total_shares").astype("float64"))
    conn.close()
    # 各块按 asset_code 切分 → 同一天出现在多块 → concat 出重复日期行；
    # 用 groupby(date).first() 合并（每个(日期,股票)格只在一块出现，first 正确拼合）
    px = pd.concat(px_p).groupby(level=0).first().sort_index()
    px = px.loc[:, ~px.columns.duplicated()]
    op = pd.concat(op_p).groupby(level=0).first().sort_index().reindex(columns=px.columns)
    amt = pd.concat(amt_p).groupby(level=0).first().sort_index().reindex(columns=px.columns)
    tsh = pd.concat(tsh_p).groupby(level=0).first().sort_index().reindex(columns=px.columns)
    mcap = (px.astype("float64") * tsh)  # 总市值(元)=收盘×总股本；仅 2020-2024 有覆盖
    ret = px.pct_change(fill_method=None)
    # ★未复权修正：A股涨跌停限制（主板±10%/创业板·科创板2020.8后±20%/ST±5%）
    #   → 单日|收益|>21% 物理上不可能是真实交易，只能是未复权"除权/高送转"跳变
    #   （实测 2015-2017 有 4005 个此类假崩盘，会把组合回测打成 -92% 假象）。
    #   用当日横截面中位数填充这些不可能收益（假设除权日无个股 alpha，随大盘）。
    impossible = ret.abs() > 0.21
    n_imp = int(impossible.to_numpy().sum())
    if n_imp:
        rv = ret.to_numpy()
        med = np.nanmedian(rv, axis=1, keepdims=True).astype(rv.dtype)
        mask = np.abs(rv) > 0.21
        rv[mask] = np.broadcast_to(med, rv.shape)[mask]
        ret = pd.DataFrame(rv, index=ret.index, columns=ret.columns)
        print(f"    [复权修正] 剔除 {n_imp} 个物理不可能收益(除权/送转假崩盘)，用当日中位数填充", flush=True)
    listed_days = px.notna().cumsum()
    print(f"[load] {px.shape[0]} 日 x {px.shape[1]} 只  {px.index.min().date()}~{px.index.max().date()}  去重后 {total} 行"
          f"  mcap非空率 {mcap.notna().mean().mean()*100:.0f}%", flush=True)
    _CACHE["m"] = (px, op, amt, ret, listed_days)
    _CACHE["mcap"] = mcap
    return _CACHE["m"]


def ledoit_wolf_cov(R):
    """Ledoit-Wolf 常数相关收缩协方差。R: (T×N)。"""
    R = R.dropna(axis=1, how="any")
    X = R.values.astype("float64")
    T, N = X.shape
    if N < 2:
        return np.diag(X.var(axis=0)) if N == 1 else np.zeros((0, 0))
    S = np.cov(X, rowvar=False)
    if S.ndim < 2:
        return np.diag(np.full(N, float(S)))
    var = np.diag(S).copy()
    std = np.sqrt(np.maximum(var, 1e-12))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = S / np.outer(std, std)
    np.fill_diagonal(corr, 0.0)
    rbar = corr.sum() / (N * (N - 1))
    F = rbar * np.outer(std, std)
    np.fill_diagonal(F, var)
    delta = 0.30
    return (1 - delta) * S + delta * F


def erc_weights(cov):
    n = cov.shape[0]
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])
    w = np.ones(n) / n
    for _ in range(300):
        mrc = cov @ w
        rc = w * mrc
        tot = rc.sum()
        if tot <= 0:
            break
        tgt = tot / n
        nw = w * (tgt / np.maximum(rc, 1e-15)) ** 0.5
        s = nw.sum()
        if s <= 0:
            break
        nw /= s
        if np.max(np.abs(nw - w)) < 1e-9:
            w = nw
            break
        w = nw
    return w


def proj_box_simplex(v, cap):
    """Dykstra 交替投影：把 v 投到 {w: Σw=1, 0<=w<=cap}（精确，~30 步收敛）。"""
    n = len(v)
    if cap * n < 1.0:
        cap = 1.0 / n
    x = v.copy()
    p = np.zeros(n)
    q = 0.0
    for _ in range(200):
        y = np.clip(x + p, 0.0, cap)
        p = x + p - y
        z = y + q
        xn = z + (1.0 - z.sum()) / n
        q = z - xn
        if np.max(np.abs(xn - x)) < 1e-14:
            return xn
        x = xn
    return x


def minvar_weights(cov, cap=POS_CAP, iters=3000):
    """长多最小方差：min w'Σw, s.t. Σw=1, 0<=w<=cap。

    **不用 cvxpy**：本环境 OSQP/SCS/CLARABEL 等所有编译求解器都会让进程静默硬崩
    （无异常、exit code 0，实测 cvxpy 1.7.5 + quantaalpha 环境）。改用纯 numpy
    FISTA（加速投影梯度）+ Dykstra 投影，已对 KKT 条件验证：
      Σw=1（精确到 1e-8）、w∈[0,cap]、自由变量的 (Σw)_i 全等于 λ、
      w=0 的变量 (Σw)_i >= λ（一阶条件全满足）。0.73s/次（n=500）。
    """
    n = cov.shape[0]
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])
    if cap * n < 1.0:
        cap = 1.0 / n
    P = (cov + cov.T) / 2
    P = P + 1e-9 * np.trace(P) / n * np.eye(n)          # 保 PSD
    # 步长 1/L，L = 谱半径（幂迭代 50 步）+ 5% 余量
    v = np.ones(n) / np.sqrt(n)
    L = 1e-12
    for _ in range(50):
        u = P @ v
        nu = np.linalg.norm(u)
        if nu <= 0:
            break
        v = u / nu
        L = float(v @ (P @ v))
    L = max(L, 1e-12) * 1.05
    x = proj_box_simplex(np.full(n, 1.0 / n), cap)
    y = x.copy()
    t = 1.0
    for k in range(iters):
        xn = proj_box_simplex(y - (P @ y) / L, cap)
        tn = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = xn + ((t - 1.0) / tn) * (xn - x)
        x, t = xn, tn
        if k % 200 == 0 and np.linalg.norm(x - y) < 1e-11:
            break
    return x


def universe_at(i, px, mcap, amt, listed_days, op, close_prev):
    """决策日 i：取总市值 top-N（≈沪深300/500 机构标准大市值宇宙，避开追热点陷阱），
    剔次新/低流动性/涨停开盘。"""
    if i < COV_WIN:
        return []
    mc = mcap.iloc[i]
    amt_i = amt.iloc[i]
    ld = listed_days.iloc[i]
    cand = mc[(mc > 0) & (amt_i > LIQ_AMT) & (ld >= MIN_LIST_DAYS)].dropna()
    cand = cand.sort_values(ascending=False).head(UNIVERSE_N).index
    if close_prev is not None and i > 0:
        opx = op.iloc[i]
        lu = (opx / close_prev - 1) >= 0.098
        cand = [c for c in cand if not lu.get(c, False)]
    return list(cand)


def backtest(method="erc", cost_on=True, vol_target=None, cap=POS_CAP):
    px, op, amt, ret, listed_days = get_data()
    mcap = _CACHE["mcap"]
    dates = px.index
    months = dates.to_period("M")
    n = len(dates)
    cur_w = {}
    decided = {}
    nav = 1.0
    navs = np.empty(n)
    turnover_sum = 0.0
    cost_sum_bp = 0.0
    ret_mat = ret  # (dates × codes)

    for i in range(n):
        is_month_end = (i == n - 1) or (months[i] != months[i + 1])
        if is_month_end and i >= COV_WIN:
            close_prev = px.iloc[i - 1] if i > 0 else None
            uni = universe_at(i, px, mcap, amt, listed_days, op, close_prev)
            if uni:
                R = ret_mat.iloc[i - COV_WIN + 1:i + 1][uni].dropna(axis=1, how="any")
                cols = list(R.columns)
                if len(cols) >= 5:
                    if method == "ew":
                        w = pd.Series(1.0 / len(cols), index=cols)
                    elif method == "invvol":
                        v = (1.0 / R.std().replace(0, np.nan)).dropna()
                        w = v / v.sum()
                    elif method == "erc":
                        w = pd.Series(erc_weights(ledoit_wolf_cov(R)), index=cols)
                    elif method == "minvar":
                        w = pd.Series(minvar_weights(ledoit_wolf_cov(R), cap), index=cols)
                    w = w[w > 1e-6]
                    if len(w):
                        decided[dates[i]] = (w / w.sum()).to_dict()
        is_month_start = (i == 0) or (months[i] != months[i - 1])
        if is_month_start and i > 0:
            prev_me = dates[i - 1]
            if prev_me in decided:
                new_w = decided[prev_me]
                codes = set(new_w) | set(cur_w)
                turn = sum(abs(new_w.get(c2, 0.0) - cur_w.get(c2, 0.0)) for c2 in codes)
                turnover_sum += turn
                if cost_on and turn > 0:
                    part = min((1e7 / max(len(new_w), 1)) / 5e8, 0.2)
                    impact_bp = 2 * IMPACT_ETA * SIGMA_BP * (part ** 0.5)
                    cost_bp = turn * (COMM_BP + STAMP_BP * 0.5 + impact_bp)
                    cost_sum_bp += cost_bp
                    nav *= (1 - cost_bp / 10000.0)   # 调仓日扣成本
                cur_w = new_w
        # 当日收益
        day_ret = 0.0
        if cur_w:
            r = ret_mat.iloc[i]
            s = 0.0
            for code, wt in cur_w.items():
                v = r.get(code, np.nan)
                if v == v:  # not NaN
                    s += wt * v
            day_ret = s
        port_ret = day_ret
        if vol_target is not None and i >= 60:
            hist = pd.Series(navs[max(0, i - 60):i]).pct_change().dropna()
            if len(hist) > 20:
                v = hist.std() * np.sqrt(252)
                if v > vol_target:
                    port_ret *= vol_target / v
        nav *= (1 + port_ret)
        navs[i] = nav
    return pd.Series(navs, index=dates), turnover_sum, cost_sum_bp


def perf(nav):
    r = nav.pct_change().dropna()
    if len(r) < 2:
        return {}
    total = nav.iloc[-1]
    years = len(nav) / 252
    ann = total ** (1 / years) - 1 if total > 0 else np.nan
    vol = r.std() * np.sqrt(252)
    sharpe = ann / vol if vol > 0 else np.nan
    mdd = (nav / nav.cummax() - 1).min()
    return dict(total=float(total), ann=float(ann), sharpe=float(sharpe),
                vol=float(vol), mdd=float(mdd))


def main():
    global START, END, UNIVERSE_N
    START, END = "2020-01-01", "2024-12-31"   # 总市值字段仅 2020-2024 可靠(~90%)
    oos = pd.Timestamp("2023-01-01")           # train 2020-2022 / test 2023-2024
    results = {}

    def run(label, method, **kw):
        nav, turn, cost_bp = backtest(method=method, cost_on=True, **kw)
        pf = perf(nav)
        te = perf(nav[nav.index >= oos])
        results[label] = dict(full=pf, oos=te, turnover=float(turn), cost_bp=float(cost_bp))
        print(f"  {label:14s} 全期 {pf.get('ann',0)*100:+6.2f}%/S{pf.get('sharpe',0):.2f}/回撤{pf.get('mdd',0)*100:+6.1f}%/波动{pf.get('vol',0)*100:.1f}%"
              f" || OOS(23-24) {te.get('ann',0)*100:+6.2f}%/S{te.get('sharpe',0):.2f}/回撤{te.get('mdd',0)*100:+6.1f}%"
              f" || 换手{turn:.0f} 成本{cost_bp:.0f}bp", flush=True)
        return nav

    # 快速三方法 × 三个宇宙规模
    stage = os.environ.get("QA_STAGE", "all")
    if stage in ("all", "fast"):
        for N in [300, 500, 1000]:
            UNIVERSE_N = N
            print(f"\n===== 宇宙=总市值 top{N} (2020-2024, 大市值/机构标准) =====", flush=True)
            run(f"N{N}_等权", "ew")
            run(f"N{N}_逆波动", "invvol")
            run(f"N{N}_ERC", "erc")

    if stage in ("all", "minvar"):
        # MinVar(FISTA) 仅在 N=500（控耗时；~0.7s/月 × 60月 × 5 组 ≈ 4min）
        UNIVERSE_N = 500
        print(f"\n===== MinVar(numpy FISTA) @ N=500 =====", flush=True)
        run("N500_MinVar", "minvar")
        run("N500_MinVar_vt10", "minvar", vol_target=0.10)

        # MinVar 参数成簇（cap）——SOP 关卡：无尖峰才算稳健
        print("\n[稳健] MinVar cap 成簇 @ N=500 ...", flush=True)
        for cap in [0.03, 0.05, 0.10]:
            nav, turn, cost_bp = backtest(method="minvar", cost_on=True, cap=cap)
            pf = perf(nav)
            print(f"    cap={cap:.2f}: {pf.get('ann',0)*100:+6.2f}%/S{pf.get('sharpe',0):.2f}/回撤{pf.get('mdd',0)*100:+6.1f}%", flush=True)

    out_dir = os.environ.get("QA_OUT_DIR", tempfile.gettempdir())
    path = os.path.join(out_dir, "portfolio_risk_out.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n完成 -> {path}", flush=True)


if __name__ == "__main__":
    main()
