"""高收益 sleeve 探索：高beta池 / TopK 动量集中。

背景：已验证的跨资产趋势跟随（全池38只 sma200/inv_vol）年化 10.0% / Sharpe 1.03 /
回撤 −9.4%，是「能稳定赚、回撤可控」的底盘。但它的收益被两块稀释：
  (1) 池里含低波动的国债/货基，逆波动权重天然把 ~60% 仓位压在现金类；
  (2) 持有「全部」通过趋势过滤的 ETF，过度分散。
于是提两条假设（都必须走防过拟合三重测试，任一不过即判不可用）：
  H1 高beta池：剔除纯固收/货币 ETF，只在股+商品+黄金+境外上跑同样的 200日SMA 趋势跟随。
  H2 TopK集中：在通过趋势过滤的 ETF 中，按 6-1 动量取前 K 只集中持有。

铁律：基准必须与选股池同口径（同池等权），禁止拿全池等权去比高beta池策略。
"""
import numpy as np
import pandas as pd
import psycopg2

from .config import DB, ETF_COMM
from .io_db import DataError, load_etf_universe
from .metrics import metrics
from .etf_guard import etf_equal_weight

# 纯固收/货币 ETF（60日年化波动 < 5%，现金类）。
# 注：511380 可转债ETF 波动 ~9% 属混合型，保留在高beta池内。
BOND_CASH = {"511010.SH", "511090.SH", "511220.SH", "511880.SH"}


def is_bond_cash(code):
    """是否为纯固收/货币类 ETF（现金替代）。"""
    return code in BOND_CASH


def _load_etf(cur, codes, start, end):
    rows = []
    for code in codes:
        cur.execute("""SELECT trade_date, open, high, low, close, volume
                       FROM bars_1d WHERE asset_code=%s AND asset_type='etf'
                       AND trade_date>=%s AND trade_date<=%s ORDER BY trade_date""",
                    (code, start, end))
        for r in cur.fetchall():
            rows.append((code, r[0], r[1], r[2], r[3], r[4], r[5]))
    df = pd.DataFrame(rows, columns=["code", "date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    out = {}
    for code, g in df.groupby("code"):
        out[code] = g.sort_values("date").set_index("date")
    return out


def highbeta_abs_momentum(codes, start, end, signal="sma200", weight="equal",
                          topk=None, mom_lb=126, skip=21, cost=ETF_COMM, sma_w=200):
    """趋势跟随 + 可选 TopK 动量集中。

    signal: 'sma200' 收盘价>200日SMA 才通过过滤（与底盘一致）。
    weight: 'equal' 等权 / 'inv_vol' 逆波动率。
    topk:   None=持有全部通过者；K=在通过者中按 6-1(mom_lb/skip) 动量取前 K 只。
    返回 dict(nav(Series), ann, vol, sharpe, mdd, time_in_market, n_rebal)。
    """
    con = psycopg2.connect(**DB)
    cur = con.cursor()
    try:
        data = _load_etf(cur, codes, start, end)
    finally:
        con.close()
    if not data:
        raise DataError("该区间内无任何ETF数据")
    codes = list(data.keys())   # 只用实际有数据的标的，避免子窗口子集 KeyError

    all_days = sorted(set().union(*[set(g.index) for g in data.values()]))
    warmup = sma_w if signal == "sma200" else 200
    if topk:
        warmup = max(warmup, mom_lb + skip)
    if len(all_days) < warmup + 30:
        raise DataError(f"数据不足({len(all_days)} < {warmup + 30})")

    # 预计算每ETF每日：趋势信号 + 动量（用于 TopK 排序）
    sig, mom = {}, {}
    for code, g in data.items():
        c = g["close"].astype(float)
        if signal == "sma200":
            sig[code] = (c > c.rolling(sma_w).mean()).astype(float).values
        else:  # absmom: 12-1 动量 > 0
            lb = 252
            sig[code] = (c.shift(skip) / c.shift(skip + lb) - 1 > 0).astype(float).values
        mom[code] = (c.shift(skip) / c.shift(skip + mom_lb) - 1).values

    cash = 1.0
    nav, nav_dates, inmkt = [], [], []
    prev_w = {}
    last_month = None
    n_rebal = 0
    for i, d in enumerate(all_days):
        if i > 0:
            prev_d = all_days[i - 1]
            do_rb = (i > warmup) and (last_month is None
                                      or d.month != last_month.month
                                      or d.year != last_month.year)
            if do_rb:
                si = i - 1          # 用前一日信号，次月生效（无前视）
                passed = [c for c in codes
                          if si < len(sig[c]) and np.isfinite(sig[c][si]) and sig[c][si] > 0]
                if topk and len(passed) > topk:
                    scored = [(mom[c][si], c) for c in passed
                              if si < len(mom[c]) and np.isfinite(mom[c][si])]
                    scored.sort(reverse=True)   # 动量降序取前 K
                    held = [c for _, c in scored[:topk]]
                    if not held:                # 动量全 NaN 则退回全部通过者
                        held = passed
                else:
                    held = passed
                if held:
                    if weight == "inv_vol":
                        w = {}
                        for code in held:
                            ret = data[code]["close"].astype(float).pct_change()
                            vol = ret.iloc[max(0, si - 59):si + 1].std() * np.sqrt(252)
                            w[code] = (1.0 / max(vol, 0.05)) if np.isfinite(vol) and vol > 0 else 0.0
                        tot = sum(w.values())
                        target = {c: w[c] / tot for c in held} if tot > 0 else {}
                    else:
                        target = {c: 1.0 / len(held) for c in held}
                else:
                    target = {}
                turnover = sum(abs(target.get(c, 0.0) - prev_w.get(c, 0.0))
                               for c in set(list(target) + list(prev_w)))
                cash *= (1 - turnover * cost)
                prev_w = target
                last_month = d
                n_rebal += 1
            # 当日收益（用昨日权重，T+1 对齐）
            day_ret = 0.0
            if prev_w:
                tot_w = 0.0
                for code, w in prev_w.items():
                    if w <= 0:
                        continue
                    g = data[code]
                    ct = g["close"].get(d)
                    cp = g["close"].get(prev_d)
                    if ct is not None and cp is not None and pd.notna(ct) and pd.notna(cp) and cp > 0:
                        day_ret += w * (ct / cp - 1)
                        tot_w += w
                if tot_w > 0:
                    day_ret /= tot_w
            cash *= (1 + day_ret)
        nav.append(cash)
        nav_dates.append(d)
        inmkt.append(1.0 if prev_w else 0.0)

    nav_s = pd.Series(nav, index=nav_dates)
    m = metrics(nav_s.pct_change().dropna())
    m["nav"] = nav_s
    m["time_in_market"] = float(np.mean(inmkt))
    m["n_rebal"] = n_rebal
    return m


def _stat(r):
    """统一取指标。etf_equal_weight 的基线无 time_in_market/n_rebal，用默认补齐
    （等权买入持有 = 全程在市、0 次调仓），否则 walk-forward 基线会 KeyError。"""
    return dict(ann=r["ann"], sharpe=r["sharpe"], mdd=r["mdd"],
                time_in_market=r.get("time_in_market", 1.0), n_rebal=r.get("n_rebal", 0))


def run_higher(start="2019-01-01", end="2026-09-30", clean_only=True):
    """全周期对比：全池底盘 vs 高beta池各配置，各自对同池等权基线。"""
    codes_all = load_etf_universe(start, end, clean_only)
    if not codes_all:
        raise DataError("ETF 池为空")
    hb = [c for c in codes_all if not is_bond_cash(c)]
    out = dict(n_all=len(codes_all), n_hb=len(hb), start=start, end=end,
               baselines={}, configs={})

    # 同池基线（铁律：基准与选股池同口径）
    b_all = etf_equal_weight(codes_all, start, end, gate_code=None)
    b_hb = etf_equal_weight(hb, start, end, gate_code=None)
    out["baselines"]["equal_all"] = dict(ann=b_all["ann"], sharpe=b_all["sharpe"],
                                         mdd=b_all["mdd"], time_in_market=1.0, n_rebal=0)
    out["baselines"]["equal_hb"] = dict(ann=b_hb["ann"], sharpe=b_hb["sharpe"],
                                        mdd=b_hb["mdd"], time_in_market=1.0, n_rebal=0)

    plans = [
        ("底盘 全池 sma200/inv_vol", codes_all, dict(signal="sma200", weight="inv_vol")),
        ("H1 高beta sma200/equal", hb, dict(signal="sma200", weight="equal")),
        ("H1 高beta sma200/inv_vol", hb, dict(signal="sma200", weight="inv_vol")),
        ("H2 高beta topk3/equal", hb, dict(signal="sma200", weight="equal", topk=3)),
        ("H2 高beta topk3/inv_vol", hb, dict(signal="sma200", weight="inv_vol", topk=3)),
        ("H2 高beta topk5/equal", hb, dict(signal="sma200", weight="equal", topk=5)),
        ("H2 高beta topk5/inv_vol", hb, dict(signal="sma200", weight="inv_vol", topk=5)),
    ]
    for name, pool, kw in plans:
        r = highbeta_abs_momentum(pool, start, end, **kw)
        out["configs"][name] = _stat(r)
    return out


# ---------------- 防过拟合三重测试（针对高收益 sleeve） ----------------
def walk_forward_higher(start_all="2019-01-01", end="2026-09-30",
                        train_win=("2019-01-01", "2022-12-31"),
                        test_win=("2023-01-01", "2026-09-30"),
                        topks=(None, 3, 5), preset_topk=3, clean_only=True):
    """训练窗(2019-2022)内选参 -> 完全隔离测试窗(2023-2026)验证，基线=高beta池等权。

    topks 控制可选参数网格：topks=(None,) 即只验 H1（高beta池，不做TopK集中）。
    """
    codes = load_etf_universe(start_all, end, clean_only)
    hb = [c for c in codes if not is_bond_cash(c)]
    grid = []
    for w in ("equal", "inv_vol"):
        for k in topks:
            grid.append((w, k))
    # 训练窗选参
    best, best_sh = None, -9e9
    for w, k in grid:
        try:
            r = highbeta_abs_momentum(hb, train_win[0], train_win[1],
                                      signal="sma200", weight=w, topk=k)
        except DataError:
            continue
        if r["sharpe"] > best_sh:
            best, best_sh = (w, k), r["sharpe"]
    tr = highbeta_abs_momentum(hb, train_win[0], train_win[1],
                               signal="sma200", weight=best[0], topk=best[1])
    # 隔离测试窗
    tb = highbeta_abs_momentum(hb, test_win[0], test_win[1],
                               signal="sma200", weight=best[0], topk=best[1])
    tp = highbeta_abs_momentum(hb, test_win[0], test_win[1],
                               signal="sma200", weight="inv_vol", topk=preset_topk)  # 预设（未用测试窗选）
    base = etf_equal_weight(hb, test_win[0], test_win[1], gate_code=None)
    return dict(n_hb=len(hb), train_win=train_win, test_win=test_win,
                train=dict(config=f"sma200/{best[0]}/topk{best[1]}", **_stat(tr)),
                test_best=dict(config=f"sma200/{best[0]}/topk{best[1]}", **_stat(tb)),
                test_pre=dict(config="sma200/inv_vol/topk3", **_stat(tp)),
                test_base=dict(config="equal_hb", **_stat(base)))


def param_topk(start="2019-01-01", end="2026-09-30", ks=(3, 5, 8, 12), clean_only=True):
    """TopK 参数敏感：K 变化时 Sharpe 应成簇，单点尖峰=过拟合。"""
    codes = load_etf_universe(start, end, clean_only)
    hb = [c for c in codes if not is_bond_cash(c)]
    rows = []
    for k in ks:
        for w in ("equal", "inv_vol"):
            r = highbeta_abs_momentum(hb, start, end, signal="sma200", weight=w, topk=k)
            rows.append(dict(topk=k, weight=w, **_stat(r)))
    return rows


def cost_sens_higher(start="2019-01-01", end="2026-09-30", topk=3, weight="inv_vol",
                     costs=(0.0001, 0.0003, 0.0005, 0.001), clean_only=True):
    """成本敏感：万1~千1 是否仍跑赢高beta池等权基线。"""
    codes = load_etf_universe(start, end, clean_only)
    hb = [c for c in codes if not is_bond_cash(c)]
    base = etf_equal_weight(hb, start, end, gate_code=None)
    rows = []
    for c in costs:
        r = highbeta_abs_momentum(hb, start, end, signal="sma200",
                                  weight=weight, topk=topk, cost=c)
        rows.append(dict(cost=c, **_stat(r), beats=bool(r["sharpe"] > base["sharpe"]),
                         base_sharpe=base["sharpe"]))
    return rows
