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
from .method_absmom import abs_momentum_etf

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


# ---------------- 收益-回撤前沿（核心-卫星混合） ----------------
def blend_absmomentum(codes_all, start, end, blend, signal="sma200",
                      weight="inv_vol", cost=ETF_COMM, sma_w=200):
    """核心-卫星混合：底盘(全池 sma200/inv_vol) 与 H1(高beta池 sma200/inv_vol)
    按 blend(0~1, H1 卫星权重) 组合。

    模型：投资者把 (1-blend) 配到底盘 sleeve、blend 配到 H1 sleeve；每个 sleeve
    各自月频调仓、内部可转现金（不是把两个池合并成一个池）。组合日收益 =
    (1-blend)*底盘日收益 + blend*H1日收益，再复利成 NAV。
    blend=1.0 即纯 H1；blend=0.0 即纯底盘。返回 dict(nav, ann, vol, sharpe, mdd)。
    """
    codes = list(codes_all)
    hb = [c for c in codes if not is_bond_cash(c)]
    base = abs_momentum_etf(codes, start, end, signal=signal, weight=weight,
                            cost=cost, sma_w=sma_w)
    h1 = highbeta_abs_momentum(hb, start, end, signal=signal, weight=weight,
                               cost=cost, sma_w=sma_w)
    rb = base["nav"].pct_change().dropna()
    rh = h1["nav"].pct_change().dropna()
    common = rb.index.intersection(rh.index)
    br = rb.loc[common]; hr = rh.loc[common]
    blended_ret = (1.0 - blend) * br + blend * hr
    nav = (1.0 + blended_ret).cumprod()
    m = metrics(nav.pct_change().dropna())
    m["nav"] = nav
    m["time_in_market"] = float(np.mean([base["time_in_market"], h1["time_in_market"]]))
    m["n_rebal"] = 0
    return m


def run_blend_frontier(start="2019-01-01", end="2026-09-30",
                       blends=(1.0, 0.75, 0.5, 0.25, 0.0), clean_only=True):
    """收益-回撤前沿：底盘↔H1 五档混合（全周期 2019-2026）。

    返回 dict：每行含 blend(H1权重)/base_frac/h1_frac/ann/sharpe/mdd/vol，
    外加两端同口径等权基线（base_equal=全池等权, hb_equal=高beta池等权）。
    """
    codes = load_etf_universe(start, end, clean_only)
    if not codes:
        raise DataError("ETF 池为空")
    n_all = len(codes)
    n_hb = len([c for c in codes if not is_bond_cash(c)])
    base_eq = etf_equal_weight(codes, start, end, gate_code=None)
    hb_eq = etf_equal_weight([c for c in codes if not is_bond_cash(c)], start, end, gate_code=None)
    rows = []
    for b in blends:
        r = blend_absmomentum(codes, start, end, b)
        rows.append(dict(blend=b, base_frac=1.0 - b, h1_frac=b,
                         ann=r["ann"], sharpe=r["sharpe"], mdd=r["mdd"], vol=r["vol"]))
    return dict(n_all=n_all, n_hb=n_hb, start=start, end=end, blends=blends, rows=rows,
                base_equal=dict(ann=base_eq["ann"], sharpe=base_eq["sharpe"], mdd=base_eq["mdd"]),
                hb_equal=dict(ann=hb_eq["ann"], sharpe=hb_eq["sharpe"], mdd=hb_eq["mdd"]))


def walk_forward_blend(start_all="2019-01-01", end="2026-09-30",
                       train_win=("2019-01-01", "2022-12-31"),
                       test_win=("2023-01-01", "2026-09-30"),
                       blends=(1.0, 0.75, 0.5, 0.25, 0.0), clean_only=True):
    """混合比例 walk-forward OOS：训练/测试窗各自组合，验证所选比例在隔离窗仍稳定。

    关键：blend 是用户风险偏好(非挖掘参数)，测试窗沿用同比例、不重新选参 -> 无过拟合。
    基线思想：每比例的"应跑赢对象" = 同比例混合两端等权基线（base_equal 与 hb_equal 按
    同 blend 混合）。若测试窗混合 Sharpe > 混合基线 Sharpe，说明该比例的趋势跟随增强
    在隔离窗仍有效。
    """
    codes = load_etf_universe(start_all, end, clean_only)
    hb = [c for c in codes if not is_bond_cash(c)]
    base_tr = abs_momentum_etf(codes, *train_win, signal="sma200", weight="inv_vol")["nav"]
    h1_tr = highbeta_abs_momentum(hb, *train_win, signal="sma200", weight="inv_vol")["nav"]
    base_te = abs_momentum_etf(codes, *test_win, signal="sma200", weight="inv_vol")["nav"]
    h1_te = highbeta_abs_momentum(hb, *test_win, signal="sma200", weight="inv_vol")["nav"]

    def blend_nav(bn, hn, blend):
        rb = bn.pct_change().dropna(); rh = hn.pct_change().dropna()
        common = rb.index.intersection(rh.index)
        ret = (1.0 - blend) * rb.loc[common] + blend * rh.loc[common]
        return (1.0 + ret).cumprod()

    rows_tr, rows_te = [], []
    for b in blends:
        nt = blend_nav(base_tr, h1_tr, b); mt = metrics(nt.pct_change().dropna())
        ne = blend_nav(base_te, h1_te, b); me = metrics(ne.pct_change().dropna())
        rows_tr.append(dict(blend=b, ann=mt["ann"], sharpe=mt["sharpe"], mdd=mt["mdd"]))
        rows_te.append(dict(blend=b, ann=me["ann"], sharpe=me["sharpe"], mdd=me["mdd"]))
    return dict(blends=blends, train_win=train_win, test_win=test_win,
                train=rows_tr, test=rows_te)


def rolling_walk_forward_blend(blends=(0.0, 0.25, 0.5, 1.0),
                               start_all="2019-01-01", end="2026-09-30",
                               first_test_year=2022, clean_only=True):
    """逐年滚动 OOS：固定 blend(用户风险偏好, 非挖掘参数)下，把 2022~2026 每一年单独切成
    样本外，看是否每年都赚钱。

    关键：策略参数(sma200/inv_vol)全程固定、从未在任何数据上选参 -> 对完整回测 NAV 按年
    切分日收益即诚实 OOS（无前视、无重选参）。2026 仅至 9/30 为部分年(年化仅供参考)。
    返回 dict：per_year[blend]=逐年明细, oos[blend]=全样本外合并指标。
    """
    codes = load_etf_universe(start_all, end, clean_only)
    hb = [c for c in codes if not is_bond_cash(c)]
    base_full = abs_momentum_etf(codes, start_all, end, signal="sma200",
                                 weight="inv_vol")["nav"]
    h1_full = highbeta_abs_momentum(hb, start_all, end, signal="sma200",
                                    weight="inv_vol")["nav"]
    rb = base_full.pct_change().dropna()
    rh = h1_full.pct_change().dropna()
    common = rb.index.intersection(rh.index)
    rb = rb.loc[common]; rh = rh.loc[common]
    end_year = pd.Timestamp(end).year
    per_year = {b: [] for b in blends}
    oos_ret = {b: [] for b in blends}
    for Y in range(first_test_year, end_year + 1):
        ystart = f"{Y}-01-01"
        yend = end if Y == end_year else f"{Y}-12-31"
        mask = (rb.index >= ystart) & (rb.index <= yend)
        if mask.sum() < 20:
            continue
        for b in blends:
            br = (1.0 - b) * rb[mask] + b * rh[mask]
            m = metrics(br)
            per_year[b].append(dict(year=Y, partial=(Y == end_year),
                                    ann=m["ann"], sharpe=m["sharpe"], mdd=m["mdd"],
                                    ret=float((1 + br).prod() - 1)))
            oos_ret[b].append(br.values)
    oos = {}
    for b in blends:
        allr = np.concatenate(oos_ret[b]) if oos_ret[b] else np.array([])
        oos[b] = metrics(allr) if len(allr) else dict(ann=0.0, sharpe=0.0, mdd=0.0)
    return dict(blends=blends, first_test_year=first_test_year, end_year=end_year,
                per_year=per_year, oos=oos)


def leverage_scenario(codes_all, start="2019-01-01", end="2026-09-30",
                      leverages=(1.0, 1.2, 1.3, 1.5), financing=0.06,
                      signal="sma200", weight="inv_vol", blend=0.0, sma_w=200,
                      clean_only=True):
    """在低回撤底盘上施加适度杠杆(执行层 L4 杠杆, 非 L2 因子过拟合)，看收益/回撤/夏普。

    杠杆模型：日收益 = L*策略日收益 - (L-1)*融资利率/252（借入部分按日付息）。
    真实约束(融资利率5~8%、维持担保比例、券商额度、防御期空仓仍付杠杆利息)未建模，
    此处融资利率取 6% 中值仅作上限示意。blend>0 时先混 H1 再杠杆。
    """
    if blend <= 0:
        nav = abs_momentum_etf(codes_all, start, end, signal=signal,
                               weight=weight, sma_w=sma_w)["nav"]
    else:
        hb = [c for c in codes_all if not is_bond_cash(c)]
        nav = blend_absmomentum(codes_all, start, end, blend,
                                signal=signal, weight=weight, sma_w=sma_w)["nav"]
    ret = nav.pct_change().dropna()
    rows = []
    for L in leverages:
        lev_ret = L * ret - (L - 1) * financing / 252.0
        m = metrics(lev_ret)
        rows.append(dict(L=L, ann=m["ann"], sharpe=m["sharpe"], mdd=m["mdd"], vol=m["vol"]))
    return dict(financing=financing, blend=blend, rows=rows)


def vol_target_blend(codes_all, start="2019-01-01", end="2026-09-30", blend=0.0,
                     target_vol=0.10, vol_win=60, max_lev=1.5, financing=0.06,
                     signal="sma200", weight="inv_vol", sma_w=200, clean_only=True):
    """进阶：在混合策略上叠加波动率目标化(执行层 L4, 合法风控非因子过拟合)。

    机制：用 vol_win(60) 日滚动波动估组合年化波动，日杠杆 = min(target_vol/滚动波动, max_lev)；
    用**前一日**波动估计(无前视)，高波动期自动降仓(转现金)、低波动期加仓(≤max_lev)。
    仅对杠杆部分(lev>1)计融资利息。返回 dict(nav, ann, vol, sharpe, mdd, avg_lev)。
    """
    if blend <= 0:
        nav0 = abs_momentum_etf(codes_all, start, end, signal=signal,
                                weight=weight, sma_w=sma_w)["nav"]
    else:
        hb = [c for c in codes_all if not is_bond_cash(c)]
        nav0 = blend_absmomentum(codes_all, start, end, blend,
                                 signal=signal, weight=weight, sma_w=sma_w)["nav"]
    ret = nav0.pct_change().dropna()
    roll = ret.rolling(vol_win).std() * np.sqrt(252)
    lev = (target_vol / roll).clip(upper=max_lev).shift(1).fillna(1.0)  # 前一日估计, 避免前视
    fin_drag = (lev - 1.0).clip(lower=0.0) * financing / 252.0
    lev_ret = lev * ret - fin_drag
    nav = (1 + lev_ret).cumprod()
    m = metrics(lev_ret.dropna())
    m["nav"] = nav
    m["avg_lev"] = float(lev.mean())
    return m


def vol_target_walk_forward(codes_all, blend=0.0, target_vol=0.10, vol_win=60,
                            max_lev=1.5, financing=0.06,
                            train_win=("2019-01-01", "2022-12-31"),
                            test_win=("2023-01-01", "2026-09-30"), clean_only=True):
    """vol targeting 的 walk-forward OOS：参数(target_vol/vol_win/max_lev)全程固定、未选参，
    直接切训练/测试窗验证样本外是否仍稳定。返回 dict(blends?, train, test)。"""
    def run(win):
        if blend <= 0:
            nav0 = abs_momentum_etf(codes_all, win[0], win[1], signal="sma200",
                                    weight="inv_vol")["nav"]
        else:
            hb = [c for c in codes_all if not is_bond_cash(c)]
            nav0 = blend_absmomentum(codes_all, win[0], win[1], blend)["nav"]
        ret = nav0.pct_change().dropna()
        roll = ret.rolling(vol_win).std() * np.sqrt(252)
        lev = (target_vol / roll).clip(upper=max_lev).shift(1).fillna(1.0)
        lev_ret = lev * ret - (lev - 1.0).clip(lower=0.0) * financing / 252.0
        return metrics(lev_ret.dropna())
    tr = run(train_win); te = run(test_win)
    return dict(blend=blend, target_vol=target_vol, max_lev=max_lev,
                train=dict(ann=tr["ann"], sharpe=tr["sharpe"], mdd=tr["mdd"], vol=tr["vol"]),
                test=dict(ann=te["ann"], sharpe=te["sharpe"], mdd=te["mdd"], vol=te["vol"]))


def bootstrap_blend(codes_all, start="2019-01-01", end="2026-09-30", blend=0.0,
                    n=1000, block=21, seed=20261012, signal="sma200", weight="inv_vol",
                    sma_w=200, clean_only=True):
    """块 Bootstrap 显著性检验：把日收益按 block(21日=1月) 整块重采样 n 次，重算年化/夏普/回撤，
    得 5%/95% 置信区间。若年化 CI 下限仍>0，说明"稳定赚"非运气(统计显著)。
    块采样保留收益自相关，比普通 bootstrap 更贴近真实时序。
    """
    if blend <= 0:
        nav0 = abs_momentum_etf(codes_all, start, end, signal=signal,
                                weight=weight, sma_w=sma_w)["nav"]
    else:
        hb = [c for c in codes_all if not is_bond_cash(c)]
        nav0 = blend_absmomentum(codes_all, start, end, blend,
                                 signal=signal, weight=weight, sma_w=sma_w)["nav"]
    ret = nav0.pct_change().dropna().values
    rng = np.random.default_rng(seed)
    nb = max(1, len(ret) // block)
    ann_l, sh_l, mdd_l = [], [], []
    for _ in range(n):
        idx = rng.integers(0, nb, size=nb)
        sample = np.concatenate([ret[i * block:(i + 1) * block] for i in idx])[:len(ret)]
        m = metrics(sample)
        ann_l.append(m["ann"]); sh_l.append(m["sharpe"]); mdd_l.append(m["mdd"])
    ann_l = np.array(ann_l); sh_l = np.array(sh_l); mdd_l = np.array(mdd_l)
    return dict(n=n, block=block,
                ann_ci=(float(np.percentile(ann_l, 5)), float(np.percentile(ann_l, 95))),
                sharpe_ci=(float(np.percentile(sh_l, 5)), float(np.percentile(sh_l, 95))),
                mdd_ci=(float(np.percentile(mdd_l, 5)), float(np.percentile(mdd_l, 95))),
                ann_mean=float(ann_l.mean()), sharpe_mean=float(sh_l.mean()))
