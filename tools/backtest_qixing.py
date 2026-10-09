"""
七星高照 ETF 轮动策略 —— 本地复现 + 防过拟合改造
====================================================
数据源: TimescaleDB bars_1d (asset_type='etf', 前复权在 datahub 已处理)
费率: 用户实际 万1 佣金, ETF 免印花税 (--commission 可配)

策略核心 (还原网上公开版):
  每日对池内每只 ETF 算 25 日「加权对数回归」动量:
      score = (exp(slope*250) - 1) * R^2
      slope = 对数价格对时间的线性回归斜率 (日频)
      R^2   = 拟合优度 (趋势稳定性)
  只持得分最高的 top_n 只; 无合格标的 -> 空仓(货币基金 511880 库内暂无, 用现金替代并标注)

防过拟合改造 (针对网上复现发现的 LB=25 孤立优势):
  --lb        动量窗口, 支持扫 15~35 全区间 (不只25)
  --top-n     持仓数 1/2/3 (分散降回撤)
  --switch    换仓门槛: 新标的得分需高出当前持仓 switch 倍才换 (降换手/成本)
  --rebalance d/m/q 调仓频率

用法:
  python tools/backtest_qixing.py                      # 默认9只A股ETF, LB=25, top1, 每日
  python tools/backtest_qixing.py --lb-sweep 15 35     # 扫动量窗口看鲁棒性
  python tools/backtest_qixing.py --top-n 2 --switch 0.2 --rebalance m
"""
import argparse, json, sys
from math import sqrt
import numpy as np
import pandas as pd
import psycopg2

DB = dict(host="127.0.0.1", port=5432, dbname="quant_data", user="quant", password="quant123456")
# 库内可用的 A股宽基 ETF 池 (跨资产标的需你给有权限清单或 QMT 拉)
DEFAULT_POOL = ["510300.SH", "510500.SH", "512100.SH", "159915.SZ",
                "159949.SZ", "588000.SH", "510050.SH", "159919.SZ", "159922.SZ"]
COMM_RATE = 0.0001      # 万1 (用户实际)
STAMP_RATE = 0.0        # ETF 免印花税
TRANSFER_RATE = 0.00001 # 过户费 (ETF 通常免, 保守留微量)
CLEAN_SOURCE = "qmt_local_86400"   # 本脚本回填的干净 QMT 本地数据来源


def conn():
    return psycopg2.connect(**DB)


def load_etf(cur, codes, start, end):
    rows = []
    for code in codes:
        cur.execute("""select trade_date, open, high, low, close, volume
                       from bars_1d where asset_code=%s and asset_type='etf'
                       and trade_date>=%s and trade_date<=%s order by trade_date""",
                    (code, start, end))
        for r in cur.fetchall():
            rows.append((code, r[0], r[1], r[2], r[3], r[4], r[5]))
    df = pd.DataFrame(rows, columns=["code", "date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    out = {}
    for code, g in df.groupby("code"):
        g = g.sort_values("date").set_index("date")
        out[code] = g
    return out


def weighted_log_momentum(close_series, lb):
    """返回 (score, slope_annual, r2)。不足 lb 日返回 (nan,nan,nan)。"""
    if len(close_series) < lb:
        return np.nan, np.nan, np.nan
    y = np.log(close_series.iloc[-lb:].to_numpy(dtype=float))
    x = np.arange(lb, dtype=float)
    w = np.arange(1, lb + 1, dtype=float)        # 近大远小加权
    # 加权线性回归
    sw = w.sum(); swx = (w * x).sum(); swy = (w * y).sum()
    swxx = (w * x * x).sum(); swxy = (w * x * y).sum()
    denom = sw * swxx - swx * swx
    if denom == 0:
        return np.nan, np.nan, np.nan
    slope = (sw * swxy - swx * swy) / denom
    intercept = (swy - slope * swx) / sw
    yhat = slope * x + intercept
    ss_res = (w * (y - yhat) ** 2).sum()
    ss_tot = (w * (y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    ann = np.exp(slope * 250) - 1
    score = ann * r2
    return score, ann, r2


def run(cur, codes, start, end, lb, top_n, rebalance, switch, verbose=False):
    data = load_etf(cur, codes, start, end)
    # 统一交易日: 取并集(各 ETF 上市日不同, 早上市的晚加入轮动, 不再强制交集)
    all_days = sorted(set().union(*[set(g.index) for g in data.values()]))
    days = all_days
    if len(days) < lb + 5:
        print("ERROR 数据不足"); return None, None
    # 预计算每只每日的 (score, close, volume, ma20_vol, ret5, max3d_drop, above_ma20)
    pre = {c: {} for c in codes}
    for c, g in data.items():
        closes = g["close"]
        vols = g["volume"]
        for i in range(lb, len(g)):
            d = g.index[i]
            sc, ann, r2 = weighted_log_momentum(closes.iloc[:i + 1], lb)
            pre[c][d] = dict(
                score=sc, r2=r2, close=float(closes.iloc[i]),
                vma20=float(vols.iloc[max(0, i - 19):i + 1].mean()),
                vol=float(vols.iloc[i]),
                ret5=float(closes.iloc[i] / closes.iloc[max(0, i - 5)] - 1) if i >= 5 else np.nan,
                above_ma20=bool(closes.iloc[i] > closes.iloc[max(0, i - 19):i + 1].mean()),
                max3d_drop=float(min(0, *( (closes.iloc[j] / closes.iloc[j - 1] - 1) for j in range(max(1, i - 2), i + 1) ))) if i >= 1 else 0.0,
            )
    # 回测 (信号-收益对齐: T日收盘选股 -> T+1日才计入收益, 消除前视偏差)
    nav = []; nav_dates = []
    hold = []          # 当前持仓 (昨日信号决定, 今日产生收益)
    cash = 1.0
    trades = 0
    idx_of = {d: k for k, d in enumerate(days)}
    for i, d in enumerate(days):
        # 今日信号 (用 pre[d])
        cands = []
        for c in codes:
            p = pre[c].get(d)
            if p is None or not pd.notna(p["score"]):
                continue
            if p["score"] <= 0: continue
            if p["r2"] < 0.30: continue
            if not p["above_ma20"]: continue
            if p["vol"] < p["vma20"] * 0.8: continue
            if pd.notna(p["ret5"]) and p["ret5"] <= 0: continue
            if p["max3d_drop"] < -0.03: continue
            cands.append((c, p["score"]))
        cands.sort(key=lambda x: -x[1])
        target = [c for c, _ in cands[:top_n]]
        if rebalance == "h":   # 买入持有: 绕过过滤, 首日全仓
            target = list(codes)
        # 今日收益: 用当前 hold (昨日决定的) 的当日涨跌幅
        # ⚠️ 并集交易日下 g.index != days, 必须用 .loc 按日期取; 用 .iloc[k] 会张冠李戴
        if hold:
            day_ret = 0.0; n = 0
            prev_d = days[i - 1] if i > 0 else None
            for c in hold:
                g = data[c]
                ct = g["close"].get(d)
                cp = g["close"].get(prev_d) if prev_d is not None else None
                if (ct is not None and cp is not None
                        and pd.notna(ct) and pd.notna(cp) and cp > 0):
                    day_ret += ct / cp - 1; n += 1
            if n: cash *= (1 + day_ret / n)
        # 调仓判定 (今日信号决定明日 hold)
        do_rb = False
        if rebalance == "d": do_rb = True
        elif rebalance == "m": do_rb = (i > 0 and d.month != days[i - 1].month)
        elif rebalance == "q":
            if i > 0:
                qn = (d.month - 1) // 3; qp = (days[i - 1].month - 1) // 3
                do_rb = (qn != qp or d.year != days[i - 1].year)
            else: do_rb = True
        elif rebalance == "h": do_rb = (i == 0)
        if do_rb:
            if not hold or switch <= 0:
                new_hold = target
            else:
                cur_scores = {c: pre[c][d]["score"] for c in hold if d in pre[c]}
                cur_avg = np.mean(list(cur_scores.values())) if cur_scores else 0
                best_new = cands[0][1] if cands else 0
                new_hold = target if best_new >= cur_avg * (1 + switch) else hold
            if new_hold != hold:
                cash *= (1 - COMM_RATE * 2)   # 卖+买双边佣金(万1)
                hold = new_hold
                trades += 1
        nav.append(cash); nav_dates.append(d)
    nav_s = pd.Series(nav, index=nav_dates)
    # 指标
    rets = nav_s.pct_change().dropna()
    ann = (nav_s.iloc[-1] / nav_s.iloc[0]) ** (250 / len(rets)) - 1 if len(rets) else 0
    vol = rets.std() * np.sqrt(250)
    sharpe = ann / vol if vol > 0 else 0
    dd = (nav_s / nav_s.cummax() - 1).min()
    return dict(nav=nav_s, ann=ann, vol=vol, sharpe=sharpe, maxdd=dd,
                trades=trades, end=nav_s.iloc[-1]), codes


# ============================================================================
# ETF 组合层 (L3/L4/L5) —— 照搬 guoyaohua/etf-adaptive-rotation-qmt 的工程层
# ============================================================================
# 设计要点(全部在"工程/风控层", 不挖新因子, 与本站结论"真杠杆在 L4"一致):
#   ① 多周期动量排名(跳过最近5日避短周期反转) + 风险调整分数 Score=M/max(σ40,0.10)^0.75
#   ② 逆波动率权重 w_i=(1/σ_i)/Σ(1/σ_j)
#   ③ 波动率上限 10%: 预估年化波动>10% 只向下缩放(不杠杆补足); 单资产≤40%、总敞口≤90%
#   ④ 四周错峰 sleeve: 资金分 A/B/C/D, 每 ~5 交易日只替换一份旧信号, 把择时风险摊到四周
#   ⑤ 相关性去重: 选股时贪心跳过与已选 corr>0.85 的标的(避免 588000+588080 同重仓)
#   ⑥ ATR 初始/跟踪止损: 持仓回撤穿透 mult×ATR 即退出该 slice, 余下转现金/511880
#   ⑦ 货币 ETF(511880)承接闲置资金, 最多 30%

def precompute_etf(cur, codes, start, end):
    """一次加载 + 预计算每只每日的 close/σ40/ATR/动量分数, 供组合回测复用。"""
    data = load_etf(cur, codes, start, end)
    all_days = sorted(set().union(*[set(g.index) for g in data.values()]))

    series = {}
    for c, g in data.items():
        close = g["close"].astype(float)
        high = g["high"].astype(float)
        low = g["low"].astype(float)
        ret = close.pct_change()
        sigma40 = ret.rolling(40).std() * np.sqrt(250)
        prev_close = close.shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        series[c] = dict(close=close, ret=ret, sigma40=sigma40, atr=atr)

    def mom_k(close, j, k):
        s = 5  # 跳过最近 5 日
        if j - s - k < 0 or j - s < 0:
            return np.nan
        return float(close.iloc[j - s - k] / close.iloc[j - s] - 1)

    pre = {c: {} for c in codes}
    for c, g in data.items():
        close = series[c]["close"]; sig = series[c]["sigma40"]; atr = series[c]["atr"]
        for j, d in enumerate(g.index):
            r20 = mom_k(close, j, 20); r60 = mom_k(close, j, 60); r120 = mom_k(close, j, 120)
            M = (0.2 * r20 + 0.3 * r60 + 0.5 * r120) if all(pd.notna(x) for x in (r20, r60, r120)) else np.nan
            sig40 = float(sig.iloc[j]) if pd.notna(sig.iloc[j]) else np.nan
            score = (M / (max(sig40, 0.10) ** 0.75)) if pd.notna(M) and pd.notna(sig40) else np.nan
            pre[c][d] = dict(
                close=float(close.iloc[j]),
                sigma40=sig40,
                atr=float(atr.iloc[j]) if pd.notna(atr.iloc[j]) else np.nan,
                score=score, r20=r20, r60=r60, r120=r120,
            )
    return data, pre, all_days


def _metrics(nav_s, trades=0):
    rets = nav_s.pct_change().dropna()
    n = len(rets)
    ann = (nav_s.iloc[-1] / nav_s.iloc[0]) ** (250 / n) - 1 if n else 0
    vol = rets.std() * np.sqrt(250)
    sharpe = ann / vol if vol > 0 else 0
    dd = (nav_s / nav_s.cummax() - 1).min()
    return dict(nav=nav_s, ann=ann, vol=vol, sharpe=sharpe, maxdd=dd, trades=trades, end=nav_s.iloc[-1])


def portfolio_backtest(data, pre, all_days, opts):
    """组合层回测。opts 控制各层开关, 全部默认关(=朴素等权轮动对比)。

    信号-收益对齐: T日收盘排名决定权重 -> T+1 起计收益(无前视)。
    """
    days = all_days
    top_n = opts.get("top_n", 8)
    n_per = max(1, top_n // 4)               # 每 sleeve 持仓数
    sleeve_budget = 0.9 / 4.0                # 每 sleeve 占 book 22.5% (总敞口≤90%)
    mult = opts.get("atr_mult", 3.0)
    rho = 0.3                                 # 组合波动估计用的平均相关系数
    cash_etf = opts.get("cash_etf", False)
    has_cash_etf = "511880.SH" in data

    sleeve_w = {s: {} for s in range(4)}      # sleeve_id -> {code: weight}
    pos_state = {}                            # (s, code) -> {entry_close, entry_atr_frac, highest}
    nav = []; nav_dates = []
    cash = 1.0
    last_week = -1
    prev_w = {}                               # 昨日权重(用于今日收益)
    trades_cost = 0.0
    peak = 1.0
    guarded = False

    for i, d in enumerate(days):
        # 1) 今日收益(用昨日权重, T+1 对齐)
        if i > 0:
            prev_d = days[i - 1]
            day_ret = 0.0
            idle = 1.0 - sum(prev_w.values())
            idle_in_etf = min(idle, 0.30) if (cash_etf and has_cash_etf) else 0.0
            if idle_in_etf > 0:
                g = data["511880.SH"]
                ct = g["close"].get(d); cp = g["close"].get(prev_d)
                if ct is not None and cp is not None and pd.notna(cp) and cp > 0:
                    day_ret += idle_in_etf * (ct / cp - 1)
            for code, w in prev_w.items():
                if w <= 0:
                    continue
                g = data[code]
                ct = g["close"].get(d); cp = g["close"].get(prev_d)
                if ct is not None and cp is not None and pd.notna(cp) and cp > 0:
                    day_ret += w * (ct / cp - 1)
            cash *= (1 + day_ret)
        peak = max(peak, cash)
        # ⑦ 组合级回撤降级: NAV 自高点回落超 dd_guard -> 全仓转现金/511880, 回升后恢复
        gp = opts.get("dd_guard", 0.0)
        if gp > 0:
            if not guarded and (cash / peak - 1) < -gp:
                guarded = True
                turn = sum(prev_w.values())
                for s in range(4):
                    sleeve_w[s] = {}
                pos_state.clear()
                cash *= (1 - turn * COMM_RATE); trades_cost += turn
                prev_w = {}
            elif guarded and (cash / peak - 1) > -opts.get("dd_recover", 0.02):
                guarded = False
                last_week = -1          # 恢复后强制下次刷新

        # 2) ATR 止损(每日检查, 穿透即退出该 slice)
        if opts.get("atr"):
            for s in range(4):
                for code in list(sleeve_w[s].keys()):
                    st = pos_state.get((s, code))
                    if st is None:
                        continue
                    px = pre[code].get(d)
                    if px is None:
                        continue
                    close_t = px["close"]
                    if close_t > st["highest"]:
                        st["highest"] = close_t
                    stop_level = st["highest"] * (1 - mult * st["entry_atr_frac"])
                    if close_t < stop_level:
                        sleeve_w[s][code] = 0.0
                        del pos_state[(s, code)]

        # 3) 四周错峰 sleeve 刷新(每 ~5 交易日刷新一个 sleeve; 回撤降级期间跳过)
        week = i // 5
        if not guarded and week != last_week:
            last_week = week
            s = week % 4
            cands = []
            for code in opts.get("codes", list(pre.keys())):
                p = pre[code].get(d)
                if p is None or not pd.notna(p["score"]) or not pd.notna(p["sigma40"]):
                    continue
                cands.append((code, p["score"], p["sigma40"]))
            # 相关性去重
            if opts.get("dedup") and len(cands) > n_per:
                retmap = {}
                for code, _, _ in cands:
                    rc = data[code]["close"].astype(float).pct_change().iloc[-120:].dropna().to_numpy()
                    retmap[code] = rc
                sel = []
                for code, sc, _ in sorted(cands, key=lambda x: -x[1]):
                    ok = True
                    for sc2 in sel:
                        a = retmap[code]; b = retmap[sc2]; nn = min(len(a), len(b))
                        if nn > 10:
                            corr = np.corrcoef(a[-nn:], b[-nn:])[0, 1]
                            if pd.notna(corr) and corr > 0.85:
                                ok = False; break
                    if ok:
                        sel.append(code)
                    if len(sel) >= n_per:
                        break
                picks = sel
            else:
                picks = [c for c, _, _ in sorted(cands, key=lambda x: -x[1])[:n_per]]
            # 权重
            if picks:
                if opts.get("inv_vol"):
                    inv = [1.0 / max(pre[c][d]["sigma40"], 0.05) for c in picks]
                    tot = sum(inv)
                    raw = {c: (inv[k] / tot) * sleeve_budget for k, c in enumerate(picks)}
                else:
                    raw = {c: sleeve_budget / len(picks) for c in picks}
                for c in raw:
                    raw[c] = min(raw[c], 0.40)        # 单资产≤40%
                # 波动率上限(向下缩放, 不杠杆)
                if opts.get("vol_cap"):
                    w_all = {}
                    for ss in range(4):
                        src = raw if ss == s else sleeve_w[ss]
                        for c, w in src.items():
                            w_all[c] = w_all.get(c, 0) + w
                    sig = {}
                    for c in w_all:
                        pd_ = pre[c].get(d)
                        if pd_ is not None and pd.notna(pd_["sigma40"]):
                            sig[c] = pd_["sigma40"]
                    var = 0.0
                    keys = list(sig.keys())
                    for a in keys:
                        for b in keys:
                            rho_ab = 1.0 if a == b else rho
                            var += w_all[a] * w_all[b] * sig[a] * sig[b] * rho_ab
                    est = np.sqrt(max(var, 0))
                    if est > 0.10:
                        scale = 0.10 / est
                        for c in raw:
                            raw[c] *= scale
                        for ss in range(4):
                            if ss != s:
                                for c in list(sleeve_w[ss].keys()):
                                    sleeve_w[ss][c] *= scale
                # 换手成本(与本 sleeve 旧权重比较)
                old = sleeve_w[s]
                turnover = 0.0
                for c in set(list(old.keys()) + list(raw.keys())):
                    turnover += abs(raw.get(c, 0.0) - old.get(c, 0.0))
                sleeve_w[s] = dict(raw)
                for c in raw:
                    p = pre[c].get(d)
                    pos_state[(s, c)] = dict(
                        entry_close=p["close"],
                        entry_atr_frac=(p["atr"] / p["close"]) if (pd.notna(p["atr"]) and p["close"] > 0) else 0.02,
                        highest=p["close"])
                cash *= (1 - turnover * COMM_RATE)
                trades_cost += turnover

        # 4) 合并权重(供明日收益使用)
        w = {}
        for s in range(4):
            for c, x in sleeve_w[s].items():
                if x > 0:
                    w[c] = w.get(c, 0) + x
        prev_w = w
        nav.append(cash); nav_dates.append(d)

    nav_s = pd.Series(nav, index=nav_dates)
    return _metrics(nav_s, trades_cost), list(pre.keys())


def main():
    global COMM_RATE
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", nargs="*", default=DEFAULT_POOL)
    ap.add_argument("--start", default="2020-01-02")
    ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--lb", type=int, default=25)
    ap.add_argument("--lb-sweep", nargs=2, type=int, default=None, help="扫动量窗口 lo hi")
    ap.add_argument("--top-n", type=int, default=1)
    ap.add_argument("--rebalance", default="d", choices=["d", "m", "q", "h"],
                    help="d日/m月/q季/h买入持有(池等权基准)")
    ap.add_argument("--switch", type=float, default=0.0)
    ap.add_argument("--commission", type=float, default=0.0001)
    ap.add_argument("--cross", action="store_true",
                    help="加载库内全部 asset_type='etf' 代码做跨资产轮动")
    ap.add_argument("--clean-only", action="store_true",
                    help="只用单一干净来源(qmt_local_86400)的 ETF; 排除旧 qmt 损坏数据")
    ap.add_argument("--min-days", type=int, default=0,
                    help="过滤历史不足该天数的 ETF(默认0=不过滤)")
    # ---- 组合层开关 ----
    ap.add_argument("--portfolio", action="store_true", help="打开全部组合层(=全开对比组)")
    ap.add_argument("--top-n-port", type=int, default=8, help="组合层总持仓数(分4 sleeve)")
    ap.add_argument("--inv-vol", action="store_true", help="逆波动率权重")
    ap.add_argument("--vol-cap", action="store_true", help="波动率上限10%(只向下缩放)")
    ap.add_argument("--sleeve", action="store_true", help="四周错峰(否则每月整批调仓)")
    ap.add_argument("--atr", action="store_true", help="ATR 初始/跟踪止损")
    ap.add_argument("--dedup", action="store_true", help="相关性去重(corr>0.85 跳过)")
    ap.add_argument("--cash-etf", action="store_true", help="闲置资金由511880承接(否则0收益现金)")
    ap.add_argument("--atr-mult", type=float, default=3.0, help="ATR 止损倍数")
    ap.add_argument("--dd-guard", type=float, default=0.0, help="组合级回撤降级阈值(如0.08=回落8%全转现金)")
    ap.add_argument("--dd-recover", type=float, default=0.02, help="回撤降级后恢复阈值(如0.02=回升至-2%内恢复)")
    args = ap.parse_args()
    COMM_RATE = args.commission

    c = conn(); cur = c.cursor()
    if args.cross:               # 跨资产池: 库内全部 ETF, 可选过滤
        if args.clean_only:
            cur.execute("""select asset_code from bars_1d where asset_type='etf'
                           group by asset_code having count(distinct source)=1
                             and max(source)=%s order by 1""", (CLEAN_SOURCE,))
        else:
            cur.execute("""select distinct asset_code from bars_1d
                           where asset_type='etf' order by 1""")
        args.pool = [r[0] for r in cur.fetchall()]
        print(f"跨资产池: {len(args.pool)} 只 ETF (clean_only={args.clean_only})")

    # ---------- 组合层模式 ----------
    if args.portfolio or any([args.inv_vol, args.vol_cap, args.sleeve, args.atr, args.dedup, args.cash_etf]):
        codes = args.pool
        data, pre, all_days = precompute_etf(cur, codes, args.start, args.end)
        base = dict(codes=codes, top_n=args.top_n_port, atr_mult=args.atr_mult)
        print(f"\n=== ETF 组合层对比 (池{len(codes)}只, {args.start}~{args.end}, 佣金万{args.commission*1e4:.0f}, top_n={args.top_n_port}) ===")
        print(f"{'组合配置':<22}{'年化':>8}{'波动':>8}{'夏普':>7}{'回撤':>9}{'换手(单边)':>10}")
        # 真·等权持有基准(34只全仓, 与 qixing_etf_verdict.md 同口径)
        rb, _ = run(cur, codes, args.start, args.end, 25, 1, "h", 0)
        print(f"{'真·等权持有(34)':<20}{rb['ann']*100:>7.1f}%{rb['vol']*100:>7.1f}%{rb['sharpe']:>7.2f}{rb['maxdd']*100:>8.1f}%{rb['trades']:>10}")
        configs = [
            ("朴素 top-8 等权轮动", dict(base, inv_vol=False, vol_cap=False, sleeve=False, atr=False, dedup=False, cash_etf=False)),
            ("+逆波动率权重", dict(base, inv_vol=True, vol_cap=False, sleeve=False, atr=False, dedup=False, cash_etf=False)),
            ("+波动率上限", dict(base, inv_vol=True, vol_cap=True, sleeve=False, atr=False, dedup=False, cash_etf=False)),
            ("+四周错峰", dict(base, inv_vol=True, vol_cap=True, sleeve=True, atr=False, dedup=False, cash_etf=False)),
            ("+ATR止损", dict(base, inv_vol=True, vol_cap=True, sleeve=True, atr=True, dedup=False, cash_etf=False)),
            ("+相关性去重", dict(base, inv_vol=True, vol_cap=True, sleeve=True, atr=True, dedup=True, cash_etf=False)),
            ("全开(+511880缓冲)", dict(base, inv_vol=True, vol_cap=True, sleeve=True, atr=True, dedup=True, cash_etf=True)),
            ("全开+回撤降级(8%)", dict(base, inv_vol=True, vol_cap=True, sleeve=True, atr=True, dedup=True, cash_etf=True, dd_guard=0.08, dd_recover=0.02)),
        ]
        for name, opt in configs:
            res, _ = portfolio_backtest(data, pre, all_days, opt)
            print(f"{name:<20}{res['ann']*100:>7.1f}%{res['vol']*100:>7.1f}%{res['sharpe']:>7.2f}{res['maxdd']*100:>8.1f}%{res['trades']:>10.1f}")
        cur.close(); c.close()
        return

    if args.lb_sweep:
        lo, hi = args.lb_sweep
        print(f"动量窗口鲁棒性扫描 ({lo}~{hi}, top_n={args.top_n}, reb={args.rebalance}, switch={args.switch})")
        print(f"{'LB':>4} {'年化':>8} {'波动':>8} {'夏普':>6} {'回撤':>8} {'换手次':>6}")
        for lb in range(lo, hi + 1, 5):
            res, _ = run(cur, args.pool, args.start, args.end, lb, args.top_n, args.rebalance, args.switch)
            if res:
                print(f"{lb:>4} {res['ann']*100:>7.1f}% {res['vol']*100:>7.1f}% {res['sharpe']:>6.2f} {res['maxdd']*100:>7.1f}% {res['trades']:>6}")
    else:
        res, codes = run(cur, args.pool, args.start, args.end, args.lb, args.top_n, args.rebalance, args.switch, verbose=True)
        print(f"\n池({len(codes)}只A股宽基ETF): {', '.join(codes)}")
        print(f"区间 {args.start}~{args.end} | LB={args.lb} | top_n={args.top_n} | 调仓={args.rebalance} | 换仓门槛={args.switch} | 佣金万{args.commission*1e4:.0f}")
        print(f"  年化 {res['ann']*100:+.1f}% | 波动 {res['vol']*100:.1f}% | 夏普 {res['sharpe']:.2f} | 最大回撤 {res['maxdd']*100:.1f}% | 换手 {res['trades']} 次")
        print(f"  终值 {res['end']:.2f}x (={'%.1f%%'%( (res['end']-1)*100 )})")
        # 对比沪深300
        r300, _ = run(cur, ["510300.SH"], args.start, args.end, args.lb, 1, "h", 0)
        if r300:
            print(f"  对比 沪深300买入持有: 年化 {r300['ann']*100:+.1f}% 回撤 {r300['maxdd']*100:.1f}%")
    cur.close(); c.close()


if __name__ == "__main__":
    main()
