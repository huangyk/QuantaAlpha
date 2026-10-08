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
    args = ap.parse_args()
    global COMM_RATE
    COMM_RATE = args.commission

    c = conn(); cur = c.cursor()
    if args.cross:               # 跨资产池: 库内全部 ETF, 可选过滤
        if args.clean_only:
            # 旧 'qmt' 数据存在整段缺失(2025)与未修拆份, 只取单一干净来源的标的
            cur.execute("""select asset_code from bars_1d where asset_type='etf'
                           group by asset_code having count(distinct source)=1
                             and max(source)=%s order by 1""", (CLEAN_SOURCE,))
        else:
            cur.execute("""select distinct asset_code from bars_1d
                           where asset_type='etf' order by 1""")
        args.pool = [r[0] for r in cur.fetchall()]
        print(f"跨资产池: {len(args.pool)} 只 ETF (clean_only={args.clean_only})")
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
