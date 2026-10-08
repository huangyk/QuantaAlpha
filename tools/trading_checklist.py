"""5 步交易决策清单（CLI，可跑、可核对）

把用户的手动玩法固化为一条「有状态、带验证」的数据链路：
    1 看大盘  → 指数环境(趋势/波动率) + 全市宽度
    2 选个股  → 流动性/次新/涨停过滤 + 动量排序的候选池
    3 看位置  → 价 vs MA20/MA60、距 20 日高/52 周高、支撑位
    4 分时痕迹 + 筹码分布 → bars_5m 量能剖面 + bars_1d 重建 CYQ 成本分布
    5 找止盈  → 止损/目标/R:R

数据源：TimescaleDB(quant_data) bars_1d(全量 2011~2026-09-30) / bars_5m(仅 2026-01-05 起)
口径说明（务必看清）：
    - 创业板指(399006)在 bars_1d 中为空，大盘环境只用 沪深300/中证500/中证1000/上证。
    - 筹码分布(CYQ)用 每日成交量×100(股) 在收盘价上累加 + 指数衰减重建，
      不依赖 float_shares（实测多数票该字段为 null）；属近似，非券商真实持仓成本。
    - 分时痕迹仅能看 as_of 当日(且 bars_5m 仅 2026 年有数据)。
    - 本清单是「决策辅助」，不是「买卖指令」：第 2/3/4 步给出候选与上下文，最终由人拍板。

用法：
    python tools/trading_checklist.py                 # 默认 as-of=库最新日, 跑全市场候选
    python tools/trading_checklist.py --top 12        # 候选数
    python tools/trading_checklist.py --watch 600000.SH 000905.SH   # 只看指定票(跳过第2步自动选)
    python tools/trading_checklist.py --min-amount 3e7 --no-5m      # 调流动性门槛/跳过分时
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

DB = dict(host="127.0.0.1", port=5432, dbname="quant_data", user="quant", password="quant123456")

# 大盘环境用的宽基指数（bars_1d 中实际存在的）
INDICES = {
    "000300.SH": "沪深300",
    "000905.SH": "中证500",
    "000852.SH": "中证1000",
    "000001.SH": "上证指数",
}
CYQ_WINDOW = 120          # 筹码分布回看交易日
CYQ_HALFLIFE = 20         # 筹码衰减半衰期(交易日)
CYQ_BINS = 40             # 成本分布分箱数
MIN_LIST_DAYS = 250        # 剔除次新股/数据不足
STOP_PCT = 0.08            #  fallback 止损幅度


def conn():
    return psycopg2.connect(**DB)


def sql(cur, q, params=()):
    cur.execute(q, params)
    return cur.fetchall()


def get_latest_date(cur):
    return sql(cur, "select max(trade_date) from bars_1d")[0][0]


# ─────────────────────────── Step 1 看大盘 ───────────────────────────
def step1_market(cur, as_of):
    as_of_d = as_of.date() if hasattr(as_of, "date") else as_of
    # 指数序列（近 130 日）
    idx_rows = sql(cur,
        """select asset_code, trade_date, close from bars_1d
           where asset_code = any(%s) and trade_date >= %s
           order by asset_code, trade_date""",
        (list(INDICES), as_of - pd.Timedelta(days=130)))
    idx = pd.DataFrame(idx_rows, columns=["code", "date", "close"])
    env = {}
    for code, name in INDICES.items():
        s = idx[idx.code == code].sort_values("date").set_index("date")["close"].sort_index()
        if len(s) < 60:
            continue
        ma20, ma60 = s.rolling(20).mean().iloc[-1], s.rolling(60).mean().iloc[-1]
        ret20 = s.iloc[-1] / s.iloc[-20] - 1
        vol = s.pct_change().rolling(20).std().iloc[-1] * np.sqrt(242)
        above_ma20 = s.iloc[-1] > ma20
        label = "强" if (above_ma20 and ret20 > 0) else ("弱" if (not above_ma20 and ret20 < 0) else "震荡")
        env[name] = dict(close=s.iloc[-1], ma20=ma20, ma60=ma60,
                         ret20=ret20, vol=vol, label=label)
    # 全市宽度：% 个股站上 MA20（用近 65 日数据估算）
    stock_close = sql(cur,
        """select asset_code, trade_date, close from bars_1d
           where asset_type='stock' and trade_date >= %s order by asset_code, trade_date""",
        (as_of - pd.Timedelta(days=65),))
    sc = pd.DataFrame(stock_close, columns=["code", "date", "close"])
    sc["date"] = pd.to_datetime(sc["date"])
    breadth = np.nan
    if len(sc):
        g = sc.sort_values(["code", "date"])
        last_close = g.groupby("code")["close"].last()
        ma20 = g.groupby("code")["close"].apply(lambda x: x.tail(20).mean() if len(x) >= 20 else np.nan)
        both = ma20.dropna().index.intersection(last_close.index)
        if len(both):
            breadth = float((last_close[both] > ma20[both]).mean() * 100)
    return env, breadth


# ─────────────────────────── Step 2 选个股 ───────────────────────────
def step2_universe(cur, as_of, min_amount, topn):
    as_of_d = as_of.date() if hasattr(as_of, "date") else as_of
    # 近 25 日 OHLC+amount（算流动性 / 涨停过滤）
    base = sql(cur,
        """select asset_code, trade_date, open, close, amount from bars_1d
           where asset_type='stock' and trade_date >= %s
           order by asset_code, trade_date""",
        (as_of - pd.Timedelta(days=35),))
    b = pd.DataFrame(base, columns=["code", "date", "open", "close", "amount"])
    # 上市天数（数据长度）过滤
    cnt = sql(cur,
        """select asset_code, count(*) from bars_1d where asset_type='stock'
           group by asset_code having count(*) >= %s""", (MIN_LIST_DAYS,))
    mature = {r[0] for r in cnt}
    cand = {}
    for code, g in b.groupby("code"):
        if code not in mature:
            continue
        g = g.sort_values("date")
        if g.date.iloc[-1] != as_of_d:
            continue  # as_of 无数据（停牌/新）
        amt_med = g["amount"].tail(20).median()
        if not (amt_med >= min_amount):
            continue
        prev_close = g["close"].iloc[-2] if len(g) >= 2 else np.nan
        if pd.notna(prev_close) and prev_close > 0:
            limit_up = (g["open"].iloc[-1] / prev_close - 1) >= 0.098
        else:
            limit_up = False
        if limit_up:
            continue  # 涨停开盘买不进
        ret20 = g["close"].iloc[-1] / g["close"].iloc[-20] - 1 if len(g) >= 20 else np.nan
        cand[code] = dict(amount_med=amt_med, close=g["close"].iloc[-1],
                           ret20=ret20, prev_close=prev_close)
    df = pd.DataFrame.from_dict(cand, orient="index")
    if df.empty:
        return df
    df = df.sort_values("ret20", ascending=False)
    return df.head(topn)


# ─────────────────────── Step 3/4 历史(位置+筹码) ───────────────────────
def fetch_history(cur, codes, as_of, days):
    rows = sql(cur,
        """select asset_code, trade_date, open, high, low, close, volume
           from bars_1d where asset_code = any(%s) and trade_date >= %s
           order by asset_code, trade_date""",
        (list(codes), as_of - pd.Timedelta(days=days)))
    return pd.DataFrame(rows, columns=["code", "date", "open", "high", "low", "close", "volume"])


def step3_position(hist, as_of):
    as_of_d = as_of.date() if hasattr(as_of, "date") else as_of
    out = {}
    for code, g in hist.groupby("code"):
        g = g.sort_values("date")
        if g.date.iloc[-1] != as_of_d or len(g) < 60:
            continue
        c = g["close"]
        ma20 = c.rolling(20).mean().iloc[-1]
        ma60 = c.rolling(60).mean().iloc[-1]
        hi20 = c.tail(20).max()
        hi252 = c.tail(252).max() if len(g) >= 252 else c.max()
        lo20 = c.tail(20).min()
        lo10 = c.tail(10).min()
        lo5 = c.tail(5).min()
        cur = c.iloc[-1]
        out[code] = dict(
            close=cur, ma20=ma20, ma60=ma60,
            pct_ma20=(cur / ma20 - 1) * 100,
            pct_ma60=(cur / ma60 - 1) * 100,
            pct_from_hi20=(cur / hi20 - 1) * 100,
            pct_from_hi252=(cur / hi252 - 1) * 100,
            support_ma20=ma20, resist_hi20=hi20,
            low20=lo20, low10=lo10, low5=lo5,
        )
    return out


def step4_chips(hist, as_of):
    """CYQ 成本分布：每日 volume×100(股) 置于收盘价，指数衰减累加。"""
    out = {}
    decay = 0.5 ** (1.0 / CYQ_HALFLIFE)
    for code, g in hist.groupby("code"):
        g = g.sort_values("date").tail(CYQ_WINDOW)
        if len(g) < 30:
            continue
        cur = g["close"].iloc[-1]
        lo, hi = g["low"].min(), g["high"].max()
        if hi <= lo:
            continue
        edges = np.linspace(lo, hi, CYQ_BINS + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        mass = np.zeros(CYQ_BINS)
        n = len(g)
        for i, (_, row) in enumerate(g.iterrows()):
            w = (decay ** (n - 1 - i)) * (row["volume"] * 100.0)
            if w <= 0 or pd.isna(row["close"]):
                continue
            b = int(np.searchsorted(edges, row["close"]) - 1)
            b = min(max(b, 0), CYQ_BINS - 1)
            mass[b] += w
        tot = mass.sum()
        if tot <= 0:
            continue
        profit = mass[centers < cur].sum() / tot * 100
        avg_cost = float((centers * mass).sum() / tot)
        # 90% 成本集中度：从 avg_cost 向两侧扩到累计 90%
        order = np.argsort(np.abs(centers - avg_cost))
        cum = 0.0
        lo_p = hi_p = avg_cost
        for b in order:
            cum += mass[b]
            lo_p = min(lo_p, centers[b])
            hi_p = max(hi_p, centers[b])
            if cum / tot >= 0.90:
                break
        peak = centers[np.argmax(mass)]
        out[code] = dict(profit_ratio=profit, avg_cost=avg_cost,
                         cost90_low=lo_p, cost90_high=hi_p,
                         peak=peak, conc90=(hi_p - lo_p) / avg_cost * 100 if avg_cost else np.nan)
    return out


def step4_intraday(cur, codes, as_of):
    rows = sql(cur,
        """select asset_code, ts, open, high, low, close, volume, amount
           from bars_5m where asset_code = any(%s) and trade_date = %s
           order by asset_code, ts""",
        (list(codes), as_of))
    df = pd.DataFrame(rows, columns=["code", "ts", "open", "high", "low", "close", "volume", "amount"])
    out = {}
    if df.empty:
        return out
    for code, g in df.groupby("code"):
        g = g.sort_values("ts")
        if g.empty:
            continue
        g["hour"] = pd.to_datetime(g["ts"]).dt.strftime("%H:%M")
        prof = g.groupby("hour")["volume"].sum()
        heaviest = prof.idxmax()
        # 5m volume 单位是"手"(×100=股)，VWAP = 金额/股数
        vwap = g["amount"].sum() / (g["volume"].sum() * 100.0) if g["volume"].sum() > 0 else np.nan
        close = g["close"].iloc[-1]
        vs_vwap = (close / vwap - 1) * 1e4 if vwap else np.nan
        last6 = g.tail(6)
        eod_ret = (last6["close"].iloc[-1] / last6["close"].iloc[0] - 1) * 100 if len(last6) > 1 else np.nan
        out[code] = dict(heaviest_bucket=heaviest, vs_vwap=vs_vwap, eod_ret=eod_ret,
                         n_bars=len(g))
    return out


# ─────────────────────────── Step 5 找止盈 ───────────────────────────
def step5_exit(position, chips):
    out = {}
    for code, p in position.items():
        entry = p["close"]
        # 尾随止损：取最近支撑(MA20下1% / 5日低 / 最大15%亏损)中最紧者
        floor = entry * (1 - 0.15)
        stop = min(p["support_ma20"] * 0.99, p["low5"] * 0.999, floor)
        stop = min(stop, entry * 0.999)  # 必须在 entry 下方
        # 目标：20日高；若已在高位则至少 +10%
        target = max(p["resist_hi20"], entry * 1.10)
        risk = entry - stop
        reward = target - entry
        rr = reward / risk if risk > 0 else np.nan
        out[code] = dict(entry=entry, stop=stop, target=target,
                         risk_pct=risk / entry * 100, reward_pct=reward / entry * 100,
                         rr=rr)
    return out


# ─────────────────────────── 报告 ───────────────────────────
def fmt_pct(x):
    return f"{x*100:+.1f}%" if isinstance(x, (int, float, np.floating)) and pd.notna(x) else "  n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default=None, help="截止日 YYYY-MM-DD，默认库最新日")
    ap.add_argument("--top", type=int, default=12, help="第2步候选数")
    ap.add_argument("--min-amount", type=float, default=3e7, help="日均成交额下限(元)，默认3000万")
    ap.add_argument("--watch", nargs="*", default=None, help="指定票(跳过自动选股)，如 600000.SH")
    ap.add_argument("--no-5m", action="store_true", help="跳过分时痕迹")
    args = ap.parse_args()

    # 同时写盘，便于核对留痕
    out_path = None
    tee = None
    try:
        res_dir = Path("data/results")
        res_dir.mkdir(parents=True, exist_ok=True)
        out_path = res_dir / f"trading_checklist_{args.as_of or 'latest'}.txt"

        class Tee:
            def __init__(self, f):
                self.f = f
                self._o = sys.stdout

            def write(self, t):
                self._o.write(t)
                self.f.write(t)

            def flush(self):
                self._o.flush()
                self.f.flush()

        tee = Tee(open(out_path, "w", encoding="utf-8"))
        sys.stdout = tee
    except Exception:
        pass

    c = conn()
    cur = c.cursor()
    as_of = (pd.Timestamp(args.as_of) if args.as_of
             else pd.Timestamp(get_latest_date(cur)))
    print(f"═══ 5步交易决策清单 ═══  as-of: {as_of.date()}  ═══\n")

    # Step 1
    env, breadth = step1_market(cur, as_of)
    print("【1 看大盘】")
    for name, e in env.items():
        print(f"  {name:6s} {e['label']}  收{e['close']:.2f}  "
              f"vsMA20{(e['close']/e['ma20']-1)*100:+.1f}%  vsMA60{(e['close']/e['ma60']-1)*100:+.1f}%  "
              f"20日{e['ret20']*100:+.1f}%  年化波动{e['vol']*100:.0f}%")
    print(f"  全市宽度(站上MA20): {breadth:.1f}%" if pd.notna(breadth) else "  全市宽度: n/a")
    print()

    # Step 2
    if args.watch:
        codes = [x.upper() for x in args.watch]
        print(f"【2 选个股】--watch 模式，指定 {len(codes)} 只：{', '.join(codes)}\n")
    else:
        uni = step2_universe(cur, as_of, args.min_amount, args.top)
        if uni.empty:
            print("【2 选个股】无候选（流动性/次新/涨停过滤后为空）\n")
            cur.close(); c.close()
            print(f"\n报告已保存: {out_path}")
            sys.stdout = sys.__stdout__
            return
        codes = list(uni.index)
        print(f"【2 选个股】候选 {len(codes)} 只（流动性≥{args.min_amount/1e4:.0f}万, 剔除次新/涨停, 按20日动量排序）:")
        for code, r in uni.iterrows():
            print(f"  {code}  收{r['close']:.2f}  20日动量{fmt_pct(r['ret20'])}  日均额{r['amount_med']/1e4:.0f}万")
        print()

    # Step 3/4 历史
    hist = fetch_history(cur, codes, as_of, max(255, CYQ_WINDOW + 5))
    position = step3_position(hist, as_of)
    chips = step4_chips(hist, as_of)
    intraday = {} if args.no_5m else step4_intraday(cur, codes, as_of)
    exit_ = step5_exit(position, chips)

    print("【3 看位置 / 4 筹码+分时 / 5 止盈】\n")
    for code in codes:
        p = position.get(code)
        if not p:
            print(f"  {code}: 历史不足，跳过\n"); continue
        ch = chips.get(code, {})
        idy = intraday.get(code, {})
        ex = exit_.get(code, {})
        print(f"  ▶ {code}  收{p['close']:.2f}")
        print(f"    位置: MA20{p['pct_ma20']:+.1f}%  MA60{p['pct_ma60']:+.1f}%  "
              f"距20日高{p['pct_from_hi20']:+.1f}%  距52周高{p['pct_from_hi252']:+.1f}%")
        if ch:
            print(f"    筹码: 获利盘{ch['profit_ratio']:.0f}%  平均成本{ch['avg_cost']:.2f}  "
                  f"90%成本[{ch['cost90_low']:.2f},{ch['cost90_high']:.2f}] 峰{ch['peak']:.2f} 集中度{ch['conc90']:.1f}%")
        else:
            print("    筹码: n/a(历史不足)")
        if idy:
            print(f"    分时: 最大量能{idy['heaviest_bucket']}  收-VWAP{idy['vs_vwap']:+.0f}bp  "
                  f"尾盘30分{idy['eod_ret']:+.2f}%  ({idy['n_bars']}根5m)")
        elif not args.no_5m:
            print("    分时: 当日5m无数据")
        if ex:
            rr_s = f"{ex['rr']:.2f}" if pd.notna(ex['rr']) else "n/a"
            print(f"    止盈: 止损{ex['stop']:.2f}(-{ex['risk_pct']:.1f}%)  目标{ex['target']:.2f}(+{ex['reward_pct']:.1f}%)  R:R={rr_s}")
        print()

    cur.close(); c.close()
    print(f"\n报告已保存: {out_path}")
    sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()
