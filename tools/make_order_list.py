"""生成下个月度调仓的「下单清单」——把研究结论直接变成可在 QMT/Ptrade 抄的单子。

逻辑与 quantaalpha/strategy/pool_hold.py 一致（2026-10-08 修正后口径）：
  1. 按 60 日成交额中位数给全 A 排序，取【中后段流动性】1001~3000 名
     （修正池构建 bug 后：低流动性长期跑赢，中后段 +2.89%/年 最优；Top500 仅 +2.02%）
  2. 剔除：数据历史<500日(次新近似) / 涨停开盘(买不进) / 停牌无成交 / 数据残缺日
  3. 等权，单票≤5%，流动性配额 w ≤ 2%×日成交额/账户（防冲击）
  4. 用最新交易日收盘价作参考价，算 100 股整数手数

输出：data/results/order_list_<YYYYMMDD>.csv + 终端打印。
用法：
  python tools/make_order_list.py                 # 默认 1e7 账户、流动性 1001~3000
  python tools/make_order_list.py --account 5e6 --lo 501 --hi 1500
  python tools/make_order_list.py --as-of 20260930 # 指定建仓日（默认取库内最新交易日）

⚠️ 这是「底仓/beta 策略」不是 alpha：预期年化 ~+2.5%（修正后无偏口径），2021 后转负。
   详见 docs/trading_plan.html。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

DB = dict(host="127.0.0.1", port=5432, dbname="quant_data", user="quant", password="quant123456")
OUT = Path("data/results")
OUT.mkdir(parents=True, exist_ok=True)

LOT = 100
MAX_WEIGHT = 0.05
LIQ_CAP = 0.02          # 单票买入额 ≤ 2% × 日成交额
MIN_LIST_DAYS = 500     # 近似剔除次新股
WINDOW = 60             # 流动性排序窗口


def load(db, as_of=None):
    cur = db.cursor()
    # 最新交易日
    cur.execute("SELECT max(trade_date) FROM bars_1d")
    latest = pd.Timestamp(cur.fetchone()[0])
    if as_of:
        latest = pd.Timestamp(as_of)
    #  maturity: 每个标的的历史数据天数（近似上市时长）
    cur.execute("SELECT asset_code, count(*) AS n FROM bars_1d GROUP BY 1")
    mat = pd.DataFrame(cur.fetchall(), columns=["asset_code", "n"])
    mat = mat.set_index("asset_code")["n"]
    # 近 70 日价量（用于流动性排序 + 涨停开盘过滤）
    lo = (latest - pd.Timedelta(days=110)).strftime("%Y-%m-%d")
    hi = latest.strftime("%Y-%m-%d")
    cur.execute(
        """SELECT asset_code, trade_date, open, close, amount
           FROM bars_1d WHERE trade_date BETWEEN %s AND %s AND amount > 0
           ORDER BY asset_code, trade_date""",
        (lo, hi),
    )
    rows = cur.fetchall()
    cur.close()
    px = pd.DataFrame(rows, columns=["asset_code", "trade_date", "open", "close", "amount"])
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    return latest, mat, px


def build(latest, mat, px, account, lo_rank, hi_rank):
    # 流动性：每票近 60 日成交额中位数
    amt60 = (px.groupby("asset_code")["amount"]
               .apply(lambda s: s.tail(WINDOW).median()))
    # 最新一日
    last = px[px.trade_date == latest].set_index("asset_code")
    # 前一日收盘（涨停判断用）
    prev = (px[px.trade_date < latest].groupby("asset_code")["close"].last())

    uni = amt60.dropna().sort_values(ascending=False).index
    # 取流动性排名 [lo_rank, hi_rank]（1-indexed，1=最流动）
    sel = uni[lo_rank - 1: hi_rank]
    df = pd.DataFrame(index=sel)
    df["amt60"] = amt60.reindex(sel)
    df["mature"] = mat.reindex(sel).fillna(0) >= MIN_LIST_DAYS
    df["open_t"] = last["open"].reindex(sel)
    df["close_t"] = last["close"].reindex(sel)
    df["prev_close"] = prev.reindex(sel)
    # 过滤
    df = df[df["mature"]]
    df = df[df["amt60"] > 0]
    lim_up = (df["open_t"] / df["prev_close"] - 1) >= 0.098
    df = df[~lim_up.fillna(False)]
    df = df.dropna(subset=["close_t", "open_t", "amt60"])

    # 等权 + 约束
    n = len(df)
    if n == 0:
        return None
    w = pd.Series(1.0 / n, index=df.index)
    w = w.clip(upper=MAX_WEIGHT)
    cap = (LIQ_CAP * df["amt60"].reindex(w.index).fillna(0)) / account
    w = w.clip(upper=cap)
    w = w[w > 0]
    w = w / w.sum()

    out = pd.DataFrame(index=w.index)
    out["weight"] = w
    ref = df["close_t"].reindex(w.index)
    out["ref_price"] = ref
    out["daily_amount"] = df["amt60"].reindex(w.index)
    shares = (account * w / ref / LOT).apply(lambda x: int(x) * LOT)
    out["shares"] = shares
    out["notional"] = shares * ref
    out = out.sort_values("weight", ascending=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", type=float, default=1e7)
    ap.add_argument("--lo", type=int, default=1001, dest="lo_rank")
    ap.add_argument("--hi", type=int, default=3000, dest="hi_rank")
    ap.add_argument("--as-of", type=str, default=None)
    args = ap.parse_args()

    db = psycopg2.connect(**DB)
    latest, mat, px = load(db, args.as_of)
    db.close()
    print(f"建仓日(库内最新): {latest.date()}  账户: {args.account/1e4:.0f}万  流动性排名: {args.lo_rank}~{args.hi_rank}")

    out = build(latest, mat, px, args.account, args.lo_rank, args.hi_rank)
    if out is None:
        print("❌ 该口径下可买池为空（检查 DB 数据完整性）")
        return
    used = out["notional"].sum()
    print(f"可买 {len(out)} 只  等权单票 {100/len(out):.3f}%  计划占用 {used/1e4:.0f}万 "
          f"({used/args.account*100:.1f}% 账户)  残差现金 { (args.account-used)/1e4:.0f}万")
    print(f"单票权重上限 {MAX_WEIGHT*100:.0f}%  流动性配额上限 {LIQ_CAP*100:.0f}%×日成交额")
    print("\n前 15 只：")
    print(out.head(15).to_string(float_format=lambda x: f"{x:.4f}"))

    fn = OUT / f"order_list_{latest.strftime('%Y%m%d')}.csv"
    out.to_csv(fn)
    print(f"\n✅ 已保存下单清单: {fn}")


if __name__ == "__main__":
    main()
