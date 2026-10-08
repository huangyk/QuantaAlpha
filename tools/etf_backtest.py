# -*- coding: utf-8 -*-
"""
8万小账户 ETF 回测（修正版）：

为什么用指数而不是 DB 的 ETF 价格：
  DB 里 512100/510500 的 close 未处理份额拆分（2015 年 510500 从 1.49 跳到 7.84，
  一年 5.3 倍，而中证500当年仅 +43%；2022 年 512100 从 1.17 跳到 2.51，而中证1000当年跌 21%），
  直接用会严重高估收益。ETF 本质跟踪指数，故用库内拆干净的真实指数序列
  （000852 中证1000 / 000905 中证500 / 399006 创业板指 / 000300 沪深300）当 ETF 收益代理。
  ETF 实盘 = 指数价格收益 + 分红(~2%/年) - 管理费(~0.5%/年)，净约多 +1~1.5%/年，下文注明。

成本：单笔买入佣金 max(金额*0.00025, 5元)；持有期无换手；期末卖出扣佣金+0.05%印花税。
"""
import psycopg2
import pandas as pd
import numpy as np

DB = dict(host="127.0.0.1", port=5432, dbname="quant_data", user="quant", password="quant123456")

# 指数代码 -> (名称, 说明)
IDX = {
    "000852.SH": ("中证1000", "小盘，最贴近回测P3流动性溢价敞口"),
    "000905.SH": ("中证500",  "中盘，分散更稳"),
    "399006.SZ": ("创业板指",  "成长股，波动与弹性最高"),
    "000300.SH": ("沪深300",  "大盘蓝筹，基准对照"),
}

ACCOUNT = 8e4          # 用户真实资金
COMM_BP = 25.0         # 万2.5 佣金
STAMP_BP = 5.0         # 卖出印花税 万5
FLOOR = 5.0            # 单笔最低5元


def load_idx(code):
    conn = psycopg2.connect(**DB)
    cur = conn.cursor()
    cur.execute("SELECT trade_date, close FROM bars_1d WHERE asset_code=%s ORDER BY trade_date", (code,))
    rows = cur.fetchall()
    conn.close()
    df = pd.DataFrame(rows, columns=["date", "close"])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()["close"]


def round_trip_cost(notional, is_buy):
    comm = max(notional * COMM_BP / 1e4, FLOOR)
    stamp = notional * STAMP_BP / 1e4 if not is_buy else 0.0
    return comm + stamp


def buy_hold(code, name, note):
    px = load_idx(code)
    if px.empty:
        return None
    start = px.index[0]; end = px.index[-1]
    p0 = px.iloc[0]
    units = ACCOUNT / p0                      # 用首日均价买入（无100股限制，指数连续）
    buy_cost = round_trip_cost(units * p0, True)
    end_px = px.iloc[-1]
    end_val = units * end_px
    sell_cost = round_trip_cost(end_val, False)
    net_end = end_val - sell_cost
    # 净值曲线（盯市，初始投入=ACCOUNT）
    eq = units * px
    equity = eq.copy()
    equity.iloc[0] = ACCOUNT - (buy_cost - 0)   # 首笔已付买入费，约 ACCOUNT-5
    equity.iloc[-1] = net_end
    ret = net_end / ACCOUNT - 1.0
    yrs = (end - start).days / 365.25
    ann = (net_end / ACCOUNT) ** (1 / yrs) - 1 if net_end > 0 else float("nan")
    peak = equity.cummax()
    mdd = (equity / peak - 1.0).min()
    return dict(code=code, name=name, note=note,
                start=start.date(), end=end.date(), yrs=round(yrs, 1),
                p0=round(p0, 1), p_end=round(end_px, 1),
                units=round(units, 4), end_value=round(net_end, 0),
                total_ret=round(ret * 100, 1), ann=round(ann * 100, 1),
                mdd=round(mdd * 100, 1), equity=equity)


def main():
    print(f"账户: {ACCOUNT/1e4:.0f}万 | 成本: 万2.5佣金(最低5元)+卖出万5印花税")
    print("注: 以下为指数价格收益；ETF实盘≈指数价收+分红(~2%/年)-管理费(~0.5%/年)，净多+1~1.5%/年\n")
    results = []
    eqs = {}
    for code, (name, note) in IDX.items():
        r = buy_hold(code, name, note)
        if r:
            results.append(r); eqs[code] = r["equity"]
            print(f"=== {name} ({code}) ===")
            print(f"  区间 : {r['start']} ~ {r['end']}  ({r['yrs']}年)")
            print(f"  起点 {r['p0']} -> 终点 {r['p_end']}")
            print(f"  ★ 总收益 : {r['total_ret']:+.1f}%   年化: {r['ann']:+.1f}%")
            print(f"  ★ 最大回撤: {r['mdd']:.1f}%")
            print(f"  8万变: {r['end_value']/1e4:.2f}万\n")

    # 50/50 分散：中证500 + 中证1000
    if "000905.SH" in eqs and "000852.SH" in eqs:
        a = eqs["000905.SH"]; b = eqs["000852.SH"]
        idx = a.index.intersection(b.index)
        blended = 0.5 * a.reindex(idx) + 0.5 * b.reindex(idx)
        ret = blended.iloc[-1] / ACCOUNT - 1
        yrs = (idx[-1] - idx[0]).days / 365.25
        ann = (blended.iloc[-1] / ACCOUNT) ** (1 / yrs) - 1
        mdd = (blended / blended.cummax() - 1).min()
        print(f"=== 50/50 分散 (中证500 + 中证1000 各半仓买入持有) ===")
        print(f"  区间 : {idx[0].date()} ~ {idx[-1].date()}  ({round(yrs,1)}年)")
        print(f"  ★ 总收益 : {ret*100:+.1f}%   年化: {ann*100:+.1f}%")
        print(f"  ★ 最大回撤: {mdd*100:.1f}%")
        print(f"  8万变: {blended.iloc[-1]/1e4:.2f}万\n")
        eqs["BLEND_5050"] = blended

    print("=== 对照：8万自己做股票篮子（前期回测，已证伪）===")
    print("  季频25只: 8万→0.57万 (年化 -45.7%, 回撤 -96.1%)")
    print("  月频25只: 8万→0.04万 (年化 -32.3%, 回撤 -99.6%)")
    print("  → 同一笔8万，指数ETF方案 vs 自己持股，差距在天壤。\n")

    out = pd.DataFrame({c: eqs[c] for c in eqs})
    out.to_csv("data/results/etf_equity_curve.csv")
    pd.DataFrame([{k: v for k, v in r.items() if k != "equity"} for r in results]).to_csv(
        "data/results/etf_backtest.csv", index=False)
    print("保存: data/results/etf_backtest.csv, etf_equity_curve.csv")


if __name__ == "__main__":
    main()
