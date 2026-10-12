"""生成跨资产趋势跟随策略的下单清单（按 100 股取整，支持核心-卫星混合）。

用法：
  python -m tools.index_lab.make_order_list 80000          # 纯底盘(blend=0)
  python -m tools.index_lab.make_order_list 80000 0.25     # 底盘75% + H1(高beta)25%
  python -m tools.index_lab.make_order_list 80000 0.5      # 各半

混合后按"合并权重"单次分配股数：blended_w_i = (1-blend)*底盘权重_i + blend*H1权重_i，
避免分仓后再各自取整导致高单价债券被舍。输出 data/results/order_list_<资金>_b<H1%>.csv。
"""
import os
import sys
import csv

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from tools.index_lab.method_absmom import current_signal
from tools.index_lab.io_db import load_etf_universe
from tools.index_lab.method_higher import is_bond_cash

NAMES = {
    "511090.SH": "30年国债ETF", "511220.SH": "城投债ETF", "511880.SH": "货币ETF(日利)",
    "512890.SH": "银行ETF", "510030.SH": "180价值ETF", "513650.SH": "标普500ETF",
    "513500.SH": "标普500ETF", "513100.SH": "纳指100ETF", "159985.SZ": "豆粕ETF",
    "513520.SH": "日经225ETF", "501018.SH": "南方原油",
}


def main():
    cap = float(sys.argv[1]) if len(sys.argv) > 1 else 80000.0
    blend = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    if not (0.0 <= blend <= 1.0):
        print(f"[错误] blend 须在 [0,1]，收到 {blend}", file=sys.stderr)
        sys.exit(2)

    codes = load_etf_universe("2019-01-01", "2026-09-30", True)
    hb = [c for c in codes if not is_bond_cash(c)]
    base_sig = current_signal(codes, as_of=None, signal="sma200", weight="inv_vol")
    h1_sig = current_signal(hb, as_of=None, signal="sma200", weight="inv_vol")

    base_w = {h["code"]: h["weight"] for h in base_sig["holdings"] if h["hold"]}
    h1_w = {h["code"]: h["weight"] for h in h1_sig["holdings"] if h["hold"]}
    price = {h["code"]: h["close"] for h in base_sig["holdings"]}
    for h in h1_sig["holdings"]:
        price.setdefault(h["code"], h["close"])

    all_codes = sorted(set(base_w) | set(h1_w))
    rows = []
    for code in all_codes:
        bw = (1.0 - blend) * base_w.get(code, 0.0) + blend * h1_w.get(code, 0.0)
        if bw <= 0:
            continue
        close = price.get(code)
        if close is None or close != close or close <= 0:
            rows.append([code, NAMES.get(code, ""), "CASH", "", 0.0, 0, 0, 0.0])
            continue
        target = bw * cap
        lots = round(target / (close * 100)) if target >= close * 100 else 0
        shares = max(0, lots) * 100
        amt = shares * close
        action = "BUY" if shares > 0 else "CASH"
        rows.append([code, NAMES.get(code, ""), action, round(close, 4),
                     round(bw * 100, 2), round(target), shares, round(amt, 2)])

    buys = [r for r in rows if r[2] == "BUY"]
    cashes = [r for r in rows if r[2] == "CASH"]
    buys.sort(key=lambda r: -r[4])
    total = sum(r[7] for r in buys)
    left = cap - total

    tag = f"_b{int(blend*100)}"
    out = os.path.join(ROOT, "data", "results", f"order_list_{int(cap)}{tag}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code", "name", "action", "price", "target_weight_pct",
                    "target_yuan", "shares", "amount"])
        for r in buys:
            w.writerow(r)
        for r in cashes:
            w.writerow(r)
        w.writerow(["", "", "CASH_LEFT", "", "", round(left, 2), "", round(left, 2)])

    print(f"资金 {cap:,.0f} | H1卫星 {blend*100:.0f}% | 买入 {len(buys)} 只 | "
          f"占用 {total:,.2f} | 留现金 {left:,.2f}")
    print(f"{'代码':<10}{'名称':<14}{'动作':<6}{'现价':>9}{'目标%':>8}{'目标元':>10}{'股数':>8}{'金额':>11}")
    for r in buys:
        print(f"{r[0]:<10}{r[1]:<14}{r[2]:<6}{r[3]:>9}{r[4]:>7.1f}%{r[5]:>10,.0f}{r[6]:>8}{r[7]:>11,.2f}")
    if cashes:
        print(f"  (转现金 {len(cashes)} 只：{', '.join(r[0] for r in cashes)})")
    print(f"  -> 留现金: {left:,.2f} 元；已保存: {out}")


if __name__ == "__main__":
    main()
