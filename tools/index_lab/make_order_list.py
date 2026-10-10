"""生成跨资产趋势跟随策略的下单清单（按 100 股取整）。

用法：
  python -m tools.index_lab.make_order_list 80000
读取 data/results/current_holdings.csv（run_signal 产出），按给定资金额
把权重换算成「买多少股」，并列出转现金的标的。输出 data/results/order_list_<资金>.csv。
"""
import os
import sys
import csv

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "data", "results", "current_holdings.csv")

NAMES = {
    "511090.SH": "30年国债ETF", "511220.SH": "城投债ETF", "511880.SH": "货币ETF(日利)",
    "512890.SH": "银行ETF", "510030.SH": "180价值ETF", "513650.SH": "标普500ETF",
    "513500.SH": "标普500ETF", "513100.SH": "纳指100ETF", "159985.SZ": "豆粕ETF",
    "513520.SH": "日经225ETF", "501018.SH": "南方原油",
}


def main():
    cap = float(sys.argv[1]) if len(sys.argv) > 1 else 80000.0
    if not os.path.exists(SRC):
        print(f"[错误] 找不到 {SRC}，请先跑 `python -m tools.index_lab signal`", file=sys.stderr)
        sys.exit(2)
    rows = []
    with open(SRC, newline="") as f:
        for x in csv.DictReader(f):
            code = x["code"]
            close = float(x["close"]) if x["close"] else float("nan")
            hold = int(x["hold"])
            w = float(x["weight"])
            name = NAMES.get(code, "")
            if hold == 1 and close == close and close > 0:
                target = w * cap
                lots = max(1, round(target / (close * 100)))
                shares = lots * 100
                amt = shares * close
                rows.append([code, name, "BUY", round(close, 4),
                             round(w * 100, 2), round(target), shares, round(amt, 2)])
            else:
                rows.append([code, name, "CASH", round(close, 4) if close == close else "",
                             0.0, 0, 0, 0.0])
    buys = [r for r in rows if r[2] == "BUY"]
    cashes = [r for r in rows if r[2] == "CASH"]
    buys.sort(key=lambda r: -r[4])
    total = sum(r[7] for r in buys)
    left = cap - total

    out = os.path.join(ROOT, "data", "results", f"order_list_{int(cap)}.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code", "name", "action", "price", "target_weight_pct",
                    "target_yuan", "shares", "amount"])
        for r in buys:
            w.writerow(r)
        for r in cashes:
            w.writerow(r)
        w.writerow(["", "", "CASH_LEFT", "", "", round(left, 2), "", round(left, 2)])

    print(f"资金 {cap:,.0f} 元 | 买入 {len(buys)} 只 | 占用 {total:,.2f} | 留现金 {left:,.2f}")
    print(f"{'代码':<10}{'名称':<14}{'动作':<6}{'现价':>9}{'目标%':>8}{'目标元':>10}{'股数':>8}{'金额':>11}")
    for r in buys:
        print(f"{r[0]:<10}{r[1]:<14}{r[2]:<6}{r[3]:>9}{r[4]:>7.1f}%{r[5]:>10,.0f}{r[6]:>8}{r[7]:>11,.2f}")
    print(f"  -> 留现金: {left:,.2f} 元（遇不足1手时向下取整，余款留仓）")
    print(f"  -> 已保存: {out}")


if __name__ == "__main__":
    main()
