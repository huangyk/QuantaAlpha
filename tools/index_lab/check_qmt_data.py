# -*- coding: utf-8 -*-
"""
诊断 QMT 本机 ETF 日线数据是否够用（200日均线 + 60日波动窗口）。

为什么需要它
------------
QMT 有**两个 datadir**，内容不一致：
  - `{QMT}/datadir/`                （主目录，部分标的更全）
  - `{QMT}/userdata_mini/datadir/`  （★QMT 实际读取的目录，部分标的残缺）
同一标的在两处行数可能差几十倍（例：513080.SH mini=31 / datadir=1532）。
**QMT 用 mini 那份** → 200日均线历史不足 → 信号几乎不变 → 回测结果失真。

用法
----
    python tools/index_lab/check_qmt_data.py
    python tools/index_lab/check_qmt_data.py --min-rows 260
"""
import argparse
import os
import struct
import datetime

QMT_HOME = r"D:\中航证券QMT实盘-交易端"
DATADIRS = [
    ("datadir(主)", os.path.join(QMT_HOME, "datadir")),
    ("mini(实际读取)", os.path.join(QMT_HOME, "userdata_mini", "datadir")),
]
# 从粘贴版策略里读 POOL，避免两处维护
POOL_FILE = os.path.join(QMT_HOME, "absmom_策略代码_可直接粘贴.txt")


def load_pool():
    """从可粘贴策略代码里解析 POOL 列表。"""
    if not os.path.exists(POOL_FILE):
        return []
    t = open(POOL_FILE, "rb").read().decode("utf-8-sig")
    i = t.find("POOL = [")
    j = t.find("]", i)
    if i < 0 or j < 0:
        return []
    body = t[i + len("POOL = ["):j].replace("\n", "")
    return [x.strip().strip('"') for x in body.split(",") if x.strip()]


def rows_of(root, code):
    """返回该 datadir 下标的的日线行数与起始日期，无文件返回 (None, None)。"""
    num, ex = code.split(".")
    p = os.path.join(root, ex, "86400", num + ".DAT")
    if not os.path.exists(p):
        return None, None
    n = (os.path.getsize(p) - 8) // 64
    if n <= 0:
        return 0, None
    with open(p, "rb") as f:
        f.seek(8)
        ts0 = struct.unpack("<i", f.read(4))[0]
    d0 = datetime.datetime.utcfromtimestamp(ts0) + datetime.timedelta(hours=8)
    return n, d0.date()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-rows", type=int, default=260,
                    help="策略需要的最小行数(默认260 = 200日SMA + 60日波动)")
    args = ap.parse_args()

    pool = load_pool()
    if not pool:
        print("[X] 找不到 POOL（%s 不存在？）" % POOL_FILE)
        return

    print("=" * 74)
    print("QMT ETF 数据体检  最小需要 %d 行" % args.min_rows)
    print("=" * 74)
    print("%-12s %-16s %-16s %s" % ("代码", "datadir(主)", "mini(实际)", "判定"))

    bad = []
    for c in pool:
        cells = []
        ns = {}
        for tag, root in DATADIRS:
            n, d0 = rows_of(root, c)
            ns[tag] = n
            cells.append("%s" % ("无" if n is None else "%d行 %s" % (n, d0)))
        used = ns["mini(实际读取)"]
        if used is None:
            verdict = "[X] mini无数据"
        elif used < args.min_rows:
            verdict = "[!] 历史不足%d行" % args.min_rows
        elif ns["datadir(主)"] and ns["datadir(主)"] > used * 1.5:
            verdict = "[!] mini数据偏少(主目录有%d行)" % ns["datadir(主)"]
        else:
            verdict = "OK"
        if verdict != "OK":
            bad.append((c, ns, verdict))
        print("%-12s %-16s %-16s %s" % (c, cells[0], cells[1], verdict))

    print()
    print("=" * 74)
    if bad:
        # 只列真正会让 200日均线失效的（不足 min_rows），
        # 「mini 偏少但已过线」的另作提示，避免用户做无用功
        must = [b for b in bad if b[2] != "OK" and b[1]["mini(实际读取)"] is not None
                and b[1]["mini(实际读取)"] < args.min_rows]
        soft = [b for b in bad if b not in must]

        if must:
            print("【必须补】%d 只历史不足 %d 行 -> 200日均线算不出，信号会失真"
                  % (len(must), args.min_rows))
            print()
            print("%-12s %-9s %-12s %-12s %s"
                  % ("代码", "当前行数", "当前起始", "建议下载起始", "主目录可参考"))
            print("-" * 74)
            for c, ns, v in must:
                nb = ns["mini(实际读取)"]
                _, dmini = rows_of(DATADIRS[1][1], c)
                na, dmain = rows_of(DATADIRS[0][1], c)
                # 260 个交易日 ≈ 380 自然日，往前留足
                sug = (dmini - datetime.timedelta(days=380)) if dmini else "2019-01-01"
                print("%-12s %-9d %-12s %-12s %s"
                      % (c, nb or 0, dmini or "?", sug,
                         ("%d行 %s" % (na, dmain)) if na else "无"))
            print()
            print("做法：QMT【行情】→【历史数据下载】→ 勾选上面 %d 只 → 日线" % len(must))
            print("     → 起始日期填「建议下载起始」列（或统一填 2019-01-01 更省事）")
            print("     → 勾「补充」/「全量」后开始下载")
            print("若下载后仍不足：QMT 服务器未提供更早数据，需切到带「迅投」字样的行情服务器。")
            print()

        if soft:
            print("【提示】另 %d 只虽已过线，但 QMT 实际读取的数据明显偏少：" % len(soft))
            print("       （能跑，但历史窗口短，回测早期信号可能偏差）")
            for c, ns, v in soft[:12]:
                na = ns["datadir(主)"]
                print("       %-12s QMT实际 %s行，主目录 %s行"
                      % (c, ns["mini(实际读取)"], na))
            if len(soft) > 12:
                print("       ... 其余 %d 只略" % (len(soft) - 12))
            print()
    else:
        print("全部 %d 只历史充足，可以正常回测。" % len(pool))


if __name__ == "__main__":
    main()
