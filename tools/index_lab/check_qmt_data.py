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
        print("有问题 %d / %d 只：" % (len(bad), len(pool)))
        for c, ns, v in bad:
            print("  %-12s %s" % (c, v))
        print()
        print("处理办法：QMT【行情】→【历史数据下载】→ 选这些 ETF → 日线 → 补足到 %d 行以上"
              % args.min_rows)
        print("若下载后仍不足，说明该 QMT 服务器未提供更早数据（需切到带「迅投」字样的服务器）。")
    else:
        print("全部 %d 只历史充足，可以正常回测。" % len(pool))


if __name__ == "__main__":
    main()
