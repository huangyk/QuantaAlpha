# -*- coding: utf-8 -*-
"""从 QMT 本地 datadir 回填跨资产 ETF 池日线到 TimescaleDB bars_1d。

为什么不用 xtquant API / akshare:
    - xtquant.get_market_data 在沙箱里 segfault(需 QMT 客户端运行), 跑不了。
    - akshare 是外部源, 用户硬规则: QMT 取不到才说, 不能用 akshare 替代。
    - 本机 QMT 实盘终端的 datadir 里有完整日线文件, 直接读 = 最权威的 QMT 数据。

文件格式 (实测):
    D:/中航证券QMT实盘-交易端/datadir/{SH,SZ}/86400/<6位代码>.DAT
    - 8 字节头部(版本/魔数), 之后每条记录 64 字节 = 16 × int32(小端)
    - 记录字段: [0]=Unix时间戳(秒), [1..4]=开/高/低/收 (×1000),
                [6]=成交量(手), [8]=成交额(元), 其余保留/标记
    - 价格是「已复权」连续序列(个别如 511220 城投债仍含拆份跳变 -> 本脚本检测修正)

复权修正:
    QMT datadir 实测为未复权原始价(个别 ETF 有份额拆分造成 9 倍假暴跌)。
    用「开盘价/昨收」检测 round-ratio 拆份(0.1/0.2/0.25/0.5/2/4/5/10),
    后向复权(最新价不变, 历史价乘累计拆份因子)使序列连续, 动量计算才干净。
    (动量用对数收益, 对任意一致乘性复权不变, 只拆份必须修)

用法:
    python tools/backfill_etf_qmt_local.py
    python tools/backfill_etf_qmt_local.py --start 2015-01-01
"""
import argparse
import os
import sys
import struct
import datetime as dt
from datetime import timezone

import psycopg2

DB = dict(host="127.0.0.1", port=5432, dbname="quant_data",
          user="quant", password="quant123456")
QMT_DATADIR = r"D:/中航证券QMT实盘-交易端/datadir"
# 两个数据目录都可能存在该 code 的 .DAT：miniQMT 实时写入 userdata_mini(新鲜,
# 但部分标的历史短), 旧 datadir 历史完整。回填时合并两者取全集(去重+按时间排序)。
QMT_DATADIRS = [
    os.path.join(r"D:/中航证券QMT实盘-交易端/userdata_mini/datadir"),  # 运行中 miniQMT 实时写入
    r"D:/中航证券QMT实盘-交易端/datadir",                                # 旧回填目录(历史全)
]

# 七星高照 V1.7.2 跨资产 ETF 池(39 只; 库内 datadir 实测覆盖 34 只)
POOL = {
    "518880.SH": "黄金ETF",   "159985.SZ": "豆粕ETF",    "161226.SZ": "白银LOF",
    "501018.SH": "南方原油",  "512400.SH": "有色金属",   "159981.SZ": "能源化工",
    "513100.SH": "纳指ETF",   "513500.SH": "标普500ETF", "513520.SH": "日经ETF",
    "513030.SH": "德国ETF",   "513080.SH": "法国CAC40",  "513980.SH": "沙特ETF",
    "513050.SH": "中概互联",  "513650.SH": "美国50",     "513330.SH": "恒生互联网",
    "513730.SH": "东南亚科技", "513130.SH": "恒生科技",  "510900.SH": "恒生ETF",
    "513660.SH": "港股通红利", "513090.SH": "香港证券",  "513180.SH": "恒生科技指数",
    "510300.SH": "沪深300ETF", "510500.SH": "中证500ETF", "159915.SZ": "创业板ETF",
    "588000.SH": "科创50ETF",  "512100.SH": "中证1000ETF", "588080.SH": "科创板50",
    "159949.SZ": "创业板50",   "512500.SH": "中证500增强", "563300.SH": "中证2000",
    "512890.SH": "红利低波",  "159967.SZ": "创业板成长", "510030.SH": "价值ETF",
    "159324.SZ": "自由现金流", "511380.SH": "可转债ETF", "511010.SH": "国债ETF",
    "511880.SH": "银华日利(货币)", "511220.SH": "城投债ETF", "511090.SH": "30年国债",
}


def market_of(code):
    """6 位代码 -> SH/SZ。ETF: SH=51/50/58/56/11; SZ=15/16/12/00/30。"""
    return "SZ" if code[:2] in ("15", "16", "12", "00", "30") else "SH"


ROUND_RATIOS = [0.1, 0.125, 0.2, 0.25, 1 / 3, 0.5, 0.8, 1.25, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0]
PLAUSIBLE_LO, PLAUSIBLE_HI = 0.65, 1.5   # 正常单日收益率区间(ETF 单日不可能超出)


def snap_ratio(r):
    """把异常比例吸附到最近的整数拆分比例, 吸附不上就用实际比例。"""
    best, bd = None, 1e9
    for f in ROUND_RATIOS:
        d = abs(r - f) / f
        if d < bd:
            bd, best = d, f
    return best if bd <= 0.25 else r


def detect_split(close_i, close_prev):
    """返回拆份因子(默认1.0=无拆份)。

    判据改用「收盘/昨收」(实测开盘判据会漏判):
        ETF 单日不可能涨跌超过 +/-50%, 超出 -35%/+50% 一定不是真实行情,
        而是份额拆分/折算或数据错误 -> 按比例修正使序列连续。
    """
    if close_prev is None or close_i is None or close_prev <= 0:
        return 1.0
    r = close_i / close_prev
    if PLAUSIBLE_LO <= r <= PLAUSIBLE_HI:
        return 1.0                      # 正常波动
    return snap_ratio(r)


def parse_dat(path):
    """返回 list[(date, open, high, low, close, volume, amount)] 后向复权价。"""
    b = open(path, "rb").read()
    rec = 64
    off = 8
    n = (len(b) - off) // rec
    if n <= 0:
        return []
    ts_, o, h, l, c, v, a = [], [], [], [], [], [], []
    for i in range(n):
        vals = struct.unpack("<" + "i" * 16, b[off + i * rec: off + i * rec + 64])
        ts_.append(vals[0])
        o.append(vals[1] / 1000.0)
        h.append(vals[2] / 1000.0)
        l.append(vals[3] / 1000.0)
        c.append(vals[4] / 1000.0)
        v.append(vals[6])
        a.append(vals[8])
    # 后向复权: 最新价不变, 历史价乘累计拆份因子
    mf = 1.0
    adj_o, adj_h, adj_l, adj_c = [0.0] * n, [0.0] * n, [0.0] * n, [0.0] * n
    splits = 0
    for i in range(n - 1, -1, -1):
        adj_o[i] = o[i] * mf
        adj_h[i] = h[i] * mf
        adj_l[i] = l[i] * mf
        adj_c[i] = c[i] * mf
        if i > 0:
            s = detect_split(c[i], c[i - 1])
            if s != 1.0:
                mf *= s
                splits += 1
    rows = []
    for i in range(n):
        d = dt.datetime.fromtimestamp(ts_[i], tz=timezone.utc).date()
        rows.append((d, adj_o[i], adj_h[i], adj_l[i], adj_c[i], v[i], a[i]))
    return rows, splits


def read_raw_records(path):
    """读取单个 .DAT 的全部原始 16-int 记录(未复权)。无则返回 []。"""
    if not os.path.exists(path):
        return []
    b = open(path, "rb").read()
    off, rec = 8, 64
    n = (len(b) - off) // rec
    if n <= 0:
        return []
    return [struct.unpack("<" + "i" * 16, b[off + i * rec: off + i * rec + 64])
            for i in range(n)]


def parse_code(code):
    """合并所有 QMT_DATADIRS 中该 code 的原始记录 -> 后向复权 rows + 拆份数。

    旧 datadir 历史全但停在 9/29；userdata_mini 有 10 月新鲜数据但部分标的历史短。
    合并+去重(同 ts)+按时间排序后统一做拆份修正, 得到完整且最新的序列。
    """
    raw = []
    for d in QMT_DATADIRS:
        p = os.path.join(d, market_of(code[:6]), "86400", code[:6] + ".DAT")
        raw.extend(read_raw_records(p))
    if not raw:
        return [], 0
    seen = {}
    for v in raw:
        seen[v[0]] = v                      # 同时间戳去重
    vals = [seen[k] for k in sorted(seen)]
    n = len(vals)
    o = [v[1] / 1000.0 for v in vals]
    h = [v[2] / 1000.0 for v in vals]
    l = [v[3] / 1000.0 for v in vals]
    c = [v[4] / 1000.0 for v in vals]
    v = [v[6] for v in vals]
    a = [v[8] for v in vals]
    mf = 1.0
    adj_o, adj_h, adj_l, adj_c = [0.0] * n, [0.0] * n, [0.0] * n, [0.0] * n
    splits = 0
    for i in range(n - 1, -1, -1):
        adj_o[i] = o[i] * mf; adj_h[i] = h[i] * mf
        adj_l[i] = l[i] * mf; adj_c[i] = c[i] * mf
        if i > 0:
            s = detect_split(c[i], c[i - 1])
            if s != 1.0:
                mf *= s; splits += 1
    rows = []
    for i in range(n):
        d = dt.datetime.fromtimestamp(vals[i][0], tz=timezone.utc).date()
        rows.append((d, adj_o[i], adj_h[i], adj_l[i], adj_c[i], v[i], a[i]))
    return rows, splits


def upsert(cur, code, rows):
    """按 (asset_code, trade_date) 去重写入。

    为什么要先删再插:
        表唯一键是 (asset_code, ts), 而历史 'qmt' 行的 ts 混用 00:00 / 15:00
        两种时刻 -> 只靠 ts 对齐会撞不上, 同一 trade_date 插出第二行。
        改为: 先删掉同一 (asset_code, trade_date) 的所有旧行, 再插本脚本的一致数据;
        datadir 没覆盖的更早期历史不在 rows 里, 不受影响。
    """
    dates = sorted({d for d, *_ in rows if d is not None})
    if dates:
        cur.execute("""delete from bars_1d
                       where asset_code=%s and trade_date = any(%s::date[])""",
                    (code, dates))
    n = 0
    for d, o, h, l, c, v, a in rows:
        if d is None or c is None or c <= 0:
            continue
        # ⚠️ 表既有数据的约定: ts = 交易日 00:00 北京时间(+08), 不是 UTC。
        #    用错时区会让同一 trade_date 插出第二行(唯一键是 asset_code+ts)。
        ts = dt.datetime.combine(d, dt.time(0, 0),
                                 tzinfo=dt.timezone(dt.timedelta(hours=8)))
        cur.execute("""
            insert into bars_1d (asset_code, asset_type, ts, trade_date, open, high,
                                 low, close, volume, amount, source)
            values (%s,'etf',%s,%s,%s,%s,%s,%s,%s,%s,'qmt_local_86400')
            on conflict (asset_code, ts) do update
              set trade_date=excluded.trade_date, open=excluded.open,
                  high=excluded.high, low=excluded.low, close=excluded.close,
                  volume=excluded.volume, amount=excluded.amount, source=excluded.source
        """, (code, ts, d, o, h, l, c, v, a))
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-legacy", action="store_true",
                    help="清掉被回填标的的非本次来源旧行(推荐: 避免复权基准拼接产生假收益)")
    args = ap.parse_args()

    conn = psycopg2.connect(**DB)
    cur = conn.cursor()
    ok, miss, total_rows = 0, [], 0
    for code, name in POOL.items():
        rows, splits = parse_code(code)
        if not rows:
            print(f"  {code} {name:<10} 所有 datadir 缺失(跳过)")
            miss.append(code)
            continue
        n = upsert(cur, code, rows)
        if args.clean_legacy:
            # 删掉该标的的非本次来源残留行: 旧 'qmt' 数据的复权基准与本脚本不同,
            # 拼接处会产生假收益(实测 +176%/-87% 级别), 必须整码统一来源。
            cur.execute("""delete from bars_1d
                           where asset_code=%s and asset_type='etf'
                             and source <> 'qmt_local_86400'""", (code,))
            legacy = cur.rowcount
        else:
            legacy = 0
        conn.commit()
        ok += 1
        total_rows += n
        first, last = rows[0][0], rows[-1][0]
        flag = f"  [修{splits}拆份]" if splits else ""
        if legacy:
            flag += f"  [清{legacy}旧行]"
        print(f"  {code} {name:<10} 入库 {n:>4} 行  {first}~{last}{flag}")
    cur.close()
    conn.close()
    print(f"\n完成: {ok}/{len(POOL)} 只入库, 共 {total_rows} 行; datadir缺失: {miss or '无'}")


if __name__ == "__main__":
    sys.exit(main())
