# -*- coding: utf-8 -*-
"""回填七星高照ETF轮动策略的ETF池日线数据(前复权) 到 TimescaleDB bars_1d。

背景:
    本地 TimescaleDB 只有 9 只 ETF(全是 A 股宽基), 七星高照 39 池的
    黄金/纳指/原油/港股/债券/货币等跨资产标的全部缺失, 无法本地复现轮动回测。
    本脚本用 QMT(xtdata) 拉取池内 ETF 的前复权日线并 upsert 入库。

⚠️ 必须在 QMT 正常运行的机器上跑(沙箱内 xtdata.get_market_data 会 segfault)。
   用法:  python tools/backfill_etf_qmt.py            # 全池 2020-01-01 至今
          python tools/backfill_etf_qmt.py --start 2015-01-01

复权说明:
    xtdata.get_market_data(dividend_type='front') 返回前复权价, 动量策略必需
    —— 未复权数据里 513100 有 -80.6% 的虚假暴跌(份额合并), 会打断轮动逻辑。

数据池: 七星高照 V1.7.2 的 39 只跨资产 ETF(聚宽社区源策略, agents-quant.com
    2026-07 复现文章核对)。个别代码若在 QMT 无数据会自动跳过并打印, 可自行增删。
"""
import argparse
import sys
import datetime as dt

import psycopg2

DB = dict(host="127.0.0.1", port=5432, dbname="quant_data",
          user="quant", password="quant123456")

# ─── 七星高照 ETF 池(按资产类别; 与原版 39 只对齐, 个别冷门代码若拉不到会跳过) ───
POOL = {
    # 大宗商品 (6)
    "518880.SH": "黄金ETF",   "159985.SZ": "豆粕ETF",    "161226.SZ": "白银LOF",
    "501018.SH": "南方原油",  "512400.SH": "有色金属ETF", "159981.SZ": "能源化工ETF",
    # 国际市场 (10)
    "513100.SH": "纳指ETF",   "513500.SH": "标普500ETF", "513520.SH": "日经ETF",
    "513030.SH": "德国ETF",   "513080.SH": "法国CAC40",  "513980.SH": "沙特ETF",
    "513050.SH": "中概互联",  "513650.SH": "美国50",     "513330.SH": "恒生互联网",
    "513730.SH": "东南亚科技",
    # 港股 (5)
    "513130.SH": "恒生科技",  "510900.SH": "恒生ETF",    "513660.SH": "港股通红利",
    "513090.SH": "香港证券",  "513180.SH": "恒生科技指数",
    # A股指数 (9)
    "510300.SH": "沪深300ETF", "510500.SH": "中证500ETF", "159915.SZ": "创业板ETF",
    "588000.SH": "科创50ETF",  "512100.SH": "中证1000ETF", "588080.SH": "科创板50",
    "159949.SZ": "创业板50",   "512500.SH": "中证500增强", "563300.SH": "中证2000",
    # 风格 (4)
    "512890.SH": "红利低波",  "159967.SZ": "创业板成长",  "510030.SH": "价值ETF",
    "159324.SZ": "自由现金流",
    # 债券/货币 (4-5)
    "511380.SH": "可转债ETF", "511010.SH": "国债ETF",    "511880.SH": "银华日利(货币)",
    "511220.SH": "城投债ETF", "511090.SH": "30年国债",
}


def upsert(cur, code, rows):
    """rows: [(date, open, high, low, close, volume, amount), ...] 前复权。"""
    n = 0
    for d, o, h, l, c, v, a in rows:
        if d is None or c is None or c <= 0:
            continue
        cur.execute("""
            insert into bars_1d (asset_code, asset_type, trade_date, open, high,
                                 low, close, volume, amount, source)
            values (%s,'etf',%s,%s,%s,%s,%s,%s,%s,'qmt_etf_front')
            on conflict (asset_code, trade_date) do update
              set open=excluded.open, high=excluded.high, low=excluded.low,
                  close=excluded.close, volume=excluded.volume,
                  amount=excluded.amount, source=excluded.source
        """, (code, d, o, h, l, c, v, a))
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-01")
    args = ap.parse_args()

    from xtquant import xtdata           # noqa: 仅QMT机器可import
    import pandas as pd

    conn = psycopg2.connect(**DB)
    cur = conn.cursor()

    ok, miss = 0, []
    start_s = args.start.replace("-", "")
    for code, name in POOL.items():
        try:
            xtdata.download_history_data(code, period="day",
                                         start_time=start_s, incrementally=True)
            panels = xtdata.get_market_data(
                [], [code], period="day", dividend_type="front",
                start_time=start_s,
                end_time=dt.date.today().strftime("%Y%m%d"))
            pan = panels[code]
            rows = []
            for ts, row in pan.iterrows():
                d = pd.Timestamp(str(ts)).date()
                rows.append((d, row.get("open"), row.get("high"), row.get("low"),
                             row.get("close"), row.get("volume"), row.get("amount")))
            n = upsert(cur, code, rows)
            conn.commit()
            print(f"  {code} {name:<12} 入库 {n} 行")
            ok += 1 if n else 0
            if n == 0:
                miss.append(code)
        except Exception as e:                      # noqa: 拉不到的跳过
            print(f"  {code} {name:<12} 失败: {e}")
            miss.append(code)
            conn.rollback()
    cur.close(); conn.close()
    print(f"\n完成: {ok}/{len(POOL)} 只入库; 拉不到/空的: {miss or '无'}")
    if miss:
        print("→ 这些代码可手动核对后从 POOL 里增删。")


if __name__ == "__main__":
    sys.exit(main())
