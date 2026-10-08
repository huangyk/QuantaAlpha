"""批量拉取 2025-2026 的换手率 / 振幅（akshare = 东方财富）。

为什么需要拉：
    换手率因子是项目里最强的信号（RankIC |t| 8~11，三段同号），
    但 DB 里的 turnover_rate 只覆盖 2020-2024 → **2025-2026 无法验证**。
    而 2025-2026 正是战法失效的关键区间，不验证就无法判断信号是否已死。

为什么不用 QMT（已实测排除）：
    1. `get_market_data_ex` 只返回 open/high/low/close/volume/amount + preClose/suspendFlag，
       **turnoverRate / totalValue / totalVolume 全部 MISS**
    2. `get_financial_data` 的 9 个报表（BalanceInfo/IncomeInfo/CapitalStockStructure/…）
       **全部返回 shape=(0,0)** —— mini 版 QMT 不提供财务数据（那是完整版的功能）
    3. xtquant 版本 250807 的签名是 `table_list`（不是旧文档的 report_name）

数据源对比已验证一致：akshare 的 600000 在 2026-09-30 成交量 1474848
与 QMT/DB 完全相同（逐位），可放心使用。

用法：
    python tools/fetch_turnover_akshare.py --top 500 --workers 8
    python tools/fetch_turnover_akshare.py --all --workers 8
输出：data/cache/akshare_turnover_{tag}.csv.gz
"""
from __future__ import annotations

import argparse
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

OUT = Path('data/cache')
PANEL = Path('data/cache/panel_2011.parquet')
START, END = '20250101', '20260930'
_print_lock = threading.Lock()


def log(*a):
    with _print_lock:
        print(*a, flush=True)


def pick_codes(mode: str, n: int):
    """按 2026-09 成交额排名取前 n 只；mode='all' 取全部。"""
    P = pd.read_parquet(PANEL)
    if not isinstance(P.columns, pd.MultiIndex):
        P.columns = pd.MultiIndex.from_tuples(
            [tuple(c) if isinstance(c, tuple) else (c, c) for c in P.columns])
    P.index = pd.DatetimeIndex(P.index.get_level_values(0))
    codes = list(P.columns.get_level_values(1).unique())
    if mode == 'all':
        return codes
    last = AMT = P['amount'].loc['2026-06-01':].median().sort_values(ascending=False)
    rank = last[last.index.isin(codes)]
    sel = list(rank.index[:n])
    log(f'候选池: {len(codes)} 只 → 取成交额前 {len(sel)} 只（末位日均成交额 {rank.iloc[-1]/1e8:.2f} 亿）')
    return sel


_P = None


def fetch_one(code: str, retries: int = 2):
    # ⚠️ 每个 worker 线程都要自己初始化 akshare（线程池不会继承主线程的全局变量）
    global _P
    if _P is None:
        import akshare as ak
        _P = ak
    sym = code.split('.')[0]
    for k in range(retries + 1):
        try:
            df = _P.stock_zh_a_hist(symbol=sym, period='daily',
                                    start_date=START, end_date=END, adjust="qfq")
            if df is None or len(df) == 0:
                return code, None
            out = pd.DataFrame({
                'datetime': pd.to_datetime(df['日期']),
                'turnover_rate': pd.to_numeric(df['换手率'], errors='coerce'),
                'amplitude': pd.to_numeric(df['振幅'], errors='coerce'),
                'pct_chg': pd.to_numeric(df['涨跌幅'], errors='coerce'),
                'amount_ak': pd.to_numeric(df['成交额'], errors='coerce'),
            })
            out['instrument'] = code
            return code, out
        except Exception as e:
            if k == retries:
                return code, f'ERR {type(e).__name__}'
            time.sleep(0.8 * (k + 1))
    return code, 'ERR'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--top', type=int, default=500, help='取成交额前 N 只')
    ap.add_argument('--all', action='store_true', help='全部 5200+ 只')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()

    codes = pick_codes('all' if args.all else 'top', args.top)
    tag = args.tag or ('all' if args.all else f'top{args.top}')
    log(f'待拉取 {len(codes)} 只，区间 {START}~{END}，并发 {args.workers}')

    t0 = time.time()
    frames, fails = [], []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, c): c for c in codes}
        for fu in as_completed(futs):
            code, r = fu.result()
            done += 1
            if r is None:
                fails.append((code, 'empty'))
            elif isinstance(r, str):
                fails.append((code, r))
            else:
                frames.append(r)
            if done % 100 == 0:
                el = time.time() - t0
                rate = done / el
                left = (len(codes) - done) / max(rate, 1e-9)
                log(f'  {done}/{len(codes)}  成功率 {len(frames)/max(done,1):.1%}  '
                    f'{rate:.1f} 只/秒  预计剩余 {left/60:.1f} 分钟')

    if frames:
        out = pd.concat(frames, ignore_index=True)
        p = OUT / f'akshare_turnover_{tag}.csv.gz'
        out.to_csv(p, index=False, compression='gzip')
        log(f'\n保存 {p}（{p.stat().st_size/1e6:.1f} MB）')
        log(f'  股票数 {out["instrument"].nunique()}  行数 {len(out):,}  '
            f'区间 {out["datetime"].min().date()} ~ {out["datetime"].max().date()}')
        log(f'  换手率非NaN {out["turnover_rate"].notna().sum():,}')
    if fails:
        log(f'  失败 {len(fails)} 只，样例: {fails[:5]}')
    log(f'总耗时 {(time.time()-t0)/60:.1f} 分钟')


if __name__ == '__main__':
    main()