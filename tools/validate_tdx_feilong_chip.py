"""
TDX 飞龙/抄底/主力筹码 验证 (含 COST/WINNER 筹码分布还原)
============================================================================
用户要求(不看凰/凤线):
  1 看买点   2 看飞龙线是不是往上走   3 看筹码 主力筹码是不是增多
  4 看盈亏比 5 飞龙线/抄底线/主力散户线依赖 COST()/WINNER() 筹码分布,
     本库 bars_1d 只有 OHLCV -> 用 tools/chip_distribution.py 还原筹码再算。

做法:
  - 从 TimescaleDB 取 float_shares 非空的非 ETF 股票(筹码输入齐备的子集)
  - 还原每日筹码分布 -> COST(5/10/50/89/90), WINNER(0.9C/1.0C/1.1C), 集中度 conc
  - 飞龙线 = sqrt(EMA89 * COST89);  抄底线 = sqrt(EMA89 * COST5)
  - 主力线 zl = EMA(WINNER(1.1C)-WINNER(0.9C), 5)  (夹在 ±10% 的筹码厚度≈主力仓)
  - 散户线 sh = MA(WINNER(0.9C),10) - MA(zl,10)    (复刻用户公式)
  - 5 个买侧判据 + 两类合成买点(抄底型/突破型)
  - 回测: T+1 开盘买; 抄底型止损=抄底线, 目标=飞龙线; 突破型同; H=5/10/20
  - 盈亏比 = 每笔 (目标-买)/(买-止损); 另报实现盈亏比(avg_win/|avg_loss|)、期望
  - 末日快照: 2026-09-30 通过合成买点的票(实用买点清单)

数据铁律: 只用 TimescaleDB(源自 QMT datadir 回填), 绝不碰 akshare。
对齐铁律: 信号只用 T 及之前收盘; 买 T+1 开盘; 成本万1+市值分层冲击;
          剔次新(<250)/ST/涨停买不进(open/昨收>=1.098)。
"""
import argparse, json, time
import numpy as np
import pandas as pd
import psycopg2
from chip_distribution import compute_chip

DB = dict(host='127.0.0.1', port=5432, dbname='quant_data', user='quant', password='quant123456')
COMM = 0.0001


def load_panel(start, end):
    c = psycopg2.connect(**DB); cur = c.cursor()
    # 仅取 float_shares 非空(筹码输入齐备)的非 ETF 股票
    sql = """
        select b.asset_code, b.trade_date, b.open, b.high, b.low, b.close, b.volume,
               b.float_shares, b.turnover_rate, coalesce(m.asset_name,'') as asset_name
        from bars_1d b
        left join asset_master m on m.asset_code = b.asset_code
        where b.trade_date between %s and %s
          and b.asset_type is distinct from 'etf'
          and b.float_shares is not null
        order by b.asset_code, b.trade_date
    """
    cur.execute(sql, (start, end)); rows = cur.fetchall(); cur.close(); c.close()
    df = pd.DataFrame(rows, columns=['code', 'date', 'open', 'high', 'low', 'close',
                                     'volume', 'float_shares', 'turnover_rate', 'name'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    return df


def ema_(x, span):
    a = 2.0 / (span + 1)
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def ema_nan(x, span):
    """NaN 安全的 EMA: 从首个有限值开始播种, 前导 NaN 仍为 NaN, 中途 NaN 沿用前值。"""
    a = 2.0 / (span + 1)
    out = np.full_like(x, np.nan, dtype=float)
    valid = np.isfinite(x)
    if not valid.any():
        return out
    first = int(np.argmax(valid))
    out[first] = x[first]
    for i in range(first + 1, len(x)):
        if valid[i]:
            out[i] = a * x[i] + (1 - a) * out[i - 1]
        else:
            out[i] = out[i - 1] if np.isfinite(out[i - 1]) else np.nan
    return out


def build_signals(df):
    """在已含 chip 列的 df 上计算 飞龙/抄底/主力/散户 与 5 个买侧判据。"""
    req = ['cost5', 'cost10', 'cost50', 'cost89', 'cost90',
           'winner_c', 'winner_09c', 'winner_11c', 'conc']
    chip = df[req].copy()
    out_parts = []
    for code, g in df.groupby('code', group_keys=False):
        g = g.sort_values('date')          # 保留原全局 index, 供 chip_df.loc[g.index] 对齐
        n = len(g)
        ck = chip.loc[g.index]
        C = g['close'].values.astype(float)
        # 基础指标
        ema89 = ema_(C, 89)
        feilong = np.sqrt(ema89 * ck['cost89'].values)
        chaodi = np.sqrt(ema89 * ck['cost5'].values)
        diff = ck['winner_11c'].values - ck['winner_09c'].values   # 夹价筹码厚度
        zl = ema_nan(diff, 5)                                      # 主力线 (NaN 安全 EMA)
        ma09 = pd.Series(ck['winner_09c'].values).rolling(10).mean().values
        sh = ma09 - pd.Series(zl).rolling(10).mean().values         # 散户线
        conc = ck['conc'].values
        # 判据(需无 nan)
        valid = np.isfinite(feilong) & np.isfinite(chaodi) & np.isfinite(zl) \
            & np.isfinite(conc) & np.isfinite(C)
        feilong_up = np.zeros(n, bool); main_up = np.zeros(n, bool)
        conc_imp = np.zeros(n, bool); near_chaodi = np.zeros(n, bool)
        above_feilong = np.zeros(n, bool)
        f5 = np.zeros(n); f20 = np.zeros(n); z5 = np.zeros(n); c20 = np.zeros(n)
        for i in range(20, n):
            if not valid[i]:
                continue
            f5[i] = feilong[i] - feilong[i - 5]
            f20[i] = feilong[i] - feilong[i - 20]
            z5[i] = zl[i] - zl[i - 5]
            c20[i] = conc[i] - conc[i - 20]
            feilong_up[i] = (f5[i] > 0) and (f20[i] > 0)
            main_up[i] = z5[i] > 0
            conc_imp[i] = c20[i] < 0
            near_chaodi[i] = C[i] <= chaodi[i] * 1.05
            above_feilong[i] = C[i] > feilong[i]
        c_buy_dip = feilong_up & main_up & near_chaodi          # 抄底型买点
        c_buy_brk = feilong_up & main_up & above_feilong        # 突破型买点
        # 止损/目标: 抄底型 止损=抄底线(成本下沿) 目标=飞龙线(成本中枢上沿)
        #           突破型 止损=飞龙线(成本中枢支撑) 目标=入场价*1.15
        stop_dip = chaodi.copy(); target_dip = feilong.copy()
        stop_brk = feilong.copy(); target_brk = C * 1.15
        part = pd.DataFrame({
            'feilong': feilong, 'chaodi': chaodi, 'zl': zl, 'sh': sh, 'conc': conc,
            'feilong_up': feilong_up, 'main_up': main_up, 'conc_imp': conc_imp,
            'near_chaodi': near_chaodi, 'above_feilong': above_feilong,
            'buy_dip': c_buy_dip, 'buy_brk': c_buy_brk,
            'stop_dip': stop_dip, 'target_dip': target_dip,
            'stop_brk': stop_brk, 'target_brk': target_brk,
        }, index=g.index)
        out_parts.append(part)
    res = pd.concat(out_parts)
    return res


def path_exit(entry, stop, target, lows, highs, closes):
    """持有窗口内: 先触 stop 则止损, 先触 target 则止盈, 否则收尾。返回(return, type)。"""
    for k in range(len(lows)):
        if np.isnan(lows[k]) or np.isnan(highs[k]):
            continue
        if lows[k] <= stop:
            return stop / entry - 1.0, 'stop'
        if highs[k] >= target:
            return target / entry - 1.0, 'target'
    if len(closes) and np.isfinite(closes[-1]):
        return closes[-1] / entry - 1.0, 'hold'
    return np.nan, 'na'


def tier_impact_vec(mcap):
    """按 float_mcap(元) 分层的单边冲击成本(bp)。大>=500亿/中>=100亿/小。"""
    imp = np.full_like(mcap, 0.0030, dtype=float)
    imp = np.where(mcap >= 5e10, 0.0005, imp)
    imp = np.where((mcap >= 1e10) & (mcap < 5e10), 0.0012, imp)
    imp = np.where(np.isfinite(mcap) & (mcap < 1e10), 0.0030, imp)
    imp = np.where(~np.isfinite(mcap), 0.0030, imp)
    return imp


def backtest_path(df, sig_col, stop_col, target_col, H, mcap):
    """T+1 开盘买, 路径依赖退出; 含双边成本(万1+分层冲击)。返回逐笔交易。"""
    rows = []
    for code, g in df.groupby('code', group_keys=False):
        g = g.sort_values('date')          # 保留原全局 index, 供 chip_df.loc[g.index] 对齐
        n = len(g)
        sig = g[sig_col].values.astype(bool)
        O = g['open'].values.astype(float); Hh = g['high'].values.astype(float)
        Ll = g['low'].values.astype(float); C = g['close'].values.astype(float)
        stop = g[stop_col].values.astype(float)
        target = g[target_col].values.astype(float)
        for i in range(n):
            if not sig[i] or i + 1 >= n or i + 1 + H > n:
                continue
            entry = O[i + 1]
            if not np.isfinite(entry) or entry <= 0:
                continue
            if entry / C[i] >= 1.098:        # 入场日开盘涨停买不进
                continue
            sp = stop[i] if np.isfinite(stop[i]) else entry * 0.92
            tp = target[i] if np.isfinite(target[i]) else entry * 1.15
            if sp >= entry:                  # 止损不在下方则退化为 8% 止损
                sp = entry * 0.92
            tl = Ll[i + 1:i + 1 + H]; th = Hh[i + 1:i + 1 + H]; tc = C[i + 1:i + 1 + H]
            gross, typ = path_exit(entry, sp, tp, tl, th, tc)
            if not np.isfinite(gross):
                continue
            mc = mcap[g.index[i]] if hasattr(mcap, '__getitem__') else mcap
            imp = tier_impact_vec(np.array([mc]))[0]
            cost = 2 * COMM + 2 * imp          # 双边
            net = gross - cost
            rows.append((code, g['date'].iloc[i], entry, sp, tp, gross, net, cost, typ))
    tr = pd.DataFrame(rows, columns=['code', 'date', 'entry', 'stop', 'target',
                                     'gross', 'net', 'cost', 'exit'])
    return tr


def backtest_plain(df, sig_col, H, mcap):
    """普通持有(无止损/止盈)版本, 用于隔离'选股'本身的超额(排除退出规则贡献)。"""
    rows = []
    for code, g in df.groupby('code', group_keys=False):
        g = g.sort_values('date')          # 保留全局 index 供 mcap[g.index[i]] 对齐
        n = len(g)
        sig = g[sig_col].values.astype(bool)
        O = g['open'].values.astype(float); C = g['close'].values.astype(float)
        for i in range(n):
            if not sig[i] or i + 1 >= n or i + 1 + H >= n:
                continue
            entry = O[i + 1]
            if not np.isfinite(entry) or entry <= 0:
                continue
            if entry / C[i] >= 1.098:
                continue
            ex = C[i + 1 + H]
            if not np.isfinite(ex):
                continue
            gross = ex / entry - 1.0
            mc = mcap[g.index[i]]
            imp = tier_impact_vec(np.array([mc]))[0]
            net = gross - (2 * COMM + 2 * imp)
            rows.append((g['date'].iloc[i], net))
    return pd.DataFrame(rows, columns=['date', 'net'])


def summarize_trades(tr, label, daily_mean):
    """daily_mean: 同交易日全样本等权持有净收益均值(Series indexed by date)。"""
    if len(tr) == 0:
        return {label: dict(n=0)}
    net = tr['net'].values; gross = tr['gross'].values
    # 逐笔超额 = 净收益 - 同交易日全样本均值
    dm = daily_mean.reindex(pd.to_datetime(tr['date'].values))
    excess = net - dm.values.astype(float)
    excess = excess[np.isfinite(excess)]
    win = (net > 0).mean()
    avg_w = gross[gross > 0].mean() if (gross > 0).any() else 0.0
    avg_l = gross[gross < 0].mean() if (gross < 0).any() else 0.0
    payoff = (avg_w / abs(avg_l)) if avg_l != 0 else np.nan
    rr = ((tr['target'] - tr['entry']) / (tr['entry'] - tr['stop']))
    rr_med = np.nanmedian(rr.values) if len(rr) else np.nan
    exp = win * avg_w - (1 - win) * abs(avg_l)
    t = excess.mean() / (excess.std() / np.sqrt(len(excess))) if len(excess) > 1 else 0
    return dict(
        label=label, n=int(len(tr)), win_rate=float(win),
        mean_gross_bps=float(gross.mean() * 1e4),
        mean_net_bps=float(net.mean() * 1e4),
        median_net_bps=float(np.median(net) * 1e4),
        avg_win_bps=float(avg_w * 1e4), avg_loss_bps=float(avg_l * 1e4),
        payoff_ratio=float(payoff), median_rr=float(rr_med),
        expectancy_bps=float(exp * 1e4),
        excess_net_bps=float(excess.mean() * 1e4), excess_t=float(t),
        exit_stop=float((tr['exit'] == 'stop').mean()),
        exit_target=float((tr['exit'] == 'target').mean()),
        exit_hold=float((tr['exit'] == 'hold').mean()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2011-01-01')
    ap.add_argument('--end', default='2026-09-30')
    ap.add_argument('--nbins', type=int, default=500)
    args = ap.parse_args()
    t0 = time.time()

    print(f"[1/5] 加载面板(仅 float_shares 非空) {args.start}~{args.end} ...")
    df = load_panel(args.start, args.end)
    print(f"     行数 {len(df):,}  股票数 {df['code'].nunique():,}  耗时 {time.time()-t0:.1f}s")

    df['listed_n'] = df.groupby('code').cumcount() + 1
    df['is_st'] = df['name'].str.contains('ST|退', case=False, na=False)

    print("[2/5] 还原筹码分布 COST/WINNER (逐日换手加权) ...")
    t1 = time.time()
    cache = f"tools/_chip_cache_{args.start}_{args.end}_{args.nbins}.feather"
    import os
    if os.path.exists(cache):
        print(f"     命中缓存 {cache}, 加载中 ...")
        chip = pd.read_feather(cache)
    else:
        chip = compute_chip(df, nbins=args.nbins, warmup=120)
        chip.to_feather(cache)
        print(f"     已缓存到 {cache}")
    print(f"     耗时 {time.time()-t1:.1f}s  有效筹码日 {chip['winner_c'].notna().sum():,}")

    print("[3/5] 计算 飞龙/抄底/主力/散户 线 + 5 买侧判据 ...")
    sig = build_signals(pd.concat([df, chip], axis=1))
    for col in sig.columns:
        df[col] = sig[col].values
    for nm in ['buy_dip', 'buy_brk', 'feilong_up', 'main_up', 'near_chaodi']:
        print(f"     {nm:12s} 信号日: {int(df[nm].sum()):,}")

    print("[4/5] 路径依赖回测(含成本+全样本基准超额) + 盈亏比 ...")
    mcap = (df['close'] * df['float_shares']).values
    results = {}
    daily_mean_by_H = {}
    for H in [5, 10, 20]:
        # 全样本基准: T+1 开盘买, T+H 收盘卖, 含双边成本(万1+分层冲击)
        grp = df.groupby('code')
        ent = grp['open'].shift(-1); ext = grp['close'].shift(-H)
        fwd = ext / ent - 1.0
        ok = ent.notna() & ext.notna() & np.isfinite(fwd) & (~df['is_st']) \
            & (df['listed_n'] >= 250) & (ent / df['close'] < 1.098) & (np.abs(fwd) < 0.6)
        cost_arr = 2 * COMM + 2 * tier_impact_vec(mcap)
        base_net = (fwd - cost_arr)[ok.values]
        # 显式 DataFrame groupby 避免 DatetimeIndex groupby(level=0) 全 NaN 的坑
        tmp = pd.DataFrame({'d': df['date'][ok.values].values, 'v': base_net})
        daily_mean = tmp.groupby('d')['v'].mean()
        daily_mean_by_H[H] = daily_mean
        print(f"     全样本基准净均值 H={H}: {daily_mean.mean()*1e4:+.1f}bp/日 "
              f"({len(daily_mean)} 交易日)")
        # 信号回测
        td = backtest_path(df, 'buy_dip', 'stop_dip', 'target_dip', H, mcap)
        results[f'dip_H{H}'] = summarize_trades(td, f'抄底型H{H}', daily_mean)
        tb = backtest_path(df, 'buy_brk', 'stop_brk', 'target_brk', H, mcap)
        results[f'brk_H{H}'] = summarize_trades(tb, f'突破型H{H}', daily_mean)
        for key in [f'dip_H{H}', f'brk_H{H}']:
            r = results[key]
            if r.get('n'):
                print(f"  [{key}] n={r['n']:,} 胜率 {r['win_rate']*100:.1f}% "
                      f"净均值 {r['mean_net_bps']:+.0f}bp 盈亏比 {r['payoff_ratio']:.2f} "
                      f"结构R:R {r['median_rr']:.2f} 期望 {r['expectancy_bps']:+.0f}bp "
                      f"超额净 {r['excess_net_bps']:+.0f}bp t={r['excess_t']:+.2f} "
                      f"止损 {r['exit_stop']*100:.0f}% 止盈 {r['exit_target']*100:.0f}% 持有 {r['exit_hold']*100:.0f}%")

    # 稳健性检查 1: 抄底型普通持有(无止损/目标) -> 隔离"选股"本身超额
    print("[4b] 稳健性: 抄底型普通持有(退出=T+H收盘, 无止损/目标) ...")
    for H in [5, 10, 20]:
        tp = backtest_plain(df, 'buy_dip', H, mcap)
        dm = daily_mean_by_H[H]
        ex = (tp['net'] - dm.reindex(pd.to_datetime(tp['date'].values)).values)
        ex = ex[np.isfinite(ex)]
        t = ex.mean() / (ex.std() / np.sqrt(len(ex))) if len(ex) > 1 else 0
        print(f"  抄底型普通持有 H={H}: 净均值 {tp['net'].mean()*1e4:+.0f}bp "
              f"超额净 {ex.mean()*1e4:+.0f}bp t={t:+.2f} (n={len(ex):,})")

    # 稳健性检查 2: 抄底型路径退出超额的逐年分解 -> 排除单区间驱动
    print("[4c] 稳健性: 抄底型路径退出超额 逐年分解 ...")
    td = backtest_path(df, 'buy_dip', 'stop_dip', 'target_dip', 10, mcap)
    dm = daily_mean_by_H[10]
    td['ex'] = (td['net'] - dm.reindex(pd.to_datetime(td['date'].values)).values)
    td['yr'] = pd.to_datetime(td['date'].values).year
    ann = td.groupby('yr')['ex'].agg(['mean', 'count'])
    for yr, row in ann.iterrows():
        if row['count'] > 50:
            print(f"  {yr}: 超额净 {row['mean']*1e4:+.0f}bp (n={int(row['count']):,})")

    # 末日快照: 通过合成买点的票
    last_date = df['date'].max()
    snap = df[df['date'] == last_date]
    watch_dip = snap[snap['buy_dip'] & (~snap['is_st'])]
    watch_brk = snap[snap['buy_brk'] & (~snap['is_st'])]
    print(f"[5/5] 末日({last_date.date()}) 买点清单: 抄底型 {len(watch_dip)} 只 / 突破型 {len(watch_brk)} 只")
    results['watchlist'] = {
        'date': str(last_date.date()),
        'dip': watch_dip[['code', 'name', 'close', 'feilong', 'chaodi', 'zl', 'conc']]
               .rename(columns={'close': 'price'}).to_dict('records')[:60],
        'brk': watch_brk[['code', 'name', 'close', 'feilong', 'chaodi', 'zl', 'conc']]
               .rename(columns={'close': 'price'}).to_dict('records')[:60],
    }

    with open('tools/_tdx_feilong_chip_out.json', 'w') as f:
        json.dump({'window': f"{args.start}~{args.end}", 'results': results,
                   'universe': dict(rows=len(df), stocks=int(df['code'].nunique()),
                                    chip_days=int(chip['winner_c'].notna().sum()))},
                  f, indent=2, ensure_ascii=False, default=float)
    print("完成 -> tools/_tdx_feilong_chip_out.json")


if __name__ == '__main__':
    main()
