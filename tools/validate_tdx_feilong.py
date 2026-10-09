"""
TDX 飞龙/战法2(涨停回调) 指标验证 —— 全 A 真实面板 (TimescaleDB bars_1d)
============================================================================
用户给的通达信公式包含几类信号:
  A. 飞龙线/抄底线 = SQRT(EMA(C,89)*COST(89/5))  ← 依赖 COST() 筹码分布
  B. 主力/散户线   = WINNER() 筹码分布            ← 依赖 WINNER() 筹码分布
  C. 涨停回调买点 A1-A8 (战法2核心)               ← 纯 OHLCV, 可回测
  D. 凰线/凤线突破 BU/SEL                         ← 纯 OHLCV, 可回测
  E. AA/BB 动量出击 CROSS(AA,BB)                  ← 纯 OHLCV, 可回测

A/B 两类的 COST()/WINNER() 是通达信筹码分布函数, 需要逐日换手加权持仓成本,
本库 bars_1d 仅有 OHLCV(源自 QMT datadir 回填), 无此数据, 无法忠实复现。
(且飞龙因子此前已在全 A 面板证伪, 见 docs/factor_scorecard.md)。

本脚本只回测 C/D/E 三类纯 OHLCV 信号, 严格对齐:
  - 信号判定只用 T 日及之前收盘数据 (无前视)
  - 买入价 = T+1 开盘 (A股 T+1)
  - 持有 H 日, 退出价 = T+H 收盘
  - 成本 = 万1 双边 + 按市值分层冲击
  - 过滤: 剔次新(<250日) / ST / 涨停买不进(open/昨收>=1.098)
  - 变体: 带 8% 止损 (回应 dixi 终审"缺止损是死因")

数据铁律: 只用 TimescaleDB(源自 QMT datadir 回填), 绝不碰 akshare。
"""
import argparse, json
import numpy as np
import pandas as pd
import psycopg2

DB = dict(host='127.0.0.1', port=5432, dbname='quant_data', user='quant', password='quant123456')
COMM = 0.0001  # 万1 双边佣金


def load_panel(start, end):
    c = psycopg2.connect(**DB); cur = c.cursor()
    sql = """
        select b.asset_code, b.trade_date, b.open, b.high, b.low, b.close, b.volume,
               b.float_shares, coalesce(m.asset_name,'') as asset_name
        from bars_1d b
        left join asset_master m on m.asset_code = b.asset_code
        where b.trade_date between %s and %s
          and b.asset_type is distinct from 'etf'
        order by b.asset_code, b.trade_date
    """
    cur.execute(sql, (start, end)); rows = cur.fetchall(); cur.close(); c.close()
    df = pd.DataFrame(rows, columns=['code', 'date', 'open', 'high', 'low', 'close',
                                     'volume', 'float_shares', 'name'])
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


def sig_for_group(g):
    """单只股票内计算全部信号 (无跨股票泄漏)。返回与原 index 对齐的 DataFrame。"""
    g = g.sort_values('date').reset_index(drop=True)
    n = len(g)
    if n < 60:
        return pd.DataFrame({'zt': np.zeros(n, bool), 'A1': np.full(n, np.nan),
                              'buy_setup': np.zeros(n, bool), 'bu_phoenix': np.zeros(n, bool),
                              'chongji': np.zeros(n, bool)}, index=g.index)
    C = g['close'].values.astype(float); H = g['high'].values.astype(float)
    L = g['low'].values.astype(float); O = g['open'].values.astype(float)
    V = g['volume'].values.astype(float)
    prevC = np.roll(C, 1); prevC[0] = np.nan
    zt = (C / prevC - 1.0) * 100 > 9.5
    # BARSLAST(zt)
    idx = np.arange(n)
    zt_idx = np.where(zt, idx, np.nan)
    last_zt = np.fmax.accumulate(np.where(np.isnan(zt_idx), np.nan, zt_idx))
    A1 = idx - last_zt
    A1[np.isnan(last_zt)] = np.nan
    # MACD
    e12 = ema_(C, 12); e26 = ema_(C, 26); diff = e12 - e26
    dea = ema_(diff, 9); macd = 2 * (diff - dea)
    # 凰线 / 凤线
    xopen = (np.roll(O, 1) + np.roll(C, 1)) / 2.0; xopen[0] = np.nan
    xhigh = np.maximum(H, xopen); xlow = np.minimum(L, xopen)
    xhl = xhigh - xlow
    volality = np.full(n, np.nan)
    for i in range(8, n):
        volality[i] = xhl[i - 7:i + 1].mean()
    ma5 = np.full(n, np.nan)
    for i in range(5, n):
        ma5[i] = C[i - 4:i + 1].mean()
    huang = ma5 + volality / 2.0
    feng = ma5 - volality / 2.0
    bu_phoenix = np.zeros(n, bool)
    for i in range(1, n):
        if not np.isnan(huang[i]) and not np.isnan(huang[i - 1]):
            if C[i] > huang[i] and C[i - 1] <= huang[i - 1]:
                bu_phoenix[i] = True
    # AA/BB 出击
    jj = (C + H + L) / 3.0
    aa = ema_(jj, 20); bb = np.roll(aa, 1); bb[0] = np.nan
    chongji = np.zeros(n, bool)
    for i in range(1, n):
        if not np.isnan(aa[i]) and not np.isnan(aa[i - 1]):
            if aa[i] > bb[i] and aa[i - 1] <= bb[i - 1]:
                chongji[i] = True
    # 涨停回调 A1-A8 买点
    buy = np.zeros(n, bool)
    vma5 = np.full(n, np.nan)
    for i in range(5, n):
        vma5[i] = V[i - 4:i + 1].mean()
    vprev = np.roll(V, 1); vprev[0] = np.nan
    valid = (A1 >= 2) & (A1 <= 13) & ~np.isnan(A1)
    vi = np.where(valid)[0]
    if len(vi) > 0:
        kv = A1[vi].astype(int)
        refL = L[vi - kv]; refH = H[vi - kv - 1]
        refC = C[vi - kv - 1]; refO = O[vi - kv]
        A2 = np.where(refL > refH, refC, refO)
        cA = C[vi]; lA = L[vi]; oA = O[vi]; vA = V[vi]; mcd = macd[vi]
        A4 = (cA >= A2) & (lA / A2 >= 0.97) & (lA / A2 <= 1.035)
        A5 = (vA < vprev[vi]) & (vA < vma5[vi])
        A6 = np.abs(cA / oA - 1.0) * 100 < 5
        LLVc = np.array([C[i - kv[j] + 1:i + 1].min() for j, i in enumerate(vi)])
        A7 = LLVc >= A2
        A8 = mcd > -1
        buy[vi] = A4 & A5 & A6 & A7 & A8
    return pd.DataFrame({'zt': zt, 'A1': A1, 'buy_setup': buy,
                         'bu_phoenix': bu_phoenix, 'chongji': chongji}, index=g.index)


def backtest(df, mask, H, stop=False, sl=0.08):
    """给定信号 mask(判定于 T 日), 计算 T+1 开盘买入、T+H 收盘退出的收益与统计。"""
    grp = df.groupby('code')
    ent = grp['open'].shift(-1)                 # T+1 开盘 (买入价)
    ext = grp['close'].shift(-H)               # T+H 收盘 (退出价)
    fwd = ext / ent - 1.0
    if stop:
        # T+1..T+H 内最低收盘, 触及 (1-sl) 即止损
        cmin = grp['close'].transform(lambda s: s[::-1].rolling(H).min()[::-1]).shift(-1)
        stopped = cmin <= ent * (1 - sl)
        fwd = np.where(stopped.fillna(False), -sl, fwd)
    # 入场有效性
    entry_limit_up = ent / df['close'] >= 1.098   # 入场日开盘涨停买不进
    ok = (mask.values if hasattr(mask, 'values') else mask) & ent.notna() & ext.notna() \
        & np.isfinite(fwd) & (~df['is_st'].values) & (df['listed_n'].values >= 250) \
        & (~entry_limit_up.values) & (np.abs(fwd) < 0.6)
    ok = pd.Series(ok, index=df.index)
    sub = df[ok].copy()
    sub['fwd'] = fwd[ok]
    # 同日全样本基准 (所有有效股同 H 收益)
    base_ok = ent.notna() & ext.notna() & np.isfinite(fwd) & (~df['is_st'].values) \
        & (df['listed_n'].values >= 250) & (np.abs(fwd) < 0.6)
    base_ok = pd.Series(base_ok, index=df.index)
    base = df[base_ok].copy()
    base['fwd'] = fwd[base_ok]
    # 按信号日聚合超额
    sig_day = sub.groupby('date')['fwd'].mean()
    uni_day = base.groupby('date')['fwd'].mean()
    both = pd.concat([sig_day.rename('sig'), uni_day.rename('uni')], axis=1).dropna()
    excess = both['sig'] - both['uni']
    # NAV (复利每日超额)
    nav = (1 + excess).cumprod()
    peak = nav.cummax()
    maxdd = (nav / peak - 1).min()
    ann = (1 + excess.mean()) ** 252 - 1
    vol = excess.std() * np.sqrt(252)
    sharpe = ann / vol if vol > 0 else 0
    t = excess.mean() / (excess.std() / np.sqrt(len(excess))) if len(excess) > 1 else 0
    return {
        'n_trades': int(ok.sum()),
        'mean_bps': float(sub['fwd'].mean() * 1e4),
        'median_bps': float(sub['fwd'].median() * 1e4),
        'win_rate': float((sub['fwd'] > 0).mean()),
        'universe_bps': float(base['fwd'].mean() * 1e4),
        'excess_bps': float(excess.mean() * 1e4),
        'excess_t': float(t),
        'excess_ann': float(ann),
        'excess_sharpe': float(sharpe),
        'excess_maxdd': float(maxdd),
        'n_days': int(len(excess)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2014-01-02')
    ap.add_argument('--end', default='2026-09-30')
    args = ap.parse_args()

    print(f"[1/4] 加载面板 {args.start}~{args.end} ...")
    df = load_panel(args.start, args.end)
    print(f"     行数 {len(df):,}  股票数 {df['code'].nunique():,}")

    df['listed_n'] = df.groupby('code').cumcount() + 1
    df['prev_close'] = df.groupby('code')['close'].shift(1)
    df['is_st'] = df['name'].str.contains('ST|退', case=False, na=False)
    df['float_mcap'] = df.groupby('code')['close'].transform(
        lambda s: (s * 1).ffill(limit=5))
    # 市值代理: close * float_shares
    df['float_mcap'] = (df['close'] * df['float_shares'])
    df['float_mcap'] = df.groupby('code')['float_mcap'].transform(lambda s: s.ffill(limit=5))

    print("[2/4] 计算信号 (纯 OHLCV) ...")
    sig = df.groupby('code', group_keys=False).apply(sig_for_group)
    for col in ['zt', 'A1', 'buy_setup', 'bu_phoenix', 'chongji']:
        df[col] = sig[col].values
    print(f"     涨停回调买点 A1-A8 信号日: {int(df['buy_setup'].sum()):,}")
    print(f"     凰线突破 BU 信号日     : {int(df['bu_phoenix'].sum()):,}")
    print(f"     AA/BB 动量出击 信号日   : {int(df['chongji'].sum()):,}")

    print("[3/4] 回测 (T+1 开盘买, 万1+冲击, 剔次新/ST/涨停买不进) ...")
    results = {}
    for name, col in [('涨停回调A1-A8', 'buy_setup'),
                      ('凰线突破BU', 'bu_phoenix'),
                      ('AA/BB动量出击', 'chongji')]:
        results[name] = {}
        for H in [5, 10, 20]:
            r = backtest(df, df[col], H, stop=False)
            results[name][f'H{H}'] = r
            print(f"  [{name}] H={H}: 交易 {r['n_trades']:,} | 均值 {r['mean_bps']:+.0f}bp "
                  f"胜率 {r['win_rate']*100:.1f}% | 超额 {r['excess_bps']:+.0f}bp "
                  f"t={r['excess_t']:+.2f} | 超额年化 {r['excess_ann']*100:+.1f}% "
                  f"夏普 {r['excess_sharpe']:.2f} 回撤 {r['excess_maxdd']*100:.1f}%")
        # 带 8% 止损 (回应 dixi 终审)
        r = backtest(df, df[col], 10, stop=True, sl=0.08)
        results[name]['H10_stop8'] = r
        print(f"  [{name}] H=10+8%止损: 均值 {r['mean_bps']:+.0f}bp 胜率 {r['win_rate']*100:.1f}% "
              f"超额 {r['excess_bps']:+.0f}bp t={r['excess_t']:+.2f}")

    print("[4/4] 完成 -> tools/_tdx_feilong_out.json")
    with open('tools/_tdx_feilong_out.json', 'w') as f:
        json.dump({'window': f"{args.start}~{args.end}", 'results': results},
                  f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
