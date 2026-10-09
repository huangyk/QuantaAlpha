"""
单票稽查: 源飞宠物(001222) 在 2026-09-17 是否触发我们的飞龙/抄底/主力买点,
以及之后真实涨跌。复用 tools/chip_distribution.py + validate_tdx_feilong_chip.py 的同一套逻辑。
"""
import sys, psycopg2, numpy as np, pandas as pd
sys.path.insert(0, r'D:/qqmt/QuantaAlpha/tools')
from chip_distribution import compute_chip
from validate_tdx_feilong_chip import build_signals, DB
from validate_tdx_feilong import sig_for_group

SIGNAL_DATE = pd.Timestamp('2026-09-17')
NAME = '源飞宠物'

# 1) 找 code
con = psycopg2.connect(**DB); cur = con.cursor()
cur.execute("SELECT asset_code, asset_name, list_date, is_st FROM asset_master WHERE asset_name LIKE %s", (f'%{NAME}%',))
master = cur.fetchall()
print('asset_master 命中:', master)
code = master[0][0]

# 2) 取该票宽窗口(保证筹码 warmup=120 天有数据)
cur.execute("""
    select b.asset_code, b.trade_date, b.open, b.high, b.low, b.close, b.volume,
           b.float_shares, b.turnover_rate
    from bars_1d b
    where b.asset_code = %s and b.trade_date between %s and %s
    order by b.trade_date
""", (code, '2024-01-01', '2026-09-30'))
rows = cur.fetchall(); cur.close(); con.close()
df = pd.DataFrame(rows, columns=['code','date','open','high','low','close','volume','float_shares','turnover_rate'])
df['date'] = pd.to_datetime(df['date'])
print(f'\n{code} {NAME}: 取到 {len(df)} 行, {df.date.min().date()}~{df.date.max().date()}')
print(f'  float_shares 非空: {df.float_shares.notna().sum()}, turnover_rate 非空: {df.turnover_rate.notna().sum()}')

# 3) 筹码 + 信号
chip = compute_chip(df, nbins=500, warmup=120)
sig = build_signals(pd.concat([df, chip], axis=1))
for col in sig.columns:
    df[col] = sig[col].values
df = df.sort_values('date').reset_index(drop=True)

# 4) 9/17 当天状态
# 4b) OHLCV-only 信号(涨停回调 A1-A8 / 凰线突破 / AA-BB 出击)
oh = sig_for_group(df)
for col in ['zt', 'A1', 'buy_setup', 'bu_phoenix', 'chongji']:
    df[col] = oh[col].values

mask = df['date'] == SIGNAL_DATE
if mask.any():
    r = df[mask].iloc[0]
    print(f'\n===== {SIGNAL_DATE.date()} 当天信号状态 =====')
    print(f'  收盘 C={r.close:.2f}  开盘 O={r.open:.2f}  高={r.high:.2f}  低={r.low:.2f}')
    print(f'  飞龙线 feilong={r.feilong:.3f}  抄底线 chaodi={r.chaodi:.3f}  '
          f'主力线 zl={r.zl:.4f}  散户线 sh={r.sh:.4f}  集中度 conc={r.conc:.4f}')
    print(f'  判据: 飞龙向上 feilong_up={bool(r.feilong_up)}  主力向上 main_up={bool(r.main_up)}  '
          f'筹码更集中 conc_imp={bool(r.conc_imp)}')
    print(f'        贴近抄底 near_chaodi={bool(r.near_chaodi)}(C<=chaodi*1.05?)  '
          f'站上飞龙 above_feilong={bool(r.above_feilong)}')
    print(f'  >>> 抄底型买点 buy_dip = {bool(r.buy_dip)}    突破型买点 buy_brk = {bool(r.buy_brk)}')
    print(f'  [OHLCV-only] 涨停 zt={bool(r.zt)}  A1(距涨停天数)={r.A1 if np.isfinite(r.A1) else "NA"}  '
          f'涨停回调买点 buy_setup={bool(r.buy_setup)}')
    print(f'             凰线突破 bu_phoenix={bool(r.bu_phoenix)}  AA-BB出击 chongji={bool(r.chongji)}')
else:
    print(f'\n⚠️ {SIGNAL_DATE.date()} 不在数据范围内/非交易日')

# 5) 最近一次信号日(前后 60 天)
sub = df[(df['date'] >= SIGNAL_DATE - pd.Timedelta(days=60)) & (df['date'] <= SIGNAL_DATE + pd.Timedelta(days=20))]
last_dip = sub[sub['buy_dip']]['date']
last_brk = sub[sub['buy_brk']]['date']
print('\n9/17 前后 60 天内 信号触发日:')
print('  抄底型:', [d.date().isoformat() for d in last_dip.tolist()] or '无')
print('  突破型:', [d.date().isoformat() for d in last_brk.tolist()] or '无')

# 6) 9/17 之后真实涨跌 (T+1 开盘买, 看后续收盘)
idx = df.index[df['date'] == SIGNAL_DATE]
if len(idx):
    i = idx[0]
    c0 = df['close'].iloc[i]
    o1 = df['open'].iloc[i+1] if i+1 < len(df) else np.nan
    print(f'\n===== 9/17 之后真实涨跌 (基准收盘 C0={c0:.2f}, T+1开盘={o1:.2f}) =====')
    print(f'  {"交易日":<12}{"收盘":>8}{"较C0%":>9}{"较T+1开%":>10}')
    for k in range(1, min(14, len(df)-i)):
        rr = df['close'].iloc[i+k]
        d = df['date'].iloc[i+k]
        p0 = (rr/c0 - 1)*100
        p1 = (rr/o1 - 1)*100 if np.isfinite(o1) else float('nan')
        print(f'  {d.date().isoformat():<12}{rr:>8.2f}{p0:>8.1f}%{p1:>9.1f}%')

# 7) 9月整月信号列一览(看清脉络)
sep = df[df['date'].dt.strftime('%Y-%m') == '2026-09']
print('\n===== 2026-09 每日关键列 =====')
print(f'  {"日期":<11}{"收盘":>8}{"飞龙":>8}{"抄底":>8}{"主力":>7}{"集中度":>8}'
      f'{"飞龙↑":>6}{"主力↑":>6}{"近抄底":>7}{"上飞龙":>7}{"买抄底":>7}{"买突破":>7}')
for _, r in sep.iterrows():
    fu = '' if not np.isfinite(r.feilong_up) else ('Y' if r.feilong_up else '.')
    mu = '' if not np.isfinite(r.main_up) else ('Y' if r.main_up else '.')
    print(f'  {r.date.date().isoformat():<11}{r.close:>8.2f}{r.feilong:>8.3f}{r.chaodi:>8.3f}'
          f'{r.zl:>7.4f}{r.conc:>8.4f}{fu:>6}{mu:>6}'
          f'{"Y" if r.near_chaodi else ".":>7}{"Y" if r.above_feilong else ".":>7}'
          f'{"Y" if r.buy_dip else ".":>7}{"Y" if r.buy_brk else ".":>7}')
