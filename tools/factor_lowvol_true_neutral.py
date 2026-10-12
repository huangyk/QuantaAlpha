"""
低波动异象的「真中性化」研究 —— 全 A 真实面板 (TimescaleDB bars_1d)
=========================================================================
动机: factor_neutral_test.py 只做了市值中性化, 得到 IC +0.0332/t=6.17,
判定低波动是「真 alpha 信号但单多亏钱、须多空、A股做空受限难变现」。
但没做行业中性化 —— 低波动股可能天然聚集在银行/公用事业等防御行业,
若行业中性化后 IC 消失, 说明它只是行业偏见而非个股 alpha。

本脚本在真实全 A 面板上补齐「真中性化」:
  - 信号: 低波动 signal = -vol20 (20日已实现波动率)
  - 收益: T+1 开盘收益 open_{t+1}/open_t - 1 (无前视)
  - 过滤: 次新(<250日) / 涨停买不进(open/昨收>=1.098) / 跌停卖不出 /
          ST(asset_master.is_st) / 停牌近似 / |收益|>50%
  - 中性化三层:
      (1) raw          = 横截面 -rank(vol20)
      (2) size-neutral  = raw 对 size_rank 横截面回归残差
      (3) ind+size-neu  = 先按行业去均值, 再行业内对 size_rank 去残差
                          (Frisch-Waugh-Lovell 等价于 行业哑变量+size 联合回归)
  - 组合: 每日十分位; 多空(第1组多/第10组空) 日频+月频;
          另测「可实盘 long-only」: 仅在流动盘(市值前50%)内取低波动组, 月频再平衡
  - 成本: 万1 双边 + 冲击(按 float_mcap 分层单边 30/12/5bp), 1x/2x/3x 压力

数据铁律: 只用 TimescaleDB(源自 QMT datadir 回填), 绝不碰 akshare。
"""
import argparse, json, sys
import numpy as np
import pandas as pd
import psycopg2

DB = dict(host='127.0.0.1', port=5432, dbname='quant_data', user='quant', password='quant123456')
COMM = 0.0001  # 万1 双边佣金


def load_panel(start, end):
    c = psycopg2.connect(**DB)
    cur = c.cursor()
    sql = """
        select b.asset_code, b.trade_date, b.open, b.close, b.float_shares,
               coalesce(m.is_st, false) as is_st
        from bars_1d b
        left join asset_master m on m.asset_code = b.asset_code
        where b.trade_date between %s and %s
          and (b.asset_type is distinct from 'etf')
        order by b.asset_code, b.trade_date
    """
    cur.execute(sql, (start, end))
    rows = cur.fetchall()
    cur.close(); c.close()
    df = pd.DataFrame(rows, columns=['code', 'date', 'open', 'close', 'float_shares', 'is_st'])
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values(['code', 'date']).reset_index(drop=True)


def load_industry():
    """股票->行业(GICS) 当前映射(静态标签, 行业重分类极少)。"""
    c = psycopg2.connect(**DB)
    cur = c.cursor()
    cur.execute("""
        select sm.asset_code, sm.sector_code
        from sector_membership sm
        join sector_master s on s.sector_code = sm.sector_code
        where s.sector_type = 'industry' and sm.end_date is null
    """)
    m = {r[0]: r[1] for r in cur.fetchall()}
    cur.close(); c.close()
    return m


def xsec_rank(s, dates):
    return s.groupby(dates).rank(pct=True)


def daily_ic(panel, sigcol):
    ic = panel.groupby('date').apply(
        lambda g: g[sigcol].corr(g['tgt_rank']) if len(g) > 10 else np.nan)
    ic = ic.dropna()
    if len(ic) < 2 or ic.std() == 0:
        return ic.mean(), 0.0, ic
    t = ic.mean() / (ic.std() / np.sqrt(len(ic)))
    return ic.mean(), t, ic


def stats(rets):
    r = np.array(rets, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return 0, 0, 0, 0, 0
    ann = (1 + r.mean()) ** 252 - 1
    vol = r.std() * np.sqrt(252)
    sharpe = ann / vol if vol > 0 else 0
    nav = np.cumprod(1 + r)
    peak = np.maximum.accumulate(nav)
    maxdd = (nav / peak - 1).min()
    return ann, vol, sharpe, maxdd, len(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2011-01-04')
    ap.add_argument('--end', default='2026-09-30')
    ap.add_argument('--vol-win', type=int, default=20)
    args = ap.parse_args()

    print(f"[1/6] 加载面板 {args.start}~{args.end} ...")
    df = load_panel(args.start, args.end)
    print(f"     行数 {len(df):,}  股票数 {df['code'].nunique():,}")
    ind = load_industry()
    print(f"     行业映射覆盖 {len(ind):,} 只")

    grp = df.groupby('code')
    df['listed_n'] = grp.cumcount() + 1
    df['prev_close'] = grp['close'].shift(1)
    df['ret_fwd'] = grp['open'].shift(-1) / df['open'] - 1.0
    df['logret'] = np.log(grp['close'].transform(lambda s: s / s.shift(1)))
    df['vol20'] = grp['logret'].transform(
        lambda s: s.rolling(args.vol_win, min_periods=args.vol_win).std())
    df['float_mcap'] = df['close'] * df['float_shares']
    # 指标 ffill(limit=5) 防停牌放大; 收益用原始
    df['vol20'] = grp['vol20'].transform(lambda s: s.ffill(limit=5))
    df['float_mcap'] = grp['float_mcap'].transform(lambda s: s.ffill(limit=5))
    df['limit_up'] = (df['open'] / df['prev_close'] >= 1.098)
    df['limit_dn'] = (df['open'] / df['prev_close'] <= 0.902)
    df['industry'] = df['code'].map(ind)

    valid = (df['listed_n'] >= 250) & (~df['limit_up']) & (~df['limit_dn']) & (~df['is_st']) \
        & df['vol20'].notna() & df['float_mcap'].notna() & df['ret_fwd'].notna() \
        & (df['open'] > 0) & (df['ret_fwd'].abs() < 0.5)
    panel = df[valid].copy()
    print(f"     有效样本 {len(panel):,} (剔除次新/涨跌停/ST/缺失)")

    panel['sig_raw'] = -xsec_rank(panel['vol20'], panel['date'])     # 低波动=高信号
    panel['size_rank'] = xsec_rank(panel['float_mcap'], panel['date'])
    panel['tgt_rank'] = xsec_rank(panel['ret_fwd'], panel['date'])

    # (2) size-neutral
    resid = []
    for d, g in panel.groupby('date'):
        x = g['size_rank'].values; y = g['sig_raw'].values
        X = np.column_stack([np.ones_like(x), x])
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ beta
        except Exception:
            r = y
        resid.append(pd.Series(r, index=g.index))
    panel['sig_sz'] = pd.concat(resid)

    # (3) ind+size-neutral: 先行业去均值, 再行业内对 size 去残差 (FWL)
    panel['sig_raw_dm'] = panel.groupby(['date', 'industry'])['sig_raw'].transform(lambda s: s - s.mean())
    panel['size_dm'] = panel.groupby(['date', 'industry'])['size_rank'].transform(lambda s: s - s.mean())
    resid2 = []
    for d, g in panel.groupby('date'):
        x = g['size_dm'].values; y = g['sig_raw_dm'].values
        X = np.column_stack([np.ones_like(x), x])
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ beta
        except Exception:
            r = y
        resid2.append(pd.Series(r, index=g.index))
    panel['sig_ind'] = pd.concat(resid2)

    print("[2/6] IC 检验 ...")
    ic_raw, t_raw, _ = daily_ic(panel, 'sig_raw')
    ic_sz, t_sz, _ = daily_ic(panel, 'sig_sz')
    # ind+size 只用有行业标签的样本
    pm = panel[panel['industry'].notna()]
    ic_ind, t_ind, _ = daily_ic(pm, 'sig_ind')
    print(f"     原始 IC          : {ic_raw:+.4f}  t={t_raw:+.2f}")
    print(f"     市值中性 IC      : {ic_sz:+.4f}  t={t_sz:+.2f}")
    print(f"     行业+市值中性 IC : {ic_ind:+.4f}  t={t_ind:+.2f}  (样本 {len(pm):,})")

    # 组合: 冲击分层
    q20 = panel['float_mcap'].quantile(0.2); q80 = panel['float_mcap'].quantile(0.8)
    def impact(mcap):
        if mcap < q20: return 0.0030
        elif mcap < q80: return 0.0012
        return 0.0005

    print("[3/6] 多空组合 (日频再平衡, T+1开盘) ...")
    # 约定: sig_ind 越高 = 波动率越低 = 应做多。
    dates = sorted(panel['date'].unique())
    ls_d, ls_broad_d, lo_liq_d, ls_rw_d = [], [], [], []   # 极端/宽口/流动盘极端/排名加权
    for d in dates:
        g = panel[panel['date'] == d]
        if len(g) < 50: continue
        sr = g['sig_ind'].rank(pct=True)             # 高 sig = 低波动
        # 极端: 最低波动十分位做多 / 最高波动十分位做空
        ls_d.append(g[sr >= 0.9]['ret_fwd'].mean() - g[sr <= 0.1]['ret_fwd'].mean() - 2 * COMM)
        # 宽口: 低波动半(后50%)做多 / 高波动半(前50%)做空 —— 捕捉单调关系
        ls_broad_d.append(g[sr >= 0.5]['ret_fwd'].mean() - g[sr < 0.5]['ret_fwd'].mean() - 2 * COMM)
        # 排名加权多空: 权重=(rank_pct-0.5), 直接把 rank IC 转成零成本组合收益
        # 这是"正 rank IC 是否有经济 payoff"的最直接检验
        w = (sr - 0.5)
        ls_rw_d.append((w * g['ret_fwd']).sum() / len(g) - 2 * COMM)
        # 流动盘极端: 市值前50% 内最低波动十分位
        liq = g[g['float_mcap'] >= g['float_mcap'].median()]
        if len(liq) >= 20:
            srl = liq['sig_ind'].rank(pct=True)
            lo_liq_d.append(liq[srl >= 0.9]['ret_fwd'].mean()
                            - COMM - impact(liq[srl >= 0.9]['float_mcap'].median()))

    # 月频再平衡: 宽口多空 + 大盘低波动多头(可实盘, 不用做空)
    print("[4/6] 多空组合 (月频再平衡) ...")
    panel['ym'] = panel['date'].dt.to_period('M')
    ls_m, lo_large_m = [], []
    for ym, g in panel.groupby('ym'):
        d0 = g['date'].min()
        g0 = g[g['date'] == d0]
        if len(g0) < 50: continue
        sr0 = g0['sig_ind'].rank(pct=True)
        # 宽口月频
        long_set = set(g0[sr0 >= 0.5]['code']); short_set = set(g0[sr0 < 0.5]['code'])
        ls_m.append(g[g['code'].isin(long_set)]['ret_fwd'].mean() - g[g['code'].isin(short_set)]['ret_fwd'].mean() - 2 * COMM)
        # 大盘低波动多头: 流动性前300 中取低波动半, 月频持有(可实盘)
        top = g0.nlargest(300, 'float_mcap')
        if len(top) >= 50:
            strk = top['sig_ind'].rank(pct=True)
            long_set2 = set(top[strk >= 0.5]['code'])
            lo_large_m.append(g[g['code'].isin(long_set2)]['ret_fwd'].mean() - COMM - 0.0005)

    print("[5/6] 成本敏感 (宽口多空日频, 1x/2x/3x) ...")
    def lsb_with_cost(mult):
        out = []
        for d in dates:
            g = panel[panel['date'] == d]
            if len(g) < 50: continue
            sr = g['sig_ind'].rank(pct=True)
            out.append(g[sr >= 0.5]['ret_fwd'].mean() - g[sr < 0.5]['ret_fwd'].mean() - 2 * COMM * mult)
        return stats(out)
    s1 = lsb_with_cost(1); s2 = lsb_with_cost(2); s3 = lsb_with_cost(3)

    def pct(x): return f"{x*100:+.1f}%"
    ls_s = stats(ls_d); lsb_s = stats(ls_broad_d); lo_s = stats(lo_liq_d)
    lsrw_s = stats(ls_rw_d); lsm_s = stats(ls_m); lol_s = stats(lo_large_m)
    print(f"     极端多空日频 : 年化 {pct(ls_s[0])} 夏普 {ls_s[2]:.2f} 回撤 {pct(ls_s[3])} 样本日 {ls_s[4]}")
    print(f"     宽口多空日频 : 年化 {pct(lsb_s[0])} 夏普 {lsb_s[2]:.2f} 回撤 {pct(lsb_s[3])} 样本日 {lsb_s[4]}")
    print(f"     排名加权多空 : 年化 {pct(lsrw_s[0])} 夏普 {lsrw_s[2]:.2f} 回撤 {pct(lsrw_s[3])} 样本日 {lsrw_s[4]}")
    print(f"     宽口多空月频 : 年化 {pct(lsm_s[0])} 夏普 {lsm_s[2]:.2f} 回撤 {pct(lsm_s[3])}")
    print(f"     long-only流动盘极端: 年化 {pct(lo_s[0])} 夏普 {lo_s[2]:.2f} 回撤 {pct(lo_s[3])}")
    print(f"     long-only大盘低波动: 年化 {pct(lol_s[0])} 夏普 {lol_s[2]:.2f} 回撤 {pct(lol_s[3])}")
    print(f"     成本 1x/2x/3x : 夏普 {s1[2]:.2f}/{s2[2]:.2f}/{s3[2]:.2f}  年化 {pct(s1[0])}/{pct(s2[0])}/{pct(s3[0])}")

    print("[6/6] 判定 ...")
    alive = "存活 ✅" if abs(t_ind) >= 2 and ic_ind > 0 else "被行业中性化杀死 ❌"
    print(f"     行业+市值中性化 IC 仍显著? {alive} (IC={ic_ind:+.4f} t={t_ind:+.2f})")
    out = {
        'window': f"{args.start}~{args.end}",
        'ic_raw': round(ic_raw, 4), 't_raw': round(t_raw, 2),
        'ic_sz': round(ic_sz, 4), 't_sz': round(t_sz, 2),
        'ic_ind': round(ic_ind, 4), 't_ind': round(t_ind, 2),
        'long_short_daily_extreme': {'ann': round(ls_s[0], 4), 'sharpe': round(ls_s[2], 2), 'maxdd': round(ls_s[3], 4)},
        'long_short_daily_broad': {'ann': round(lsb_s[0], 4), 'sharpe': round(lsb_s[2], 2), 'maxdd': round(lsb_s[3], 4)},
        'long_short_daily_rankweighted': {'ann': round(lsrw_s[0], 4), 'sharpe': round(lsrw_s[2], 2), 'maxdd': round(lsrw_s[3], 4)},
        'long_short_monthly_broad': {'ann': round(lsm_s[0], 4), 'sharpe': round(lsm_s[2], 2), 'maxdd': round(lsm_s[3], 4)},
        'long_only_liquid_extreme': {'ann': round(lo_s[0], 4), 'sharpe': round(lo_s[2], 2), 'maxdd': round(lo_s[3], 4)},
        'long_only_largecap_lowvol': {'ann': round(lol_s[0], 4), 'sharpe': round(lol_s[2], 2), 'maxdd': round(lol_s[3], 4)},
        'cost_sensitivity': {'1x': {'sharpe': round(s1[2], 2), 'ann': round(s1[0], 4)},
                              '2x': {'sharpe': round(s2[2], 2), 'ann': round(s2[0], 4)},
                              '3x': {'sharpe': round(s3[2], 2), 'ann': round(s3[0], 4)}},
    }
    with open('tools/_lowvol_true_neutral_out.json', 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("完成 -> tools/_lowvol_true_neutral_out.json")


if __name__ == '__main__':
    main()
