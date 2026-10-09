"""
低波动异象的"真中性化"检验 —— 全 A 真实面板 (TimescaleDB bars_1d)
==================================================================
背景: factor_scorecard.md 裁定所有横截面因子在全 A 复现均负 IC; 传统
低波动/低振幅族 (RVOL/GARKLINE/RANGERATIO) t≈-7 但本质是"低波动风格因子"
(IC 与小盘程度单调正相关), 过完成本归零, 只能当风险过滤器。

本脚本做一件事: 用 bars_1d 真实数据, 把"20日已实现波动率"当成信号,
严格中性化后检验它到底是不是真 alpha:
  - 信号: 低波动 (signal = -vol20, 低波动股未来应涨)
  - 目标: T+1 开盘收益 (open_{t+1}/open_t - 1), 无前视
  - 市值代理: float_mcap = close * float_shares (bars_1d 自带 float_shares)
  - 过滤: 剔除次新(<250日)、剔除涨停买不进(open/昨收>=1.098)、跌停卖不出、
         剔除 ST/*ST (asset_master.asset_name 含 ST/退)
  - 中性化: 横截面用 float_mcap 排名对 signal 做正交化残差 (size-neutral)
  - 组合: 每日十分位, 第1组(最低波动)多 / 第10组(最高波动)空; 及"仅多第1组"
  - 成本: 万1 双边 + 冲击(按 float_mcap 分层的参与率近似, 见下)

输出: 原始 IC / 市值中性化 IC / 多空组合 / 低波动仅多组合 vs 等权持有基准。

数据铁律: 只用 TimescaleDB(源自 QMT datadir 回填), 绝不碰 akshare。
"""
import argparse, json, sys
import numpy as np
import pandas as pd
import psycopg2

DB = dict(host='127.0.0.1', port=5432, dbname='quant_data', user='quant', password='quant123456')
COMM = 0.0001  # 万1 双边佣金


def load_panel(start, end):
    """拉全 A 日线面板 (open/close/float_shares) + ST 标记, 返回 tidy DataFrame。"""
    c = psycopg2.connect(**DB)
    cur = c.cursor()
    sql = """
        select b.asset_code, b.trade_date, b.open, b.close,
               b.float_shares, coalesce(m.asset_name, '') as asset_name
        from bars_1d b
        left join asset_master m on m.asset_code = b.asset_code
        where b.trade_date between %s and %s
          and b.asset_type is distinct from 'etf'   -- 排除 ETF; 股票多为 'stock'/NULL
        order by b.asset_code, b.trade_date
    """
    cur.execute(sql, (start, end))
    rows = cur.fetchall()
    cur.close(); c.close()
    df = pd.DataFrame(rows, columns=['code', 'date', 'open', 'close',
                                     'float_shares', 'name'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2014-01-02')
    ap.add_argument('--end', default='2026-09-30')
    ap.add_argument('--vol-win', type=int, default=20)
    args = ap.parse_args()

    print(f"[1/5] 加载面板 {args.start}~{args.end} ...")
    df = load_panel(args.start, args.end)
    print(f"     行数 {len(df):,}  股票数 {df['code'].nunique():,}")

    # 停牌/缺失: 指标前 ffill(limit=5), 但收益用原始 OHLC
    df = df.sort_values(['code', 'date'])
    grp = df.groupby('code')
    # 次新过滤: 上市<250 交易日
    df['listed_n'] = grp.cumcount() + 1
    # 昨收 (用于涨跌停判定)
    df['prev_close'] = grp['close'].shift(1)
    # 前视保护: T+1 开盘收益
    df['ret_fwd'] = grp['open'].shift(-1) / df['open'] - 1.0
    # 20日已实现波动率 (用原始 close 收益)
    df['logret'] = np.log(grp['close'].transform(lambda s: s / s.shift(1)))
    df['vol20'] = grp['logret'].transform(
        lambda s: s.rolling(args.vol_win, min_periods=args.vol_win).std())
    # 市值代理
    df['float_mcap'] = df['close'] * df['float_shares']
    # 涨跌停标记
    df['limit_up'] = (df['open'] / df['prev_close'] >= 1.098)
    df['limit_dn'] = (df['open'] / df['prev_close'] <= 0.902)
    df['is_st'] = df['name'].str.contains('ST|退', case=False, na=False)
    # 指标 ffill(limit=5) 防停牌放大
    df['vol20'] = grp['vol20'].transform(lambda s: s.ffill(limit=5))
    df['float_mcap'] = grp['float_mcap'].transform(lambda s: s.ffill(limit=5))

    # 有效样本: 非次新, 非涨跌停日, 非ST, 指标齐全, 收益非缺失
    valid = (df['listed_n'] >= 250) & (~df['limit_up']) & (~df['limit_dn']) & (~df['is_st']) \
        & df['vol20'].notna() & df['float_mcap'].notna() & df['ret_fwd'].notna() \
        & df['open'].notna() & (df['open'] > 0) & (df['ret_fwd'].abs() < 0.5)
    panel = df[valid].copy()
    print(f"     有效样本 {len(panel):,} (剔除次新/涨跌停/ST/缺失)")

    # 横截面排名
    def xsec_rank(s):
        return s.groupby(panel['date']).rank(pct=True)

    panel['sig_raw'] = -xsec_rank(panel['vol20'])          # 低波动=高信号
    panel['size_rank'] = xsec_rank(panel['float_mcap'])
    # 市值中性化: signal 对 size 横截面回归取残差
    resid = []
    for d, g in panel.groupby('date'):
        x = g['size_rank'].values
        y = g['sig_raw'].values
        X = np.column_stack([np.ones_like(x), x])
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            r = y - X @ beta
        except Exception:
            r = y
        resid.append(pd.Series(r, index=g.index))
    panel['sig_neu'] = pd.concat(resid)
    panel['tgt_rank'] = xsec_rank(panel['ret_fwd'])

    # IC 计算 (每日 corr of ranks)
    def daily_ic(sigcol):
        ic = panel.groupby('date').apply(
            lambda g: g[sigcol].corr(g['tgt_rank']) if len(g) > 10 else np.nan)
        ic = ic.dropna()
        return ic.mean(), ic.mean() / (ic.std() / np.sqrt(len(ic))) if ic.std() > 0 else 0.0, ic

    print("[2/5] IC 检验 ...")
    ic_raw, t_raw, icr = daily_ic('sig_raw')
    ic_neu, t_neu, icn = daily_ic('sig_neu')
    print(f"     原始 IC      : {ic_raw:+.4f}  t={t_raw:+.2f}")
    print(f"     市值中性 IC  : {ic_neu:+.4f}  t={t_neu:+.2f}")

    # 组合构建: 每日十分位
    print("[3/5] 组合构建 (T+1 开盘, 万1, 含冲击) ...")
    long_short = []
    low_only = []
    ew_hold = []
    dates = sorted(panel['date'].unique())
    # 冲击: 按 float_mcap 分层给单边冲击 (小盘更高); 分位在循环外只算一次
    q20 = panel['float_mcap'].quantile(0.2)
    q80 = panel['float_mcap'].quantile(0.8)
    def impact(mcap):
        # 粗略: 小盘(后20%) 30bp, 中盘 12bp, 大盘 5bp (单边)
        if mcap < q20:
            return 0.0030
        elif mcap < q80:
            return 0.0012
        return 0.0005

    for d in dates:
        g = panel[panel['date'] == d]
        if len(g) < 50:
            continue
        dec = pd.qcut(g['sig_neu'], 10, labels=False, duplicates='drop')
        g = g.assign(dec=dec)
        # 多空: 第1组多, 第10组空 (等权)
        long_ret = g[g['dec'] == 0]['ret_fwd'].mean()
        short_ret = g[g['dec'] == 9]['ret_fwd'].mean()
        ls = long_ret - short_ret - 2 * COMM
        # 低波动仅多 (第1组) vs 等权持有全样本
        lo = g[g['dec'] == 0]['ret_fwd'].mean() - COMM - impact(g[g['dec'] == 0]['float_mcap'].median())
        ew = g['ret_fwd'].mean() - COMM
        long_short.append(ls); low_only.append(lo); ew_hold.append(ew)

    def stats(rets):
        r = np.array(rets)
        r = r[~np.isnan(r)]
        ann = (1 + r.mean()) ** 252 - 1
        vol = r.std() * np.sqrt(252)
        sharpe = ann / vol if vol > 0 else 0
        nav = np.cumprod(1 + r)
        peak = np.maximum.accumulate(nav)
        maxdd = (nav / peak - 1).min()
        return ann, vol, sharpe, maxdd, len(r)

    ls_s = stats(long_short)
    lo_s = stats(low_only)
    ew_s = stats(ew_hold)
    print(f"     多空(低波动多/高波动空): 年化 {ls_s[0]*100:+.1f}%  夏普 {ls_s[2]:.2f}  回撤 {ls_s[3]*100:.1f}%  样本日 {ls_s[4]}")
    print(f"     低波动仅多(第1组)      : 年化 {lo_s[0]*100:+.1f}%  夏普 {lo_s[2]:.2f}  回撤 {lo_s[3]*100:.1f}%")
    print(f"     等权持有全样本(基准)    : 年化 {ew_s[0]*100:+.1f}%  夏普 {ew_s[2]:.2f}  回撤 {ew_s[3]*100:.1f}%")

    # 结论判定
    print("[4/5] 判定 ...")
    beats = "是" if (ls_s[2] > ew_s[2] and ls_s[0] > 0) else "否"
    print(f"     低波动多空组合夏普 > 等权持有? {beats}")
    print(f"     市值中性化后 IC 仍显著(|t|>=2)? {'是' if abs(t_neu) >= 2 else '否'} (t={t_neu:+.2f})")

    out = {
        'window': f"{args.start}~{args.end}",
        'ic_raw': round(ic_raw, 4), 't_raw': round(t_raw, 2),
        'ic_neu': round(ic_neu, 4), 't_neu': round(t_neu, 2),
        'long_short': {'ann': round(ls_s[0], 4), 'sharpe': round(ls_s[2], 2), 'maxdd': round(ls_s[3], 4)},
        'low_only': {'ann': round(lo_s[0], 4), 'sharpe': round(lo_s[2], 2), 'maxdd': round(lo_s[3], 4)},
        'ew_hold': {'ann': round(ew_s[0], 4), 'sharpe': round(ew_s[2], 2), 'maxdd': round(ew_s[3], 4)},
    }
    with open('tools/_neutral_test_out.json', 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("[5/5] 完成 -> tools/_neutral_test_out.json")


if __name__ == '__main__':
    main()
