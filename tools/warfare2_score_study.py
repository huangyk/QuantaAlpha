"""战法2 打分版地基验证 —— 「事件型」vs「打分型」哪个真有 edge。

背景（本轮实测，务必记住）：
    严格布尔 AND（F3 & F4 & F2 全中）的**事件型**信号，2020-2026 全池仅 569 次
    （日均 0.34 只，85.4% 交易日无信号），各持有期超额 t 全在 ±0.7 内 → **无 alpha**。
    但同一批因子以**连续值**喂模型时历史报过 IC≈0.072 —— 差别在于布尔化丢掉了
    「距 MA20 多近」「放了多少量」这些强度信息，且样本被砍掉 99%。

本脚本把三因子改成连续打分（保留强度信息，每天全池都有分），严格验证：
    f3 = (MA20 - MA60)/MA60              多头排列强度（越大多头越强）
    f4 = -|C - MA20|/MA20                贴近 MA20 程度（回踩低吸：越贴近越好）
    f2 = VOL / MA5(VOL)                  放量强度
    合成：截面 rank 等权相加（M0）与 LGB（训练内验证段挑选）

验证口径（与飞龙 V2 同标准，防自欺）：
    - 收益：可交易 open(T+1) 买 → open(T+2) 卖（T 收盘出分）
    - 基准：同池等权（liquid500 等权）——不是沪深300
    - 必看：逐日 rank IC + t、**十分位组单调性**、分年度、含成本 top-K 超额

输出：data/results/warfare2_score_*.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import qlib

PROVIDER = 'D:/qqmt/qlib_data/cn_real_full'
START, END = '2020-01-01', '2026-09-30'
FIELDS = ['$open', '$high', '$low', '$close', '$volume']
FEATS = ['f3_bull', 'f4_pullback', 'f2_vol']
COST = 0.002
SPLIT = '2025-01-01'          # 训练/测试分界（测试 2025 起，约 1.8 年）
OUT = Path('data/results')


def load():
    qlib.init(provider_uri=PROVIDER, region='cn')
    from qlib.data import D
    codes = D.list_instruments(D.instruments('liquid500'), start_time=START, end_time=END, as_list=True)
    df = D.features(codes, FIELDS, start_time=START, end_time=END, freq='day').sort_index()
    panels = {}
    for i, g in df.groupby(level='instrument'):
        panels[i] = g.droplevel('instrument').sort_index() if isinstance(g.index, pd.MultiIndex) else g.sort_index()
    return panels


def build_panel(panels):
    """⚠️ 关键修复：先对停牌日做价格延续（ffill），再算滚动窗口。

    诊断发现的 bug：cn_real_full 里一只票全年可能只缺 29 天（停牌），但
    `rolling(60, min_periods=60)` 要求窗口内**非 NaN 计数达 60**，于是
    「停牌 1 天 → 其后 59 个交易日的 MA60 全为 NaN」。实测某票 1636 天有收盘价，
    MA60 却只有 402 天有值 —— 样本被砍掉 75%，f3 每日可用个股只剩 47~176 只，
    所有依赖 MA20/MA60 的结论都被这个空洞污染（前几轮结论据此作废）。
    修复：ffill(limit=5) 补停牌价格后再算均线；但 **fwd 仍用原始 open**，
    保证只有真实可交易的日子才有收益（停牌日不可交易）。
    """
    frames = []
    for inst, g in panels.items():
        raw_open = g['$open'].copy()          # 原始开盘价：判断真实可交易
        g = g.ffill(limit=5)
        c, o, v = g['$close'], g['$open'], g['$volume']
        ma20 = c.rolling(20, min_periods=20).mean()
        ma60 = c.rolling(60, min_periods=60).mean()
        ma5v = v.rolling(5, min_periods=5).mean()
        d = pd.DataFrame(index=g.index)
        d['f3_bull'] = (ma20 - ma60) / ma60
        d['f4_pullback'] = -((c - ma20).abs() / ma20)          # 越接近 0 越贴近 MA20
        d['f2_vol'] = v / ma5v
        d['is_bull'] = (ma20 > ma60) & (c > ma20)              # 战法2 的方向约束（多头 + 站上MA20）
        # T收盘出分 → T+1开盘买 → T+2开盘卖；用**原始** open，停牌日自然为 NaN
        d['fwd'] = raw_open.shift(-2) / raw_open.shift(-1) - 1.0
        d['close'] = c
        d = d.reset_index()
        d['instrument'] = inst
        frames.append(d.set_index(['datetime', 'instrument']))
    return pd.concat(frames).sort_index()


def cs_rank(df, cols=FEATS):
    return df[cols].groupby(level='datetime').rank(pct=True)


def ic_series(panel, dates, pred_col='pred'):
    out = {}
    for dt in dates:
        try:
            sub = panel.xs(dt, level='datetime')
        except KeyError:
            continue
        sub = sub.dropna(subset=[pred_col, 'fwd'])
        if len(sub) < 50:
            continue
        out[dt] = sub[pred_col].corr(sub['fwd'], method='spearman')
    return pd.Series(out).dropna()


def ic_stats(s):
    if len(s) < 5:
        return {}
    return {'IC均值': s.mean(), 'ICIR': s.mean() / s.std(ddof=1),
            't': s.mean() / (s.std(ddof=1) / np.sqrt(len(s))),
            'IC>0占比': float((s > 0).mean()), '天数': len(s)}


def run_topk(panel, dates, bench, topk=30, mask_bull=True, every=1, pred_col='pred'):
    """每日 top-K 等权，可加方向约束 is_bull（MA20>MA60 且 C>MA20）。"""
    hold, rows = None, []
    for k, dt in enumerate(dates):
        try:
            sub = panel.xs(dt, level='datetime')
        except KeyError:
            continue
        sub = sub.dropna(subset=[pred_col, 'fwd'])
        if mask_bull:
            sub = sub[sub['is_bull'].fillna(False)]
        if len(sub) < topk:
            continue
        if k % every != 0:
            if hold is None:
                continue
            cur = hold & set(sub.index)
            if not cur:
                continue
            sel = sub.loc[list(cur)]
            rows.append({'date': dt, 'net': sel['fwd'].mean(), 'turnover': 0.0})
            continue
        sel = sub.nlargest(topk, pred_col)
        cur = set(sel.index)
        turn = 1.0 if hold is None else 1.0 - len(cur & hold) / max(len(cur), 1)
        rows.append({'date': dt, 'net': sel['fwd'].mean() - turn * COST, 'turnover': turn})
        hold = cur
    r = pd.DataFrame(rows).set_index('date')
    r['bench'] = bench.reindex(r.index)
    return r


def metrics(rets, bench):
    rets = rets.dropna()
    if len(rets) == 0:
        return {}
    ex = rets - bench.reindex(rets.index).fillna(0)
    cum, ec = rets.cumsum(), ex.cumsum()
    return {'年化': rets.mean() * 252, '年化超额': ex.mean() * 252,
            '超额IR': ex.mean() / ex.std(ddof=1) * np.sqrt(252) if ex.std(ddof=1) > 0 else np.nan,
            '超额t': ex.mean() / (ex.std(ddof=1) / np.sqrt(len(ex))) if ex.std(ddof=1) > 0 else np.nan,
            '最大回撤': (cum - cum.cummax()).min(), '日胜率': float((rets > 0).mean()),
            '超额日胜率': float((ex > 0).mean()), '天数': len(rets)}


def decile_table(panel, dates, bench, pred_col='pred', tag='', save=None):
    """十分位组。**D1 = 预测分最高组**（rank 降序后切分，label 0 即最高）。
    ⚠️ 早期版本用 rank(method='first') 升序切分却把 label 0 标成 D1，方向说反了，此处已修正。"""
    grp_ret = {}
    for dt in dates:
        try:
            sub = panel.xs(dt, level='datetime')
        except KeyError:
            continue
        sub = sub.dropna(subset=[pred_col, 'fwd'])
        if len(sub) < 50:
            continue
        try:
            lab = pd.qcut(sub[pred_col].rank(ascending=False, method='first'), 10, labels=False)
        except ValueError:
            continue
        grp_ret[dt] = sub.groupby(lab)['fwd'].mean()
    G = pd.DataFrame(grp_ret).T.sort_index()
    if G.empty:
        print("  分位组样本不足")
        return pd.DataFrame()
    print(f"\n=== {tag} 十分位组（D1=分最高 → D10=分最低）超额 vs 同池等权 ===")
    rows = []
    for gcol in sorted(G.columns):
        s = G[gcol].dropna()
        m = metrics(s, bench)
        rows.append({'组': f'D{gcol+1}', '年化超额': m.get('年化超额'), '超额t': m.get('超额t')})
        print(f"  D{gcol+1:<2d} 年化超额={m.get('年化超额', float('nan'))*100:+7.2f}%  "
              f"t={m.get('超额t', float('nan')):+.2f}")
    d1, d10 = metrics(G[0].dropna(), bench), metrics(G[9].dropna(), bench)
    print(f"  → D1(最高)−D10(最低) = {(d1['年化超额']-d10['年化超额'])*100:+.2f}%"
          f"（>0 且自上而下递减 = 真 alpha）")
    df = pd.DataFrame(rows)
    if save:
        df.to_csv(OUT / save, index=False)
    return df


def topk_table(panel, dates, bench, pred_col='pred', tag='', save=None):
    print(f"\n=== {tag} top-K 组合（含成本 {COST*100:.1f}%/全换，基准=池等权） ===")
    out = []
    for topk in (10, 20, 30, 50):
        for every in (1, 2, 3, 5):
            R = run_topk(panel, dates, bench, topk=topk, every=every, pred_col=pred_col)
            m = metrics(R['net'], bench)
            out.append({'topk': topk, '调仓周期': every, '日均换手': R['turnover'].mean(),
                        '年化': m.get('年化'), '年化超额': m.get('年化超额'), '超额IR': m.get('超额IR'),
                        '超额t': m.get('超额t'), '最大回撤': m.get('最大回撤')})
            print(f"  K={topk:<3d} every={every}  换手={R['turnover'].mean():.2f}  "
                  f"年化超额={m.get('年化超额', float('nan'))*100:+7.2f}%  "
                  f"IR={m.get('超额IR', float('nan')):+.3f}  t={m.get('超额t', float('nan')):+.2f}  "
                  f"回撤={m.get('最大回撤', float('nan'))*100:+.2f}%")
    df = pd.DataFrame(out)
    if save:
        df.to_csv(OUT / save, index=False)
    return df


def main():
    panels = load()
    print(f"载入 {len(panels)} 只标的")
    panel = build_panel(panels)
    panel.to_pickle('data/results/_warfare2_score_panel.pkl')
    bq = panel.groupby(level='datetime')['fwd'].mean()          # 同池等权基准
    panel = panel.join(bq.rename('bench_eq'), on='datetime')

    dates = sorted(set(panel.index.get_level_values('datetime')))
    tr = [d for d in dates if d < pd.Timestamp(SPLIT)]
    te = [d for d in dates if d >= pd.Timestamp(SPLIT)]
    print(f"训练 {tr[0].date()}~{tr[-1].date()} ({len(tr)}天) | 测试 {te[0].date()}~{te[-1].date()} ({len(te)}天)")

    # ---- 数据覆盖诊断（决定结论可信度，必做）----
    print("\n=== 数据覆盖诊断：每日有效样本数（按年） ===")
    cov = pd.DataFrame({
        '有fwd': panel['fwd'].notna().groupby(level='datetime').sum(),
        '有f2(5日)': panel['f2_vol'].notna().groupby(level='datetime').sum(),
        '有f4(20日)': panel['f4_pullback'].notna().groupby(level='datetime').sum(),
        '有f3(60日)': panel['f3_bull'].notna().groupby(level='datetime').sum(),
    })
    print(cov.groupby(cov.index.year).mean().round(0).to_string())
    print("  （若某年'有f3'远小于'有fwd'，说明池内个股历史窗口不足，结论样本会被砍）")

    # ---- 单因子 IC（全样本）----
    print("\n=== 单因子逐日 rank IC（全样本，可交易口径） ===")
    for f in FEATS:
        s = ic_series(panel.assign(pred=panel[f]), dates)
        st = ic_stats(s)
        print(f"  {f:14s} IC={st['IC均值']:+.4f}  ICIR={st['ICIR']:+.3f}  t={st['t']:+.2f}  "
              f"IC>0={st['IC>0占比']:.1%}  n={st['天数']}")

    # ---- f4 单独：分年 + 测试期（地基因子，单独坐实）----
    print("\n=== f4_pullback（贴近MA20）分年 IC ===")
    for y in sorted(set(pd.Timestamp(d).year for d in dates)):
        dy = [d for d in dates if pd.Timestamp(d).year == y]
        s = ic_series(panel.assign(pred=panel['f4_pullback']), dy)
        st = ic_stats(s)
        if st:
            print(f"  {y}  IC={st['IC均值']:+.4f}  ICIR={st['ICIR']:+.3f}  t={st['t']:+.2f}  n={st['天数']}")

    # ---- 合成：M0 等权 vs LGB（训练内验证段挑选）----
    vcut = tr[int(len(tr) * 0.8)]
    fit_d = [d for d in tr if d < vcut]
    val_d = [d for d in tr if d >= vcut]
    fit = panel.loc[fit_d].dropna(subset=FEATS + ['fwd'])
    val = panel.loc[val_d].dropna(subset=FEATS + ['fwd'])
    m0_val = cs_rank(val).sum(axis=1)
    lg = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, num_leaves=15, subsample=0.8,
                           colsample_bytree=0.8, verbose=-1, random_state=42)
    lg.fit(cs_rank(fit), fit['fwd'])
    m1_val = pd.Series(lg.predict(cs_rank(val)), index=val.index)
    ic0 = ic_stats(ic_series(val.assign(pred=m0_val), val_d))
    ic1 = ic_stats(ic_series(val.assign(pred=m1_val), val_d))
    print(f"\n=== 训练内验证段（{len(val_d)}天）选模型 ===")
    print(f"  M0 等权合成  IC={ic0['IC均值']:+.4f} ICIR={ic0['ICIR']:+.3f} t={ic0['t']:+.2f}")
    print(f"  M1 LGB       IC={ic1['IC均值']:+.4f} ICIR={ic1['ICIR']:+.3f} t={ic1['t']:+.2f}")
    use_lgb = abs(ic1.get('ICIR', 0)) > abs(ic0.get('ICIR', 0))
    name = 'M1_LGB' if use_lgb else 'M0_等权'
    print(f"  → 选用 {name}")

    # 测试期打分（训练期全量重训 → 测试期预测）
    tr_all = panel.loc[tr].dropna(subset=FEATS + ['fwd'])
    te_df = panel.loc[te].dropna(subset=FEATS + ['fwd'])
    if use_lgb:
        lg.fit(cs_rank(tr_all), tr_all['fwd'])
        panel['pred'] = np.nan
        panel.loc[te_df.index, 'pred'] = lg.predict(cs_rank(te_df))
    else:
        panel['pred'] = cs_rank(panel[FEATS]).sum(axis=1)

    te_ic = ic_stats(ic_series(panel.loc[te], te))
    print(f"\n=== 测试期 IC（{name}） ===  IC={te_ic['IC均值']:+.4f} ICIR={te_ic['ICIR']:+.3f} "
          f"t={te_ic['t']:+.2f} n={te_ic['天数']}")

    bench_te = bq.reindex(te)
    decile_table(panel, te, bench_te, 'pred', tag=name, save='warfare2_score_decile.csv')
    topk_table(panel, te, bench_te, 'pred', tag=name, save='warfare2_score_topk.csv')

    # ---- f4 纯净版（地基因子单独验证，不掺 f3/f2 噪声）----
    panel['pred_f4'] = panel['f4_pullback'].groupby(level='datetime').rank(pct=True)
    te_ic_f4 = ic_stats(ic_series(panel.loc[te], te, pred_col='pred_f4'))
    print(f"\n=== f4 纯净版 测试期 IC ===  IC={te_ic_f4['IC均值']:+.4f} ICIR={te_ic_f4['ICIR']:+.3f} "
          f"t={te_ic_f4['t']:+.2f} n={te_ic_f4['天数']}")
    decile_table(panel, te, bench_te, 'pred_f4', tag='f4_only', save='warfare2_f4_decile.csv')
    topk_table(panel, te, bench_te, 'pred_f4', tag='f4_only', save='warfare2_f4_topk.csv')
    print("\n保存: warfare2_score_*.csv / warfare2_f4_*.csv")


if __name__ == '__main__':
    main()
