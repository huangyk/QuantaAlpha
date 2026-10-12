"""
低吸(抄底/买回调)战法 —— 全 A 真实面板事件驱动回测
=====================================================
来源: 用户资料库 reference/tdx_strategies/tdx_warfare_modular_design.md
  F4 回踩MA20(战法2, dixi 已证伪, 本次在全 A 重验)
  F6 黄金坑(战法4, 未测)
  F5 首阴(战法3, 未测)
  F7 龙回头(战法5, 未测)

方法(严格守验证铁律):
  - 数据: TimescaleDB bars_1d 全 A 非ETF, 2011-2026 真实价
  - 信号 T 日收盘判定 -> T+1 开盘买入(无前视, T+1)
  - 出场: 首个满足 {止损<= -8% / 止盈>= +10% / 最长持有20日} 的日收盘价
           T+1 约束: 买入当日不可卖, 最早 T+2 收盘出
  - 基准: 同池等权(全 A 有效宇宙每日等权 NAV), 必须同口径
  - 成本: 双边 0.25%(万1x2 + 印花0.05% + 冲击/滑点~0.15%)
  - 过滤: 剔次新(<250日)/ST/停牌(open<=0)/|日收益|>50%
  - 止损敏感性: F4 同时跑"无止损仅持有20日"对照 dixi 结论

输出: 每信号 逐笔超额均值±t(伪复制) + 日度聚合 t + 样本数
数据铁律: 仅 TimescaleDB, 不碰 akshare。
"""
import argparse, json
import numpy as np
import pandas as pd
import psycopg2

DB = dict(host='127.0.0.1', port=5432, dbname='quant_data', user='quant', password='quant123456')
COST = 0.0025          # 双边往返成本(含冲击近似)
STOP_LOSS = -0.08      # ≤8% 止损
TAKE_PROFIT = 0.10     # +10% 止盈
MAX_HOLD = 20          # 最长持有交易日


def load_panel(start, end):
    c = psycopg2.connect(**DB); cur = c.cursor()
    cur.execute("""
        select b.asset_code, b.trade_date, b.open, b.high, b.low, b.close,
               b.volume, b.float_shares, coalesce(m.is_st, false) as is_st
        from bars_1d b
        left join asset_master m on m.asset_code = b.asset_code
        where b.trade_date between %s and %s
          and b.asset_type is distinct from 'etf'
        order by b.asset_code, b.trade_date
    """, (start, end))
    rows = cur.fetchall(); cur.close(); c.close()
    df = pd.DataFrame(rows, columns=['code','date','open','high','low','close','volume','float_shares','is_st'])
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values(['code','date']).reset_index(drop=True)


def build_benchmark(panel):
    """同池等权每日 NAV + 分市值档 NAV(控小盘风格)。

    返回 dict:
      'broad' : 全 A 有效宇宙每日等权 NAV (Series, 与信号同口径)
      'size'  : DataFrame[date x 10档] 每档等权 NAV(按当日 float_mcap 十分位)
    均使用 panel['dret'](已在 main 计算的日收益) 与 panel['size_bucket']。
    """
    g = panel.groupby('code')
    panel['dret'] = g['close'].transform(lambda s: s / s.shift(1) - 1)
    valid_b = panel['listed_n'] >= 250
    d = panel.loc[valid_b & panel['dret'].notna() & (panel['dret'].abs() < 0.5),
                  ['date', 'dret', 'size_bucket']].copy()
    # 广基等权
    daily_broad = d.groupby('date')['dret'].mean()
    nav_broad = (1 + daily_broad).cumprod()
    # 分市值档等权(同日同档内等权, 天然免幸存者偏差)
    daily_size = d.groupby(['date', 'size_bucket'])['dret'].mean().unstack('size_bucket')
    nav_size = (1 + daily_size).cumprod()
    return {'broad': nav_broad, 'size': nav_size}


def make_signals(panel):
    """逐股算指标 + 4 个低吸信号, 返回附带信号的 panel(全部按股分组, 无跨股泄漏)。"""
    g = panel.groupby('code')
    panel['sma20'] = g['close'].transform(lambda s: s.rolling(20, min_periods=20).mean())
    panel['sma60'] = g['close'].transform(lambda s: s.rolling(60, min_periods=60).mean())
    panel['sma10'] = g['close'].transform(lambda s: s.rolling(10, min_periods=10).mean())
    panel['tsmin20'] = g['close'].transform(lambda s: s.rolling(20, min_periods=20).min())
    panel['svol5'] = g['volume'].transform(lambda s: s.rolling(5, min_periods=5).mean())
    panel['max60'] = g['close'].transform(lambda s: s.rolling(60, min_periods=60).max())
    # ffill(5) 防停牌放大
    for col in ['sma20','sma60','sma10','tsmin20','max60']:
        panel[col] = g[col].transform(lambda s: s.ffill(limit=5))
    c = panel['close']; p1 = panel['prev_close']; p2 = panel['prev2_close']
    sma20 = panel['sma20']; sma60 = panel['sma60']; sma10 = panel['sma10']
    tsmin20 = panel['tsmin20']; svol5 = panel['svol5']; max60 = panel['max60']
    # F4 回踩MA20: 距MA20<3% 且 站上MA20 且 金叉(今收>MA20 且昨收<=昨MA20)
    prev_sma20 = g['sma20'].shift(1)       # 同股昨日 MA20(按股分组, 无跨股泄漏)
    panel['F4'] = ((c - sma20).abs() / sma20 < 0.03) & (c > sma20) & (p1 <= prev_sma20)
    # F6 黄金坑: 20日新低 < MA60*0.93 且 收回MA20 且 放量
    panel['F6'] = (tsmin20 < sma60 * 0.93) & (c > sma20) & (panel['volume'] > svol5 * 1.5)
    # F5 首阴: 昨涨停(昨收/前收>1.095) 且 今收阴 且 放量
    panel['F5'] = (p1 / p2 > 1.095) & (c < p1) & (panel['volume'] > svol5 * 1.5)
    # F7 龙回头: 20日前60日高 > 今60日高*1.25(曾涨>25%) 且 回踩MA20金叉MA10 且 放量
    max60_20ago = max60.shift(20)
    was_leader = max60_20ago / max60 > 1.25
    panel['F7'] = was_leader & (c > sma20) & (c > sma10) & (panel['volume'] > svol5 * 1.3)
    return panel


def simulate_trades(panel, bench, signal_col, use_stop=True):
    """事件驱动回测: 返回逐笔 (exit_date, excess_broad, excess_size, trade_ret)。

    时序对齐(严格守铁律): T 日收盘出信号 -> T+1 开盘买入 ->
    买入当日不可卖(最早 T+2 收盘出) -> 出场取首个满足
    {止损≤-8% / 止盈≥+10% / 最长持有20交易日} 的日收盘价。
    excess_broad = 对同池等权基准的超额; excess_size = 对同市值档等权基准的超额(控小盘风格)。
    """
    nav_broad = bench['broad']; nav_size = bench['size']
    nav_map = {ts.date(): v for ts, v in nav_broad.items()}
    # 每档 NAV 也按 date 建 dict
    size_maps = {b: {ts.date(): v for ts, v in nav_size[b].items()} for b in nav_size.columns}
    trades = []
    cnt_pass = cnt_navmiss = cnt_append = 0
    for code, g in panel.groupby('code'):
        g = g.sort_values('date')
        arr_open = g['open'].values; arr_high = g['high'].values
        arr_low = g['low'].values; arr_close = g['close'].values
        arr_sig = g[signal_col].values; arr_date = g['date'].values
        arr_bucket = g['size_bucket'].values
        n = len(g)
        for i in range(n):
            # 需要 i+1 能买入(T+1 开盘), 且 i+2 能卖(买入当日不可卖)
            if i + 2 > n - 1:
                break
            if not arr_sig[i] or not np.isfinite(arr_open[i + 1]) or arr_open[i + 1] <= 0:
                continue
            cnt_pass += 1
            buy = arr_open[i + 1]                       # T+1 开盘买入
            b = arr_bucket[i + 1]                       # 买入日市值档
            last_j = min(i + 1 + MAX_HOLD, n - 1)       # 最长持有 MAX_HOLD 交易日至 T+1+MAX_HOLD
            exited = False
            for j in range(i + 2, last_j + 1):          # 最早 T+2 收盘出
                if not np.isfinite(arr_close[j]) or arr_close[j] <= 0:
                    continue
                hi = arr_high[j] / buy - 1; lo = arr_low[j] / buy - 1
                exit_price = None
                if use_stop:
                    if lo <= STOP_LOSS:
                        exit_price = buy * (1 + STOP_LOSS); ej = j; exited = True; break
                    if hi >= TAKE_PROFIT:
                        exit_price = buy * (1 + TAKE_PROFIT); ej = j; exited = True; break
                # 未触发: 继续; 循环末自然到期
            if not exited:
                ej = last_j
                exit_price = arr_close[ej]
            if exit_price is None or not np.isfinite(exit_price):
                continue
            trade_ret = exit_price / buy - 1 - COST
            d0 = pd.Timestamp(arr_date[i + 1]).date()   # 买入日(基准起点)
            d1 = pd.Timestamp(arr_date[ej]).date()
            nv0 = nav_map.get(d0); nv1 = nav_map.get(d1)
            if nv0 is None or nv1 is None or nv0 <= 0:
                cnt_navmiss += 1
                continue
            bench_ret = nv1 / nv0 - 1
            excess = trade_ret - bench_ret
            # 市值档基准
            sm = size_maps.get(b)
            if sm is not None:
                sv0 = sm.get(d0); sv1 = sm.get(d1)
                if sv0 is not None and sv1 is not None and sv0 > 0:
                    excess_size = trade_ret - (sv1 / sv0 - 1)
                else:
                    excess_size = np.nan
            else:
                excess_size = np.nan
            trades.append((pd.Timestamp(arr_date[ej]), excess, excess_size, trade_ret, pd.Timestamp(arr_date[i + 1])))
            cnt_append += 1
    print(f"     [debug {signal_col}] 初筛通过 {cnt_pass} | nav缺失跳过 {cnt_navmiss} | 建仓 {cnt_append}")
    return trades


def agg(trades):
    if not trades:
        return dict(n=0, pt_mean=0.0, pt_t=0.0, day_mean=0.0, day_t=0.0,
                    pt_size=0.0, pt_size_t=0.0, day_size=0.0, day_size_t=0.0, win=0.0)
    ex = np.array([t[1] for t in trades])
    exs = np.array([t[2] for t in trades])
    # 逐笔(伪复制)
    pt_mean = ex.mean(); pt_t = pt_mean / (ex.std()/np.sqrt(len(ex))) if ex.std()>0 else 0.0
    # 日度聚合(按出场日)
    s = pd.Series(ex, index=[t[0] for t in trades]).groupby(level=0).mean()
    day_mean = s.mean(); day_t = day_mean / (s.std()/np.sqrt(len(s))) if s.std()>0 else 0.0
    # 市值中性
    m = ~np.isnan(exs)
    if m.sum() > 0:
        exs_v = exs[m]
        pt_size = exs_v.mean(); pt_size_t = pt_size / (exs_v.std()/np.sqrt(len(exs_v))) if exs_v.std()>0 else 0.0
        ss = pd.Series(exs_v, index=[t[0] for t in trades if not np.isnan(t[2])]).groupby(level=0).mean()
        day_size = ss.mean(); day_size_t = day_size / (ss.std()/np.sqrt(len(ss))) if ss.std()>0 else 0.0
    else:
        pt_size = pt_size_t = day_size = day_size_t = 0.0
    win = (ex > 0).mean()
    return dict(n=len(trades), pt_mean=float(pt_mean), pt_t=float(pt_t),
                day_mean=float(day_mean), day_t=float(day_t),
                pt_size=float(pt_size), pt_size_t=float(pt_size_t),
                day_size=float(day_size), day_size_t=float(day_size_t),
                win=float(win))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2011-01-04')
    ap.add_argument('--end', default='2026-09-30')
    args = ap.parse_args()
    print(f"[1] 加载面板 {args.start}~{args.end} ...")
    df = load_panel(args.start, args.end)
    print(f"     行数 {len(df):,} 股票 {df['code'].nunique():,}")
    grp = df.groupby('code')
    df['listed_n'] = grp.cumcount() + 1
    df['prev_close'] = grp['close'].shift(1)
    df['prev2_close'] = grp['close'].shift(2)
    df['limit_up'] = (df['open'] / df['prev_close'] >= 1.098)
    df['limit_dn'] = (df['open'] / df['prev_close'] <= 0.902)
    df['float_mcap'] = df['close'] * df['float_shares']
    valid = (df['listed_n'] >= 250) & (~df['is_st']) & (df['open'] > 0) \
        & df['close'].notna() & df['float_shares'].notna()
    panel = df[valid].copy()
    print(f"     有效样本 {len(panel):,}")
    # 市值十分位档(按当日横截面, 天然免幸存者偏差)
    panel['size_bucket'] = panel.groupby('date')['float_mcap'].transform(
        lambda s: pd.qcut(s.rank(method='first'), 10, labels=False, duplicates='drop'))

    print("[2] 建同池等权基准 + 分市值档基准 ...")
    bench = build_benchmark(panel)
    nav = bench['broad']
    print(f"     基准 NAV 终点 {nav.iloc[-1]:.2f} (全期等权年化 {(nav.iloc[-1]**(252/len(nav))-1):+.1%})")

    print("[3] 算指标 + 低吸信号 ...")
    panel = make_signals(panel)
    for col in ['F4','F6','F5','F7']:
        cnt = int(panel[col].sum())
        print(f"     {col} 信号天数(全A) : {cnt:,}")

    print("[4] 事件驱动回测(带≤8%止损/+10%止盈/最长20日, T+1) ...")
    out = {}
    tr_f4 = None
    for col in ['F4','F6','F5','F7']:
        tr = simulate_trades(panel, bench, col, use_stop=True)
        if col == 'F4':
            tr_f4 = tr
        a = agg(tr)
        out[col] = a
        print(f"     {col} (止损版): 笔数 {a['n']:,} | 逐笔超额 {a['pt_mean']*1e4:+.2f}bp t={a['pt_t']:+.2f} | "
              f"日度 t={a['day_t']:+.2f} | 胜率 {a['win']:.1%}")
        print(f"          └ 市值中性: 逐笔 {a['pt_size']*1e4:+.2f}bp t={a['pt_size_t']:+.2f} | "
              f"日度 t={a['day_size_t']:+.2f}")

    print("[5] F4 无止损对照(仅持有20日) —— 验 dixi '无止损=自杀' ...")
    tr0 = simulate_trades(panel, bench, 'F4', use_stop=False)
    a0 = agg(tr0)
    out['F4_nostop'] = a0
    print(f"     F4 (无止损): 笔数 {a0['n']:,} | 逐笔超额 {a0['pt_mean']*1e4:+.2f}bp t={a0['pt_t']:+.2f} | "
          f"日度 t={a0['day_t']:+.2f} | 胜率 {a0['win']:.1%}")
    print(f"          └ 市值中性: 逐笔 {a0['pt_size']*1e4:+.2f}bp t={a0['pt_size_t']:+.2f} | "
          f"日度 t={a0['day_size_t']:+.2f}")

    print("[6] 随机入场对照(同套止损/止盈, 入场日随机) —— 隔离机制收益 vs F4 择时 ...")
    tr_rnd = simulate_random_control(panel, bench, n_total=out['F4']['n'], seed=20241012)
    a_rnd = agg(tr_rnd)
    out['RANDOM'] = a_rnd
    print(f"     随机(止损版): 笔数 {a_rnd['n']:,} | 逐笔超额 {a_rnd['pt_mean']*1e4:+.2f}bp t={a_rnd['pt_t']:+.2f} | "
          f"日度 t={a_rnd['day_t']:+.2f} | 胜率 {a_rnd['win']:.1%}")
    print(f"          └ 市值中性: 逐笔 {a_rnd['pt_size']*1e4:+.2f}bp t={a_rnd['pt_size_t']:+.2f} | "
          f"日度 t={a_rnd['day_size_t']:+.2f}")

    print("[7] 入场年拆分(OOS/regime 检验) F4 vs 随机 ...")
    yf = yearly_breakdown(tr_f4)     # F4 止损版 trades (显式捕获, 非循环尾变量)
    yr = yearly_breakdown(tr_rnd)
    out['F4_yearly'] = yf; out['RANDOM_yearly'] = yr
    print(f"     年   F4笔数  F4逐笔bp(t)        F4市值中性bp(t)      随机逐笔bp(t)")
    for y in sorted(set(list(yf) + list(yr))):
        f = yf.get(y, {}); r = yr.get(y, {})
        print(f"     {y}  {f.get('n',0):>7,}  {f.get('pt_bp',0):+8.2f}({f.get('pt_t',0):+6.2f})  "
              f"{f.get('pt_size_bp',0):+8.2f}({f.get('pt_size_t',0):+6.2f})   "
              f"{r.get('pt_bp',0):+8.2f}({r.get('pt_t',0):+6.2f})")

    with open('tools/_dixi_fullpanel_out.json','w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("完成 -> tools/_dixi_fullpanel_out.json")


def simulate_random_control(panel, bench, n_total, seed=0, per_stock_cap=50):
    """随机入场对照: 同一套止损/止盈/持有规则, 但入场日随机(非 F4 信号日)。
    用于隔离'止损/止盈机制本身带来的收益' vs 'F4 择时 alpha'。
    选股频率与 F4 不同(均匀随机), 仅作无信号基线。
    """
    rng = np.random.default_rng(seed)
    nav_broad = bench['broad']; nav_size = bench['size']
    nav_map = {ts.date(): v for ts, v in nav_broad.items()}
    size_maps = {b: {ts.date(): v for ts, v in nav_size[b].items()} for b in nav_size.columns}
    trades = []
    for code, g in panel.groupby('code'):
        g = g.sort_values('date')
        arr_open = g['open'].values; arr_high = g['high'].values
        arr_low = g['low'].values; arr_close = g['close'].values
        arr_date = g['date'].values; arr_bucket = g['size_bucket'].values
        n = len(g)
        feas = [k for k in range(n) if k + 2 <= n - 1 and np.isfinite(arr_open[k + 1]) and arr_open[k + 1] > 0]
        if not feas:
            continue
        picks = rng.choice(feas, size=min(per_stock_cap, len(feas)), replace=False)
        for i in picks:
            buy = arr_open[i + 1]; b = arr_bucket[i + 1]
            last_j = min(i + 1 + MAX_HOLD, n - 1)
            exited = False; ej = last_j; exit_price = arr_close[last_j]
            for j in range(i + 2, last_j + 1):
                if not np.isfinite(arr_close[j]) or arr_close[j] <= 0:
                    continue
                hi = arr_high[j] / buy - 1; lo = arr_low[j] / buy - 1
                if lo <= STOP_LOSS:
                    exit_price = buy * (1 + STOP_LOSS); ej = j; exited = True; break
                if hi >= TAKE_PROFIT:
                    exit_price = buy * (1 + TAKE_PROFIT); ej = j; exited = True; break
            if exit_price is None or not np.isfinite(exit_price):
                continue
            trade_ret = exit_price / buy - 1 - COST
            d0 = pd.Timestamp(arr_date[i + 1]).date(); d1 = pd.Timestamp(arr_date[ej]).date()
            nv0 = nav_map.get(d0); nv1 = nav_map.get(d1)
            if nv0 is None or nv1 is None or nv0 <= 0:
                continue
            bench_ret = nv1 / nv0 - 1; excess = trade_ret - bench_ret
            sm = size_maps.get(b)
            if sm is not None:
                sv0 = sm.get(d0); sv1 = sm.get(d1)
                excess_size = trade_ret - (sv1 / sv0 - 1) if (sv0 is not None and sv1 is not None and sv0 > 0) else np.nan
            else:
                excess_size = np.nan
            trades.append((pd.Timestamp(arr_date[ej]), excess, excess_size, trade_ret, pd.Timestamp(arr_date[i + 1])))
        if len(trades) >= n_total:
            break
    return trades


def yearly_breakdown(trades):
    """按入场年拆分逐笔超额(广基+市值中性), 用于 OOS/regime 检验。"""
    rows = {}
    df = pd.DataFrame(trades, columns=['exit', 'ex', 'exs', 'ret', 'entry'])
    df['yr'] = df['entry'].dt.year
    for yr, g in df.groupby('yr'):
        ex = g['ex'].values; exs = g['exs'].values[~np.isnan(g['exs'].values)]
        m = ex.mean(); t = m / (ex.std()/np.sqrt(len(ex))) if ex.std() > 0 else 0.0
        if len(exs) > 0:
            ms = exs.mean(); ts = ms / (exs.std()/np.sqrt(len(exs))) if exs.std() > 0 else 0.0
        else:
            ms = ts = 0.0
        rows[int(yr)] = dict(n=len(g), pt_bp=float(m*1e4), pt_t=float(t),
                             pt_size_bp=float(ms*1e4), pt_size_t=float(ts))
    return rows


if __name__ == '__main__':
    main()
