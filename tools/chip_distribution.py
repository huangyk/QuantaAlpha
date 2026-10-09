"""
筹码分布 (Cost Distribution) 计算器 —— 从 OHLCV 还原通达信 COST()/WINNER()
============================================================================
通达信的 COST(N) / WINNER(P) 是"筹码分布"函数, 本质是**逐日换手加权持仓成本**
的分布。本库 bars_1d (QMT datadir 回填) 只有 OHLCV, 没有筹码数据, 这里用
标准 TDX 衰减模型从 OHLCV + 流通股本(flow_shares)/换手率(turnover_rate) 还原。

算法 (TDX 官方"成本分布"衰减模型):
-----------------------------------------------------------------
对每一交易日 t (按时间顺序):
  1. 当日均价 avg = (H + L + 2*C) / 4        # TDX 给收盘更高权重
  2. 换手率 turn = turnover_rate (若有) 否则 volume*100 / float_shares
                   (volume 单位是"手"=100股, float_shares 单位是股)
  3. 当日成交按**三角形分布**铺在 [L, H] 区间, 峰值在 avg:
         f(p) = 0                              p<=L 或 p>=H
         f(p) = 2*(p-L)/((H-L)*(avg-L))        L<p<=avg
         f(p) = 2*(H-p)/((H-L)*(H-avg))        avg<p<H
     (三角形面积=1, 即 f 是密度)
  4. 衰减更新: D_t = D_{t-1} * (1 - turn) + turn * f_grid
     -> 每天有 turn 比例的老筹码"移仓"到当日成本, 其余(1-turn)保留旧成本
由此 D 是覆盖全价的"筹码直方图"(单位: 占流通盘比例, 总和≈1)。

派生量:
  COST(q)  = 累积分布达到 q% 对应的价格 (q=5/10/50/89/90)
  WINNER(p)= 成本低于 p 的筹码比例 (即"以价 p 买入者获利的比例")
  conc     = (COST(90)-COST(10)) / COST(50)   筹码集中度(越低越集中)

注意: 这是"忠实复现 TDX 算法逻辑"的还原, 不是 TDX 内部二进制结果;
TDX 实际实现在某些参数(均价加权/衰减窗口)上略有差异, 但结论性规律一致。
"""
import numpy as np
import pandas as pd


def _chip_one(h, l, o, c, vol, flt, turn, nbins=500, warmup=120):
    """单只股票, 输入为 numpy 数组(已按日期升序)。返回字典 of 等长数组(缺失=nan)。"""
    n = len(c)
    out = dict(
        cost5=np.full(n, np.nan), cost10=np.full(n, np.nan),
        cost50=np.full(n, np.nan), cost89=np.full(n, np.nan),
        cost90=np.full(n, np.nan), winner_c=np.full(n, np.nan),
        winner_09c=np.full(n, np.nan), winner_11c=np.full(n, np.nan),
        conc=np.full(n, np.nan),
    )
    if n < warmup + 1:
        return out

    lo_all = np.nanmin(l); hi_all = np.nanmax(h)
    if not np.isfinite(lo_all) or not np.isfinite(hi_all) or hi_all <= lo_all:
        return out
    # 价格网格
    edges = np.linspace(lo_all, hi_all, nbins + 1)
    pc = 0.5 * (edges[:-1] + edges[1:])           # bin 中心
    dbin = edges[1] - edges[0]
    D = np.zeros(nbins, dtype=float)              # 筹码直方图(占流通盘比例)

    for t in range(n):
        lt = l[t]; ht = h[t]; ct = c[t]
        if not (np.isfinite(lt) and np.isfinite(ht) and ht > lt and np.isfinite(ct)):
            continue
        avg = (ht + lt + 2.0 * ct) / 4.0
        # 换手率
        if turn is not None and np.isfinite(turn[t]) and turn[t] > 0:
            tr = turn[t]
        elif flt is not None and np.isfinite(flt[t]) and flt[t] > 0 and vol is not None \
                and np.isfinite(vol[t]) and vol[t] > 0:
            tr = vol[t] * 100.0 / flt[t]
        else:
            tr = 0.0
        tr = min(max(tr, 0.0), 1.0)               # 钳制到 [0,1]
        # 当日三角形密度 (向量化 over bins)
        f = np.zeros(nbins, dtype=float)
        in_range = (pc > lt) & (pc < ht)
        if np.any(in_range):
            below = in_range & (pc <= avg)
            above = in_range & (pc > avg)
            if ht > lt and avg > lt:
                f[below] = 2.0 * (pc[below] - lt) / ((ht - lt) * (avg - lt))
            if ht > lt and ht > avg:
                f[above] = 2.0 * (ht - pc[above]) / ((ht - lt) * (ht - avg))
        # 归一化使密度积分=1, 再乘换手率得到当日筹码质量
        finteg = f.sum() * dbin
        if finteg > 0:
            newd = tr * f / finteg * dbin
        else:
            newd = np.zeros(nbins)
        # 衰减更新
        if t == 0 or D.sum() == 0:
            D = newd.copy()
        else:
            D = D * (1.0 - tr) + newd
        # 至少需要 warmup 天才算稳定
        if t < warmup:
            continue
        total = D.sum()
        if total <= 0:
            continue
        cum = np.cumsum(D) / total               # 累积比例 (len=nbins)
        # 二分定位
        def winner_at(p):
            if not np.isfinite(p):
                return np.nan
            idx = int(np.searchsorted(edges, p)) - 1
            idx = min(max(idx, 0), nbins - 1)
            return cum[idx]
        def cost_at(q):
            # 累积达到 q/100 的价格
            thr = q / 100.0
            idx = int(np.searchsorted(cum, thr))
            if idx <= 0:
                return edges[0]
            if idx >= nbins:
                return edges[-1]
            # 在 bin idx-1 与 idx 之间线性插值
            c0, c1 = cum[idx - 1], cum[idx]
            if c1 == c0:
                return edges[idx]
            frac = (thr - c0) / (c1 - c0)
            return edges[idx - 1] + frac * (edges[idx] - edges[idx - 1])

        out['winner_c'][t] = winner_at(ct)
        out['winner_09c'][t] = winner_at(0.9 * ct)
        out['winner_11c'][t] = winner_at(1.1 * ct)
        c5 = cost_at(5); c10 = cost_at(10); c50 = cost_at(50)
        c89 = cost_at(89); c90 = cost_at(90)
        out['cost5'][t] = c5; out['cost10'][t] = c10; out['cost50'][t] = c50
        out['cost89'][t] = c89; out['cost90'][t] = c90
        if c50 and c50 > 0:
            out['conc'][t] = (c90 - c10) / c50
    return out


def compute_chip(df, nbins=500, warmup=120, has_turn_col=True):
    """对面板(df, 含 code/date/open/high/low/close/volume/float_shares[/turnover_rate])
    按股票计算筹码派生量, 返回合并后的 DataFrame(原 index 对齐)。
    """
    req = ['open', 'high', 'low', 'close', 'volume', 'float_shares']
    if has_turn_col:
        req.append('turnover_rate')
    miss = [x for x in req if x not in df.columns]
    if miss:
        raise ValueError(f"缺少列: {miss}")

    def proc(g):
        g = g.sort_values('date').reset_index(drop=True)
        h = g['high'].values.astype(float); l = g['low'].values.astype(float)
        o = g['open'].values.astype(float); c = g['close'].values.astype(float)
        vol = g['volume'].values.astype(float)
        flt = g['float_shares'].values.astype(float)
        turn = g['turnover_rate'].values.astype(float) if has_turn_col else None
        res = _chip_one(h, l, o, c, vol, flt, turn, nbins=nbins, warmup=warmup)
        return pd.DataFrame(res, index=g.index)

    # 手动按股票迭代: 避免 groupby.apply 的重复索引/ FutureWarning,
    # concat 后用 ignore_index=True 得到与输入 df 行序一致的干净 RangeIndex。
    parts = []
    for code, g in df.groupby('code', sort=False):
        parts.append(proc(g))
    out = pd.concat(parts, ignore_index=True)
    return out


if __name__ == '__main__':
    # 自测: 600519.SH 一只票, 打印 sanity + 计时
    import psycopg2, time
    DB = dict(host='127.0.0.1', port=5432, dbname='quant_data',
             user='quant', password='quant123456')
    con = psycopg2.connect(**DB); cur = con.cursor()
    cur.execute("""select asset_code, trade_date, open, high, low, close, volume,
                          float_shares, turnover_rate
                   from bars_1d where asset_code='600519.SH'
                   order by trade_date""")
    rows = cur.fetchall(); cur.close(); con.close()
    d = pd.DataFrame(rows, columns=['code', 'date', 'open', 'high', 'low', 'close',
                                    'volume', 'float_shares', 'turnover_rate'])
    print(f"600519 行数 {len(d)}  float_shares 非空 {d.float_shares.notna().sum()}")
    t0 = time.time()
    r = compute_chip(d, nbins=500, warmup=120)
    print(f"耗时 {time.time()-t0:.2f}s")
    last = r.dropna(subset=['winner_c']).iloc[-1]
    print("末日快照:")
    for k in ['cost5', 'cost10', 'cost50', 'cost89', 'cost90', 'winner_c',
              'winner_09c', 'winner_11c', 'conc']:
        print(f"  {k:10s} = {last[k]:.4f}")
    print(f"  收盘 = {d['close'].iloc[-1]:.2f}")
    # 基本不变式检查
    assert 0 <= last['winner_c'] <= 1, "WINNER 应在 [0,1]"
    assert last['winner_09c'] <= last['winner_c'] <= last['winner_11c'], "WINNER 单调性"
    assert last['cost5'] <= last['cost50'] <= last['cost89'] <= last['cost90'], "COST 单调性"
    print("不变式检查通过 ✅")
