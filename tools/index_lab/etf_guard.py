"""功能 B：ETF 等权持有的 MA60 降回撤保护实测。

把指数层发现的唯一真亮点（创业板/中证500 快 MA60 降回撤）接进 ETF 等权持有：
用宽基指数(默认中证500)的 MA60 趋势门控——指数在 MA60 上方才持有 ETF，
跌破则转现金——测对夏普/回撤的实际影响。信号-收益对齐：用前一日收盘判 MA，
次日均生效，无前视。
"""
import numpy as np
import pandas as pd
import psycopg2

from .config import DB, CLEAN_SOURCE, ETF_COMM, GATE_INDEX, GATE_MA
from .io_db import DataError
from .metrics import metrics


def _load_etf(cur, codes, start, end):
    rows = []
    for code in codes:
        cur.execute("""SELECT trade_date, open, high, low, close, volume
                       FROM bars_1d WHERE asset_code=%s AND asset_type='etf'
                       AND trade_date>=%s AND trade_date<=%s ORDER BY trade_date""",
                    (code, start, end))
        for r in cur.fetchall():
            rows.append((code, r[0], r[1], r[2], r[3], r[4], r[5]))
    df = pd.DataFrame(rows, columns=["code", "date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    out = {}
    for code, g in df.groupby("code"):
        out[code] = g.sort_values("date").set_index("date")
    return out


def etf_equal_weight(codes, start, end, gate_code=None, gate_ma=GATE_MA, cost=ETF_COMM):
    """ETF 等权持有回测。gate_code=None 时为纯等权持有；否则叠加 MA(gate_ma) 趋势门控。

    返回 dict(nav, ann, vol, sharpe, mdd, n_switches)。
    """
    con = psycopg2.connect(**DB); cur = con.cursor()
    try:
        data = _load_etf(cur, codes, start, end)
        gate = None
        if gate_code:
            cur.execute("""SELECT trade_date, close FROM bars_1d
                           WHERE asset_code=%s AND trade_date>=%s AND trade_date<=%s
                           ORDER BY trade_date""", (gate_code, start, end))
            g = pd.DataFrame(cur.fetchall(), columns=["date", "close"]).set_index("date")
            g.index = pd.to_datetime(g.index)
            gate = g["close"]
    finally:
        con.close()

    all_days = sorted(set().union(*[set(g.index) for g in data.values()]))
    if len(all_days) < gate_ma + 5:
        raise DataError(f"交易日不足({len(all_days)} < {gate_ma + 5})")

    # 门控仓位（T+1 对齐：用前一日收判定）
    gate_pos = np.ones(len(all_days))
    if gate is not None:
        gc = gate.reindex(pd.Index(all_days)).ffill()
        gma = gc.rolling(gate_ma).mean().values
        for t in range(1, len(all_days)):
            gate_pos[t] = 1.0 if (np.isfinite(gma[t - 1]) and gc.iloc[t - 1] > gma[t - 1]) else 0.0

    cash = 1.0; nav = []; nav_dates = []; last_gate = 1.0
    for i, d in enumerate(all_days):
        if i > 0:
            prev_d = all_days[i - 1]
            day_ret = 0.0; tot_w = 0.0
            for code, g in data.items():
                ct = g["close"].get(d); cp = g["close"].get(prev_d)
                if ct is not None and cp is not None and pd.notna(ct) and pd.notna(cp) and cp > 0:
                    day_ret += (ct / cp - 1); tot_w += 1.0
            if tot_w > 0:
                day_ret /= tot_w
            day_ret *= gate_pos[i]
            if gate_pos[i] != last_gate:        # 门控切换：整仓级佣金一次
                cash *= (1 - cost)
            last_gate = gate_pos[i]
            cash *= (1 + day_ret)
        nav.append(cash); nav_dates.append(d)

    nav_s = pd.Series(nav, index=nav_dates)
    m = metrics(nav_s.pct_change().dropna())
    m["n_switches"] = int(np.sum(np.diff(gate_pos) != 0))
    m["nav"] = nav_s
    return m


def run(start="2019-01-01", end="2026-09-30", gate=GATE_INDEX,
        clean_only=True, extra_gates=None):
    """对比：纯等权 vs 中证500 MA60 门控 vs (可选)创业板 MA60 门控。返回 dict。"""
    from .io_db import load_etf_universe
    codes = load_etf_universe(start, end, clean_only)
    if not codes:
        raise DataError("ETF 池为空")
    out = dict(module="etf_guard", n_etf=len(codes), start=start, end=end, gates={})
    base = etf_equal_weight(codes, start, end, gate_code=None)
    out["equal_weight"] = dict(ann=base["ann"], sharpe=base["sharpe"],
                               mdd=base["mdd"], n_switches=base["n_switches"])
    # 主门控
    gname, gcode = gate
    g = etf_equal_weight(codes, start, end, gate_code=gcode, gate_ma=GATE_MA)
    out["gates"][gname] = dict(ann=g["ann"], sharpe=g["sharpe"],
                               mdd=g["mdd"], n_switches=g["n_switches"])
    # 额外门控（如创业板）
    for ename, ecode in (extra_gates or []):
        e = etf_equal_weight(codes, start, end, gate_code=ecode, gate_ma=GATE_MA)
        out["gates"][ename] = dict(ann=e["ann"], sharpe=e["sharpe"],
                                   mdd=e["mdd"], n_switches=e["n_switches"])
    return out
