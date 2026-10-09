"""好方法稳健性实验：防过拟合佐证（walk-forward OOS + 参数敏感 + 成本敏感）。

项目铁律：判定因子好不好用一律以全A面板为准，不认窄样本IC；LB扫描呈噪声带即过拟合。
本模块对唯一验证有效的跨资产趋势跟随(sma200+inv_vol)做三重压力测试：
  1) walk-forward：训练窗选参 -> 完全隔离的测试窗验证（防偷看未来 / 防前视）
  2) 参数敏感：SMA窗口 60~250 全程 Sharpe 是否成簇（单点尖峰=过拟合）
  3) 成本敏感：万1~千1 佣金下是否仍跑赢等权基线
"""
import numpy as np
import pandas as pd

from .config import ETF_COMM
from .io_db import DataError, load_etf_universe
from .etf_guard import etf_equal_weight
from .method_absmom import abs_momentum_etf


def walk_forward(codes, train=("2019-01-01", "2022-12-31"),
                 test=("2023-01-01", "2026-09-30")):
    """训练窗选参(按Sharpe)，测试窗完全隔离验证。返回 dict。"""
    grid = [(s, w, sw) for s in ("absmom", "sma200")
            for w in ("equal", "inv_vol")
            for sw in (100, 150, 200, 250)]
    best = None
    for s, w, sw in grid:
        try:
            r = abs_momentum_etf(codes, train[0], train[1], signal=s, weight=w, sma_w=sw)
        except DataError:
            continue
        if best is None or r["sharpe"] > best["sharpe"]:
            best = dict(sig=s, w=w, sw=sw, **r)
    if best is None:
        raise DataError("训练窗无可用配置")
    # 测试窗：训练选出的最优配置 + 预设 sma200/inv_vol(未用任何数据选) + 等权基线
    test_best = abs_momentum_etf(codes, test[0], test[1],
                                 signal=best["sig"], weight=best["w"], sma_w=best["sw"])
    test_pre = abs_momentum_etf(codes, test[0], test[1], signal="sma200", weight="inv_vol")
    base = etf_equal_weight(codes, test[0], test[1], gate_code=None)
    return dict(
        train=dict(config=f"{best['sig']}/{best['w']}/sma{best['sw']}",
                   sharpe=best["sharpe"], ann=best["ann"], mdd=best["mdd"]),
        test_best=dict(config=f"{best['sig']}/{best['w']}/sma{best['sw']}",
                        sharpe=test_best["sharpe"], ann=test_best["ann"], mdd=test_best["mdd"]),
        test_pre=dict(config="sma200/inv_vol",
                      sharpe=test_pre["sharpe"], ann=test_pre["ann"], mdd=test_pre["mdd"]),
        test_base=dict(ann=base["ann"], sharpe=base["sharpe"], mdd=base["mdd"]),
        train_win=train, test_win=test,
    )


def param_sensitivity(codes, start="2019-01-01", end="2026-09-30",
                      sma_ws=(60, 100, 150, 200, 250),
                      weights=("equal", "inv_vol")):
    """SMA窗口 60~250 全程表现；若 Sharpe 成簇(无单点尖峰)即非过拟合。"""
    rows = []
    for w in weights:
        for sw in sma_ws:
            try:
                r = abs_momentum_etf(codes, start, end, signal="sma200", weight=w, sma_w=sw)
            except DataError:
                continue
            rows.append(dict(weight=w, sma_w=sw, ann=r["ann"], sharpe=r["sharpe"],
                             mdd=r["mdd"], time_in_market=r["time_in_market"]))
    return rows


def cost_sensitivity(codes, start="2019-01-01", end="2026-09-30",
                     costs=(0.0001, 0.0003, 0.0005, 0.001),
                     signal="sma200", weight="inv_vol"):
    """sma200/inv_vol 在不同佣金(万1~千1)下的稳健性。"""
    rows = []
    for c in costs:
        r = abs_momentum_etf(codes, start, end, signal=signal, weight=weight, cost=c)
        base = etf_equal_weight(codes, start, end, gate_code=None)
        rows.append(dict(cost=c, ann=r["ann"], sharpe=r["sharpe"], mdd=r["mdd"],
                         base_sharpe=base["sharpe"], beats=bool(r["sharpe"] > base["sharpe"])))
    return rows


def run_robust(train=("2019-01-01", "2022-12-31"), test=("2023-01-01", "2026-09-30"),
               start="2019-01-01", end="2026-09-30", clean_only=True):
    """跑三重压力测试；返回 dict(n_etf, wf, param, cost)。"""
    codes = load_etf_universe(start, end, clean_only)
    if not codes:
        raise DataError("ETF 池为空")
    return dict(n_etf=len(codes),
                wf=walk_forward(codes, train, test),
                param=param_sensitivity(codes, start, end),
                cost=cost_sensitivity(codes, start, end))
