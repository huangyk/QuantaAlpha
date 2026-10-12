"""ML / DL 择时叠加层：在已验证的趋势策略上，用模型判断"下个月该不该持有"。

动机：用户指出前面全是古典规则(SMA200+逆波动)，没有机器学习/深度学习。
本模块把 ML/DL 作为**择时叠加层**（不替换已验证策略，只在月频调仓点决定是否持有），
严格 expanding-window walk-forward（每个测试月用此前全部历史训练、预测下月、再实现收益，
无前视）。模型：逻辑回归 / LightGBM(机器学习) / 小 PyTorch MLP(深度学习)。

铁律：样本仅 7.5 年月度(~90点)，深度学习易背数 -> MLP 刻意做小(16-8隐藏、dropout)并
walk-forward；若 ML/DL 不击败"始终持有"朴素基准，则如实报告(与项目既有的"ML因子证伪"一致)。
"""
import numpy as np
import pandas as pd
import psycopg2

from .config import DB, ETF_COMM
from .io_db import DataError, load_etf_universe
from .method_absmom import abs_momentum_etf, _load_etf
from .method_higher import is_bond_cash, blend_absmomentum
from .etf_guard import etf_equal_weight
from .metrics import metrics


def build_ml_panel(codes_all, start, end, blend=0.0):
    """构造月度 ML 面板：特征(月末, 仅用≤当月数据) + 目标(下月趋势策略收益符号)。

    特征：breadth(全池%站上200SMA) / 等权指数12-1动量 / 趋势策略12-1动量 /
    近6月波动 / 近12月回撤 / 上月收益 / 近3月收益。
    目标 y：趋势策略下月收益 >0 取1，否则0（标签在当月，特征无前视）。
    """
    if blend <= 0:
        trend = abs_momentum_etf(codes_all, start, end, signal="sma200",
                                 weight="inv_vol")["nav"]
    else:
        hb = [c for c in codes_all if not is_bond_cash(c)]
        trend = blend_absmomentum(codes_all, start, end, blend)["nav"]
    ew = etf_equal_weight(codes_all, start, end, gate_code=None)["nav"]

    con = psycopg2.connect(**DB); cur = con.cursor()
    try:
        data = _load_etf(cur, codes_all, start, end)
    finally:
        con.close()
    codes_list = list(data.keys())
    close_df = pd.DataFrame({c: data[c]["close"].astype(float) for c in codes_list})
    sma200 = close_df.rolling(200).mean()
    breadth_daily = (close_df > sma200).mean(axis=1)
    breadth_m = breadth_daily.resample("ME").last()

    trend_m = trend.resample("ME").last()
    ew_m = ew.resample("ME").last()
    trend_ret = trend_m.pct_change()
    ew_ret = ew_m.pct_change()

    feat = pd.DataFrame(index=trend_m.index)
    feat["breadth"] = breadth_m.reindex(trend_m.index)
    feat["ew_mom12"] = ew_m.pct_change(12)
    feat["trend_mom12"] = trend_m.pct_change(12)
    feat["vol6m"] = trend_ret.rolling(6).std() * np.sqrt(12)
    feat["dd12m"] = trend_m / trend_m.rolling(12).max() - 1
    feat["ret1m"] = trend_ret
    feat["ret3m"] = trend_ret.rolling(3).sum()
    feat["y"] = (trend_ret.shift(-1) > 0).astype(int)  # 下月符号，标签在当月
    feat = feat.dropna()
    return feat, trend_ret


def _fit_predict(kind, Xtr, ytr, Xte):
    """返回 (pred, prob)：pred=二元择时(0/1)，prob=下月上涨概率(连续, 供软加权)。

    注意：本环境 sklearn.LogisticRegression.fit() 会静默 segfault(二进制兼容bug)，
    故 ML 模型用 lightgbm/xgboost(正常)，DL 用 torch(正常)。
    """
    if kind == "gbm":
        import lightgbm as lgb
        m = lgb.LGBMClassifier(n_estimators=50, max_depth=3, num_leaves=7,
                               min_child_samples=5, verbose=-1).fit(Xtr, ytr)
        return int(m.predict(Xte)[0]), float(m.predict_proba(Xte)[0, 1])
    if kind == "xgb":
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1,
                              verbosity=0).fit(Xtr, ytr)
        return int(m.predict(Xte)[0]), float(m.predict_proba(Xte)[0, 1])
    if kind == "mlp":
        import torch
        Xtr_t = torch.tensor(np.asarray(Xtr), dtype=torch.float32)
        ytr_t = torch.tensor(np.asarray(ytr), dtype=torch.float32).unsqueeze(1)
        Xte_t = torch.tensor(np.asarray(Xte), dtype=torch.float32)
        torch.manual_seed(0)
        net = torch.nn.Sequential(
            torch.nn.Linear(Xtr.shape[1], 16), torch.nn.ReLU(), torch.nn.Dropout(0.3),
            torch.nn.Linear(16, 8), torch.nn.ReLU(), torch.nn.Dropout(0.3),
            torch.nn.Linear(8, 1), torch.nn.Sigmoid())
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        lossf = torch.nn.BCELoss()
        for _ in range(200):
            opt.zero_grad(); loss = lossf(net(Xtr_t), ytr_t); loss.backward(); opt.step()
        with torch.no_grad():
            p = net(Xte_t).item()
        return (1 if p >= 0.5 else 0), p
    raise ValueError(kind)


def walk_forward_ml(start="2019-01-01", end="2026-09-30", blend=0.0,
                    test_start="2023-01-01", models=("gbm", "xgb", "mlp"),
                    constant_fracs=(0.5, 0.6, 0.7), clean_only=True):
    """expanding-window walk-forward：每个测试月用此前全部历史训练、预测下月是否持有。

    对比：
      - always = 始终持有趋势策略(测试窗月度收益)
      - 各模型 hard = 预测1才全仓、否则现金；soft = 暴露=上涨概率(连续0~1)
      - constant_fracs = 空基准：永远持 f 比例风险仓(其余现金)，用于证伪"软加权=真择时 alpha"
        若 ML soft ≈ 某恒定比例，则说明只是降杠杆而非预测本事。
    返回 dict：每模型 hard/soft 指标 + always + constant + 各月决策数。
    """
    codes = load_etf_universe(start, end, clean_only)
    if not codes:
        raise DataError("ETF 池为空")
    panel, trend_ret = build_ml_panel(codes, start, end, blend)
    months = list(panel.index)
    X = panel.drop(columns=["y"]).values.astype(float)
    y = panel["y"].values.astype(int)

    port_hard = {k: [] for k in models}   # 二元择时：预测1才全仓，否则现金
    port_soft = {k: [] for k in models}   # 软加权择时：暴露=上涨概率，连续(0~1)
    hold_pos = {k: 0 for k in models}
    hold_tot = {k: 0 for k in models}
    exp_sum = {k: 0.0 for k in models}
    always = []
    for i in range(len(months)):
        if months[i] < pd.Timestamp(test_start):
            continue
        if i + 1 >= len(months):
            break
        outcome = float(trend_ret.loc[months[i + 1]])  # 下月趋势收益(预测后才"实现")
        tr_i = list(range(i))
        if len(tr_i) < 12:
            continue
        Xtr, ytr = X[tr_i], y[tr_i]
        Xte = X[i:i + 1]
        for k in models:
            pred, prob = _fit_predict(k, Xtr, ytr, Xte)
            hold_tot[k] += 1
            exp_sum[k] += prob
            if pred == 1:
                hold_pos[k] += 1
                port_hard[k].append(outcome)
            else:
                port_hard[k].append(0.0)
            # 软加权：收益 = 概率×风险收益 + (1−概率)×0(现金)，连续暴露降低 whip-saw
            port_soft[k].append(prob * outcome)
        always.append(outcome)

    out = dict(blend=blend, test_start=test_start, n_decisions=len(always),
               models={}, always=None, constant={})
    if always:
        # always / port 均为月度收益序列 -> 必须 freq='m'，否则按日频年化会错误放大
        a = metrics(pd.Series(always), freq="m")
        out["always"] = dict(ann=a["ann"], sharpe=a["sharpe"], mdd=a["mdd"])
        # 恒定比例空基准：暴露恒为 f，收益 = f×风险收益(检验"软加权是否只是降杠杆")
        always_s = pd.Series(always)
        for f in constant_fracs:
            mc = metrics(always_s * f, freq="m")
            out["constant"][f] = dict(ann=mc["ann"], sharpe=mc["sharpe"], mdd=mc["mdd"])
    for k in models:
        if port_hard[k]:
            mh = metrics(pd.Series(port_hard[k]), freq="m")
            ms = metrics(pd.Series(port_soft[k]), freq="m")
            out["models"][k] = dict(
                hard=dict(ann=mh["ann"], sharpe=mh["sharpe"], mdd=mh["mdd"],
                          hit=hold_pos[k] / hold_tot[k] if hold_tot[k] else 0.0,
                          hold_rate=hold_pos[k] / hold_tot[k] if hold_tot[k] else 0.0),
                soft=dict(ann=ms["ann"], sharpe=ms["sharpe"], mdd=ms["mdd"],
                          avg_exp=exp_sum[k] / hold_tot[k] if hold_tot[k] else 0.0))
    return out
