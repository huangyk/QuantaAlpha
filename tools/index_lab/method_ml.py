"""ML / DL 择时叠加层：在已验证的趋势策略上，用模型判断"下个月该不该持有"。

动机：用户指出前面全是古典规则(SMA200+逆波动)，没有机器学习/深度学习。
本模块把 ML/DL 作为**择时叠加层**（不替换已验证策略，只在月频调仓点决定是否持有），
严格 expanding-window walk-forward（每个测试月用此前全部历史训练、预测下月、再实现收益，
无前视）。模型：LightGBM(机器学习) / XGBoost(机器学习) / 小 PyTorch MLP(深度学习)。

特征两套：build_ml_panel(v1, 7维) / build_ml_panel_v2(增强: 多期限宽度+风险偏好+
期限利差+黄金+横截面离散度+波动率之波动, 仅用现有ETF数据无外部源)。
逐年滚动 OOS(rolling_walk_forward_ml) + 恒定比例空基准证伪软加权=降杠杆而非真择时alpha。

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
    feat["ew_mom12"] = ew_m.pct_change(12, fill_method=None)
    feat["trend_mom12"] = trend_m.pct_change(12, fill_method=None)
    feat["vol6m"] = trend_ret.rolling(6).std() * np.sqrt(12)
    feat["dd12m"] = trend_m / trend_m.rolling(12).max() - 1
    feat["ret1m"] = trend_ret
    feat["ret3m"] = trend_ret.rolling(3).sum()
    feat["y"] = (trend_ret.shift(-1) > 0).astype(int)  # 下月符号，标签在当月
    feat = feat.dropna()
    return feat, trend_ret


def _fit_predict(kind, Xtr, ytr, Xte, epochs=200):
    """返回 (pred, prob)：pred=二元择时(0/1)，prob=下月上涨概率(连续, 供软加权)。

    epochs: MLP 训练轮数（周频大量重训时可调小以提速；默认200保持月频原行为）。
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
        for _ in range(epochs):
            opt.zero_grad(); loss = lossf(net(Xtr_t), ytr_t); loss.backward(); opt.step()
        with torch.no_grad():
            p = net(Xte_t).item()
        return (1 if p >= 0.5 else 0), p
    raise ValueError(kind)


def _wf_ml_collect(panel, trend_ret, test_start, models):
    """walk-forward 核心：返回 [(year, outcome, {k:(pred,prob)}), ...] 供整体/逐年聚合复用。

    每个测试月用此前全部历史训练、预测下月是否持有；outcome 是下月趋势收益(预测后才实现)，无前视。
    """
    months = list(panel.index)
    X = panel.drop(columns=["y"]).values.astype(float)
    y = panel["y"].values.astype(int)
    rec = []
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
        yr = months[i].year
        preds = {}
        for k in models:
            pred, prob = _fit_predict(k, Xtr, ytr, Xte)
            preds[k] = (pred, prob)
        rec.append((yr, outcome, preds))
    return rec


def walk_forward_ml(start="2019-01-01", end="2026-09-30", blend=0.0,
                    test_start="2023-01-01", models=("gbm", "xgb", "mlp"),
                    constant_fracs=(0.5, 0.6, 0.7), clean_only=True,
                    panel_fn=build_ml_panel):
    """expanding-window walk-forward：每个测试月用此前全部历史训练、预测下月是否持有。

    对比：
      - always = 始终持有趋势策略(测试窗月度收益)
      - 各模型 hard = 预测1才全仓、否则现金；soft = 暴露=上涨概率(连续0~1)
      - constant_fracs = 空基准：永远持 f 比例风险仓(其余现金)，用于证伪"软加权=真择时 alpha"
      - yearly = 按日历年切的 OOS 子窗(逐年稳定性，呼应 blend 的 rolling_walk_forward)
    返回 dict：always / models(hard,soft) / constant / yearly + 决策数。
    """
    codes = load_etf_universe(start, end, clean_only)
    if not codes:
        raise DataError("ETF 池为空")
    panel, trend_ret = panel_fn(codes, start, end, blend)
    rec = _wf_ml_collect(panel, trend_ret, test_start, models)

    always = [r[1] for r in rec]
    port_hard = {k: [] for k in models}
    port_soft = {k: [] for k in models}
    hold_pos = {k: 0 for k in models}
    hold_tot = {k: 0 for k in models}
    exp_sum = {k: 0.0 for k in models}
    always_y = {}            # year -> list
    soft_y = {k: {} for k in models}
    for yr, outcome, preds in rec:
        always_y.setdefault(yr, []).append(outcome)
        for k in models:
            pred, prob = preds[k]
            hold_tot[k] += 1
            exp_sum[k] += prob
            if pred == 1:
                hold_pos[k] += 1
                port_hard[k].append(outcome)
            else:
                port_hard[k].append(0.0)
            port_soft[k].append(prob * outcome)
            soft_y[k].setdefault(yr, []).append(prob * outcome)

    out = dict(blend=blend, test_start=test_start, n_decisions=len(always),
               models={}, always=None, constant={}, yearly={})
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
    # 逐年 OOS 子窗(各年决策用此前全部历史，天然无前视)
    for yr in sorted(always_y):
        aym = metrics(pd.Series(always_y[yr]), freq="m")
        yd = dict(always=dict(ann=aym["ann"], sharpe=aym["sharpe"], mdd=aym["mdd"]),
                  soft={}, const60=dict(**metrics(pd.Series(always_y[yr]) * 0.6, freq="m")))
        for k in models:
            if soft_y[k].get(yr):
                sm = metrics(pd.Series(soft_y[k][yr]), freq="m")
                yd["soft"][k] = dict(ann=sm["ann"], sharpe=sm["sharpe"], mdd=sm["mdd"])
        out["yearly"][yr] = yd
    return out


def rolling_walk_forward_ml(start="2019-01-01", end="2026-09-30", blend=0.0,
                            first_test_year=2022, models=("gbm", "xgb", "mlp"),
                            clean_only=True, panel_fn=build_ml_panel):
    """逐年滚动 OOS 稳定性：从 first_test_year 起每个日历年独立看(训练窗=此前全部历史)。

    直接复用 walk_forward_ml 的 yearly 切片；返回 {yearly, always, models, test_start}。
    """
    r = walk_forward_ml(start=start, end=end, blend=blend,
                        test_start=f"{first_test_year}-01-01", models=models,
                        clean_only=clean_only, panel_fn=panel_fn)
    return dict(yearly=r["yearly"], always=r["always"], models=r["models"],
                test_start=r["test_start"], n_decisions=r["n_decisions"])


GOLD_CODES = {"518880.SH"}
TERM_LONG = "511090.SH"
TERM_SHORT = "511010.SH"


def _group_ew_nav(close_df, codes_subset):
    """组内等权净值(各标的日收益等权平均后复利)，用于风险偏好/期限利差/黄金代理。

    用收益复利而非'除以首值'归一化——规避部分 ETF 上市晚导致首行全 NaN 使整条 nav 作废的坑。
    """
    if not codes_subset:
        return None
    sub = close_df[[c for c in codes_subset if c in close_df.columns]]
    if sub.empty:
        return None
    ret = sub.pct_change().fillna(0.0)          # 首日收益记0；上市前 NaN 经 skipna 排除
    ew_ret = ret.mean(axis=1)                    # 默认 skipna=True，等权用当时可得标的
    nav = (1.0 + ew_ret).cumprod()
    return nav


def build_ml_panel_v2(codes_all, start, end, blend=0.0):
    """增强特征面板(仅用现有 ETF 数据，无外部源)：

    多期限宽度 breadth50/100/200；全池动量 ew_mom12；风险偏好 risk_appetite=
    非债券组12-1动量 − 债券组12-1动量；黄金动量 gold_mom3；期限利差 term_spread=
    30年国债3月动量 − 短国债3月动量；横截面离散度 dispersion(6月收益截面std)；
    波动率之波动 vol_of_vol；近6月波动 vol6m；近12月回撤 dd12m；上月/近3月收益。
    目标 y：下月趋势策略收益>0。全部月末、无前视。
    """
    if blend <= 0:
        trend = abs_momentum_etf(codes_all, start, end, signal="sma200",
                                 weight="inv_vol")["nav"]
    else:
        trend = blend_absmomentum(codes_all, start, end, blend)["nav"]

    con = psycopg2.connect(**DB); cur = con.cursor()
    try:
        data = _load_etf(cur, codes_all, start, end)
    finally:
        con.close()
    codes_list = list(data.keys())
    close_df = pd.DataFrame({c: data[c]["close"].astype(float) for c in codes_list})

    bond = [c for c in codes_list if is_bond_cash(c)]
    nonbond = [c for c in codes_list if not is_bond_cash(c)]
    # 多期限宽度
    breadth = {}
    for w in (50, 100, 200):
        sma = close_df.rolling(w).mean()
        breadth[w] = (close_df > sma).mean(axis=1).resample("ME").last()
    # 组动量
    bond_nav = _group_ew_nav(close_df, bond)
    nonbond_nav = _group_ew_nav(close_df, nonbond)
    bond_m = bond_nav.resample("ME").last() if bond_nav is not None else None
    nonbond_m = nonbond_nav.resample("ME").last() if nonbond_nav is not None else None
    # 黄金 / 期限利差(单标的动量)
    gold_nav = _group_ew_nav(close_df, [c for c in GOLD_CODES if c in codes_list])
    gold_m = gold_nav.resample("ME").last() if gold_nav is not None else None
    tl = close_df[TERM_LONG].resample("ME").last() if TERM_LONG in close_df else None
    ts = close_df[TERM_SHORT].resample("ME").last() if TERM_SHORT in close_df else None

    trend_m = trend.resample("ME").last()
    trend_ret = trend_m.pct_change()
    # 横截面离散度：每月各 ETF 收益，6月滚动截面 std
    ret_m = close_df.resample("ME").last().pct_change()
    disp = ret_m.rolling(6).std().mean(axis=1)

    feat = pd.DataFrame(index=trend_m.index)
    feat["breadth50"] = breadth[50].reindex(trend_m.index)
    feat["breadth100"] = breadth[100].reindex(trend_m.index)
    feat["breadth200"] = breadth[200].reindex(trend_m.index)
    feat["ew_mom12"] = trend_m.pct_change(12, fill_method=None)
    if nonbond_m is not None and bond_m is not None:
        feat["risk_appetite"] = nonbond_m.pct_change(12, fill_method=None) - bond_m.pct_change(12, fill_method=None)
    if gold_m is not None:
        feat["gold_mom3"] = gold_m.pct_change(3, fill_method=None)
    if tl is not None and ts is not None:
        feat["term_spread"] = tl.pct_change(3, fill_method=None) - ts.pct_change(3, fill_method=None)
    feat["dispersion"] = disp.reindex(trend_m.index)
    feat["vol6m"] = trend_ret.rolling(6).std() * np.sqrt(12)
    feat["vol_of_vol"] = feat["vol6m"].rolling(6).std()
    feat["dd12m"] = trend_m / trend_m.rolling(12).max() - 1
    feat["ret1m"] = trend_ret
    feat["ret3m"] = trend_ret.rolling(3).sum()
    feat["y"] = (trend_ret.shift(-1) > 0).astype(int)
    feat = feat.dropna()
    return feat, trend_ret
