"""ML 择时层（周频，806 周大样本）——检验"ML 是否有 alpha"。

设计（对齐项目铁律）：
- 严格 expanding-window walk-forward：第 t 周用此前全部历史训练，预测第 t+1 周，收益在预测后才实现，无前视。
- 特征只用 ≤t 周数据；标签 y[t] = 第 t+1 周等权指数收益>0。
- 关键证伪：与「恒定暴露」空基准对比。若 ML 软暴露 ≈ 某恒定比例，说明只是降杠杆而非预测本事。
- 防过拟合：训练窗内选阈，测试窗只评估一次；报告逐年与 t 值。
"""
import numpy as np
import pandas as pd

from tools.index_lab.method_ml import _fit_predict   # 复用 LightGBM/XGBoost/MLP
from . import features as F
from .backtest import weekly_metrics, yearly_table


def build_weekly_features(wk, elig):
    """周频特征（只用 ≤当周数据）：多期限宽度 + 等权指数动量/波动/回撤 + 上周收益。"""
    idx = F.equal_weight_index(wk, elig)
    ret = idx.pct_change(fill_method=None)
    feat = pd.DataFrame(index=wk.index)
    for ma in (10, 20, 40, 52):
        feat[f"breadth{ma}"] = F.breadth(wk, elig, ma_weeks=ma)
    feat["mom4"] = idx.pct_change(4)
    feat["mom13"] = idx.pct_change(13)
    feat["mom52"] = idx.pct_change(52)
    feat["vol13"] = ret.rolling(13).std() * np.sqrt(52)
    feat["dd52"] = idx / idx.rolling(52).max() - 1
    feat["ret1"] = ret
    return feat


def walk_forward_weekly_ml(wk, elig, models=("gbm", "xgb", "mlp"),
                           min_train=104, test_start="2015-01-01", cash_ann=0.025,
                           mlp_epochs=60):
    """周频 ML 择时 walk-forward。返回各模型 hard/soft 收益 + 基线。"""
    feat = build_weekly_features(wk, elig).dropna()   # 去早期NaN(52周动量/宽度需热身)
    idx = F.equal_weight_index(wk, elig)
    stock_ret = idx.pct_change(fill_method=None).reindex(feat.index)
    cash_w = (1 + cash_ann) ** (1 / 52) - 1
    X = feat.values.astype(float)
    months = list(feat.index)
    n = len(months)

    hard = {k: [] for k in models}
    soft = {k: [] for k in models}
    dates, stock_r = [], []
    for i in range(n):
        if months[i] < pd.Timestamp(test_start) or i + 1 >= n:
            continue
        if i < min_train:
            continue
        # 训练只用截至第 i 周已知的标签：特征周 t(0..i-1) 对应标签=第 t+1 周收益>0
        # y = stock_ret[1..i]（i 个），Xtr = X[0..i-1]（i 个），严格无前视
        y = (stock_ret.iloc[1:i + 1] > 0).values.astype(int)
        Xtr = X[:i]
        Xte = X[i:i + 1]
        realized = stock_ret.iloc[i + 1]
        if not np.isfinite(realized) or len(y) < 12 or len(set(y.tolist())) < 2:
            continue
        for k in models:
            ep = mlp_epochs if k == "mlp" else 200
            pred, prob = _fit_predict(k, Xtr, y, Xte, epochs=ep)
            hard[k].append(realized if pred == 1 else cash_w)
            soft[k].append(prob * realized + (1 - prob) * cash_w)
        dates.append(months[i + 1]); stock_r.append(realized)
    dates = pd.DatetimeIndex(dates); stock_r = pd.Series(stock_r, index=dates)
    out = dict(dates=dates, n=len(dates), models={}, baselines={})
    sr = stock_r
    out["baselines"]["长期持股"] = sr
    out["baselines"]["长期持现金"] = pd.Series(cash_w, index=dates)
    out["baselines"]["固定60股40现"] = 0.6 * sr + 0.4 * cash_w
    for k in models:
        out["models"][k] = dict(hard=pd.Series(hard[k], index=dates),
                                soft=pd.Series(soft[k], index=dates))
    return out


def report(out, cash_ann=0.025):
    """对比 ML vs 基线，并给出恒定暴露空基准（证伪 ML 是否只是降杠杆）。"""
    dates = out["dates"]; sr = out["baselines"]["长期持股"]
    cash_w = (1 + cash_ann) ** (1 / 52) - 1
    rows = []
    def add(label, s):
        m = weekly_metrics(s); yr = yearly_table(s)
        rows.append(dict(cfg=label, ann=m["ann"], sharpe=m["sharpe"], mdd=m["mdd"],
                         正年=f"{(yr>0).sum()}/{len(yr)}"))
    add("长期持股", sr)
    add("固定60股40现", 0.6 * sr + 0.4 * cash_w)
    for k, v in out["models"].items():
        add(f"{k}-硬择时", v["hard"]); add(f"{k}-软暴露", v["soft"])
    # 恒定暴露空基准
    for f in (0.4, 0.5, 0.6, 0.7, 0.8):
        add(f"恒定{f*100:.0f}%暴露", f * sr + (1 - f) * cash_w)
    return pd.DataFrame(rows)
