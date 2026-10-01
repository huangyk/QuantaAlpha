#!/usr/bin/env python3
"""
方向① (Part C v2) —— 因子权重方案（用 Part B 真实【pipeline ICIR】，非 Part A 原始 IC）。
pipeline ICIR = LGBM + CSRankNorm 后模型预测对标签的 ICIR，反映因子"被正确使用时"的真实贡献。
原始 IC（线性、0/1 信号）多为负/近零，会误导权重；故以 pipeline ICIR 为准。
相关性矩阵沿用 Part A 真实算出的面板相关性。
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path("data/results/direction1")
summary = json.loads(open(OUT / "backtests/single_factor_summary.json", encoding="utf-8").read())
corr = pd.read_csv(OUT / "corr_matrix_testwin.csv", index_col=0)

# 按 Part A 面板顺序对齐因子（与 corr 矩阵一致）
factors = list(corr.columns)
icir = {s["factor"]: s["metrics"]["ICIR"] for s in summary}
mu = np.array([icir[f] for f in factors], dtype=float)

def show(name, w):
    s = np.array(w, dtype=float); tot = s.sum()
    pct = (s / tot * 100) if tot != 0 else s * 0
    print(f"\n[{name}] (sum={tot:.3f})")
    for f, p in zip(factors, pct):
        print(f"   {f:20s} {p:+7.1f}%")

# 1) ICIR 归一（多头，全正）
w1 = mu / mu.sum()
show("1) pipeline-ICIR 归一(多头)", w1)

# 2) 相关性调整 w ∝ C^{-1} mu
C = corr.loc[factors, factors].values.astype(float)
try:
    Cinv = np.linalg.inv(C)
    w2_raw = Cinv @ mu
    show("2a) 相关性调整(含负)", w2_raw)
    w2 = np.clip(w2_raw, 0, None)
    if w2.sum() > 0: w2 = w2 / w2.sum()
    show("2b) 相关性调整(多头clip)", w2)
except np.linalg.LinAlgError:
    w2 = w1.copy(); print("C 奇异，回退")

# 3) 实用：按 IR（信息比，风险调整后收益）归一，强调实战贡献
ir = {s["factor"]: s["metrics"]["information_ratio"] for s in summary}
v = np.array([ir[f] for f in factors], dtype=float)
w3 = v / v.sum()
show("3) 按组合 IR 归一(实用)", w3)

result = {
    "basis": "Part B pipeline ICIR（LGBM+CSRankNorm）；相关性=Part A 面板相关性",
    "factors": factors,
    "pipeline_ICIR": {f: float(icir[f]) for f in factors},
    "portfolio_IR": {f: float(ir[f]) for f in factors},
    "scheme1_icir_norm_pct": {f: float(x) for f, x in zip(factors, w1/mu.sum()*100)},
    "scheme2_corr_adj_pct": {f: float(x*100) for f, x in zip(factors, w2)},
    "scheme3_portfolio_IR_pct": {f: float(x*100) for f, x in zip(factors, w3)},
    "note": "以上为研究型手动权重；生产部署权重由 LGBM 复合（战法2）学出（ICIR=1.037）。",
}
with open(OUT / "weights_v2.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print("\nSaved weights_v2.json")
