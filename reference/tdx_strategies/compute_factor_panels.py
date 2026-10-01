#!/usr/bin/env python3
"""
方向① (Part A) —— 因子面板 / 单因子 IC / 因子间相关性 计算（真实数据，cn_real_full 全 A）。

设计说明：
- 数据：cn_real_full（全 A ~5210 只，2020-01-02~2026-08-17）。
- 因子面板：用项目既有的 CustomFactorCalculator（AlphaAgent expr_parser + function_lib），
  与 BacktestRunner 回测用的是同一条因子计算链路（不是 test_factor_panel.py 里直接 D.features 的失败路径）。
- 计算窗口：测试窗口 2025-01-02~2026-08-17（样本外 OOS，用于权重设计最诚实）。
  用独立 cache 目录 + use_cache=False，避免旧缓存（全周期）被错误复用（缓存键只含表达式、不含窗口）。
- 标签：与回测完全一致 —— qlib D.features("Ref($close,-2)/Ref($close,-1)-1")。
- 输出：factor_panels.pkl / ic_per_factor.csv / corr_matrix.csv / panel_summary.json
"""
import os, sys, json, time
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]   # D:/qqmt/QuantaAlpha
sys.path.insert(0, str(PROJ))

import numpy as np
import pandas as pd

# 独立缓存目录，强制重算（避免窗口不一致的旧缓存复用）
FRESH_CACHE = PROJ / "data/results/factor_cache_corr"
FRESH_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["FACTOR_CACHE_DIR"] = str(FRESH_CACHE)
os.environ["JOBLIB_START_METHOD"] = "loky"

import qlib
from qlib.data import D

PROVIDER = "~/.qlib/qlib_data/cn_real_full"
TEST_START, TEST_END = "2025-01-02", "2026-08-17"
LABEL_EXPR = "Ref($close, -2) / Ref($close, -1) - 1"

# 9 个原子因子（战法1/2/4/6 拆解）
FACTOR_JSONS = [
    PROJ / "data/factorlib/tdx_warfare2_factors.json",   # F2 F3 F4
    PROJ / "data/factorlib/tdx_warfare146_factors.json", # F1 F6 F8 F9
]

def load_all_factors():
    factors = []
    for jp in FACTOR_JSONS:
        with open(jp, "r", encoding="utf-8") as f:
            data = json.load(f)
        for fid, info in data["factors"].items():
            factors.append({
                "factor_name": info["factor_name"],
                "factor_expression": info["factor_expression"],
                "src": jp.stem,
            })
    return factors

def main():
    t0 = time.time()
    qlib.init(provider_uri=PROVIDER, region="cn")
    inst = D.instruments("all")

    # ---- 标签（与回测一致）----
    label_df = D.features(inst, [LABEL_EXPR], start_time=TEST_START, end_time=TEST_END, freq="day")
    label_df.columns = ["LABEL0"]
    print(f"[label] shape={label_df.shape}")

    # ---- 因子面板：复用 CustomFactorCalculator ----
    from quantaalpha.backtest.custom_factor_calculator import CustomFactorCalculator, get_qlib_stock_data
    cfg = {
        "data": {
            "provider_uri": PROVIDER, "region": "cn", "market": "all",
            "start_time": TEST_START, "end_time": TEST_END,
        }
    }
    print("Loading OHLCV (test window, all-A) ...")
    data_df = get_qlib_stock_data(cfg)
    print(f"[ohlcv] shape={data_df.shape}  (loaded in {time.time()-t0:.1f}s)")

    calc = CustomFactorCalculator(data_df=data_df, cache_dir=FRESH_CACHE,
                                  auto_extract_cache=False, config=cfg)

    factors = load_all_factors()
    panels = {}
    for f in factors:
        name = f["factor_name"]
        t1 = time.time()
        ser = calc.calculate_factor(name, f["factor_expression"])
        if ser is None or ser.isna().all():
            print(f"  [WARN] factor {name} 计算失败/全 NaN，跳过")
            continue
        panels[name] = ser
        print(f"  [ok] {name:20s} non-null={ser.notna().sum():>9d}  fire_rate={ser.mean():.4f}  ({time.time()-t1:.1f}s)")

    if not panels:
        raise RuntimeError("没有任何因子面板计算成功")

    panel_df = pd.DataFrame(panels)
    print(f"[panels] shape={panel_df.shape}")

    # ---- 对齐标签 ----
    common = panel_df.index.intersection(label_df.index)
    panel_df = panel_df.loc[common]
    label_al = label_df.loc[common]["LABEL0"]
    print(f"[aligned] rows={len(panel_df)}")

    # ---- 单因子 IC / ICIR（截面 Pearson；Rank IC = Spearman）----
    dates = panel_df.index.get_level_values(0).unique()
    ic_rows = []
    for name in panel_df.columns:
        fcol = panel_df[name]
        per_day_ic, per_day_ric = [], []
        for d in dates:
            try:
                idx = fcol.xs(d, level=0).dropna().index
            except Exception:
                idx = fcol.loc[d].dropna().index
            sub_f = fcol.loc[d].reindex(idx)
            sub_l = label_al.loc[d].reindex(idx)
            m = sub_f.notna() & sub_l.notna()
            if m.sum() < 10:
                continue
            x = sub_f[m].astype(float).values
            y = sub_l[m].astype(float).values
            if np.std(x) < 1e-12 or np.std(y) < 1e-12:
                continue
            ic = np.corrcoef(x, y)[0, 1]
            ric = pd.Series(x).rank().corr(pd.Series(y).rank())
            if not np.isnan(ic):
                per_day_ic.append(ic)
            if not np.isnan(ric):
                per_day_ric.append(ric)
        ic_arr = np.array(per_day_ic)
        ric_arr = np.array(per_day_ric)
        ic_mean = float(ic_arr.mean()) if len(ic_arr) else float("nan")
        ric_mean = float(ric_arr.mean()) if len(ric_arr) else float("nan")
        icir = ic_mean / ic_arr.std() if len(ic_arr) and ic_arr.std() > 0 else 0.0
        ricir = ric_mean / ric_arr.std() if len(ric_arr) and ric_arr.std() > 0 else 0.0
        ic_rows.append({
            "factor": name,
            "IC": ic_mean, "ICIR": icir,
            "Rank_IC": ric_mean, "Rank_ICIR": ricir,
            "n_days": len(ic_arr),
            "fire_rate": float(fcol.mean()),
        })
        print(f"  {name:20s} IC={ic_mean:+.4f} ICIR={icir:+.3f} RankIC={ric_mean:+.4f} RankICIR={ricir:+.3f}")

    ic_df = pd.DataFrame(ic_rows).sort_values("ICIR", ascending=False)

    # ---- 因子间相关性（Pooled Pearson，二分信号）----
    corr = panel_df.corr(method="pearson")
    print("[corr matrix]")
    print(corr.round(3).to_string())

    # ---- 保存 ----
    out = PROJ / "data/results/direction1"
    out.mkdir(parents=True, exist_ok=True)
    panel_df.to_pickle(out / "factor_panels_testwin.pkl")
    ic_df.to_csv(out / "ic_per_factor_testwin.csv", index=False)
    corr.to_csv(out / "corr_matrix_testwin.csv")
    summary = {
        "window": f"{TEST_START}~{TEST_END}",
        "universe": "all (~5210)",
        "dataset": "cn_real_full",
        "label": LABEL_EXPR,
        "n_factors": len(panel_df.columns),
        "n_rows": len(panel_df),
        "ic_table": ic_df.to_dict(orient="records"),
        "corr": corr.round(4).to_dict(),
        "elapsed_seconds": time.time() - t0,
    }
    with open(out / "panel_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nDONE in {time.time()-t0:.1f}s. Outputs in {out}")
    print(ic_df[["factor", "IC", "ICIR", "Rank_IC", "Rank_ICIR", "fire_rate"]].to_string(index=False))

if __name__ == "__main__":
    main()
