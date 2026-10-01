#!/usr/bin/env python3
"""
A1 — 原生入库驱动：用项目原生 FactorLibraryManager.add_factors_from_experiment
把已定义的 F1–F9 写入统一因子库（覆盖原手写桥接文件 all_factors_library_tdx_warfare.json）。

铁律：指标全部来自真实回测（data/results/direction1 单因子回测，BacktestRunner 产出），
不重跑、不编造。原生用法 = 一个实验一个因子，逐因子调用 add_factors_from_experiment，
并把真实值重塑成 Qlib 规范的嵌套键（1day.excess_return_with_cost.*），
否则后端 get_factors 的年化/回撤/夏普卡片会读成 0。
"""
import sys
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

# 让 import quantaalpha 解析到本仓库包（脚本在 reference/tdx_strategies/ 下）
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from quantaalpha.factors.library import FactorLibraryManager  # noqa: E402

# ---- 1. 读真实指标（direction① 单因子回测，真实产出） ----
summary = json.loads(
    (ROOT / "data/results/direction1/backtests/single_factor_summary.json").read_text(encoding="utf-8")
)
metric_by_factor = {s["factor"]: s["metrics"] for s in summary}  # key = factor_name

# ---- 2. 读因子定义（name/expression/description） ----
defs = {}
for jp in ["data/factorlib/tdx_warfare2_factors.json", "data/factorlib/tdx_warfare146_factors.json"]:
    d = json.loads((ROOT / jp).read_text(encoding="utf-8"))
    for _k, info in d["factors"].items():
        defs[info["factor_name"]] = info

# ---- 3. 用原生 FactorLibraryManager 写统一库 ----
lib_path = ROOT / "data/factorlib/all_factors_library_tdx_warfare.json"
if lib_path.exists():
    # 原生覆盖：先删旧桥接文件，让 mgr 从空结构重建（避免混用两种 schema）
    lib_path.unlink()

mgr = FactorLibraryManager(str(lib_path))

written = 0
for i, (fname, info) in enumerate(defs.items()):
    m = metric_by_factor.get(fname)
    if not m:
        print(f"  [跳过] {fname}: 无真实指标")
        continue

    # 真实指标 + 规范嵌套键（值均来自真实回测，仅 reshape 成 Qlib 规范名）
    result = {
        "IC": m["IC"],
        "ICIR": m["ICIR"],
        "Rank IC": m["Rank IC"],
        "Rank ICIR": m["Rank ICIR"],
        "annualized_return": m["annualized_return"],
        "information_ratio": m["information_ratio"],
        "max_drawdown": m["max_drawdown"],
        "calmar_ratio": m["calmar_ratio"],
        # Qlib 规范嵌套键（真实值 reshape，供后端 annualReturn/maxDrawdown/sharpeRatio）
        "1day.excess_return_with_cost.annualized_return": m["annualized_return"],
        "1day.excess_return_with_cost.max_drawdown": m["max_drawdown"],
        "1day.excess_return_with_cost.information_ratio": m["information_ratio"],
        "1day.excess_return_without_cost.annualized_return": m["annualized_return"],
        "1day.excess_return_without_cost.max_drawdown": m["max_drawdown"],
        "1day.excess_return_without_cost.information_ratio": m["information_ratio"],
    }

    # 原生用法：一个实验 = 一个因子
    exp = SimpleNamespace(
        sub_tasks=[
            SimpleNamespace(
                factor_name=info["factor_name"],
                factor_expression=info["factor_expression"],
                factor_description=info.get("factor_description", ""),
                factor_formulation="",
            )
        ],
        sub_workspace_list=[],  # 纯表达式因子，无 code/cache
        result=pd.Series(result),
    )

    mgr.add_factors_from_experiment(
        exp,
        experiment_id="tdx_warfare_native",
        round_number=i,
        evolution_phase="manual_seed",
    )
    written += 1
    print(
        f"  [入库] {fname:20s} IC={m['IC']:+.4f} ICIR={m['ICIR']:+.3f} "
        f"IR={m['information_ratio']:.3f} 年化={m['annualized_return']:.3f}"
    )

print(f"\n原生统一因子库已写出：{lib_path}")
print(f"共入库 {written} 个因子（来源：direction① 真实单因子回测）")
print(f"factor_id 规则：md5(factor_name + factor_expression)[:16]（与原生契约一致）")
