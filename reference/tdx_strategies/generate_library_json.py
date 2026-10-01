#!/usr/bin/env python3
"""
[已归档/弃用] 本桥接脚本已被 reference/tdx_strategies/native_ingest.py 取代。
F1–F9 现通过该原生脚本经 FactorLibraryManager.add_factors_from_experiment 入库（真实 direction① 指标），
不再需要本手写桥接。保留仅作历史参考。

方向①→Web 数据桥：把已挖的 F1–F9（真实单因子指标）写成后端 /api/v1/factors 认得的
data/factorlib/all_factors_library_*.json 格式。纯本地文件桥接，非原生 mining 流水线产出。
真实指标来源：data/results/direction1/backtests/single_factor_summary.json
因子定义来源：data/factorlib/tdx_warfare2_factors.json + tdx_warfare146_factors.json
"""
import json, hashlib
from datetime import datetime
from pathlib import Path

ROOT = Path(".")
summary = json.loads(open("data/results/direction1/backtests/single_factor_summary.json", encoding="utf-8").read())
metric_by_factor = {s["factor"]: s["metrics"] for s in summary}

# 合并因子定义
defs = {}
for jp in ["data/factorlib/tdx_warfare2_factors.json", "data/factorlib/tdx_warfare146_factors.json"]:
    d = json.loads(open(jp, encoding="utf-8").read())
    for fid, info in d["factors"].items():
        defs[info["factor_name"]] = info

now = datetime.now().isoformat()
factors = {}
for fname, info in defs.items():
    m = metric_by_factor.get(fname)
    if not m:
        continue
    fid = hashlib.md5(f"{fname}_{info['factor_expression']}".encode()).hexdigest()[:16]
    bt = {
        "IC": m["IC"], "ICIR": m["ICIR"],
        "Rank IC": m["Rank IC"], "Rank ICIR": m["Rank ICIR"],
        "information_ratio": m["information_ratio"],
        "annualized_return": m["annualized_return"],
        "max_drawdown": m["max_drawdown"],
        # 同时满足后端 get_factors 的嵌套键（annualReturn/maxDrawdown/sharpeRatio 读取路径）
        "1day.excess_return_without_cost.annualized_return": m["annualized_return"],
        "1day.excess_return_with_cost.annualized_return": m["annualized_return"],
        "1day.excess_return_without_cost.max_drawdown": m["max_drawdown"],
        "1day.excess_return_with_cost.max_drawdown": m["max_drawdown"],
        "1day.excess_return_without_cost.information_ratio": m["information_ratio"],
        "1day.excess_return_with_cost.information_ratio": m["information_ratio"],
    }
    factors[fid] = {
        "factor_id": fid,
        "factor_name": info["factor_name"],
        "factor_expression": info["factor_expression"],
        "factor_description": info["factor_description"],
        "factor_formulation": "",
        "backtest_results": bt,
        "cache_location": {},
        "metadata": {
            "experiment_id": "tdx_warfare_bridge",
            "round_number": 0,
            "evolution_phase": "manual_bridge",
            "created_at": now,
            "source": "tdx 战法拆解（方向①），真实单因子回测指标",
        },
        "evolution_metadata": {"round": 0, "direction_index": ""},
        "added_at": now,
    }

out = {
    "metadata": {
        "created_at": now,
        "last_updated": now,
        "total_factors": len(factors),
        "version": "1.0",
        "note": "桥接文件：由 reference/tdx_strategies/generate_library_json.py 从真实单因子回测指标生成；"
                "非 quantaalpha factors/experiment 原生挖掘产出。原生产出应走 FactorLibraryManager.add_factors_from_experiment。",
    },
    "factors": factors,
}
out_path = ROOT / "data/factorlib/all_factors_library_tdx_warfare.json"
out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"写入 {out_path}：{len(factors)} 个因子")
for fid, f in factors.items():
    print(f"  {f['factor_name']:20s} IC={f['backtest_results']['IC']:+.4f} ICIR={f['backtest_results']['ICIR']:+.3f} IR={f['backtest_results']['information_ratio']:.3f}")
