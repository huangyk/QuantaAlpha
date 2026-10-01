"""Backtest driver for 通达信战法 2 (均线回踩低吸) — 多周期验证 (方向②).

LLM-free: reads tdx_warfare2_factors.json, computes factors via AlphaAgent expr_parser,
trains lgb, runs TopK backtest on cn_real_full (2020-2026).
Outputs metrics to data/results/backtest_warfare2_multiyear/backtest_metrics.json.
"""
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJ, ".env"))

from quantaalpha.backtest.runner import BacktestRunner

cfg = os.environ.get("BT_CONFIG") or os.path.join(PROJ, "configs", "backtest_warfare2_multiyear.yaml")
fj = [os.path.join(PROJ, "data", "factorlib", "tdx_warfare2_factors.json")]

if __name__ == "__main__":
    r = BacktestRunner(cfg)
    metrics = r.run(
        factor_source="custom",
        factor_json=fj,
        experiment_name="tdx_warfare2_multiyear",
        output_name="tdx_warfare2_multiyear",
    )
    print("===BACKTEST_METRICS===")
    print(metrics)
