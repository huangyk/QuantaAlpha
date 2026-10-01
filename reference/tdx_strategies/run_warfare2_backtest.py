"""Backtest driver for 通达信战法 2 (均线回踩低吸) on QuantaAlpha.

Uses BacktestRunner directly (LLM-free): reads tdx_warfare2_factors.json,
computes factors via AlphaAgent expr_parser, trains lgb, runs TopK backtest.
Outputs metrics to data/results/backtest_v2_results/backtest_metrics.json.
"""
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJ, ".env"))

from quantaalpha.backtest.runner import BacktestRunner

cfg = os.path.join(PROJ, "configs", "backtest_warfare2.yaml")
fj = [os.path.join(PROJ, "data", "factorlib", "tdx_warfare2_factors.json")]

if __name__ == "__main__":
    r = BacktestRunner(cfg)
    metrics = r.run(
        factor_source="custom",
        factor_json=fj,
        experiment_name="tdx_warfare2",
        output_name="tdx_warfare2",
    )
    print("===BACKTEST_METRICS===")
    print(metrics)
