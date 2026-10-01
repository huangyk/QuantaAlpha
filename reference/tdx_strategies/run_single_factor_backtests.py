#!/usr/bin/env python3
"""
方向① (Part B) —— 单因子回测（真实数据，cn_real_full 全 A，多周期）。
对 9 个原子因子逐一作为唯一特征跑 TopkDropoutStrategy 回测，与战法2 复合做横向比较。

复用 BacktestRunner（与战法2 多周期回测同一条链路）。
- F2/F3/F4 命中既有全周期缓存（data/results/factor_cache），不重算。
- F1/F6/F8/F9 现场计算（无缓存）。
输出：data/results/direction1/backtests/<factor>_backtest_metrics.json + batch_summary.json
"""
import os, sys, json, time
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))

import yaml
from quantaalpha.backtest.runner import BacktestRunner

CONFIG_PATH = PROJ / "configs/backtest_warfare2_multiyear.yaml"
FACTOR_JSONS = [
    PROJ / "data/factorlib/tdx_warfare2_factors.json",
    PROJ / "data/factorlib/tdx_warfare146_factors.json",
]
OUT_DIR = PROJ / "data/results/direction1/backtests"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_all_factors():
    out = {}
    for jp in FACTOR_JSONS:
        with open(jp, "r", encoding="utf-8") as f:
            data = json.load(f)
        for fid, info in data["factors"].items():
            out[info["factor_name"]] = info
    return out

def main():
    factors = load_all_factors()
    base_cfg = yaml.safe_load(open(CONFIG_PATH, "r", encoding="utf-8"))
    base_cfg["experiment"]["output_dir"] = str(OUT_DIR)
    base_cfg["experiment"]["output_metrics_file"] = "batch_summary.json"

    summary = []
    t_all = time.time()
    for fname, info in factors.items():
        # 单因子临时 json
        tmp = OUT_DIR / f"_tmp_{fname}.json"
        json.dump({"metadata": {"note": "single-factor"}, "factors": {fname: info}},
                  open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        t0 = time.time()
        try:
            runner = BacktestRunner(str(CONFIG_PATH))
            runner.config = base_cfg  # 内存覆盖 output_dir
            m = runner.run(
                factor_source="custom",
                factor_json=[str(tmp)],
                experiment_name=f"single_{fname}",
                output_name=fname,
            )
            entry = {"factor": fname, "metrics": m, "elapsed": time.time()-t0}
        except Exception as e:
            import traceback; traceback.print_exc()
            entry = {"factor": fname, "error": str(e)[:300], "elapsed": time.time()-t0}
        summary.append(entry)
        print(f">> {fname}: {entry.get('metrics', entry.get('error'))}  ({entry['elapsed']:.1f}s)")
        tmp.unlink(missing_ok=True)

    with open(OUT_DIR / "single_factor_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nALL DONE in {time.time()-t_all:.1f}s. Summary -> {OUT_DIR/'single_factor_summary.json'}")

if __name__ == "__main__":
    main()
