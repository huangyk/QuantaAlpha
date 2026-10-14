"""交叉核对所有 ETF 结果 JSON，确认 docs/etf_config_menu.md 里的数字无误。"""
import json
import os

d = os.environ.get("QA_OUT_DIR", ".")
files = [
    "etf_tsmom_full_out.json",
    "etf_tsmom_variants_out.json",
    "etf_vt_lambda_grid_out.json",
    "etf_pool_sensitivity_v4_out.json",
    "etf_algo_align_out.json",
    "etf_recalib_out.json",
]
for f in files:
    p = os.path.join(d, f)
    if not os.path.exists(p):
        print(f, "MISSING")
        continue
    j = json.load(open(p))
    print("===", f, "===")

    def show(k, v, indent="  "):
        if isinstance(v, dict) and "ann" in v:
            ann = v["ann"] * 100
            s = v.get("sharpe", 0)
            mdd = v.get("mdd", 0) * 100
            print(f"{indent}{k}: ann={ann:+.2f}% S={s:.2f} mdd={mdd:+.1f}%")

    if isinstance(j, dict):
        for k, v in j.items():
            show(k, v)
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    show(k2, v2, indent="    ")
