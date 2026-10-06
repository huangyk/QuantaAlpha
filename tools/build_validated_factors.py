"""从 oos_validate 一致 t 统计量表导出「已验证因子」为回测 JSON。

筛选标准（见 mining_5m_E_result.md §4 口径）：
  significant == 'yes'  且  seg_ok == True  （三段同号 + |t|>=2）
该口径下通过的是真正稳健的单因子信号，可直接喂 run_backtest 做组合回测。

用法：
  python tools/build_validated_factors.py \
      --csv data/results/oos_validate_5m_ALL_v2.csv \
      --out data/results/validated_factors.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="oos_validate 输出的全池 CSV")
    ap.add_argument("--out", required=True, help="导出的回测因子 JSON 路径")
    ap.add_argument("--min-ric", type=float, default=0.0,
                    help="额外门槛：|ric_all| >= 该值才入选（默认 0，只看 significant+seg_ok）")
    ap.add_argument("--include-flips", action="store_true",
                    help="也纳入『significant 但 valid 翻号』的因子（两端同号、中段噪声）")
    args = ap.parse_args()

    d = pd.read_csv(args.csv)
    need = (d["significant"] == "yes") & (d["seg_consistent"] == "yes")
    if args.include_flips:
        need = need | ((d["significant"] == "yes") & (d["seg_consistent"] == "no"))
    if args.min_ric:
        need = need & (d["abs_ric_all"] >= args.min_ric)
    sub = d[need].sort_values("ric_all", key=abs, ascending=False)

    factors = {}
    for _, r in sub.iterrows():
        name = str(r["factor"])
        expr = str(r.get("expression", ""))
        if not expr or expr.lower() in ("nan", "none"):
            continue
        factors[name] = {"factor_name": name, "factor_expression": expr}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"factors": factors}, out.open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"导出 {len(factors)} 条已验证因子 -> {out}")
    print(sub[["factor", "ric_all", "t_all"]].to_string(index=False,
          float_format=lambda v: f"{v:+.5f}"))


if __name__ == "__main__":
    main()
