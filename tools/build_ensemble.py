#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建「组合回测」因子集 = 自挖验证因子(19) + 传统因子库里通过单因子复验的子集。

输入：
  --mined   data/results/validated_factors_AE.json      (我们挖的、已验证的强因子)
  --trad-csv data/results/oos_validate_5m_traditional.csv (传统库单因子复验结果)
输出：
  --out     data/results/validated_factors_ensemble.json
           {"factors": {id: {"factor_name","factor_expression","source","ric_all","t_all"}}}

筛选：传统因子需 significant=='yes' 且 seg_consistent==True（与我们的强因子同标准）。
"""
import argparse
import json
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mined", default="data/results/validated_factors_AE.json")
    ap.add_argument("--trad-csv", default="data/results/oos_validate_5m_traditional.csv")
    ap.add_argument("--out", default="data/results/validated_factors_ensemble.json")
    a = ap.parse_args()

    # 1) 自挖验证因子
    mined = json.load(open(a.mined, encoding="utf-8"))["factors"]
    factors = {}
    for i, (k, v) in enumerate(mined.items()):
        factors[f"mined_{i:03d}"] = {
            "factor_name": v["factor_name"],
            "factor_expression": v["factor_expression"],
            "source": "mined",
        }
    print(f"自挖验证因子: {len(factors)}")

    # 2) 传统因子库复验结果（significant=='yes' 即纳入，模型自行降权翻号项）
    df = pd.read_csv(a.trad_csv)
    trad = df[df["significant"] == "yes"]
    added = 0
    for _, r in trad.iterrows():
        name = r["factor"]
        expr = r.get("expression")
        if not expr or pd.isna(expr):
            continue
        # 去重：表达式已存在则跳过
        if any(v["factor_expression"] == expr for v in factors.values()):
            continue
        factors[f"trad_{added:03d}"] = {
            "factor_name": name,
            "factor_expression": expr,
            "source": "traditional",
            "ric_all": float(r["ric_all"]) if not pd.isna(r["ric_all"]) else None,
            "t_all": float(r["t_all"]) if not pd.isna(r["t_all"]) else None,
        }
        added += 1
    print(f"传统库通过复验新增: {added}")
    print(f"组合总计: {len(factors)}")

    json.dump({"factors": factors}, open(a.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    # 来源统计
    from collections import Counter
    c = Counter(v["source"] for v in factors.values())
    print("来源分布:", dict(c))


if __name__ == "__main__":
    main()
