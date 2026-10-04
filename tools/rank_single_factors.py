"""按**单因子口径**给复验结果排序（而不是按合成 RankIC）。

为什么需要这个
--------------
2026-10-04 方向 D 证伪了「按合成 RankIC 选优」这条路：13 个 A×C 交叉因子在
LGB 多元模型里把合成 RankIC 推到 0.05（>基线 0.041），但作为**单因子**没有一个成立
（只有 3 个三段同号，且 |RankIC| 全 < 0.003）。原因是进化器的反馈只看合成 RankIC，
于是它持续往「和其他特征互补、但自身没有独立 alpha」的方向走。

本脚本就是对抗这个偏差的：合成 RankIC 再高也不看，排序完全基于
    1) 三段时间（train/valid/test）符号是否一致  —— 信号是不是真的
    2) |RankIC| 的量级                            —— 值不值得用
    3) 三段量级的稳定度 stab = min|seg| / max|seg| —— 会不会只是某一段冲出来的

用法
----
    python tools/rank_single_factors.py --csv data/results/oos_validate_5m_D.csv
    python tools/rank_single_factors.py --csv newest.csv --baseline older.csv \
        --out data/results/single_factor_ranking.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# 三档判据的阈值（|RankIC|）。0.008 是经验线：5min 全样本 52 个因子里，
# 三段同号且 |RIC|>=0.008 的只有 8 个，低于这个量级的与「另一种噪声」难以区分。
STRONG = 0.008
CANDIDATE = 0.003


def _sign(v) -> int:
    if pd.isna(v):
        return 0
    return 1 if v > 0 else (-1 if v < 0 else 0)


def _strip_prefix(name: str) -> str:
    """'F45_Volatility_Adj_Efficiency_20D' -> 'Volatility_Adj_Efficiency_20D'。

    oos_validate 给的名字带 F<idx>_ 前缀（idx 是它在当次池子里的序号），
    跨轮次不可比，必须剥掉才能跟历史 CSV 对上。
    """
    parts = str(name).split("_", 1)
    return parts[1] if len(parts) > 1 and parts[0].startswith("F") and parts[0][1:].isdigit() else str(name)


def rank(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["name"] = out["factor"].map(_strip_prefix)

    segs = ["ric_train", "ric_valid", "ric_test"]
    for c in segs + ["ric_all"]:
        if c not in out.columns:
            raise SystemExit(f"CSV 缺少列 {c}；请确认传的是 oos_validate 产出的文件")

    out["seg_consistent"] = [
        len({_sign(r.ric_train), _sign(r.ric_valid), _sign(r.ric_test)}) == 1 for r in out.itertuples()
    ]
    mags = out[segs].abs()
    out["stab"] = (mags.min(axis=1) / mags.max(axis=1)).round(3)
    out["abs_ric"] = out["ric_all"].abs()

    def verdict(r) -> str:
        if pd.isna(r.ric_all):
            return "复算失败"
        if not r.seg_consistent:
            return "不稳定(跨段反号)" if r.abs_ric >= CANDIDATE else "噪声"
        if r.abs_ric >= STRONG:
            return "强"
        if r.abs_ric >= CANDIDATE:
            return "候选"
        return "同号但量级过小"

    out["verdict"] = [verdict(r) for r in out.itertuples()]
    # 同为正/负方向时，按「符号一致的负号因子也值得看」的口径排：
    # 主要键是量级，次要键是稳定度。
    return out.sort_values(["abs_ric", "stab"], ascending=False)


def main() -> int:
    ap = argparse.ArgumentParser(description="按单因子口径排序复验结果")
    ap.add_argument("--csv", required=True, help="oos_validate 产出的 CSV")
    ap.add_argument("--baseline", default=None, help="上一轮 CSV，用来标出哪些因子是本轮新增")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--only-consistent", action="store_true", help="只列三段同号的")
    ap.add_argument("--out", default=None, help="写出排序后的完整 CSV")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    ranked = rank(df)

    if args.baseline:
        old = set(_strip_prefix(n) for n in pd.read_csv(args.baseline)["factor"])
        ranked["is_new"] = [n not in old for n in ranked["name"]]
    else:
        ranked["is_new"] = None

    cols = ["name", "verdict", "ric_all", "ric_train", "ric_valid", "ric_test", "stab", "seg_consistent"]
    if ranked["is_new"].notna().any():
        cols.append("is_new")

    show = ranked if not args.only_consistent else ranked[ranked["seg_consistent"]]
    print(f"\n单因子口径排序（共 {len(ranked)} 个；三段同号 {int(ranked['seg_consistent'].sum())} 个）")
    print("=" * 104)
    print(show[cols].head(args.top).to_string(index=False, float_format=lambda v: f"{v:+.5f}"))

    print("\n分档统计：")
    for k, v in ranked["verdict"].value_counts().items():
        print(f"  {k:<16} {v}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        ranked.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"\n已写：{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
