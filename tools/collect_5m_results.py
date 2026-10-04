"""收集 5min 挖掘结果：把每个 workspace 的因子名 + 指标汇总成一张表。

为什么需要
----------
挖掘过程中 read_exp_res.py 只在 workspace 里留下一份 qlib_res.csv，
pipeline 侧的轨迹池里 RankIC 可能是 None（比如校验误杀，见 commit 87d7bf7）。
而 recorder 里其实有真实指标 —— 这个工具从 workspace 反向把结果捞回来，
并按 RankIC 排序，方便挑因子。

用法
----
    python tools/collect_5m_results.py                    # 全部 5min workspace
    python tools/collect_5m_results.py --since-minutes 240 # 只看最近 4 小时
    python tools/collect_5m_results.py --top 15
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WS_ROOT = ROOT / "git_ignore_folder" / "RD-Agent_workspace"
BASELINE_RIC = 0.04130751359431582  # 4 特征基线（因子没进模型时的常数）


def _read_metrics(p: Path) -> dict:
    out = {}
    try:
        txt = p.read_text(encoding="utf-8")
    except Exception:
        return out
    for line in txt.splitlines()[1:]:
        if "," in line:
            k, v = line.split(",", 1)
            try:
                out[k.strip()] = float(v.strip())
            except ValueError:
                pass
    return out


def collect(since_minutes: int = 0) -> pd.DataFrame:
    cutoff = time.time() - since_minutes * 60 if since_minutes else 0
    rows = []
    if not WS_ROOT.exists():
        return pd.DataFrame()
    for ws in WS_ROOT.iterdir():
        if not (ws / "conf_5min_nextday.yaml").exists():
            continue  # 只要 5min 轮
        pq = ws / "combined_factors_df.parquet"
        res = ws / "qlib_res.csv"
        if not pq.exists() or not res.exists():
            continue
        if cutoff and res.stat().st_mtime < cutoff:
            continue
        try:
            cols = [c[1] for c in pd.read_parquet(pq).columns]
        except Exception:
            cols = []
        m = _read_metrics(res)
        ric = m.get("Rank IC")
        rows.append({
            "workspace": ws.name[:8],
            "n_factors": len(cols),
            "factors": " | ".join(cols),
            "RankIC": ric,
            "IC": m.get("IC"),
            "ICIR": m.get("ICIR"),
            "RankICIR": m.get("Rank ICIR"),
            "vs_baseline": (ric - BASELINE_RIC) if ric is not None else None,
            "mtime": pd.to_datetime(res.stat().st_mtime, unit="s").strftime("%m-%d %H:%M"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("RankIC", ascending=False, na_position="last")
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since-minutes", type=int, default=0)
    ap.add_argument("--top", type=int, default=0)
    ap.add_argument("--out", default="data/results/collect_5m.csv")
    args = ap.parse_args()

    df = collect(args.since_minutes)
    if df.empty:
        print("[collect_5m] 没有找到 5min 结果")
        return 0

    df = df[df["RankIC"].notna()]
    if args.top:
        df = df.head(args.top)

    print(f"[collect_5m] {len(df)} 条结果（按 RankIC 降序）")
    print(f"[collect_5m] 4 特征基线 RankIC = {BASELINE_RIC:.6f}，vs_baseline>0 表示因子有增益\n")
    show = df[["mtime", "workspace", "RankIC", "IC", "RankICIR", "vs_baseline", "factors"]]
    print(show.to_string(index=False, max_colwidth=70, float_format=lambda x: f"{x:.6f}"))

    if args.out:
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n[collect_5m] 已保存 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
