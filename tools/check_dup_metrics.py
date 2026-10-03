"""检测挖掘结果是否「指标恒定」——即不同因子却拿到同一个 RankIC。

为什么需要这个工具
------------------
2026-10-04：5min 挖掘跑了 17 条轨迹，因子表达式各不相同，但 RankIC 全部等于
0.04130751359431582（16 位小数一致）。原因是 conf 里漏了 StaticDataLoader，
模型从头到尾只吃到 4 个写死的原始特征 —— 等于 17 次都在评估同一个基线模型。
这个 bug 不会抛异常、不会报错、括号平衡、日志正常，唯一的症状就是
「不同输入的产出完全相同」。所以必须有一个专门的探针来抓它。

用法
----
    # 扫全部历史 workspace（默认）
    python tools/check_dup_metrics.py
    # 只看某次挖掘 session 的轨迹池
    python tools/check_dup_metrics.py --pool data/results/<session>/trajectory_pool.json
    # 只看最近 N 分钟内产出的 workspace
    python tools/check_dup_metrics.py --since-minutes 120

判定
----
同一 RankIC 值出现 >= 2 次 → FAIL（列出涉及哪些 workspace / 因子名）。
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WS_ROOT = ROOT / "git_ignore_folder" / "RD-Agent_workspace"


def _read_qlib_res(p: Path) -> dict:
    out = {}
    try:
        for line in p.read_text(encoding="utf-8").splitlines()[1:]:
            if "," in line:
                k, v = line.split(",", 1)
                out[k.strip()] = v.strip()
    except Exception:
        pass
    return out


def scan_workspaces(since_minutes: int = 0) -> dict:
    """返回 {rankic_value: [workspace_dir, ...]}"""
    cutoff = time.time() - since_minutes * 60 if since_minutes else 0
    groups = defaultdict(list)
    if not WS_ROOT.exists():
        return groups
    for ws in WS_ROOT.iterdir():
        res = ws / "qlib_res.csv"
        if not res.exists():
            continue
        if cutoff and res.stat().st_mtime < cutoff:
            continue
        m = _read_qlib_res(res)
        ric = m.get("Rank IC")
        if ric:
            groups[ric].append(ws.name)
    return groups


def scan_pool(pool_path: Path) -> dict:
    """从 trajectory_pool.json 里抽 (factor_name, metric) 做分组。"""
    groups = defaultdict(list)
    try:
        data = json.loads(pool_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"ERROR: cannot read {pool_path}: {exc}")
        return groups

    trajs = data if isinstance(data, list) else data.get("trajectories", [])
    for t in trajs:
        if not isinstance(t, dict):
            continue
        # 指标可能挂在 t["metric"] / t["primary_metric"] / t["feedback"] 下
        ric = None
        for key in ("metric", "primary_metric", "rank_ic", "RankIC"):
            v = t.get(key)
            if isinstance(v, (int, float)):
                ric = f"{v:.16f}"
                break
            if isinstance(v, dict) and "Rank IC" in v:
                ric = f"{v['Rank IC']:.16f}"
                break
        if ric is None:
            continue
        name = t.get("name") or t.get("factor_name") or t.get("hypothesis", "")[:40]
        groups[ric].append(str(name)[:60])
    return groups


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="", help="trajectory_pool.json 路径")
    ap.add_argument("--since-minutes", type=int, default=0, help="只看最近 N 分钟产出的 workspace")
    args = ap.parse_args()

    if args.pool:
        groups = scan_pool(Path(args.pool))
        scope = f"pool={args.pool}"
    else:
        groups = scan_workspaces(args.since_minutes)
        scope = f"workspace={WS_ROOT}" + (f" (last {args.since_minutes}min)" if args.since_minutes else "")

    total = sum(len(v) for v in groups.values())
    print(f"[check_dup_metrics] scope: {scope}")
    print(f"[check_dup_metrics] results scanned: {total}, distinct RankIC: {len(groups)}")

    dups = {k: v for k, v in groups.items() if len(v) >= 2}
    if not dups:
        print("[check_dup_metrics] PASS: no duplicated RankIC. 每个结果都有独立指标。")
        return 0

    print(f"[check_dup_metrics] FAIL: {len(dups)} 个 RankIC 值被多个结果共享 —— 极可能是「因子没进模型」")
    for ric, names in sorted(dups.items(), key=lambda kv: -len(kv[1])):
        print(f"  RankIC={ric}  出现 {len(names)} 次")
        for n in names[:8]:
            print(f"      - {n}")
        if len(names) > 8:
            print(f"      - ... 另有 {len(names)-8} 个")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
