"""快速探针对照：定位"复算成功但 IC 全 NaN"的因子到底哪一步塌了。

跑在**小样本**上（少量标的 + 一年），几条表达式几十秒就能出结果，
不用等主复验那十几分钟的全市场 D.features。

用法：
  QLIB_DATA_DIR=... PYTHONPATH=tools python tools/_nan_probe.py --test F0
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")

ap = argparse.ArgumentParser()
ap.add_argument("--test", default="all", help="F0/F5/F12/F14/all")
ap.add_argument("--n-inst", type=int, default=60)
ap.add_argument("--start", default="2025-01-01")
ap.add_argument("--end", default="2025-12-31")
args = ap.parse_args()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import qlib  # noqa: E402
from qlib.data import D  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import oos_validate as ov  # noqa: E402

qlib.init(provider_uri=os.environ.get("QLIB_DATA_DIR", ov.DEFAULT_QLIB_DIR), region="cn")

# 与主复验完全相同的归一化，保证探的是同一条路径
CANDIDATES = {
    "F0": "TS_SUM(ABS($return), 5) / (TS_STD($return, 5) + 1e-8) * "
          "SIGN(TS_STD(ABS($return), 5) - TS_QUANTILE(ABS($return), 60, 0.75)) * "
          "SIGN($volume - 1.2 * TS_MEAN($volume, 20))",
    "F5": "TS_ZSCORE($return, 10) - TS_ZSCORE($volume, 10)",
    "F12": "RANK(-1 * (TS_STD($return, 20) / (TS_STD($return, 60) + 1e-8)) * TS_SKEW($return, 20))",
    "F14": "RANK(TS_SKEW($return, 20) * (TS_STD($return, 60) / (TS_STD($return, 20) + 1e-8)))",
}

known = set(__import__("qlib.data.ops", fromlist=["Operators"]).Operators._ops.keys())

codes = D.list_instruments(D.instruments("all"), start_time=args.start, end_time=args.end, as_list=True)
codes = codes[: args.n_inst]
print(f"[probe] instruments={len(codes)}")

targets = CANDIDATES if args.test == "all" else {args.test: CANDIDATES[args.test]}
for tag, raw in targets.items():
    expr = ov.normalize_expr(raw)
    need = (ov._CSRANK_MARK in expr) or (ov._CSZ_MARK in expr)
    qlib_expr = ov._strip_csrank(ov._strip_csz(expr))
    print(f"\n[{tag}] raw   : {raw[:110]}")
    print(f"[{tag}] norm  : {qlib_expr[:160]}")
    print(f"[{tag}] csrank={ov._CSRANK_MARK in expr} csz={ov._CSZ_MARK in expr}")
    try:
        s = D.features(codes, [qlib_expr], start_time=args.start, end_time=args.end, freq="day").iloc[:, 0]
    except Exception as exc:  # noqa: BLE001
        print(f"[{tag}] qlib 求值异常: {exc}")
        continue
    print(
        f"[{tag}] index={type(s.index).__name__} nlevels={s.index.nlevels} "
        f"names={list(map(str, s.index.names))} nonnull={int(s.notna().sum())} dtype={s.dtype}"
    )
    if need:
        dtl = ov._detect_dt_level(s)
        print(f"[{tag}] dt_level(detect)={dtl}")
        if ov._CSRANK_MARK in expr:
            s2 = ov._csrank_postprocess(s, dt_level=dtl)
        else:
            s2 = ov._csz_postprocess(s, dt_level=dtl)
        print(f"[post] nonnull={int(s2.notna().sum())} dtype={s2.dtype} head={s2.iloc[:3].tolist()}")
        s = s2
    # 与主流程一致的 IC 计算，直接复现 n_days
    fwd = D.features(codes, ["Ref($close, -2) / Ref($close, -1) - 1"],
                     start_time=args.start, end_time=args.end, freq="day").iloc[:, 0]
    df = pd.concat([s.rename("fac"), fwd.rename("fwd")], axis=1).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    dtl, _ = ov.find_levels(df)
    print(f"[{tag}] joined={df.shape} dt_level={dtl}")
    sub = df[["fac", "fwd"]].dropna()
    groups = list(sub.groupby(level=dtl))
    print(f"[{tag}] groups={len(groups)}", end="")
    if groups:
        lens = [len(g) for _, g in groups]
        print(f" group_len min/med/max={min(lens)}/{int(np.median(lens))}/{max(lens)}", end="")
        vals = []
        for _, g in groups:
            if len(g) < 30:
                continue
            vals.append(g.corr(method="spearman").iloc[0, 1])
        print(f" vals_n={len(vals)} nan={int(np.isnan(vals).sum())}", end="")
    print()
