"""一次性探针：确认因子脚本在 5min daily_pv.h5 上能算出分钟级因子。

不进挖掘主流程，只是把「LLM 生成的 factor.py」这套代码在 5min 数据上跑一遍，
看产出的 index 到底是日频日期还是 5min 时间戳。

用法：
    python tools/probe_5min_factor.py [--expr "..."] [--workdir ...]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PV5 = ROOT / "git_ignore_folder" / "factor_implementation_source_data_5min" / "daily_pv.h5"

TEMPLATE = '''
import pandas as pd
import numpy as np
import os
from quantaalpha.factors.coder.expr_parser import parse_expression, parse_symbol
from quantaalpha.factors.coder.function_lib import *


def calculate_factor(expr: str, name: str):
    df = pd.read_hdf('./daily_pv.h5', key='data')

    expr = parse_symbol(expr, df.columns)
    expr = parse_expression(expr)

    for col in df.columns:
        expr = expr.replace(col[1:], f"df[\'{col}\']")

    df[name] = eval(expr)
    result = df[name].astype(np.float64)

    if os.path.exists('result.h5'):
        os.remove('result.h5')
    result.to_hdf('result.h5', key='data')

if __name__ == '__main__':
    expr = __EXPR__
    name = __NAME__
    calculate_factor(expr, name)
'''

# 注意：模板里有 f-string 的 {col[1:]}，不能用 str.format，走占位符替换
def _render(expr: str, name: str) -> str:
    return TEMPLATE.replace("__EXPR__", repr(expr)).replace("__NAME__", repr(name))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expr", default="RANK(TS_STD($return, 48) * -1)",
                    help="因子表达式；默认=日内已实现波动取负（48根=1交易日）")
    ap.add_argument("--name", default="PROBE_Intraday_RealizedVol_48")
    ap.add_argument("--workdir", default="")
    args = ap.parse_args()

    if not PV5.exists():
        print(f"ERROR: 5min 价量数据不存在: {PV5}")
        return 1

    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="fprobe_"))
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "factor.py").write_text(
        _render(args.expr, args.name), encoding="utf-8"
    )
    link = workdir / "daily_pv.h5"
    if not link.exists():
        try:
            os.symlink(str(PV5), str(link))
        except OSError:
            import shutil
            shutil.copy(PV5, link)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUTF8"] = "1"

    print(f"[probe] workdir={workdir}")
    print(f"[probe] expr={args.expr}")
    proc = subprocess.run([sys.executable, "factor.py"], cwd=str(workdir),
                          env=env, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        print("[probe] FAILED")
        print(proc.stdout[-2000:])
        print(proc.stderr[-3000:])
        return 1

    import pandas as pd
    res = pd.read_hdf(workdir / "result.h5", key="data")
    idx0 = res.index.get_level_values(0)
    print(f"[probe] OK  shape={res.shape}")
    print(f"[probe] index names={res.index.names}")
    print(f"[probe] n_timestamps={idx0.nunique()}  t_min={idx0.min()}  t_max={idx0.max()}")
    print(f"[probe] n_instruments={res.index.get_level_values(1).nunique()}")
    print(f"[probe] nan_frac={res.isna().mean():.4f}")
    print(res.head(3).round(6).to_string())
    is_intraday = idx0.nunique() > 3000
    print(f"[probe] VERDICT: {'5MIN OK' if is_intraday else 'NOT intraday - 仍是日频!'}")
    return 0 if is_intraday else 1


if __name__ == "__main__":
    raise SystemExit(main())
