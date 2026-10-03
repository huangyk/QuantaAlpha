"""死循环定位：只对单条表达式跑 normalize_expr，超时直接抓栈。用法：python tools/_hang_probe.py <n>"""
from __future__ import annotations

import faulthandler
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "oos_validate.py")

for _m in ("numpy", "pandas"):
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)

ns: dict = {}
with open(SRC, "r", encoding="utf-8") as f:
    src = f.read()
exec(compile(src, SRC, "exec"), ns)
normalize_expr = ns["normalize_expr"]

CASES = [
    "RANK(COUNT((Xvolume < 0.5 * TS_MEAN(Xvolume, 20)) && "
    "(Xclose > TS_MEAN(Xclose, 20)), 5) * ((Xclose < 0.9 * TS_MAX(Xclose, 20)) ? "
    "((TS_MAX(Xclose, 20) - Xclose) / (TS_MAX(Xclose, 20) + 1e-8)) : 0))",
    "RANK(COUNT((Xvolume < 0.5 * TS_MEAN(Xvolume, 20)) && (Xclose > TS_MEAN(Xclose, 20)), 5))",
    "COUNT((Xvolume < 0.5 * TS_MEAN(Xvolume, 20)) && (Xclose > TS_MEAN(Xclose, 20)), 5)",
    "(Xvolume < 0.5 * TS_MEAN(Xvolume, 20)) && (Xclose > TS_MEAN(Xclose, 20))",
]

n = int(sys.argv[1]) if len(sys.argv) > 1 else 0
case = CASES[n]
print("[in ]", case, flush=True)
faulthandler.dump_traceback_later(8, exit=True, repeat=False)
try:
    out = normalize_expr(case)
except Exception as exc:  # noqa: BLE001
    out = f"!! {type(exc).__name__}: {exc}"
faulthandler.cancel_dump_traceback_later()
print("[out]", out, flush=True)
