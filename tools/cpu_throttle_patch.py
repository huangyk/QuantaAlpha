"""Keep RD-Agent from pegging the desktop CPU.

- Caps OpenMP/BLAS/Torch thread pools to RD_CPU_FRACTION of logical CPUs (default 0.8).
- Sets Windows process priority to BelowNormal so interactive apps preempt RD work.

Does not change LGB hyperparameters or factor formulas — only scheduling / parallelism.
Disable: RD_CPU_THROTTLE=0
Override fraction: RD_CPU_FRACTION=0.8
Force normal priority: RD_CPU_PRIORITY=normal
"""

from __future__ import annotations

import os
import sys


_APPLIED = False


def _logical_cpus() -> int:
    return int(os.cpu_count() or 1)


def _budget_threads() -> int:
    raw = os.environ.get("RD_CPU_FRACTION", "0.8")
    try:
        frac = float(raw)
    except ValueError:
        frac = 0.8
    frac = max(0.1, min(1.0, frac))
    return max(1, int(_logical_cpus() * frac))


def _set_thread_env(n: int) -> None:
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "POLARS_MAX_THREADS",
    ):
        if not os.environ.get(key):
            os.environ[key] = str(n)


def _set_below_normal_priority() -> None:
    if sys.platform != "win32":
        return
    mode = os.environ.get("RD_CPU_PRIORITY", "below_normal").strip().lower()
    if mode in ("0", "false", "off", "normal"):
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        # 64-bit: without restype, pseudo-handle -1 is truncated → ERROR_INVALID_HANDLE (6)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
        kernel32.SetPriorityClass.restype = ctypes.c_int
        handle = kernel32.GetCurrentProcess()
        # BELOW_NORMAL_PRIORITY_CLASS
        kernel32.SetPriorityClass(handle, 0x00004000)
    except Exception:
        pass


def _cap_torch(n: int) -> None:
    try:
        import torch

        torch.set_num_threads(n)
        interop = max(1, min(4, n // 4))
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(interop)
            except Exception:
                pass
    except Exception:
        pass


def apply() -> None:
    global _APPLIED
    if _APPLIED:
        return
    if os.environ.get("RD_CPU_THROTTLE", "1").strip().lower() in ("0", "false", "off"):
        return

    n = _budget_threads()
    frac = os.environ.get("RD_CPU_FRACTION", "0.8")
    _set_thread_env(n)
    _set_below_normal_priority()
    _cap_torch(n)
    _APPLIED = True
    print(
        f"[cpu-throttle] threads<={n}/{_logical_cpus()} "
        f"(RD_CPU_FRACTION={frac}) priority=BelowNormal",
        file=sys.stderr,
        flush=True,
    )
