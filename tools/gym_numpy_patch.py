"""Avoid hard crash: old gym + NumPy 2 breaks qlib PortAna/SigAna on Windows.

Redirects `import gym` to gymnasium before qlib backtest imports it.
"""
from __future__ import annotations

import sys


def apply() -> None:
    if getattr(sys, "_gym_numpy_patched", False):
        return

    try:
        import gymnasium as gymnasium_mod

        sys.modules["gym"] = gymnasium_mod
        if "gym.spaces" not in sys.modules and hasattr(gymnasium_mod, "spaces"):
            sys.modules["gym.spaces"] = gymnasium_mod.spaces
    except Exception:
        pass

    sys._gym_numpy_patched = True  # type: ignore[attr-defined]
    print("[gym-numpy-patch] gym->gymnasium", file=sys.stderr, flush=True)
