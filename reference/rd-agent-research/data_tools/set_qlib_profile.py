"""Set active profile in config/qlib_profile.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: set_qlib_profile.py <smoke|research|...>", file=sys.stderr)
        return 2
    name = sys.argv[1]
    path = Path(__file__).resolve().parents[1] / "config" / "qlib_profile.json"
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if name not in cfg.get("profiles", {}):
        known = sorted(cfg.get("profiles", {}))
        print(f"unknown profile: {name}; known={known}", file=sys.stderr)
        return 1
    cfg["active"] = name
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"active={name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
