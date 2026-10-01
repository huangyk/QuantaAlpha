"""Regenerate RD-Agent daily_pv_*.h5 from the active Qlib profile provider."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


def main() -> int:
    # Ensure patches + local tools importable when run via conda.
    tools = Path(__file__).resolve().parent
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))

    from qlib_data_profile_patch import apply_profile_to_templates, get_active_profile

    name, profile = get_active_profile()
    provider = Path(str(profile["provider_uri"]))
    if not provider.is_dir():
        raise SystemExit(f"provider missing for profile {name!r}: {provider}")

    result = apply_profile_to_templates(force=True)
    print(json.dumps({"profile": name, "patch": result}, ensure_ascii=False, indent=2))

    import rdagent

    roots = list(getattr(rdagent, "__path__", []) or [])
    if not roots:
        raise SystemExit("cannot locate rdagent package path")
    template_dir = Path(roots[0]) / "scenarios" / "qlib" / "experiment" / "factor_data_template"
    generate_py = template_dir / "generate.py"
    if not generate_py.is_file():
        raise SystemExit(f"missing {generate_py}")

    # Run generate.py with cwd = template dir so outputs land next to it.
    import runpy

    prev = Path.cwd()
    try:
        import os

        os.chdir(template_dir)
        runpy.run_path(str(generate_py), run_name="__main__")
    finally:
        os.chdir(prev)

    all_h5 = template_dir / "daily_pv_all.h5"
    debug_h5 = template_dir / "daily_pv_debug.h5"
    if not all_h5.is_file() or not debug_h5.is_file():
        raise SystemExit(f"generate.py did not produce h5 under {template_dir}")

    repo = tools.parent
    copies = [
        repo / "git_ignore_folder" / "factor_implementation_source_data" / "daily_pv.h5",
        repo / "git_ignore_folder" / "factor_implementation_source_data_debug" / "daily_pv.h5",
    ]
    srcs = [all_h5, debug_h5]
    for dest, src in zip(copies, srcs, strict=True):
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        print(f"copied {src.name} -> {dest}")

    print(
        json.dumps(
            {
                "ok": True,
                "profile": name,
                "provider_uri": str(provider),
                "all_h5": str(all_h5),
                "debug_h5": str(debug_h5),
                "all_bytes": all_h5.stat().st_size,
                "debug_bytes": debug_h5.stat().st_size,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
