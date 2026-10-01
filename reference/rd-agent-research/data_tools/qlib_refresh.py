"""Refresh research Qlib data: optional DB export → provider → profile → h5."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _log(msg: str, log_fp) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    log_fp.write(line + "\n")
    log_fp.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh Qlib research data layers")
    parser.add_argument(
        "--with-export",
        action="store_true",
        help="先从 TimescaleDB 重导 staging（慢）",
    )
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default=None, help="默认今天")
    parser.add_argument("--max-instruments", type=int, default=500)
    parser.add_argument(
        "--staging",
        default=None,
        help="staging 目录，默认 artifacts/cn_real_staging",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="provider 输出，默认读 research profile",
    )
    parser.add_argument(
        "--skip-h5",
        action="store_true",
        help="只重建 provider，不重生 h5（需 conda rd-agent）",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    log_dir = repo / "log"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"qlib_refresh_{datetime.now():%Y%m%d_%H%M%S}.log"
    core_py = repo / ".venv312" / "Scripts" / "python.exe"
    if not core_py.is_file():
        print("缺少 .venv312，请先 .\\dev.ps1 setup-core", file=sys.stderr)
        return 1

    profile_path = repo / "config" / "qlib_profile.json"
    cfg = json.loads(profile_path.read_text(encoding="utf-8"))
    research = cfg["profiles"]["research"]
    staging = Path(args.staging) if args.staging else repo / "artifacts" / "cn_real_staging"
    provider = Path(args.provider) if args.provider else Path(research["provider_uri"])
    if args.end is None:
        args.end = datetime.now().strftime("%Y-%m-%d")

    with log_path.open("w", encoding="utf-8") as log_fp:
        _log(f"log={log_path}", log_fp)
        _log(f"staging={staging}", log_fp)
        _log(f"provider={provider}", log_fp)

        if args.with_export:
            _log("STEP1: DB → staging (with-export)", log_fp)
            if staging.exists():
                shutil.rmtree(staging)
            cmd = [
                str(core_py),
                "-m",
                "jq_research_system.qlib_adapter",
                "db",
                "--output",
                str(staging),
                "--start",
                args.start,
                "--end",
                args.end,
            ]
            _log(" ".join(cmd), log_fp)
            subprocess.check_call(cmd, cwd=str(repo))
        else:
            _log("STEP1: skip export（使用已有 staging）", log_fp)
            if not (staging / "manifest.json").is_file():
                _log("ERROR: staging 不存在，请加 --with-export", log_fp)
                return 1

        _log("STEP2: staging → provider", log_fp)
        if provider.exists():
            _log(f"删除旧 provider: {provider}", log_fp)
            shutil.rmtree(provider)
        cmd = [
            str(core_py),
            "-m",
            "jq_research_system.qlib_baseline",
            "build-provider",
            "--input",
            str(staging),
            "--output",
            str(provider),
            "--max-instruments",
            str(args.max_instruments),
            "--must-include",
            str(research.get("benchmark") or "SH600000"),
        ]
        _log(" ".join(cmd), log_fp)
        subprocess.check_call(cmd, cwd=str(repo))

        _log("STEP3: set profile=research", log_fp)
        subprocess.check_call(
            [str(core_py), str(repo / "tools" / "set_qlib_profile.py"), "research"],
            cwd=str(repo),
        )

        if args.skip_h5:
            _log("STEP4: skip h5（--skip-h5）", log_fp)
        else:
            _log("STEP4: regen h5 via conda rd-agent", log_fp)
            env = dict(**{k: v for k, v in __import__("os").environ.items()})
            env["PYTHONPATH"] = str(repo / "tools")
            cmd = [
                "conda",
                "run",
                "-n",
                "rd-agent",
                "--no-capture-output",
                "python",
                str(repo / "tools" / "qlib_regen_h5.py"),
            ]
            _log(" ".join(cmd), log_fp)
            subprocess.check_call(cmd, cwd=str(repo), env=env)

        _log("STEP5: status", log_fp)
        status = subprocess.call(
            [str(core_py), str(repo / "tools" / "qlib_data_status.py")],
            cwd=str(repo),
        )
        # status may need rdagent — prefer conda for final check
        if status != 0:
            env = dict(**{k: v for k, v in __import__("os").environ.items()})
            env["PYTHONPATH"] = str(repo / "tools")
            status = subprocess.call(
                [
                    "conda",
                    "run",
                    "-n",
                    "rd-agent",
                    "--no-capture-output",
                    "python",
                    str(repo / "tools" / "qlib_data_status.py"),
                ],
                cwd=str(repo),
                env=env,
            )
        _log(f"DONE status_exit={status} log={log_path}", log_fp)
        return 0 if status in (0, 2) else status


if __name__ == "__main__":
    raise SystemExit(main())
