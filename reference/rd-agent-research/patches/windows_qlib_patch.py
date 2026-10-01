"""Windows-safe LocalEnv: conda paths, no Docker for factor data gen."""
from __future__ import annotations

import sys

import os
import shutil
import time
from pathlib import Path
from types import MappingProxyType
from typing import Any

from rdagent.utils.env import Env, EnvResult, LocalEnv, QlibCondaConf

_MINICONDA = Path(os.environ.get("MINICONDA3", r"D:\Users\lenovo\miniconda3"))


def _conda_env_name() -> str:
    return (
        os.environ.get("CONDA_ENV_NAME")
        or os.environ.get("CONDA_DEFAULT_ENV")
        or "rd-agent"
    )


def _conda_paths() -> tuple[Path, Path]:
    env = _conda_env_name()
    root = _MINICONDA / "envs" / env
    py = root / "python.exe"
    qrun = root / "Scripts" / "qrun.exe"
    if not py.exists():
        for fallback_env in ("quantaalpha", "rd-agent"):
            root = _MINICONDA / "envs" / fallback_env
            py = root / "python.exe"
            qrun = root / "Scripts" / "qrun.exe"
            if py.exists():
                break
    return py, qrun
_READ_EXP_RES_FIXED = Path(__file__).resolve().parent / "read_exp_res_fixed.py"
_TEMPLATE = (
    Path(r"D:\Users\lenovo\miniconda3\envs\rd-agent\Lib\site-packages")
    / "rdagent"
    / "scenarios"
    / "qlib"
    / "experiment"
    / "factor_data_template"
)


def _wrap_entry(entry: str) -> str:
    python_bin, qrun_bin = _conda_paths()
    conda_env = _conda_env_name()
    text = entry.strip()
    if text.startswith("/bin/sh -c "):
        inner = text[len("/bin/sh -c ") :].strip().strip("'").strip('"')
        if inner.endswith("'"):
            inner = inner[:-1]
        if "; entry_exit_code=$?" in inner:
            inner = inner.split("; entry_exit_code=$?")[0].strip()
        if inner.startswith("timeout "):
            parts = inner.split(" ", 3)
            if len(parts) >= 4:
                inner = parts[3]
        return _wrap_entry(inner)
    if text.startswith("conda run "):
        return text
    if "read_exp_res.py" in text:
        return f'"{python_bin}" "{_READ_EXP_RES_FIXED}"'
    if text.startswith("python "):
        rest = text[len("python ") :]
        return f'"{python_bin}" {rest}'
    if text.startswith("qrun "):
        rest = text[len("qrun ") :]
        return f'"{qrun_bin}" {rest}'
    return f'conda run -n {conda_env} --no-capture-output {text}'


def _patch_factor_data_folder() -> None:
    """Avoid QTDockerEnv.prepare() when generating factor source data."""
    try:
        from rdagent.components.coder.factor_coder import config as factor_conf
        from rdagent.scenarios.qlib.experiment import utils as qlib_utils
    except Exception:
        return

    python_bin, _ = _conda_paths()
    factor_conf.FACTOR_COSTEER_SETTINGS.python_bin = str(python_bin)

    def generate_data_folder_from_qlib() -> None:
        from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS

        all_h5 = _TEMPLATE / "daily_pv_all.h5"
        debug_h5 = _TEMPLATE / "daily_pv_debug.h5"
        readme = _TEMPLATE / "README.md"
        if not all_h5.exists() or not debug_h5.exists():
            raise FileNotFoundError(
                f"Missing factor templates under {_TEMPLATE}; "
                "cannot generate data without Docker on Windows."
            )
        Path(FACTOR_COSTEER_SETTINGS.data_folder).mkdir(parents=True, exist_ok=True)
        Path(FACTOR_COSTEER_SETTINGS.data_folder_debug).mkdir(parents=True, exist_ok=True)
        shutil.copy(all_h5, Path(FACTOR_COSTEER_SETTINGS.data_folder) / "daily_pv.h5")
        shutil.copy(readme, Path(FACTOR_COSTEER_SETTINGS.data_folder) / "README.md")
        shutil.copy(debug_h5, Path(FACTOR_COSTEER_SETTINGS.data_folder_debug) / "daily_pv.h5")
        shutil.copy(readme, Path(FACTOR_COSTEER_SETTINGS.data_folder_debug) / "README.md")
        print("[windows-qlib-patch] factor data restored from package templates", file=sys.stderr, flush=True)

    qlib_utils.generate_data_folder_from_qlib = generate_data_folder_from_qlib


def apply() -> None:
    if getattr(Env, "_win_qlib_patched", False):
        return
    if os.name != "nt":
        return

    os.environ.setdefault("MODEL_COSTEER_ENV_TYPE", "conda")
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

    QlibCondaConf.model_fields["conda_env_name"].default = _conda_env_name()
    QlibCondaConf.model_fields["live_output"].default = False
    QlibCondaConf.model_rebuild(force=True)

    _patch_factor_data_folder()

    orig_run = Env.run

    def run(
        self: Env,
        entry: str | None = None,
        local_path: str = ".",
        env: dict | None = None,
        **kwargs: Any,
    ) -> EnvResult:
        if not isinstance(self, LocalEnv):
            return orig_run(self, entry=entry, local_path=local_path, env=env, **kwargs)
        if entry is None:
            entry = self.conf.default_entry
        running_extra_volume = kwargs.get("running_extra_volume", MappingProxyType({}))
        start = time.time()
        log_output, return_code = self._run(
            entry,
            local_path,
            env,
            running_extra_volume=running_extra_volume,
        )
        end = time.time()
        return EnvResult(log_output, return_code, end - start)

    def _run(self: LocalEnv, *args: Any, **kwargs: Any) -> tuple[str, int]:
        import subprocess

        self.conf.live_output = False
        _, qrun_bin = _conda_paths()
        # Prevent Unix PATH fragments from breaking Windows subprocesses.
        if self.conf.bin_path and (":" in self.conf.bin_path or self.conf.bin_path.startswith("/")):
            self.conf.bin_path = str(qrun_bin.parent)
        tools_dir = str(Path(__file__).resolve().parent)
        env = kwargs.get("env")
        if env is None and len(args) >= 3 and isinstance(args[2], dict):
            env = args[2]
        if isinstance(env, dict):
            prev = str(env.get("PYTHONPATH", "") or "")
            parts = [p for p in prev.split(os.pathsep) if p]
            if tools_dir not in parts:
                env["PYTHONPATH"] = os.pathsep.join([tools_dir, *parts]) if parts else tools_dir
        else:
            prev = str(os.environ.get("PYTHONPATH", "") or "")
            parts = [p for p in prev.split(os.pathsep) if p]
            if tools_dir not in parts:
                os.environ["PYTHONPATH"] = os.pathsep.join([tools_dir, *parts]) if parts else tools_dir
        entry = args[0] if args else kwargs.get("entry")
        local_path = args[1] if len(args) > 1 else kwargs.get("local_path", ".")
        if entry:
            entry = _wrap_entry(str(entry))
        cwd = Path(local_path).resolve() if local_path else None
        merged = {**os.environ, **(env or {})}
        timeout = self.conf.running_timeout_period
        try:
            proc = subprocess.run(
                entry,
                cwd=cwd,
                env=merged,
                shell=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
            log_output = (proc.stdout or "") + (proc.stderr or "")
            return log_output, proc.returncode
        except subprocess.TimeoutExpired as exc:
            out = (exc.stdout or "") + (exc.stderr or "")
            return out + "\n[windows-qlib-patch] command timed out", 124

    def _prepare_noop(self: LocalEnv) -> None:
        return None

    Env.run = run  # type: ignore[method-assign]
    LocalEnv._run = _run  # type: ignore[method-assign]
    LocalEnv.prepare = _prepare_noop  # type: ignore[method-assign]
    Env._win_qlib_patched = True
    py, qrun = _conda_paths()
    print(f"[windows-qlib-patch] conda={_conda_env_name()} python={py} qrun={qrun}", file=sys.stderr, flush=True)
