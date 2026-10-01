# -*- coding: utf-8 -*-
"""
Windows 兼容垫片：运行时 monkey-patch rdagent 的 LocalEnv，使其能在 Windows 跑起来。

背景（真实探测确认）：
  rdagent 的 LocalEnv 假设跑在 Linux/macOS，用了三处 Windows 没有的东西：
    1) `run()` 把 entry 包成 `/bin/sh -c 'timeout --kill-after=10 N <cmd>; ...'`
       —— Windows 没有 /bin/sh，也没有 GNU timeout；
    2) `LocalEnv._run` 用 `select.poll()` 轮询子进程 stdout/stderr
       —— Windows 原生 Python 只有 select()，且 select 只对 socket 有效，对管道无效；
    3) 卷映射用 `os.symlink` 建目录符号链接 —— Windows 下通常需管理员/开发者模式。

做法：
  - 只在 os.name == "nt" 时生效（Linux/macOS 走 rdagent 原版，零影响）；
  - 不改 site-packages 里的 rdagent 源码 —— 补丁写在自己仓库，随代码走、可版本管理、
    重装 rdagent 也不丢，只需在挖掘入口 import 一次即可（见 factor_mining.py 顶部）；
  - `run()` 在 Windows 下跳过 /bin/sh + GNU timeout，直接把原始 entry 交给 _run；
  - `_run` 用线程读流替代 select.poll，超时用看门狗线程（kill-after 语义），
    符号链接用 junction（mklink /J）兜底，PATH 分隔符按 Windows 用 `;`。

这不是猜测实现：select.poll / /bin/sh / timeout 三处均来自对
  <conda>/Lib/site-packages/rdagent/utils/env.py
的真实阅读（LocalEnv._run 第 543~607 行，run() 第 314~331 行）。
"""
from __future__ import annotations

import os
import re
import sys
import json
import subprocess
import threading
import contextlib
import time
import logging
from pathlib import Path
from typing import Any, Generator, Mapping
from types import MappingProxyType

logger = logging.getLogger("quantaalpha.patch.rdagent_windows_localenv")

# 延迟导入 rdagent，避免在非挖掘场景也强依赖
import rdagent.utils.env as _E  # noqa: E402
from rdagent.utils.env import (  # noqa: E402
    Env,
    LocalEnv,
    DockerConf,
    normalize_volumes,
    EnvResult,
)
try:
    from rdagent.utils.env import T  # 翻译/路径辅助
except Exception:  # pragma: no cover
    T = None

try:
    from rich.console import Console
    from rich.rule import Rule
    from rich.table import Table
except Exception:  # pragma: no cover
    Console = Rule = Table = None


_APPLIED = False


# --------------------------------------------------------------------------- #
# 1) run()：Windows 下跳过 /bin/sh + GNU timeout                              #
# --------------------------------------------------------------------------- #
def _windows_aware_run(
    self,
    entry: str | None = None,
    local_path: str = ".",
    env: dict | None = None,
    **kwargs: Any,
) -> "EnvResult":
    running_extra_volume = kwargs.get("running_extra_volume", {})
    if entry is None:
        entry = self.conf.default_entry

    if "|" in entry:
        logger.warning(
            "You are using a command with a shell pipeline (i.e., '|'). "
            "The exit code will reflect the result of the last command in the pipeline."
        )

    if os.name == "nt":
        # Windows：直接把原始 entry 交给 _run；超时由 _run 的看门狗线程处理。
        entry_add_timeout = entry
    else:
        # 原版 Unix 逻辑（保持一致）
        def _get_chmod_cmd(workspace_path: str) -> str:
            def _get_path_stem(path: str):
                p = Path(path)
                if not p.is_absolute() and p.parts:
                    return p.parts[0]
                return None

            find_cmd = f"find {workspace_path} -mindepth 1 -maxdepth 1"
            for name in [
                _get_path_stem(T("scenarios.data_science.share:scen.cache_path").r()) if T else None,
                _get_path_stem(T("scenarios.data_science.share:scen.input_path").r()) if T else None,
            ]:
                if name:
                    find_cmd += f" ! -name {name}"
            return f"{find_cmd} -exec chmod -R 777 {{}} +"

        if getattr(self.conf, "running_timeout_period", None) is None:
            timeout_cmd = entry
        else:
            timeout_cmd = f"timeout --kill-after=10 {self.conf.running_timeout_period} {entry}"
        entry_add_timeout = (
            f"/bin/sh -c '"
            + f"{timeout_cmd}; entry_exit_code=$?; "
            + (
                f"{_get_chmod_cmd(self.conf.mount_path)}; "
                if isinstance(self.conf, DockerConf)
                else ""
            )
            + "exit $entry_exit_code"
            + "'"
        )

    if getattr(self.conf, "enable_cache", False):
        result = self.cached_run(entry_add_timeout, local_path, env, running_extra_volume)
    else:
        # __run_with_retry 是 Env 类的私有方法（name-mangling -> _Env__run_with_retry）
        result = self._Env__run_with_retry(entry_add_timeout, local_path, env, running_extra_volume)
    return result


# --------------------------------------------------------------------------- #
# 2) _run：线程读流 + 看门狗超时 + junction 符号链接                          #
# --------------------------------------------------------------------------- #
def _win_make_link(real: str, link: str) -> None:
    """Windows 下建目录链接：优先 symlink，失败回退 junction(mklink /J)，再失败建目录。"""
    link_path = Path(link)
    real_path = Path(real)
    try:
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
    except OSError:
        pass
    if not link_path.parent.exists():
        link_path.parent.mkdir(parents=True, exist_ok=True)
    # 尝试原生目录符号链接（需开发者模式/管理员）
    try:
        link_path.symlink_to(real_path, target_is_directory=True)
        return
    except (OSError, NotImplementedError, AttributeError):
        pass
    # junction 不需要管理员权限
    r = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link_path), str(real_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if r.returncode == 0:
        return
    # 最后兜底：直接建真实目录（数据已在 real 处）
    try:
        link_path.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass


@contextlib.contextmanager
def _win_symlink_ctx(vol_map: Mapping[str, str]) -> Generator[None, None, None]:
    created: list[Path] = []
    try:
        for real, link in vol_map.items():
            _win_make_link(real, link)
            created.append(Path(link))
        yield
    finally:
        for p in created:
            try:
                if p.is_symlink():
                    p.unlink()
                elif p.exists():
                    # junction 也是 symlink 的一种，is_symlink 应为 True；兜底忽略
                    pass
            except OSError:
                pass


# --------------------------------------------------------------------------- #
# Windows 下正确的 conda 环境 PATH                                             #
#   rdagent 的 CondaConf.change_bin_path 用 `conda run ... | grep '^PATH='`，  #
#   但 Windows 的 cmd.exe 没有 grep，会抓到整段环境变量的垃圾。这里改用        #
#   `conda info --json`（Python 解析，返回原生 Windows 路径）构造。            #
# --------------------------------------------------------------------------- #
_CONDA_ENV_PATHS_CACHE: dict = {}


def _win_conda_env_paths(conda_env_name: str | None) -> list[str]:
    if conda_env_name in _CONDA_ENV_PATHS_CACHE:
        return _CONDA_ENV_PATHS_CACHE[conda_env_name]
    dirs: list[str] = []
    try:
        r = subprocess.run("conda info --json", shell=True, capture_output=True, text=True, timeout=40)
        info = json.loads(r.stdout)
        envs = info.get("envs", [])
        root = info.get("root_prefix") or info.get("conda_prefix") or ""
        target = None
        for e in envs:
            if e.rstrip("\\/").endswith(conda_env_name or ""):
                target = e
                break
        if target is None and root and conda_env_name:
            target = os.path.join(root, "envs", conda_env_name)
        if target and os.path.isdir(target):
            for sub in ["", "Scripts", "Library/bin", "Library/mingw-w64/bin", "Library/usr/bin", "bin"]:
                p = os.path.normpath(os.path.join(target, sub))
                if os.path.isdir(p):
                    dirs.append(p)
    except Exception as _e:  # pragma: no cover
        logger.warning(f"conda info 解析失败: {_e}")
    if not dirs:
        # 兜底：用当前 python 所在环境
        exe = sys.executable
        if exe:
            base = os.path.dirname(exe)
            for sub in ["", "Scripts", "Library/bin"]:
                p = os.path.normpath(os.path.join(base, sub))
                if os.path.isdir(p):
                    dirs.append(p)
    _CONDA_ENV_PATHS_CACHE[conda_env_name] = dirs
    return dirs


def _windows_localenv_run(
    self,
    entry: str | None = None,
    local_path: str | None = None,
    env: dict | None = None,
    running_extra_volume: Mapping = MappingProxyType({}),
    **kwargs: dict,
) -> tuple[str, int]:
    # ---- 卷处理（同原版结构） ----
    volumes: dict[str, Any] = {}
    if getattr(self.conf, "extra_volumes", None) is not None:
        for lp, rp in self.conf.extra_volumes.items():
            volumes[lp] = rp["bind"] if isinstance(rp, dict) else rp
        cache_path = "/tmp/sample" if "/sample/" in "".join(self.conf.extra_volumes.keys()) else "/tmp/full"
        try:
            Path(cache_path).mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        try:
            volumes[cache_path] = (
                T("scenarios.data_science.share:scen.cache_path").r() if T else ""
            )
        except Exception:
            volumes[cache_path] = ""
    for lp, rp in running_extra_volume.items():
        volumes[lp] = rp

    assert local_path is not None, "local_path should not be None"
    volumes = normalize_volumes(volumes, local_path)

    with _win_symlink_ctx(volumes):
        if env is None:
            env = {}
        # PATH 构造：
        #   - Windows：CondaConf.change_bin_path 用 `conda run ... | grep` 在 cmd 下抓到垃圾，
        #     这里改用 `conda info --json` 解析出的原生 Windows 环境目录，再并入现有 PATH。
        #   - Unix：保持 rdagent 原版逻辑。
        if os.name == "nt":
            conda_dirs = _win_conda_env_paths(getattr(self.conf, "conda_env_name", None))
            raw_parts: list[str] = list(conda_dirs)
            raw_parts += re.split(r"[:;]", os.environ.get("PATH", ""))
            win_parts = [w for w in (w.strip() for w in raw_parts) if w]
            env["PATH"] = ";".join(win_parts)
        else:
            path = [
                *self.conf.bin_path.split(":"),
                "/bin/",
                "/usr/bin/",
                *env.get("PATH", "").split(":"),
            ]
            env["PATH"] = ":".join(path)

        if entry is None:
            entry = self.conf.default_entry

        if Console is not None:
            Console().print(Rule("[bold green]LocalEnv Logs Begin (win-patched)[/bold green]", style="dark_orange"))
            table = Table(title="Run Info", show_header=False)
            table.add_column("Key", style="bold cyan")
            table.add_column("Value", style="bold magenta")
            table.add_row("Entry", entry)
            table.add_row("Local Path", local_path or "")
            table.add_row("Env", "\n".join(f"{k}:{v}" for k, v in env.items()))
            table.add_row("Volumes", "\n".join(f"{k}:\n  {v}" for k, v in volumes.items()))
            Console().print(table)

        cwd = Path(local_path).resolve() if local_path else None
        env = {k: str(v) if isinstance(v, int) else v for k, v in env.items()}

        process = subprocess.Popen(
            entry,
            cwd=cwd,
            env={**os.environ, **env},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True,
            bufsize=1,
            universal_newlines=True,
        )

        combined_output = ""
        lock = threading.Lock()

        def _reader(stream) -> None:
            nonlocal combined_output
            try:
                for line in iter(stream.readline, ""):
                    if Console is not None:
                        Console().print(line.rstrip("\n"), markup=False)
                    with lock:
                        combined_output += line
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass

        t_out = threading.Thread(target=_reader, args=(process.stdout,), daemon=True)
        t_err = threading.Thread(target=_reader, args=(process.stderr,), daemon=True)
        t_out.start()
        t_err.start()

        # 看门狗超时（模拟 timeout --kill-after）
        period = getattr(self.conf, "running_timeout_period", None)
        killed = False
        try:
            process.wait(timeout=period if period else None)
        except subprocess.TimeoutExpired:
            killed = True
            try:
                process.kill()
            except OSError:
                pass
            try:
                process.wait()
            except Exception:
                pass
        t_out.join(timeout=30)
        t_err.join(timeout=30)

        if killed:
            combined_output += (
                f"\n\nThe running time exceeds {period} seconds, so the process is killed."
            )

        return_code = process.returncode
        if Console is not None:
            Console().print(Rule("[bold green]LocalEnv Logs End (win-patched)[/bold green]", style="dark_orange"))
        return combined_output, return_code


# --------------------------------------------------------------------------- #
# 应用补丁（仅 Windows，幂等）                                                 #
# --------------------------------------------------------------------------- #
def patch_localenv_for_windows() -> bool:
    """对 rdagent LocalEnv 打 Windows 兼容补丁。返回是否实际生效。"""
    global _APPLIED
    if _APPLIED:
        return os.name == "nt"
    if os.name != "nt":
        return False

    Env.run = _windows_aware_run          # 基类：跳过 /bin/sh + GNU timeout
    LocalEnv._run = _windows_localenv_run  # 子类：线程读流 + 看门狗 + junction
    _APPLIED = True
    logger.info("[patch] rdagent LocalEnv Windows 兼容补丁已应用（monkey-patch，未改 site-packages）")
    return True
