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
import sys
import subprocess
import threading
import contextlib
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
        # PATH 处理：
        #   - Windows：**不重写 PATH**。rdagent 的 CondaConf.change_bin_path / 以及这里按
        #     conda_env_name 前置 conda 环境目录，会把 <env>\Library\bin 里的 MKL 运行时 DLL
        #     （mkl_*.dll、libiomp5md.dll 等）顶到 PATH 最前，导致 quantaalpha 的 numpy/pandas
        #     在 SigAnaRecord 计算 IC 时加载到版本不匹配的原生库而硬崩溃（exit 120，训练完成
        #     后无 traceback 直接退出）。实测：前置 quantaalpha 自己的 Library/bin 同样崩溃，
        #     而不重写 PATH 时 shell=True 也能正常跑完。下面已把 qrun/python 命令统一翻译成
        #     用 sys.executable 显式执行，因此根本不需要依赖 PATH。
        #   - Unix：保持 rdagent 原版逻辑。
        if os.name != "nt":
            path = [
                *self.conf.bin_path.split(":"),
                "/bin/",
                "/usr/bin/",
                *env.get("PATH", "").split(":"),
            ]
            env["PATH"] = ":".join(path)

        if entry is None:
            entry = self.conf.default_entry

        # --- 命令翻译（Windows 关键修复） ---
        # rdagent 默认 entry 是 "qrun conf.yaml" / "python xxx.py"（QlibFBWorkspace.execute()
        # 传 qrun；utils.generate 传 python generate.py；read_exp_res 传 python read_exp_res.py）。
        # 这些依赖 PATH 里的 qrun.exe / python.exe：
        #   - quantaalpha\Scripts\qrun.exe 是 pyinstaller 启动器，内部加载的是 rd-agent 环境的
        #     qlib，在 SigAnaRecord/PortAnaRecord 阶段退出码 127 直接崩；
        #   - 裸 `python` 可能命中其它环境的解释器。
        # 统一改成用当前 quantaalpha 的 python 显式执行，与手动验证一致
        # （python -m qlib.cli.run conf_combined_factors.yaml 退出 0、FINISHED、Rank IC ~0.06）。
        if os.name == "nt" and isinstance(entry, str):
            _e = entry.strip()
            if _e.startswith("qrun "):
                entry = f'"{sys.executable}" -m qlib.cli.run {_e[len("qrun ") :]}'
            elif _e.startswith("python "):
                entry = f'"{sys.executable}" {_e[len("python ") :]}'

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

        # 落盘调试：仅当子进程失败（非 0 退出）时，把完整输出写入 workspace 的
        # localenv_debug.log，便于定位回测 FAILED 的真实原因（runner 只截断打印前 500 字，
        # 会掩盖真正的 traceback/退出码）。成功时不落盘，避免污染每个因子 workspace。
        try:
            if local_path and process.returncode not in (0, None):
                _dbg = Path(local_path) / "localenv_debug.log"
                _dbg.write_text(combined_output, encoding="utf-8", errors="replace")
        except Exception:
            pass

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
