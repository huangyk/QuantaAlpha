#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QuantaAlpha 挖掘流水线 Windows 兼容性 / 环境自检
=================================================

用途：精确定位「端到端真实 LLM 因子挖掘」在本机为何跑不通。
本脚本**只做真实探测**（实际调用 API / 实际建环境 / 实际起子进程），
不做任何猜测或占位。每一项都给出 PASS / FAIL 与真实报错。

用法：
    # 务必用 conda quantaalpha 环境的 python
    D:/Users/lenovo/miniconda3/envs/quantaalpha/python.exe scripts/diagnose_mining_env.py

退出码：0=全部通过；1=存在阻塞项（会打印推荐路线）
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

# 应用 Windows LocalEnv 补丁（仅本机为 nt 时生效），让 check_localenv_actual 能验证修复
try:
    from quantaalpha.pipeline.patches.rdagent_windows_localenv import patch_localenv_for_windows
    _PATCH_ON = patch_localenv_for_windows()
except Exception as _pe:
    _PATCH_ON = False
    print(f"[warn] 无法加载 Windows LocalEnv 补丁: {_pe}")

RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    flag = "PASS" if ok else "FAIL"
    print(f"[{flag}] {name}" + (f" — {detail}" if detail else ""))


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------- 1. 基础信息
def check_basics() -> None:
    section("1. 运行环境")
    print(f"Python      : {sys.version.split()[0]}")
    print(f"Executable  : {sys.executable}")
    print(f"Platform    : {platform.platform()}")
    print(f"CONDA_DEFAULT_ENV = {os.environ.get('CONDA_DEFAULT_ENV')!r}")
    # 该变量缺失会导致 rdagent CondaConf 校验失败（历史 bug）
    record(
        "CONDA_DEFAULT_ENV 已设置",
        bool(os.environ.get("CONDA_DEFAULT_ENV")),
        f"值={os.environ.get('CONDA_DEFAULT_ENV')!r}（缺失会让挖掘子进程报 ValidationError）",
    )

    # quantaalpha 可 import？
    try:
        import quantaalpha  # noqa: F401

        record("quantaalpha 可导入", True, f"路径={quantaalpha.__file__}")
    except Exception as e:
        record("quantaalpha 可导入", False, f"{type(e).__name__}: {e}")

    # rdagent 可 import？
    try:
        import rdagent  # noqa: F401

        # rdagent 是命名空间包，__file__ 可能为 None，回退到 __path__
        loc = getattr(rdagent, "__file__", None) or ",".join(list(getattr(rdagent, "__path__", []))[:1])
        record("rdagent 可导入", True, f"路径={loc}")
    except Exception as e:
        record("rdagent 可导入", False, f"{type(e).__name__}: {e}")


# ------------------------------------------------- 2. select.poll（核心阻塞点）
def check_select_poll() -> None:
    section("2. select.poll（rdagent LocalEnv 的依赖，Windows 无此 API）")
    import select

    has_poll = hasattr(select, "poll")
    record("select.poll 存在", has_poll, "Windows 原生 Python 没有 poll()，只有 select()")
    print(f"    select 可用成员: {[m for m in ('select','poll','epoll','kqueue','devpoll') if hasattr(select, m)]}")

    # 直接在 rdagent LocalEnv._run 源码处确认
    try:
        import rdagent.utils.env as rd_env

        src = Path(rd_env.__file__).read_text(encoding="utf-8", errors="replace")
        idx = src.find("select.poll")
        if idx >= 0:
            line_no = src[:idx].count("\n") + 1
            record(
                "定位到 rdagent LocalEnv 中的 select.poll",
                False,
                f"{rd_env.__file__}:{line_no}（LocalEnv._run 用它做输出轮询 → Windows 必崩）",
            )
        else:
            record("定位到 rdagent LocalEnv 中的 select.poll", True, "源码中未找到（可能已修复/已替换）")
    except Exception as e:
        record("定位到 rdagent LocalEnv 中的 select.poll", False, f"{type(e).__name__}: {e}")


# ------------------------------------------- 3. Unix 入口命令（/bin/sh + timeout）
def check_unix_entry() -> None:
    section("3. Unix 入口命令：/bin/sh 与 timeout（rdagent LocalEnv 的 entry）")
    # rdagent LocalEnv 实际拼出的命令形如：
    #   /bin/sh -c 'timeout --kill-after=10 600 python ...'
    sh_path = shutil.which("sh") or shutil.which("bash")
    record("shell(sh/bash) 可解析", bool(sh_path), f"which -> {sh_path}")

    timeout_path = shutil.which("timeout")
    record("GNU timeout 可执行", bool(timeout_path), f"which -> {timeout_path}")

    # 关键：原生 Windows Python 能否直接 spawn "/bin/sh"？（路径风格是 Unix 的）
    ok = False
    detail = ""
    try:
        p = subprocess.run(
            ["/bin/sh", "-c", "echo OK"],
            capture_output=True,
            timeout=10,
        )
        ok = p.returncode == 0
        detail = f"returncode={p.returncode} stdout={p.stdout[:40]!r} stderr={p.stderr[:80]!r}"
    except Exception as e:
        detail = f"{type(e).__name__}: {e}"
    record("原生 Python 可直接 spawn /bin/sh", ok, detail)


# --------------------------------------------------- 4. 实测 rdagent LocalEnv.run
def check_localenv_actual() -> None:
    section("4. 实测：通过 rdagent LocalEnv 跑一条最简命令（真实报错来源）")
    try:
        from rdagent.utils.env import CondaConf, LocalEnv
    except Exception as e:
        record("导入 CondaConf / LocalEnv", False, f"{type(e).__name__}: {e}")
        return
    record("导入 CondaConf / LocalEnv", True, "")

    env_name = os.environ.get("CONDA_DEFAULT_ENV") or "quantaalpha"
    try:
        conf = CondaConf(conda_env_name=env_name)
        record("CondaConf 可实例化", True, f"conda_env_name={env_name}")
    except Exception as e:
        record("CondaConf 可实例化", False, f"{type(e).__name__}: {e}")
        return

    try:
        env = LocalEnv(conf=conf)
        record("LocalEnv 可实例化", True, "")
    except Exception as e:
        record("LocalEnv 可实例化", False, f"{type(e).__name__}: {e}")
        return

    # 真正跑一条：打印 python 版本
    # 注意：关掉 zip 缓存 + 用临时目录，避免把整个项目根（local_path="."）打包卡死
    try:
        import tempfile
        conf.enable_cache = False
        lp = tempfile.mkdtemp(prefix="qa_mine_test_")
        t0 = time.time()
        res = env.run(entry="python -c \"print('hello-from-localenv')\"", local_path=lp, entry_type="python")
        out = getattr(res, "stdout", None) or str(res)
        record(
            "LocalEnv.run 执行成功",
            True,
            f"{time.time() - t0:.1f}s, 输出前80字符={str(out)[:80]!r}",
        )
    except Exception as e:
        record(
            "LocalEnv.run 执行成功",
            False,
            f"{type(e).__name__}: {e}  ← 这就是挖掘任务每轮秒失败的真正原因",
        )


# ------------------------------------------------------------ 5. Qlib 数据链路
def check_qlib_data() -> None:
    section("5. Qlib 数据与 cn_data 跳转")
    qlib_dir = Path.home() / ".qlib" / "qlib_data"
    cn_data = qlib_dir / "cn_data"
    print(f"qlib_data 目录: {qlib_dir}")

    if cn_data.exists():
        target = None
        try:
            target = os.readlink(str(cn_data))
        except OSError:
            target = None
        if target:
            record("cn_data 已重定向", True, f"-> {target}（junction/symlink）")
        else:
            record("cn_data 已重定向", False, "是真实目录，未跳转（会导致 rejump/readlink 报错 4390）")
    else:
        record("cn_data 存在", False, "不存在")

    # 探测 pipepline 真正要读的数据集
    from dotenv import dotenv_values  # type: ignore

    cfg = dotenv_values(str(Path(__file__).resolve().parent.parent / "configs" / ".env"))
    qdir = cfg.get("QLIB_DATA_DIR") or os.environ.get("QLIB_DATA_DIR", "")
    record("QLIB_DATA_DIR 已配置", bool(qdir), f"={qdir}")
    if qdir and Path(qdir).exists():
        instr = Path(qdir) / "instruments" / "all.txt"
        n = sum(1 for _ in instr.open("r", encoding="utf-8", errors="replace")) if instr.exists() else 0
        has_index = False
        if instr.exists():
            txt = instr.read_text(encoding="utf-8", errors="replace")
            has_index = ("SH000300" in txt) or ("000300" in txt)
        record("目标数据集存在且有 instruments", bool(n), f"{n} 个标的；含指数(SH000300)={has_index}")
        record("存在市场指数可用于基准", has_index, "无指数 → IR 只能相对单只股票（当前用 SH600000 占位）")


# ------------------------------------------------------------------ 6. Docker
def check_docker() -> None:
    section("6. Docker 路线可行性（rdagent 原生支持的替代路径）")
    docker_cli = shutil.which("docker")
    record("docker CLI 可用", bool(docker_cli), f"-> {docker_cli}")
    if not docker_cli:
        return

    try:
        p = subprocess.run(["docker", "version", "--format", "{{.Server.Version}}"],
                           capture_output=True, text=True, timeout=25)
        record("Docker daemon 可达", p.returncode == 0,
               f"server={p.stdout.strip() or p.stderr.strip()[:80]}")
    except Exception as e:
        record("Docker daemon 可达", False, f"{type(e).__name__}: {e}")

    try:
        p = subprocess.run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
                           capture_output=True, text=True, timeout=30)
        imgs = p.stdout.split()
        hit = [i for i in imgs if "qlib" in i.lower()]
        record("存在 local_qlib 镜像", bool(hit), f"命中={hit or '无'}")
    except Exception as e:
        record("存在 local_qlib 镜像", False, f"{type(e).__name__}: {e}")

    try:
        import docker as docker_lib  # rdagent 用这个库

        try:
            c = docker_lib.from_env(timeout=10)
            c.ping()
            record("Python docker 库可连 daemon", True, "npipe/pipe 连接正常")
        except Exception as e:
            record("Python docker 库可连 daemon", False, f"{type(e).__name__}: {e}")
    except Exception as e:
        record("Python docker 库可导入", False, f"{type(e).__name__}: {e}")


# ------------------------------------------------------------- 7. 因子库真实性
def check_factor_library() -> None:
    section("7. 因子库（页面数据来源）真实性")
    root = Path(__file__).resolve().parent.parent
    lib_dir = root / "data" / "factorlib"
    files = sorted(lib_dir.glob("all_factors_library*.json")) if lib_dir.exists() else []
    record("存在可读因子库", bool(files), f"{[f.name for f in files]}")
    if not files:
        return
    import json

    try:
        raw = json.loads(files[-1].read_text(encoding="utf-8"))
        # 原生库结构：{"metadata": {...}, "factors": {<factor_id>: {...}}}
        if isinstance(raw, dict) and "factors" in raw and isinstance(raw["factors"], dict):
            items = list(raw["factors"].values())
        elif isinstance(raw, dict):
            items = [v for v in raw.values() if isinstance(v, dict) and "backtest_results" in v] or list(raw.values())
        else:
            items = raw
        record("因子库可解析", True, f"{len(items)} 个因子")

        nonzero = 0
        sample = None
        for it in items:
            it = it or {}
            br = it.get("backtest_results") or {}
            if not isinstance(br, dict):
                continue
            # 原生写法把 Qlib 规范嵌套键压平成点号字符串
            ann = (
                br.get("1day.excess_return_with_cost.annualized_return")
                or br.get("annualized_return")
                or 0
            )
            if ann:
                nonzero += 1
                if sample is None:
                    sample = (it.get("factor_name"), round(float(ann), 4))
        record(
            "因子年化收益非 0（真实指标）",
            nonzero > 0,
            f"{nonzero}/{len(items)} 个年化非 0"
            + (f"；样例={sample[0]} 年化={sample[1]}" if sample else ""),
        )
    except Exception as e:
        record("因子库可解析", False, f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------- 汇总
def summarize() -> int:
    section("汇总与结论")
    failed = [r for r in RESULTS if not r[1]]
    print(f"共 {len(RESULTS)} 项检查，PASS={len(RESULTS) - len(failed)}，FAIL={len(failed)}\n")

    if failed:
        print("阻塞项：")
        for n, _ok, d in failed:
            print(f"  - {n}: {d}")

    print("\n推荐路线：")
    localenv_ok = next((ok for n, ok, _ in RESULTS if n == "LocalEnv.run 执行成功"), False)
    if localenv_ok:
        print("  ✅ LocalEnv 在本机可用（Windows 补丁生效）→ 直接跑 `python -m quantaalpha.cli mine ...` 即可。")
    else:
        print("  ❌ LocalEnv 仍不可用 → Windows 原生跑不了原生挖掘。可选：")
        print("     A) Docker 路线：设 USE_LOCAL=false 并构建 local_qlib 镜像（rdagent 自带 Dockerfile）；")
        print("     B) 补丁路线：已提供 quantaalpha/pipeline/patches/rdagent_windows_localenv.py（monkey-patch，未改源码）；")
        print("     C) WSL 路线：在 WSL 内跑挖掘（本机 wsl.exe 曾被安全策略拦截，需你手动确认）。")
    if _PATCH_ON:
        print("  ℹ️  本次已加载 Windows LocalEnv 补丁（quantaalpha/pipeline/patches/rdagent_windows_localenv.py）。")

    print("\n说明：以上结论全部来自本机真实探测，无任何臆测数据。")
    return 0 if not failed else 1


def main() -> int:
    check_basics()
    check_select_poll()
    check_unix_entry()
    check_localenv_actual()
    check_qlib_data()
    check_docker()
    check_factor_library()
    return summarize()


if __name__ == "__main__":
    sys.exit(main())
