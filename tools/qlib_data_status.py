"""Report which Qlib/RD data layer is ready (staging → provider → h5)."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path


def _mtime(path: Path) -> str:
    if not path.exists():
        return "-"
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def _size_mb(path: Path) -> str:
    if not path.is_file():
        return "-"
    return f"{path.stat().st_size / (1024 * 1024):.1f} MB"


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  ! JSON 读失败: {path}: {exc}")
        return None


def _mark(ok: bool) -> str:
    return "OK" if ok else "缺"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass
    repo = Path(__file__).resolve().parents[1]
    profile_path = repo / "config" / "qlib_profile.json"
    staging = repo / "artifacts" / "cn_real_staging"
    cfg = _load_json(profile_path)
    if not cfg:
        print(f"[1] profile 配置: 缺  {profile_path}")
        return 1

    import os

    active = (os.environ.get("RD_QLIB_PROFILE") or "").strip() or str(cfg.get("active") or "")
    profiles = cfg.get("profiles") or {}
    if active not in profiles:
        print(f"[1] profile 配置: 缺  active={active!r} 不在 {sorted(profiles)}")
        return 1
    p = profiles[active]
    provider = Path(str(p["provider_uri"]))
    print("=" * 60)
    print("Qlib / RD 数据进度（从左到右：越后面越接近可挖因子）")
    print("=" * 60)
    print(f"当前 profile: {active}")
    print(f"期望 provider: {provider}")
    print(f"期望日期窗: {p.get('start')} → {p.get('end')}  market={p.get('market')}")
    print()

    # Layer 1: staging
    staging_manifest = staging / "manifest.json"
    st = _load_json(staging_manifest)
    staging_ok = bool(st)
    print(f"[1/4] staging CSV     [{_mark(staging_ok)}]  {staging}")
    if st:
        dr = st.get("date_range") or {}
        print(
            f"       日期 {dr.get('start')}→{dr.get('end')}  "
            f"标的 {st.get('instruments')}  "
            f"更新 {_mtime(staging_manifest)}"
        )
    else:
        print(
            "       下一步: .venv312\\Scripts\\python.exe -m jq_research_system.qlib_adapter db ..."
        )

    # Layer 2: provider
    pm_path = provider / "provider_manifest.json"
    pm = _load_json(pm_path)
    factor_bin = (
        next(provider.glob("features/*/factor.day.bin"), None) if provider.is_dir() else None
    )
    provider_ok = bool(pm) and factor_bin is not None
    print(f"[2/4] Qlib provider   [{_mark(provider_ok)}]  {provider}")
    if pm:
        dr = pm.get("date_range") or {}
        print(
            f"       日期 {dr.get('start')}→{dr.get('end')}  "
            f"选中 {pm.get('selected_instruments')}  "
            f"factor_policy={pm.get('factor_policy')}  "
            f"更新 {_mtime(pm_path)}"
        )
        src = pm.get("source_created_at")
        if src and st and st.get("created_at") and src != st.get("created_at"):
            print("       ! staging 比 provider 源更新时间不一致 → 建议重建 provider")
    else:
        print(
            "       下一步: .\\dev.ps1 qlib-provider artifacts\\cn_real_staging <provider_out> --max-instruments 500"
        )

    # Layer 3: RD templates / h5
    candidates = [
        Path(r"D:\Users\lenovo\miniconda3\envs\rd-agent\Lib\site-packages")
        / "rdagent"
        / "scenarios"
        / "qlib"
        / "experiment",
    ]
    try:
        import rdagent

        roots = list(getattr(rdagent, "__path__", []) or [])
        if roots:
            candidates.insert(0, Path(roots[0]) / "scenarios" / "qlib" / "experiment")
    except Exception:
        pass
    exp = next((c for c in candidates if c.is_dir()), None)

    h5_all = exp / "factor_data_template" / "daily_pv_all.h5" if exp else None
    gen_py = exp / "factor_data_template" / "generate.py" if exp else None
    yaml_base = exp / "factor_template" / "conf_baseline.yaml" if exp else None
    h5_ok = bool(h5_all and h5_all.is_file())
    print(f"[3/4] RD daily_pv h5  [{_mark(h5_ok)}]  {h5_all or '(找不到 rdagent 包)'}")
    if h5_all and h5_all.is_file():
        print(f"       大小 {_size_mb(h5_all)}  更新 {_mtime(h5_all)}")
        if pm_path.is_file() and h5_all.stat().st_mtime < pm_path.stat().st_mtime:
            print("       ! provider 比 h5 新 → 建议 .\\dev.ps1 qlib-regen-h5")
    if gen_py and gen_py.is_file():
        text = gen_py.read_text(encoding="utf-8")
        uri_ok = str(p["provider_uri"]).replace("\\", "/") in text.replace("\\", "/")
        print(f"       generate.py 指向当前 provider: {_mark(uri_ok)}")
    if yaml_base and yaml_base.is_file():
        ytext = yaml_base.read_text(encoding="utf-8")
        uri_ok = str(p["provider_uri"]).replace("\\", "/") in ytext.replace("\\", "/")
        print(f"       conf_baseline.yaml 指向当前 provider: {_mark(uri_ok)}")
        bt = p.get("backtest") or []
        if len(bt) == 2 and provider.is_dir():
            cal = provider / "calendars" / "day.txt"
            if cal.is_file():
                last = cal.read_text(encoding="utf-8").strip().splitlines()[-1].strip()
                if str(bt[1]) >= last:
                    print(
                        f"       ! backtest 结束日 {bt[1]} 不宜 >= 日历末日 {last} "
                        f"（会触发 PortAna IndexError → 收益 NaN）"
                    )

    # Layer 4: ready for rd-factor
    ready = staging_ok and provider_ok and h5_ok
    print(f"[4/4] 可跑 rd-factor  [{_mark(ready)}]")
    if ready:
        print("       日常: .\\dev.ps1 rd-factor 4")
        print("       观测: .\\dev.ps1 rd-status   或  .\\dev.ps1 rd-ui")
    else:
        print("       缺哪层补哪层；更新行情后可用: .\\dev.ps1 qlib-refresh")

    # Benchmark must exist in provider or PortAna returns NaN
    bench = str(p.get("benchmark") or "")
    if provider_ok and bench:
        feat = provider / "features" / bench.lower()
        bench_ok = feat.is_dir() and any(feat.glob("close.day.bin"))
        print(f"[+] 回测基准 {bench}  [{_mark(bench_ok)}]")
        if not bench_ok:
            print(
                "       ! provider 无此标的 → PortAna 收益会 NaN；改 profile.benchmark 为池内股票或导入指数"
            )

    print("=" * 60)
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
