"""Align RD-Agent Qlib templates with repo config/qlib_profile.json.

Qlib is the data boundary; RD only consumes provider_uri / market / dates.
Managed files are rewritten idempotently when the active profile changes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

_MARKER = "# managed-by: qlib_data_profile_patch"
_STATE_NAME = ".qlib_profile_applied.json"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _profile_path() -> Path:
    override = os.environ.get("RD_QLIB_PROFILE_FILE")
    if override:
        return Path(override)
    return _repo_root() / "config" / "qlib_profile.json"


def load_profile_config() -> dict[str, Any]:
    path = _profile_path()
    if not path.is_file():
        raise FileNotFoundError(f"missing qlib profile config: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("profiles"), dict) or not data["profiles"]:
        raise ValueError(f"invalid profiles in {path}")
    return data


def resolve_active_profile_name(config: dict[str, Any] | None = None) -> str:
    cfg = config or load_profile_config()
    env_name = (os.environ.get("RD_QLIB_PROFILE") or "").strip()
    name = env_name or str(cfg.get("active") or "research")
    if name not in cfg["profiles"]:
        raise KeyError(f"unknown profile {name!r}; known={sorted(cfg['profiles'])}")
    return name


def get_active_profile() -> tuple[str, dict[str, Any]]:
    cfg = load_profile_config()
    name = resolve_active_profile_name(cfg)
    profile = dict(cfg["profiles"][name])
    required = (
        "provider_uri",
        "market",
        "benchmark",
        "start",
        "end",
        "fit_end",
        "train",
        "valid",
        "test",
        "backtest",
    )
    missing = [k for k in required if k not in profile]
    if missing:
        raise ValueError(f"profile {name!r} missing keys: {missing}")
    for key in ("train", "valid", "test", "backtest"):
        pair = profile[key]
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"profile {name!r}.{key} must be [start, end]")
    return name, profile


def _fingerprint(name: str, profile: dict[str, Any]) -> str:
    payload = json.dumps({"name": name, "profile": profile}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _factor_experiment_root() -> Path:
    import rdagent

    roots = list(getattr(rdagent, "__path__", []) or [])
    if roots:
        return Path(roots[0]) / "scenarios" / "qlib" / "experiment"
    if rdagent.__file__:
        return Path(rdagent.__file__).resolve().parent / "scenarios" / "qlib" / "experiment"
    raise RuntimeError("cannot locate rdagent package path")


def _seg(pair: list[str]) -> str:
    return f"[{pair[0]}, {pair[1]}]"


def _rewrite_yaml(text: str, profile: dict[str, Any]) -> str:
    uri = str(profile["provider_uri"]).replace("\\", "/")
    market = str(profile["market"])
    benchmark = str(profile["benchmark"])
    start = str(profile["start"])
    end = str(profile["end"])
    fit_end = str(profile["fit_end"])
    train, valid, test, backtest = (
        profile["train"],
        profile["valid"],
        profile["test"],
        profile["backtest"],
    )

    text = re.sub(
        r'(provider_uri:\s*)(?:"[^"]*"|\'[^\']*\'|\S+)',
        rf'\1"{uri}"',
        text,
        count=1,
    )
    text = re.sub(r"(market:\s*&market\s+)\S+", rf"\1{market}", text, count=1)
    text = re.sub(r"(benchmark:\s*&benchmark\s+)\S+", rf"\1{benchmark}", text, count=1)

    # Generic date fields (handler / backtest windows).
    text = re.sub(r"(start_time:\s*)\d{4}-\d{2}-\d{2}", rf"\g<1>{start}", text)
    # end_time under data_handler often equals profile end; backtest has its own pair.
    # Replace all end_time then fix backtest block specifically.
    text = re.sub(r"(end_time:\s*)\d{4}-\d{2}-\d{2}", rf"\g<1>{end}", text)
    text = re.sub(
        r"(backtest:\s*\n(?:[ \t]+.*\n)*?[ \t]+start_time:\s*)\d{4}-\d{2}-\d{2}",
        rf"\g<1>{backtest[0]}",
        text,
        count=1,
    )
    text = re.sub(
        r"(backtest:\s*\n(?:[ \t]+.*\n)*?[ \t]+end_time:\s*)\d{4}-\d{2}-\d{2}",
        rf"\g<1>{backtest[1]}",
        text,
        count=1,
    )

    text = re.sub(
        r"(fit_start_time:\s*)\d{4}-\d{2}-\d{2}",
        rf"\g<1>{start}",
        text,
    )
    text = re.sub(
        r"(fit_end_time:\s*)\d{4}-\d{2}-\d{2}",
        rf"\g<1>{fit_end}",
        text,
    )

    text = re.sub(
        r"(train:\s*)\[[^\]]+\]",
        rf"\g<1>{_seg(train)}",
        text,
        count=1,
    )
    text = re.sub(
        r"(valid:\s*)\[[^\]]+\]",
        rf"\g<1>{_seg(valid)}",
        text,
        count=1,
    )
    text = re.sub(
        r"(test:\s*)\[[^\]]+\]",
        rf"\g<1>{_seg(test)}",
        text,
        count=1,
    )

    if not text.lstrip().startswith(_MARKER):
        text = f"{_MARKER}\n{text}"
    return text


def _rewrite_generate_py(text: str, profile: dict[str, Any]) -> str:
    uri = str(profile["provider_uri"]).replace("\\", "/")
    market = str(profile["market"])
    start = str(profile["start"])
    end = str(profile["end"])
    fit_end = str(profile["fit_end"])

    text = re.sub(
        r'qlib\.init\(provider_uri\s*=\s*["\'][^"\']+["\']\)',
        f'qlib.init(provider_uri="{uri}")',
        text,
        count=1,
    )
    text = re.sub(
        r'D\.instruments\(market\s*=\s*["\'][^"\']+["\']\)',
        f'D.instruments(market="{market}")',
        text,
        count=1,
    )

    # Replace D.features date windows in order: full range, then fit/debug (if present).
    pattern = r'(start_time\s*=\s*")\d{4}-\d{2}-\d{2}(",\s*end_time\s*=\s*")\d{4}-\d{2}-\d{2}(")'
    n_matches = len(list(re.finditer(pattern, text)))
    if n_matches == 0:
        raise ValueError("generate.py expected date windows in D.features(...), found 0")
    windows = [(start, end), (start, fit_end)][:n_matches]
    idx = 0

    def _replace_window(match: re.Match[str]) -> str:
        nonlocal idx
        if idx >= len(windows):
            return match.group(0)
        s, e = windows[idx]
        idx += 1
        return f"{match.group(1)}{s}{match.group(2)}{e}{match.group(3)}"

    text = re.sub(pattern, _replace_window, text, count=len(windows))
    if idx != len(windows):
        raise ValueError(f"generate.py expected {len(windows)} date windows, rewrote {idx}")

    if not text.lstrip().startswith(_MARKER):
        text = f"{_MARKER}\n{text}"
    return text


def apply_profile_to_templates(*, force: bool = False) -> dict[str, Any]:
    name, profile = get_active_profile()
    fp = _fingerprint(name, profile)
    root = _factor_experiment_root()
    state_path = root / _STATE_NAME
    if not force and state_path.is_file():
        try:
            prev = json.loads(state_path.read_text(encoding="utf-8"))
            if prev.get("fingerprint") == fp and prev.get("profile") == name:
                return {"applied": False, "profile": name, "fingerprint": fp, "reason": "unchanged"}
        except Exception:
            pass

    targets = [
        root / "factor_template" / "conf_baseline.yaml",
        root / "factor_template" / "conf_combined_factors.yaml",
        root / "factor_template" / "conf_combined_factors_sota_model.yaml",
        root / "factor_data_template" / "generate.py",
    ]
    written: list[str] = []
    for path in targets:
        if not path.is_file():
            continue
        raw = path.read_text(encoding="utf-8")
        if path.suffix == ".py":
            new = _rewrite_generate_py(raw, profile)
        else:
            new = _rewrite_yaml(raw, profile)
        if new != raw:
            path.write_text(new, encoding="utf-8", newline="\n")
            written.append(str(path))
        else:
            written.append(str(path) + " (unchanged-content)")

    state = {
        "profile": name,
        "fingerprint": fp,
        "provider_uri": profile["provider_uri"],
        "written": written,
    }
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"[qlib-data-profile-patch] applied profile={name} provider={profile['provider_uri']}",
        file=sys.stderr,
        flush=True,
    )
    return {"applied": True, "profile": name, "fingerprint": fp, "written": written}


def apply() -> None:
    try:
        import rdagent  # noqa: F401
    except Exception:
        return
    try:
        apply_profile_to_templates(force=False)
    except Exception as exc:
        print(f"[qlib-data-profile-patch] skip: {exc}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    force = "--force" in sys.argv
    print(json.dumps(apply_profile_to_templates(force=force), ensure_ascii=False, indent=2))
