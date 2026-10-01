"""Reject secrets and oversized files before they enter a Git commit."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import PurePosixPath

MAX_BYTES = int(float(os.environ.get("GIT_MAX_FILE_MB", "5")) * 1024 * 1024)
FORBIDDEN = {
    ".env",
    "settings.local.json",
    "jq_research_system/config/db_config.json",
    "jq_research_system/config/qmt_config.json",
}


def staged_paths() -> list[str]:
    output = subprocess.check_output(
        ["git", "diff", "--cached", "--diff-filter=AMCR", "--name-only", "-z"]
    )
    return [item.decode("utf-8", "replace") for item in output.split(b"\0") if item]


def index_blob_size(path: str) -> int | None:
    entry = subprocess.check_output(["git", "ls-files", "--stage", "-z", "--", path]).split(
        b"\0", 1
    )[0]
    if not entry:
        return None
    metadata = entry.split(b"\t", 1)[0].decode("ascii").split()
    mode, object_id = metadata[0], metadata[1]
    if mode == "160000":  # Gitlink/submodule; the target commit may not exist locally.
        return None
    return int(subprocess.check_output(["git", "cat-file", "-s", object_id], text=True).strip())


def main() -> int:
    errors: list[str] = []
    for path in staged_paths():
        normalized = PurePosixPath(path).as_posix()
        if normalized in FORBIDDEN or normalized.endswith("/.env"):
            errors.append(f"禁止提交本机配置或密钥文件: {normalized}")
            continue
        size = index_blob_size(path)
        if size is None:
            continue
        if size > MAX_BYTES:
            errors.append(
                f"文件过大: {normalized} ({size / 1024 / 1024:.2f} MiB > "
                f"{MAX_BYTES / 1024 / 1024:.2f} MiB)"
            )
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"staged file check passed (limit={MAX_BYTES / 1024 / 1024:.2f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
