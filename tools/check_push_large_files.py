"""Reject a push when it would introduce oversized Git blobs."""

from __future__ import annotations

import os
import subprocess
import sys

MAX_BYTES = int(float(os.environ.get("GIT_MAX_PUSH_FILE_MB", "5")) * 1024 * 1024)
ZERO_OID = "0" * 40


def new_objects(local_oid: str, remote_oid: str, remote_name: str) -> dict[str, str]:
    command = ["git", "rev-list", "--objects", local_oid]
    if remote_oid != ZERO_OID:
        command.append(f"^{remote_oid}")
    else:
        command.extend(["--not", f"--remotes={remote_name}"])
    lines = subprocess.check_output(
        command, text=True, encoding="utf-8", errors="replace"
    ).splitlines()
    return {line.split(" ", 1)[0]: line.split(" ", 1)[1] if " " in line else "" for line in lines}


def oversized_blobs(objects: dict[str, str]) -> list[tuple[int, str, str]]:
    if not objects:
        return []
    process = subprocess.Popen(
        ["git", "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    output, _ = process.communicate("\n".join(objects))
    result: list[tuple[int, str, str]] = []
    for line in output.splitlines():
        object_id, object_type, size_text = line.split(" ", 2)
        size = int(size_text)
        if object_type == "blob" and size > MAX_BYTES:
            result.append((size, objects.get(object_id, ""), object_id))
    return sorted(result, reverse=True)


def main() -> int:
    remote_name = sys.argv[1] if len(sys.argv) > 1 else "origin"
    errors: list[tuple[int, str, str]] = []
    for line in sys.stdin:
        fields = line.strip().split()
        if len(fields) != 4:
            continue
        _local_ref, local_oid, _remote_ref, remote_oid = fields
        if local_oid == ZERO_OID:
            continue
        errors.extend(oversized_blobs(new_objects(local_oid, remote_oid, remote_name)))
    if errors:
        print(
            f"Push blocked: new blobs exceed {MAX_BYTES / 1024 / 1024:.2f} MiB.",
            file=sys.stderr,
        )
        for size, path, object_id in errors[:20]:
            print(
                f"  {size / 1024 / 1024:.2f} MiB {object_id[:12]} {path or '<unknown path>'}",
                file=sys.stderr,
            )
        return 1
    print(f"push file check passed (limit={MAX_BYTES / 1024 / 1024:.2f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
