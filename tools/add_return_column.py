"""给因子源数据 daily_pv.h5 补上 $return 列，修复"LLM 生成非法表达式空转"。

背景（深挖 mining 空转的第三个堵点）：
- 喂给 LLM 的 README.md 与 prompts.yaml 都声明了 `$return: daily return of the stock`，
  但实际落盘的 daily_pv.h5 只有 ['$open','$close','$high','$low','$volume','$factor'] 六列，
  没有 $return。
- LLM 照 README 生成 `TS_STD($return, 20)` 一类表达式，执行时报 undefined variable '$return'，
  全部被 regulator 拒收 -> 因子空转、0 有效因子。

修复：
- 给 daily_pv.h5（full + debug）补上 `$return` 列 = `$close` 按 instrument 分组的日收益率
  (pct_change，首日 fillna 0)，与 README/prompt 声明保持一致。
- 这是数据侧最小改动，避免大改 prompt（prompt 里 $return 出现在多处正例，改动易遗漏）。

用法：
  python tools/add_return_column.py [h5路径1 h5路径2 ...]
  默认处理 git_ignore_folder/factor_implementation_source_data{,_debug}/daily_pv.h5
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

DEFAULT_TARGETS = [
    "git_ignore_folder/factor_implementation_source_data/daily_pv.h5",
    "git_ignore_folder/factor_implementation_source_data_debug/daily_pv.h5",
]


def add_return_column(path: Path) -> bool:
    if not path.exists():
        print(f"[skip] 不存在: {path}")
        return False
    df = pd.read_hdf(path)
    cols = list(df.columns)
    if "$return" in cols:
        print(f"[skip] 已有 $return: {path} (columns={cols})")
        return False
    # 按 instrument 分组求日收益率；先排序保证跨日顺序正确
    df = df.sort_index()
    ret = df.groupby(level="instrument")["$close"].pct_change(fill_method=None).fillna(0.0)
    df["$return"] = ret
    # 保持列顺序：$return 放在 $close 之后、$factor 之前
    ordered = [c for c in cols if c != "$factor"] + ["$return"] + (["$factor"] if "$factor" in cols else [])
    df = df[ordered]
    df.to_hdf(path, key="data")
    print(f"[done] 已补 $return: {path} -> columns={list(df.columns)}, shape={df.shape}")
    return True


def main() -> int:
    targets = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_TARGETS
    changed = 0
    for t in targets:
        if add_return_column(Path(t)):
            changed += 1
    print(f"完成：处理 {len(targets)} 个文件，改动 {changed} 个")
    return 0


if __name__ == "__main__":
    sys.exit(main())
