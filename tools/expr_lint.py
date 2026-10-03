"""因子表达式静态检查（挖出来 / 入库前跑一遍，秒级，不需要 qlib 和数据）。

存在理由（2026-10-04）：方向串里要求"波动类信号显式取负"，LLM 老老实实写了，
但写出的是 `-1 * TS_STD($return, 20) * -1 * TS_CORR($return, $volume, 20)` ——
**两个 `-1` 相乘等于没取负**。这类错误不会报错、括号也是平衡的、回测照跑，
只有等 RankIC 出来才发现符号不对，一轮挖掘白费。

所以这里做的是**纯静态的语义体检**，只报错不改表达式，判定权交给人。

用法：
    python tools/expr_lint.py "ZSCORE(-1 * TS_STD($return, 20) * -1 * TS_CORR($return, $volume, 20))"
    python tools/expr_lint.py --json data/factorlib/all_factors_library.json
    from tools.expr_lint import lint; lint(expr) -> [dict]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------- 检查项

# 乘性常量 -1：只认「紧跟 * 或括号边界」的 -1，避免把 `1e-8` / `$close - 1` 算进来。
_NEG_ONE_RE = re.compile(r"(?:(?<=^)|(?<=[\*\(\+/\-,]))\s*-\s*1\s*(?=\*|\))")

# 时序/滚动算子的窗口实参
_TS_WINDOW_RE = re.compile(r"\bTS_[A-Z_]+\s*\(\s*[^\(\)]*?,\s*(\d+)\s*\)")

# 上一轮已被真实复验证伪的 4 类模式（独立 RankIC 全在 ±0.01 内，别再挖了）
_DEPRECATED = (
    ("缩量收缩", re.compile(r"\$volume\s*<\s*0?\.?\d*\s*\*\s*TS_MEAN\(\$volume")),
    ("下轨回撤", re.compile(r"\$close\s*<\s*0?\.?\d+\s*\*\s*TS_MAX\(\$close")),
    ("波动分歧", re.compile(r"TS_STD\(\s*\$return[^\)]*\)\s*-\s*TS_STD\(\s*\$return")),
    ("孤峰(偏度/熵×波动)", re.compile(r"(?:TS_SKEW|ENTROPY)\([^\)]*\)\s*\*\s*TS_STD\(")),
)

# qlib 表达式引擎不支持 / 会静默出错的写法。
# 注意：TS_ZSCORE 不在这里 —— 它是平台合法算子，且 tools/oos_validate.py 的归一化层
# 已把它展开成 (x - Mean(x,N)) / (Std(x,N)+1e-8)，复验器能跑，不算问题。
_UNSUPPORTED = (
    ("DELAY( qlib 无此算子，须写 Ref", re.compile(r"\bDELAY\s*\(")),
    ("三元 ?: qlib 不支持", re.compile(r"\?.*:")),
    ("COUNT() 内用了 &&，须算术化", re.compile(r"COUNT\s*\([^\(\)]*&&")),
    ("字段被驼峰化成 $Return（qlib 无此字段，整列 NaN）", re.compile(r"\$Return")),
)


def _count_neg_one(expr: str) -> int:
    """统计作为**乘法因子**出现的 -1 的个数（用于判断显式取负有没有被抵消）。"""
    hits = 0
    for m in _NEG_ONE_RE.finditer(expr):
        # 排除 `e-1`（科学计数法）和 `1e-1` 之类的数字内部匹配
        prev = expr[: m.start()].rstrip()
        if prev.endswith("e") or prev.endswith("E"):
            continue
        hits += 1
    return hits


def lint(expr: str) -> list[dict]:
    """返回问题列表，每项 {level, code, msg}。level: error > warn > info。"""
    issues: list[dict] = []

    # 1) 显式取负有没有被抵消
    n = _count_neg_one(expr)
    if n >= 2 and n % 2 == 0:
        issues.append({
            "level": "error",
            "code": "NEG_CANCEL",
            "msg": (f"出现 {n} 个乘性 -1（偶数），正负抵消 = 等于没取负。"
                    f"净符号 = +1；若本意是取负，请只保留 1 个 -1。"),
        })
    elif n == 1:
        issues.append({"level": "info", "code": "NEG_OK", "msg": "已显式取负（1 个 -1），净符号 = -1。"})

    # 2) 滚动窗口太短（1~3 日的纯噪声结构）
    for w in _TS_WINDOW_RE.findall(expr):
        if int(w) <= 3:
            issues.append({
                "level": "warn",
                "code": "SHORT_WINDOW",
                "msg": f"滚动窗口 = {w}（≤3），实盘基本是噪声且换手极高，建议 ≥5、优先 10/20/60。",
            })
            break

    # 3) 已被证伪的同质模式
    for name, rx in _DEPRECATED:
        if rx.search(expr):
            issues.append({
                "level": "warn",
                "code": "DEPRECATED",
                "msg": f"命中已证伪模式「{name}」——上一轮复验其独立 RankIC 在 ±0.01 内。",
            })

    # 4) qlib 跑不动的写法
    for name, rx in _UNSUPPORTED:
        if rx.search(expr):
            issues.append({"level": "error", "code": "UNSUPPORTED", "msg": f"{name}，qlib 表达式引擎会报错或静默 NaN。"})

    return issues


def lint_many(exprs: list[str]) -> dict[str, list[dict]]:
    return {e: lint(e) for e in exprs}


# ---------------------------------------------------------------- CLI


def _print_one(expr: str, issues: list[dict]) -> None:
    mark = {"error": "ERR ", "warn": "WARN", "info": "info"}
    if not issues:
        print(f"[ok]   {expr[:88]}")
        return
    print(f"[{mark[issues[0]['level']]}] {expr[:88]}")
    for it in issues:
        print(f"        - {it['code']}: {it['msg']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="因子表达式静态体检（不跑数据）")
    ap.add_argument("expr", nargs="*", help="直接跟表达式")
    ap.add_argument("--json", help="因子库 JSON（{name: expr} 或 [{'name':..,'expression':..}]）")
    args = ap.parse_args(argv)

    items: list[tuple[str, str]] = []
    for e in args.expr:
        items.append(("-", e))
    if args.json:
        raw = json.loads(Path(args.json).read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            items += [(k, v if isinstance(v, str) else str(v)) for k, v in raw.items()]
        else:
            items += [(d.get("name", "?"), d.get("expression", "")) for d in raw]
    if not items:
        ap.print_help()
        return 1

    n_err = 0
    for name, expr in items:
        iss = lint(expr)
        n_err += sum(1 for i in iss if i["level"] == "error")
        print(f"## {name}")
        _print_one(expr, iss)
    print(f"\n合计 {len(items)} 条，error {n_err} 条。")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
