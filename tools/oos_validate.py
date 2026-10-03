"""演化因子的时序样本外（OOS）复验器。

背景 / 为什么要做这个
----------------------
AlphaAgent 演化挖掘在「全样本」上跑回测并按 RankIC 选 best，这个「选 best」的动作
本身就吃进了未来信息（selection bias）：一条轨迹在 2020-2026 全段 RankIC=0.064，
并不代表它在训练段之后仍有效。真正能拿来说话的是**时序分段**表现：

    train (2020~2023) -> valid (2024) -> test (2025~2026 纯样本外)

判读规则（避免自欺）：
  - test 段 RankIC 与 train 段同量级 (=0.6~1.0x) -> 信号真实可用；
  - test 段 RankIC 明显衰减 (<0.5x)             -> 主要是选择偏差 + 过拟合，不能上；
  - 单因子 RankIC 高但 TopK 组合扣费后年化为负 -> 换手太贵，无交易价值。

本脚本只做**评估**，不参与任何挖矿/选因子，因此不存在把 test 信息泄漏回因子的问题。

用法
----
    python tools/oos_validate.py --pool log/latest_trajectory_pool.json --k 5
    python tools/oos_validate.py --pool log/latest_trajectory_pool.json --k 3 \
        --train-start 2020-01-01 --train-end 2023-12-31 \
        --valid-start 2024-01-01 --valid-end 2024-12-31 \
        --test-start 2025-01-01 --test-end 2026-08-18 \
        --topk 50 --horizon 1 --cost-bps 5 --out data/results/oos_validate_topk.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("oos_validate")

# qlib 表达式里 $return 的标准定义（与 quantaalpha/backtest/factor_loader.py 一致）
RET_FIELD = "($close/Ref($close,1)-1)"
DEFAULT_QLIB_DIR = "C:/Users/lenovo/.qlib/qlib_data/cn_real_full"

# rdagent(TS_*) 算子 -> qlib 原生算子。
# qlib 已注册算子只有驼峰这 52 个（Abs/Mean/Std/Max/Min/Sum/Rank/Quantile/Ref/Corr/Sign...），
# 没有 TsRank / Delay / TsMean 这些 rdagent 别名，不转换会报 "operator not registered"。
# DELAY -> Ref 尤其关键：qlib 里根本没有 Delay。
OP_CONVERSIONS = {
    "DELAY": "Ref",
    "TS_DECAY_LINEAR": "WMA",
    "TS_ARGMAX": "IdxMax",
    "TS_ARGMIN": "IdxMin",
    # 纯函数运算符：qlib 注册名同样是驼峰，全大写写法需一并归一
    "ABS": "Abs",
    "MIN": "Min",
    "MAX": "Max",
    "SIGN": "Sign",
    # 其余 TS_XXX 走通用规则（TS_CORR->Corr、TS_RANK->Rank ...），见 _ts_to_camel
    "$return": RET_FIELD,
}

_CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def _ts_to_camel(name: str) -> str:
    """TS_CORR -> Corr：qlib 注册名一律驼峰，去掉 TS_ 前缀后整体首字母大写。

    注意必须用 capitalize() 而不是仅首字母大写：TS_MEAN 的尾部是 "MEAN"，
    只改首字母仍会得到 "MEAN"，qlib 里注册的是 "Mean"。
    """
    tail = name[3:] if name.upper().startswith("TS_") else name
    return tail.capitalize()


# 2026-10-03 重写：原先这套用正则 + 多轮替换来展开 TS_ZSCORE / TS_PCTCHANGE，
# 实测有两个硬伤（都曾让整条因子从没被独立复验过）：
#   (1) 实参模式 _ARG 只能吃**一层**嵌套括号，
#       ZSCORE(TS_ZSCORE(TS_STD($return,20) - TS_STD($return,60), 60) * TS_SKEW($return,20))
#       这种两级以上嵌套的实参压根匹配不上，外层展开失败 → 剩下 Zscore → 判"不可复算"跳过。
#   (2) 通用 TS_*→驼峰的转换在展开之后跑，导致 TS_ZSCORE 先被改成 Zscore 再去找展开规则，顺序反了。
# 现改为「平衡括号扫描 + 按函数名分派」的单趟重写，任意嵌套深度都吃得住。

# qlib 的 Rank 是**时序滚动**排名 (Rank(feature, N) -> rolling(N).rank(pct=True))，
# 而平台 function_lib.RANK 是**日截面**排名 (df.groupby('datetime').rank(pct=True))，
# 两者语义完全不同。裸 RANK(x) 必须走截面口径，这里先降级成占位符，算完再后处理（见 _csrank_postprocess）。
_CSRANK_MARK = "__CSRANK__"


def _match_paren(s: str, i: int) -> int:
    """s[i] 是 '(' 时返回配对的 ')' 下标，无括号返回 -1。"""
    depth, n = 0, len(s)
    while i < n:
        if s[i] == "(":
            depth += 1
        elif s[i] == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _split_top_args(s: str) -> list[str]:
    """按顶层逗号切函数实参（忽略括号内的逗号）。"""
    args, buf, depth = [], [], 0
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            args.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        args.append("".join(buf).strip())
    return args


def _as_int(s: str):
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


def _r_zscore(inner: str) -> str:
    """TS_ZSCORE(x, N) / ZSCORE(x, N) -> (x - Mean(x,N)) / (Std(x,N) + 1e-8)；无窗口走全样本。"""
    a = _split_top_args(inner)
    if len(a) >= 2 and _as_int(a[1]) is not None:
        return f"(({a[0]} - Mean({a[0]}, {a[1]})) / (Std({a[0]}, {a[1]}) + 1e-8))"
    return f"(({a[0]} - Mean({a[0]})) / (Std({a[0]}) + 1e-8))"


def _r_pctchange(inner: str) -> str:
    a = _split_top_args(inner)
    return f"({a[0]} / Ref({a[0]}, {a[1]}) - 1)"


def _r_delta(inner: str) -> str:
    """TS_DELTA/DELTA(x, p) -> qlib 无 Delta，用 Ref 还原成 x - Ref(x, p)。"""
    a = _split_top_args(inner)
    return f"({a[0]} - Ref({a[0]}, {a[1]}))"


def _r_delay(inner: str) -> str:
    """DELAY(x, p) -> 驼峰会给出 qlib 未注册的 Delay，必须显式归一成 Ref。"""
    a = _split_top_args(inner)
    return f"Ref({a[0]}, {a[1]})"


# && / || 等逻辑运算符在 qlib 里只能出现在表达式顶层，放进 COUNT(...) 这类函数实参
# 就是语法错误（实测 invalid syntax）。布尔比较结果本身是 0/1，所以改写成算术等价式。
# 下面两个函数负责从运算符位置向两侧找“最近的一个边界”：要么是一个括号（含其自身的
# 开/闭括号），要么是一个同层运算符。比“先找配对括号再取内容”稳得多 —— 后者在
# COUNT((A) && (B), 5) 这种带嵌套的实参上会把括号配对错，替换后字符串不变 → 死循环。
_OPS = set("+-*/%<>!=&|,")  # 逗号也算边界：否则 COUNT((A) && (B), 5) 会把窗口参数 "5" 吞掉


def _find_left_start(s: str, pos: int, ops: set[str] | None = None) -> int:
    """pos 是运算符起点，返回左操作数的起始下标。"""
    ops = _OPS if ops is None else ops
    depth, c = 0, pos - 1
    while c >= 0:
        ch = s[c]
        if ch == "(":
            if depth == 0:
                return c
            depth -= 1
        elif ch == ")":
            depth += 1
        elif ch in ops and depth == 0:
            return c + 1
        c -= 1
    return 0


def _open_of(s: str, close_pos: int) -> int:
    """close_pos 处是 ')'，返回配对的那个 '(' 的下标；找不到返回 -1。

    之前用「往左走，depth 归零时的那个 '('」来定位，在这个因子上永远差一层
    （COUNT((A) && (B), 5) * ((cond) ? (a) : 0) 这类嵌套里 depth 归零点不稳定）。
    改成直接搜「配对 ')' 正好是 close_pos 的那个 '('」，语义唯一、不会错层。
    """
    p = s.find("(")
    while 0 <= p < close_pos:
        if _match_paren(s, p) == close_pos:
            return p
        p = s.find("(", p + 1)
    return -1


def _find_right_end(s: str, pos: int, ops: set[str] | None = None) -> int:
    """pos 是运算符起点，返回右操作数的结束下标（开区间）。"""
    ops = _OPS if ops is None else ops
    depth, c, n = 0, pos, len(s)
    while c < n:
        ch = s[c]
        if ch == "(":
            depth += 1
        elif ch == ")":
            if depth == 0:
                return c
            depth -= 1
        elif ch in ops and depth == 0:
            return c
        c += 1
    return n


def _rewrite_and_or(expr: str) -> str:
    """布尔逻辑 && / || 改写成 qlib 认得的算术等价式。

    直接把 && 换成 and 是没用的：qlib 的 and/or 只在表达式顶层成立，
    放进 COUNT(...) 这类函数实参里就是语法错误（实测 invalid syntax）。
    布尔比较结果本身就是 0/1，所以：A && B -> A*B，A || B -> A+B-A*B。
    """
    out = expr
    for token, build in (
        ("&&", lambda a, b: f"(({a}) * ({b}))"),
        ("||", lambda a, b: f"(({a}) + ({b}) - ({a}) * ({b}))"),
    ):
        for _round in range(64):  # 硬上限：一次迭代没改到字符串就跳出，杜绝死循环
            i = out.find(token)
            if i == -1:
                break
            a_start = _find_left_start(out, i)
            b_end = _find_right_end(out, i + len(token))
            a = out[a_start:i].strip()
            b = out[i + len(token) : b_end].strip()
            if not a or not b:
                break
            new_out = out[:a_start] + build(a, b) + out[b_end:]
            if new_out == out:
                break
            out = new_out
    return out


def _r_rank(inner: str) -> str:
    """TS_RANK(x, N)：平台是时序滚动排名，qlib Rank 也是滚动排名，语义一致，直接透传并补上必填的 N。"""
    a = _split_top_args(inner)
    return f"Rank({a[0]}, {a[1]})"


def _r_csrank(inner: str) -> str:
    """裸 RANK(x)：平台是日截面排名 → 占位符，留到算完再按日分组做 rank(pct=True)。"""
    return f"{_CSRANK_MARK}({_split_top_args(inner)[0]})"


def _r_quantile(inner: str) -> str:
    """TS_QUANTILE 的实参顺序在池子里不统一（TS_QUANTILE($volume,0.25,20) 与
    TS_QUANTILE($close,20,0.2) 同时存在），而 qlib 是 Quantile(feature, N, qscore)；
    因此只要第二个实参是 (0,1) 的概率值就交换，否则原样透传。"""
    a = _split_top_args(inner)
    if len(a) != 3:
        return f"Quantile({', '.join(a)})"
    q, n = _as_int(a[1]), _as_int(a[2])
    if q is not None and 0 < q < 1 and n is not None and n > 1:
        return f"Quantile({a[0]}, {n}, {q})"
    return f"Quantile({a[0]}, {a[1]}, {a[2]})"


# 需要按函数名分派的「非原生算子」；其余 TS_* 统一走 _ts_to_camel 驼峰映射
_RULES = {
    "TS_ZSCORE": _r_zscore,
    "ZSCORE": _r_zscore,
    "ZSCORE_": _r_zscore,  # 兼容 TS_ZSCORE 之下层被提前吃掉的情况（防呆，正常用不到）
    "TS_PCTCHANGE": _r_pctchange,
    "TS_DELTA": _r_delta,
    "DELTA": _r_delta,
    "DELAY": _r_delay,
    "TS_DELAY": _r_delay,
    "TS_RANK": _r_rank,
    "RANK": _r_csrank,
    "TS_QUANTILE": _r_quantile,
}

_NUM_RE = re.compile(r"\.?\d+(?:[eE][+-]?\d+)?")


def _rewrite_calls(expr: str) -> str:
    """单趟扫描重写：平衡括号定位每个函数的实参，按 _RULES 分派或走驼峰归一。"""
    out, i, n = [], 0, len(expr)
    while i < n:
        ch = expr[i]
        if ch.isspace() or ch in "$(),+-*/%<>!=&|":
            out.append(ch)
            i += 1
            continue
        # 数字字面量（含 1e-8 这类科学计数法）整块吃掉，别把 1e-8 拆成 1 e - 8
        m = _NUM_RE.match(expr, i)
        if ch.isdigit() or ch == ".":
            out.append(m.group(0))
            i = m.end()
            continue
        j = i
        while j < n and (expr[j].isalnum() or expr[j] == "_"):
            j += 1
        if j == i:
            # 未收录的字符（: ? # " 等）必须原样吐出并前进，否则 i 不前进 → 死循环。
            # 三元改写残留的 ':' / '?' 就会走到这里。
            out.append(ch)
            i += 1
            continue
        name, k = expr[i:j], j
        while k < n and expr[k].isspace():
            k += 1
        if k < n and expr[k] == "(":
            close = _match_paren(expr, k)
            if close == -1:
                out.append(f"{name}(")
                i = k + 1
                continue
            inner = expr[k + 1 : close]
            # 实参必须递归归一：否则规则生成的串里残留的 TS_MAX/TS_STD 等不会被再扫一遍，
            # 后面对 OP_CONVERSIONS 做全局 replace 时 "MAX"->"Max" 会把 TS_MAX 啃成 TS_Max。
            # 递归深度靠 inner 严格变短保证终止。
            inner = _rewrite_calls(inner)
            fn = _RULES.get(name.upper())
            out.append(fn(inner) if fn else f"{_ts_to_camel(name)}({inner})")
            i = close + 1
            continue
        out.append(_ts_to_camel(name) if not name.isdigit() else name)
        i = j
    return "".join(out)


class TernaryNotSupported(Exception):
    """三元改写失败（分支定位不了）：整条表达式原样退回，由上层判"不可复算"后跳过。"""


_TERNARY_OPS = _OPS | {"?", ":"}


def _rewrite_ternary(expr: str) -> str:
    """三元 c ? a : b 改写成 (c)*a + (1-c)*b。

    qlib 表达式引擎没有三元运算符，只能这么近似。仅当 c 是**布尔比较**时与平台
    pandas 口径等价；若 c 是 0/1 之外的连续分数，数值会有偏差 —— 因此调用方会把
    这类因子的复算结果标记为"近似"，不与他类因子混为一谈。
    """
    out = expr
    # '?' 与 ':' 都当边界，这样三个分支的边界是一次同样规则的扫描算出来的，
    # 不再依赖“往左找配对括号”那套容易错位的实现（那版会把 cond 截成半个表达式）。
    for _ in range(32):
        if "?" not in out:
            break
        i = out.rfind("?")  # 从最内层（最右）开始，嵌套自然由内向外展开
        # 条件边界不能靠“往左找第一个边界”：那会把比较运算符 "<" 当成起点，
        # 把 (Xclose < 0.9 * X) 截成 "0.9 * X"。
        # 正确做法是找 '?' 左边紧邻的那个 ')'，再向内配对它那个 '('。
        c = i - 1
        while c >= 0 and out[c].isspace():
            c -= 1
        if os.environ.get("OOSV_DEBUG_TERN") == "1":
            logger.warning("[tern] i=%d c=%d out[c]=%r tail=%r", i, c, out[c] if c >= 0 else None, out[max(0, i - 40) : i])
        c_start = _open_of(out, c)
        if os.environ.get("OOSV_DEBUG_TERN") == "1":
            logger.warning(
                "[tern] c_start=%d cond_raw=%r", c_start, out[c_start:i].strip()[:120]
            )
        if c < 0 or out[c] != ")":
            raise TernaryNotSupported("条件不是括号包起来的比较式")
        c_start = _open_of(out, c)
        if c_start < 0:
            raise TernaryNotSupported("找不到条件左括号")
        cond_raw = out[c_start:i].strip()
        # 条件必须是自带括号的比较式（本项目的因子都这么写）。裸比较式宁可判定
        # “不可复算”跳过，也不要把 Xclose < 0.9 * X 截成半个表达式假装算过。
        if not (
            cond_raw.startswith("(")
            and cond_raw.endswith(")")
            and _match_paren(cond_raw, 0) == len(cond_raw) - 1
        ):
            raise TernaryNotSupported("条件不是括号包起来的比较式")
        cond = cond_raw[1:-1].strip()
        t_end = _find_right_end(out, i + 1, _TERNARY_OPS)  # 停在 ':' / ')' / 运算符
        true_b = out[i + 1 : t_end].strip()
        f_start = t_end + 1
        while f_start < len(out) and out[f_start].isspace():
            f_start += 1
        f_end = _find_right_end(out, f_start, _TERNARY_OPS)
        false_b = out[f_start:f_end].strip()
        if not cond or not true_b or not false_b:
            raise TernaryNotSupported("三元分支为空")
        rebuilt = f"(({cond}) * ({true_b}) + (1 - ({cond})) * ({false_b}))"
        out = out[:c_start] + rebuilt + out[f_end:]
    if "?" in out or ":" in out:
        raise TernaryNotSupported("三元未完全展开")
    return out


def normalize_expr(expr: str) -> str:
    """把平台简写表达式还原成 qlib 原生表达式（口径与挖掘时一致）。"""
    # 逻辑运算：平台/LLM 写的是 Python 方言 && / ||，qlib 表达式引擎只认 and / or。
    # 不换会在 parse_field 阶段直接 invalid syntax 崩掉整条（实测
    # Count(($volume<0.5*Mean($volume,20))&&($close>Mean($close,20)),5) 就是这么挂的）。
    result = _rewrite_and_or(expr)
    try:
        result = _rewrite_ternary(result)
    except TernaryNotSupported as exc:
        # 保留三元原样，让 is_qlib_compatible 判定为不可复算并如实跳过，
        # 而不是产出一个语义已变的表达式假装算过。这里必须打日志，
        # 否则静默返回会让因子“凭空消失”，排查时完全看不到线索。
        logger.warning("三元改写失败，因子按不可复算跳过：%s | 原文 %s", exc, expr)
        return result
    result = _rewrite_calls(result)
    for old, new in OP_CONVERSIONS.items():
        result = result.replace(old, new)
    return result


def _strip_csrank(expr: str) -> str:
    """把 __CSRANK__(x) 还原成 x（平衡地去掉外层括号）。

    注意不能简单 `expr.replace("__CSRANK__(", "")`：那只会删掉左半边，
    留下一个多余的 ')'，qlib 直接语法错。这里逐层找配对右括号再摘干净。
    """
    out = expr
    while _CSRANK_MARK in out:
        i = out.index(f"{_CSRANK_MARK}(")
        start = i + len(_CSRANK_MARK) + 1
        depth, j = 1, start
        while j < len(out) and depth:
            if out[j] == "(":
                depth += 1
            elif out[j] == ")":
                depth -= 1
            j += 1
        # j 停在当前 CSRANK 的配对 ')' 之后一位；内容区间是 [start, j-1)
        out = out[:i] + out[start : j - 1] + out[j:]
    return out


def _detect_dt_level(s: pd.Series, default: int = 0) -> int:
    """定位 MultiIndex 里 datetime 在第几层：优先按 index 名，退化取唯一值最少的那层。"""
    if not isinstance(s.index, pd.MultiIndex):
        return default
    names = [str(x).lower() for x in s.index.names]
    if "datetime" in names:
        return names.index("datetime")
    counts = [s.index.get_level_values(i).nunique() for i in range(s.index.nlevels)]
    return int(np.argmin(counts))


def _csrank_postprocess(s: pd.Series, dt_level: int = 0) -> pd.Series:
    """把 __CSRANK__(x) 的占位符替换成真正的日截面排名。

    平台 function_lib.RANK = df.groupby('datetime').rank(pct=True)，
    这里严格照做（pandas 的 rank 默认跳过 NaN，与平台一致）。
    """
    if dt_level >= 0:
        idx = s.index if not isinstance(s.index, pd.MultiIndex) else None
        if isinstance(s.index, pd.MultiIndex) and dt_level < s.index.nlevels:
            return s.groupby(level=dt_level).rank(pct=True)
    return s.rank(pct=True)


def unsupported_operators(expr: str, known: set[str]) -> list[str]:
    """找出表达式里 qlib 引擎不认识的算子名（用于提前跳过，避免整批 D.features 崩掉）。"""
    # __CSRANK__ 只是截面排名的占位符（后面会平衡剥离），不能算未注册算子，
    # 否则所有用裸 RANK(...) 的因子都会被判"不可复算"跳过 —— 池子里就有十几个。
    ops = {m for m in _CALL_RE.findall(expr)} - {"Feature", "PFeature", _CSRANK_MARK}
    return sorted(o for o in ops if o not in known)


# ---------------------------------------------------------------- factor loading
def is_qlib_compatible(expr: str, known: set[str]) -> bool:
    """能否用 qlib 表达式引擎复算：排除三元 ? : 以及 qlib 未注册的算子。

    这两类因子平台实际是走 pandas 因子计算器（见 backtest/factor_calculator.py）算的，
    本脚本用 qlib 引擎，因此跳过并如实记录，不做假结果。
    """
    if "?" in expr or ":" in expr:
        return False
    return not unsupported_operators(expr, known)


def load_top_factors(pool_path: str, k: int, known: set[str]) -> tuple[list[dict], int]:
    """从轨迹池里按 RankIC 取前 k 条轨迹，返回其全部因子（按表达式去重）。"""
    with open(pool_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    per_traj = []
    for tid, t in (data.get("trajectories") or {}).items():
        metrics = t.get("backtest_metrics") or {}
        ric = metrics.get("RankIC")
        if ric is None:
            continue
        per_traj.append((float(ric), tid, t.get("phase"), t.get("factors") or []))

    per_traj.sort(key=lambda x: -x[0])

    out, seen, skipped = [], set(), 0
    for ric, tid, phase, factors in per_traj[:k]:
        for f in factors:
            expr = (f.get("expression") or "").strip()
            if not expr or expr in seen:
                continue
            # 必须先归一化再检查算子，否则 TS_MEAN/ABS/DELAY 这些 rdagent 别名全会被误判成未注册
            norm_expr = normalize_expr(expr)
            if not is_qlib_compatible(norm_expr, known):
                skipped += 1
                logger.warning(
                    "跳过不可复算因子 [%s]：%s",
                    f.get("name"), unsupported_operators(norm_expr, known) or ["ternary ? :"],
                )
                continue
            seen.add(expr)
            out.append(
                {
                    "trajectory_id": tid,
                    "phase": phase,
                    "traj_rank_ic": ric,
                    "name": f.get("name"),
                    "expression": expr,
                }
            )
    return out, skipped


# ---------------------------------------------------------------- metric helpers
def find_levels(df: pd.DataFrame) -> tuple[int, int]:
    """定位 (datetime 层, instrument 层)。

    qlib 的 MultiIndex 层序不保证（可能是 (datetime, instrument) 也可能是 (instrument, datetime)），
    写死 level=0 会把「按个股分组」当成「按日分组」，成交量少时表现为每天只有 1~2 行、RankIC 完全失真。
    因此按 index 名 datetime 定位，退化时取唯一值个数最少的那层（交易日数 << 标的数）。
    """
    names = list(df.index.names)
    dt_level = None
    if "datetime" in names:
        dt_level = names.index("datetime")
    else:
        for i in range(len(names)):
            if df.index.get_level_values(i).nunique() < 1000:
                dt_level = i
                break
    if dt_level is None:
        dt_level = 0
    ins_level = (dt_level + 1) % len(names) if len(names) > 1 else -1
    return dt_level, ins_level


def calc_daily_ic(df: pd.DataFrame, factor_col: str, fwd_col: str, min_cnt: int = 30, dt_level: int = 0) -> pd.Series:
    """按日截面相关（spearman = RankIC；pearson = IC），返回daily 序列。"""
    sub = df[[factor_col, fwd_col]].replace([np.inf, -np.inf], np.nan).dropna()
    res = {}
    for day, g in sub.groupby(level=dt_level):
        if len(g) < min_cnt:
            continue
        c = g.corr(method="spearman")
        val = c.iloc[0, 1] if factor_col != fwd_col else c
        if isinstance(val, pd.DataFrame):
            val = val.iloc[0, 1]
        res[day] = val
    out = pd.Series(res).dropna()
    # qlib 给的日期可能是 str 也可能是 Timestamp，统一成 Timestamp 才能跟配置段边界比较
    out.index = pd.to_datetime(out.index, errors="coerce")
    return out.dropna()


def icir(ic: pd.Series, ann: int = 252) -> float:
    """ICIR = 均值 / 标准差 * sqrt(年化交易日数)。"""
    if ic.empty or ic.std() == 0:
        return float("nan")
    return float(ic.mean() / ic.std() * np.sqrt(ann))


def topk_equity_curve(
    df: pd.DataFrame, factor_col: str, fwd_col: str, topk: int, cost_bps: float,
    dt_level: int = 0, ins_level: int = 1, rebalance: int = 1,
) -> pd.DataFrame:
    """每日截面取因子值最高 topk 只等权持有一期，按日换手扣费。

    rebalance=N 表示每 N 个交易日才重新选一次股，中间沿用上次的持仓。
    这是这套因子能不能落地的关健变量：日频全速换手时换手率常在 70% 以上，
    几 bps 成本就把 RankIC 换来的那点收益吃光，必须验证降频后还剩多少。

    返回 (ret, turnover, cost) 日频序列。
    """
    rows = []
    prev_hold: set[str] | None = None
    cur_hold: set[str] | None = None
    for i, (day, g) in enumerate(
        df[[factor_col, fwd_col]].replace([np.inf, -np.inf], np.nan).dropna().groupby(level=dt_level)
    ):
        ranked = g.sort_values(factor_col, ascending=False)
        # 只在调仓日重选股；非调仓日沿用上次持仓，不产生换手也就不扣费。
        # 注意条件不能写成 `rebalance > 1 and i % rebalance == 0`：
        # rb=1 时 i%1==0 恒真而 rb>1 恒假，会让 rb=1 退化成「只选一次、永久持股」，
        # 换手率变 0、成本全不扣，整套日频基线就废了。
        if cur_hold is None or i % rebalance == 0:
            # Index 没有 .head()，切片即可
            cur_hold = set(ranked.index.get_level_values(ins_level)[:topk])
        hold = cur_hold
        held = ranked[ranked.index.get_level_values(ins_level).isin(hold)]
        fwd_ret = float(held[fwd_col].mean()) if len(held) else float("nan")
        if prev_hold is None:
            turnover = 1.0
        else:
            turnover = 1.0 - len(hold & prev_hold) / max(len(hold), 1)
        cost = turnover * cost_bps / 1e4
        rows.append({"datetime": day, "ret": fwd_ret, "turnover": turnover, "cost": cost})
        prev_hold = hold

    out = pd.DataFrame(rows).set_index("datetime")
    out.index = pd.to_datetime(out.index, errors="coerce")
    out["net_ret"] = out["ret"] - out["cost"]
    return out.dropna(subset=["net_ret"])


def perf_stats(equity: pd.Series, periods_per_year: int = 252) -> dict:
    """净收益序列 -> 年化 / 波动 / 夏普 / 最大回撤。"""
    if equity.empty:
        return {"annualized": float("nan"), "vol": float("nan"), "sharpe": float("nan"), "max_drawdown": float("nan")}
    curve = (1 + equity).cumprod()
    ann = float((1 + equity).mean() ** periods_per_year - 1)
    vol = float(equity.std() * np.sqrt(periods_per_year))
    sharpe = float(equity.mean() / equity.std() * np.sqrt(periods_per_year)) if equity.std() else float("nan")
    mdd = float((curve / curve.cummax() - 1).min())
    return {"annualized": ann, "vol": vol, "sharpe": sharpe, "max_drawdown": mdd}


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="演化因子时序样本外复验")
    ap.add_argument("--pool", required=True, help="轨迹池 json（如 log/latest_trajectory_pool.json）")
    ap.add_argument("--k", type=int, default=3, help="取 RankIC 最高的前 k 条轨迹")
    ap.add_argument("--qlib-dir", default=os.environ.get("QLIB_DATA_DIR", DEFAULT_QLIB_DIR))
    ap.add_argument("--market", default="all", help="qlib instruments")
    ap.add_argument("--train-start", default="2020-01-01")
    ap.add_argument("--train-end", default="2023-12-31")
    ap.add_argument("--valid-start", default="2024-01-01")
    ap.add_argument("--valid-end", default="2024-12-31")
    ap.add_argument("--test-start", default="2025-01-01")
    ap.add_argument("--test-end", default="2026-08-18")
    ap.add_argument("--horizon", type=int, default=1, help="前瞻期（交易日），1=次日收益")
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument(
        "--rebalance", type=int, default=1,
        help="调仓周期（交易日）：1=日频、5=每周、10=每两周。降频是这套因子落地的关键变量",
    )
    ap.add_argument("--cost-bps", type=float, default=5.0, help="单边成本（基点），A股双边常见 5~10")
    ap.add_argument("--out", default="data/results/oos_validate.csv")
    args = ap.parse_args()

    import qlib
    from qlib.data import D
    from qlib.data.ops import Operators

    qlib.init(provider_uri=args.qlib_dir, region="cn")
    known_ops = set(Operators._ops.keys())
    logger.info("qlib 已注册算子 %d 个", len(known_ops))

    # ---- 一次性算全时段因子值 ----
    start = args.train_start
    end = args.test_end

    # 注意：本项目 qlib 被 patch 过，list_instruments 不直接收 market 参数，
    # 必须先 D.instruments(market) 拿到 instruments 对象再传（见 backtest/runner.py:591）。
    codes = D.list_instruments(D.instruments(args.market), start_time=start, end_time=end, as_list=True)
    logger.info("instruments=%d, provider=%s", len(codes), args.qlib_dir)

    factors, skipped = load_top_factors(args.pool, args.k, known_ops)
    if not factors:
        logger.error("池里没有带 RankIC 的轨迹，或字段格式不匹配")
        return 2
    logger.info(
        "载入 %d 个待复验因子（来自 RankIC top-%d 轨迹）%s",
        len(factors), args.k,
        f"，跳过 {skipped} 个含三元算子的因子" if skipped else "",
    )
    exprs = [normalize_expr(f["expression"]) for f in factors]
    names = [f"F{i}_{f['name']}" for i, f in enumerate(factors)]

    # 逐个因子单独试算：qlib 对算子元数要求严格（如 Rank 必须给 N），
    # 混在一次 D.features 里任何一个算子炸掉都会拖垮整批，这里逐个隔离、失败即跳过并如实记录。
    logger.info("计算因子截面（%d 表达式）...", len(exprs))
    # __CSRANK__ 后处理要知道 datetime 在第几层（qlib MultiIndex 层序不保证）
    dt_level_hint = _detect_dt_level(series_list[0]) if series_list else 0

    series_list, ok_names = [], []
    for name, expr, raw_expr in zip(names, exprs, [f["expression"] for f in factors]):
        # 截面排名没法塞进 qlib 表达式（qlib 的 Rank 是时序滚动），
        # 所以把 __CSRANK__(x) 先还原成 x 算值，再按日分组做 rank(pct=True)。
        need_csrank = _CSRANK_MARK in expr
        qlib_expr = _strip_csrank(expr) if need_csrank else expr
        try:
            s = D.features(codes, [qlib_expr], start_time=start, end_time=end, freq="day").iloc[:, 0]
            if need_csrank:
                s = _csrank_postprocess(s, dt_level=dt_level_hint)
            s.name = name
            series_list.append(s)
            ok_names.append(name)
        except Exception as exc:  # noqa: BLE001 - 复算失败要如实记录而不是中断整批
            logger.warning("因子 %s 复算失败（跳过）：%s", name, exc)
    if not series_list:
        logger.error("没有任何因子能用 qlib 引擎复算，终止")
        return 3
    f_df = pd.concat(series_list, axis=1)
    names = ok_names
    logger.info("factor shape=%s（成功复算 %d/%d）", f_df.shape, len(ok_names), len(exprs))

    # ---- 前瞻收益 ----
    # 必须与平台口径一致：configs/backtest.yaml 的 dataset.label 就是
    #   Ref($close, -2) / Ref($close, -1) - 1
    # 语义是「t 日收盘出信号、t+1 才成交」，所以 t 日信号要对应 t+1 -> t+2 的收益。
    # 若退化成 Ref($close,-h)/$close-1（让 t 日信号直接赚 t->t+1 的钱），
    # 实测会把 RankIC 抬高约 60%（同一因子 0.0174 vs 0.0281），且对短周期因子
    # 会放大到符号反转，复验结论完全不可比。
    fwd_expr = f"Ref($close, -{args.horizon + 1}) / Ref($close, -{args.horizon}) - 1"
    fwd = D.features(codes, [fwd_expr], start_time=start, end_time=end, freq="day").iloc[:, 0]
    fwd.name = "fwd"
    logger.info("fwd shape=%s", fwd.shape)

    df = f_df.join(fwd, how="inner")
    dt_level, ins_level = find_levels(df)
    logger.info(
        "joined shape=%s；index levels=%s -> datetime层=%d, instrument层=%d",
        df.shape, [str(x) for x in df.index.names], dt_level, ins_level,
    )

    # ---- 分段统计 ----
    segments = [
        ("train", args.train_start, args.train_end),
        ("valid", args.valid_start, args.valid_end),
        ("test", args.test_start, args.test_end),
    ]

    summary = []
    per_day_ric = {}
    for name in names:
        ric_all = calc_daily_ic(df, name, "fwd", dt_level=dt_level)
        row = {"factor": name, "traj_rank_ic": None}
        logger.info(
            "[diag] %s: n_days=%d mean=%.5f std=%.5f", name, len(ric_all), ric_all.mean(), ric_all.std()
        )
        row["ric_all"] = float(ric_all.mean()) if not ric_all.empty else float("nan")
        row["icir_all"] = icir(ric_all)
        per_day_ric[name] = ric_all
        for seg, s, e in segments:
            # index 是 datetime，配置的段边界是字符串，必须统一成 Timestamp 再切
            ric = ric_all[(ric_all.index >= pd.Timestamp(s)) & (ric_all.index <= pd.Timestamp(e))]
            row[f"ric_{seg}"] = float(ric.mean()) if not ric.empty else float("nan")
            row[f"icir_{seg}"] = icir(ric)
        summary.append(row)
        # 关联回轨迹 RankIC
        idx = int(name.split("_")[0][1:])
        summary[-1]["traj_rank_ic"] = factors[idx]["traj_rank_ic"]
        summary[-1]["phase"] = factors[idx]["phase"]
        summary[-1]["expression"] = factors[idx]["expression"]

    summ_df = pd.DataFrame(summary)

    # ---- TopK 组合 + 扣费敏感性 ----
    top_name = summ_df.sort_values("ric_all", ascending=False)["factor"].iloc[0]
    logger.info("组合基准因子: %s", top_name)
    # gross 必须是「零成本」那一档的年化。
    # 注意 perf_stats 吃的是 net_ret（已扣费），同一序列无论在哪个 bps 下都已经是净收益，
    # 若把 st["annualized"] 同时赋给 gross_ann，毛/净两列会恒等、扣费前后的毛收益对比彻底失真。
    def _seg_stats(eq: pd.DataFrame, seg: str, s: str, e: str) -> dict:
        q = eq[(eq.index >= pd.Timestamp(s)) & (eq.index <= pd.Timestamp(e))]
        return {
            "series": q["net_ret"],
            "days": len(q),
            "turnover": float(q["turnover"].mean()) if not q.empty else float("nan"),
        }

    sens_rows = []
    for rb in [1, 5, 10]:
        # 每个调仓周期都要有自己的零成本基准，否则 gross 会串档
        gross_eq = topk_equity_curve(df, top_name, "fwd", args.topk, 0.0, dt_level, ins_level, rb)
        gross_cache = {seg: _seg_stats(gross_eq, seg, s, e) for seg, s, e in segments}
        for bps in [0.0, 3.0, 5.0, 10.0, 20.0]:
            eq = topk_equity_curve(df, top_name, "fwd", args.topk, bps, dt_level, ins_level, rb)
            for seg, s, e in segments:
                q = _seg_stats(eq, seg, s, e)
                st = perf_stats(q["series"])
                g = gross_cache[seg]
                sens_rows.append(
                    {
                        "rebalance": rb,
                        "cost_bps": bps,
                        "segment": seg,
                        "days": q["days"],
                        "avg_turnover": q["turnover"],
                        "gross_ann": perf_stats(g["series"])["annualized"],
                        "net_ann": st["annualized"],
                        "net_minus_gross": st["annualized"] - perf_stats(g["series"])["annualized"],
                        "sharpe": st["sharpe"],
                        "max_drawdown": st["max_drawdown"],
                    }
                )
    sens_df = pd.DataFrame(sens_rows)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    summ_df.to_csv(args.out, index=False, encoding="utf-8-sig")
    sens_path = str(Path(args.out).with_name(Path(args.out).stem + "_cost.csv"))
    sens_df.to_csv(sens_path, index=False, encoding="utf-8-sig")

    # ---- 打印 ----
    print("\n" + "=" * 96)
    print(f"样本外复验（horizon={args.horizon}, topk={args.topk}, cost={args.cost_bps}bps单边）")
    print("=" * 96)
    show = summ_df[["factor", "phase", "traj_rank_ic", "ric_all", "icir_all", "ric_train", "ric_valid", "ric_test"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n衰减比 test/train:")
    for _, r in summ_df.iterrows():
        if r["ric_train"] and not pd.isna(r["ric_train"]) and r["ric_train"] != 0:
            print(f"  {r['factor']:<45} {r['ric_test']/r['ric_train']:.2f}x")
    print("\n扣费 × 调仓周期敏感性（基准因子 %s）: 净年化" % top_name)
    order = [0.0, 3.0, 5.0, 10.0, 20.0]
    for rb in [1, 5, 10]:
        sub = sens_df[sens_df.rebalance == rb]
        print(f"\n  调仓周期 = 每 {rb} 个交易日（turnover 均值 %.3f）"
              % sub[sub.segment == "train"]["avg_turnover"].mean())
        print(sub.pivot_table(index="cost_bps", columns="segment", values="net_ann")
              .reindex(order)
              .to_string(float_format=lambda v: f"{v:.4f}"))
    print("\n成本吃掉的比例 (1 - net/gross):")
    for rb in [1, 5, 10]:
        sub = sens_df[sens_df.rebalance == rb]
        ratio = pd.DataFrame(
            {
                seg: 1 - sub[(sub.segment == seg)].set_index("cost_bps")["net_ann"]
                / sub[(sub.segment == seg)].set_index("cost_bps")["gross_ann"]
                for seg in ("train", "valid", "test")
            }
        ).reindex(order)
        print(f"\n  调仓周期 = 每 {rb} 个交易日")
        print(ratio.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\n结果已写：{args.out}\n           {sens_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
