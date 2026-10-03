"""归一化层自测：只验证表达式改写，不碰 qlib / pandas（秒级返回）。

用法：python tools/_norm_selftest.py

注意：oos_validate.py 在模块顶层 `import numpy as pd/pandas as pd`，而这台机器上
导入 pandas 单次要数分钟（沙箱对每次文件读取都写审计日志）。但归一化层
（_rewrite_calls / _rewrite_ternary / normalize_expr）其实不依赖 pandas
（`from __future__ import annotations` 让类型注解只是字符串），
所以这里把源码截到 numpy 导入之前 exec，绕开那个慢导入。

bash 里 $ 会被展开，这里用 Xclose/Xreturn 这类占位字段名（归一化不解析字段名，
用 X 前缀同样能验证括号配对 / 参数映射 / 算子改写是否正确）。
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "oos_validate.py")

with open(SRC, "r", encoding="utf-8") as f:
    src = f.read()

# 把 numpy / pandas 换成空壳塞进 sys.modules，让模块顶层 import 直接命中缓存，
# 既绕开慢导入，又拿到完整的归一化函数（它们在 numpy 导入之后才定义）。
import types  # noqa: E402

for _m in ("numpy", "pandas"):
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)

ns: dict = {}
exec(compile(src, SRC, "exec"), ns)  # noqa: S102 - 自测脚本，源码为本文件同目录下的工具模块
normalize_expr = ns["normalize_expr"]

TESTS = [
    # 1) 两级以上嵌套的裸 Zscore（本轮从未被复验过的最优因子）
    "ZSCORE(TS_ZSCORE(TS_STD(Xreturn, 20) - TS_STD(Xreturn, 60), 60) * TS_SKEW(Xreturn, 20))",
    # 2) 三元
    "(Xclose < 0.9 * TS_MAX(Xclose, 20)) ? "
    "((TS_MAX(Xclose, 20) - Xclose) / (TS_MAX(Xclose, 20) + 1e-8)) : 0",
    # 3) 裸 RANK -> 截面排名占位符
    "RANK( (Xclose / DELAY(Xclose, 5)) * (Xvolume / (TS_MEAN(Xvolume, 20) + 1e-8)) )",
    # 4) 裸 RANK 套 COUNT + 逻辑运算 &&  + 三元
    "RANK(COUNT((Xvolume < 0.5 * TS_MEAN(Xvolume, 20)) && "
    "(Xclose > TS_MEAN(Xclose, 20)), 5) * ((Xclose < 0.9 * TS_MAX(Xclose, 20)) ? "
    "((TS_MAX(Xclose, 20) - Xclose) / (TS_MAX(Xclose, 20) + 1e-8)) : 0))",
    # 5) TS_QUANTILE 概率在前（需要交换）
    "TS_SUM(ABS(Xreturn), 5) * SIGN(TS_STD(ABS(Xreturn), 5) - TS_QUANTILE(ABS(Xreturn), 60, 0.75))",
    # 6) TS_QUANTILE 窗口在前（不需要交换）
    "RANK(TS_QUANTILE(Xclose, 20, 0.2) / (TS_MEAN(Xvolume, 5) / (TS_STD(Xvolume, 15) + 1e-8)))",
    # 7) TS_RANK 时序 -> qlib Rank(x, N)
    "TS_RANK(ABS(TS_STD(Xclose, 20)), 60) - TS_RANK(Xclose / DELAY(Xclose, 5), 60)",
    # 8) TS_DELTA + 嵌套 RANK + TS_PCTCHANGE
    "-1 * TS_CORR(RANK(DELTA(Xclose, 5) / DELAY(Xclose, 5)), "
    "RANK(TS_STD(Xreturn, 10)), 5)",
    # 9) 回归：普通 TS_ZSCORE
    "TS_ZSCORE(Xclose, 10) - TS_ZSCORE(Xvolume, 10)",
    # 10) 回归：TS_PCTCHANGE + TS_ZSCORE
    "TS_PCTCHANGE(Xclose, 3) * TS_ZSCORE(Xhigh - Xlow, 20)",
    # 11) MAX(0, x) 常量在首位（平台是 ReLU，qlib 的 Max 会炸在 int.get_extended_window_size）
    "SIGN(TS_SUM(Xreturn, 5) / (TS_STD(Xreturn, 5) + 1e-8)) * MAX(0, "
    "Xvolume / TS_MEAN(Xvolume, 20) - 1.2)",
    # 12) 双特征逐元素 MIN（平台是 MIN(close,open)，qlib 的 Min 同样会是滚动语义）
    "MIN(Xclose, Xopen) - Xlow / (Xhigh - Xlow + 1e-8) * "
    "((MIN(Xclose, Xopen) - Xlow) / (Xhigh - Xlow + 1e-8) > 0.4 && "
    "Xclose >= (Xhigh + Xlow) / 2)",
    # 13) 回归：MAX 常量 + 内层 TS_MAX 滚动语义不能混
    "MAX(0, TS_MAX(Xclose, 20) - Xclose)",
    # 14) 单参 ZSCORE 是横截面标准化（factor_calculator 的 OPERATIONS_DOC 明写）
    "ZSCORE(TS_ZSCORE(TS_STD(Xreturn, 20) - TS_STD(Xreturn, 60), 60) * TS_SKEW(Xreturn, 20))",
    # 15) 回归：双参仍是时序滚动 z-score，不能被 14) 的截面分支吃掉
    "ZSCORE(Xclose, 20)",
    # 16) $return 必须整块保留，再由 OP_CONVERSIONS 展开成真实字段（见下方断言）
    "TS_ZSCORE($return, 10) - TS_ZSCORE($volume, 10)",
    # 17) $return 嵌套在其他算子里同样不能被驼峰化
    "TS_SUM(ABS($return), 5) / (TS_STD($return, 5) + 1e-8)",
]

# 截面排名占位符的剥离（去掉外层括号，必须平衡）
_strip = ns["_strip_csrank"]
for raw in (
    "__CSRANK__(Max(Xclose, 20) - Xclose)",
    "Corr(__CSRANK__(A), __CSRANK__(B), 5)",
    "__CSRANK__(Corr(__CSRANK__(A), __CSRANK__(B), 5))",
):
    stripped = _strip(raw)
    balanced = stripped.count("(") == stripped.count(")")
    assert balanced, f"括号不平衡: {raw} -> {stripped}"
    assert "__CSRANK__" not in stripped, f"占位符没剥干净: {raw} -> {stripped}"

# 截面 z-score 占位符的剥离同样必须平衡、剥净
_strip_z = ns["_strip_csz"]
for raw in ("__CSZ__(Std(Xreturn, 20) - Std(Xreturn, 60))",
            "Abs(__CSZ__(Xclose)) + 1"):
    z = _strip_z(raw)
    assert z.count("(") == z.count(")"), f"括号不平衡: {raw} -> {z}"
    assert "__CSZ__" not in z, f"占位符没剥干净: {raw} -> {z}"
lines = ["# _strip_csrank / _strip_csz 平衡性/剥净测试通过"]
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
for idx, t in enumerate(TESTS):
    print(f"[progress] test {idx + 1}/{len(TESTS)} ...", flush=True)
    try:
        out = normalize_expr(t)
    except Exception as exc:  # noqa: BLE001
        out = f"!! {type(exc).__name__}: {exc}"
    lines.append("IN : " + t)
    lines.append("OUT: " + out)
    lines.append("-" * 70)
    print("OUT: " + out[:160], flush=True)

# 关键断言：$return 会被 OP_CONVERSIONS 展开成真实字段，
# 若被驼峰化成 $Return，qlib 没有这个字段 → 整列 NaN 且不抛异常（实测坑过 4 条因子）。
for t in ("TS_ZSCORE($return, 10) - TS_ZSCORE($volume, 10)",
          "TS_SUM(ABS($return), 5) / (TS_STD($return, 5) + 1e-8)"):
    o = normalize_expr(t)
    assert "$Return" not in o, f"$return 被驼峰化成 $Return 了（qlib 无此字段，整列 NaN）: {t} -> {o}"
    assert "$close" in o, f"$return 没被展开成真实字段: {t} -> {o}"

out_path = os.path.join(os.path.dirname(HERE), "data", "results", "norm_test.txt")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
print("\n".join(lines))
print("\n[OK] ->", out_path)
