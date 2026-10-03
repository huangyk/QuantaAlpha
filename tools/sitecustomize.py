"""Auto-applied because Invoke-RDAgent puts tools/ on PYTHONPATH.

Must no-op if rdagent is not installed (e.g. conda env rdagent4qlib inheriting PYTHONPATH).

2026-10-03：顺手压掉一类噪声日志。各 patch 模块在被 import 时会
`print("[xxx-patch] applied", file=sys.stderr)`，而长任务（mine / 回测）里这些模块会被
反复加载 —— 实测一次 mine 的日志共 2394 行，其中 **1205 行** 是这种重复提示，把真正
有用的 direction / hypothesis / RankIC 进度行全挤没了。下面在 import 阶段给 stderr 套一层
按 tag 去重的过滤器，每个 patch 只留第一条。
"""

import sys


class _OnceStderr:
    """按 tag 去重地放行 stderr 的包装：同一个 patch 标签只放行第一次。"""

    __slots__ = ("_stream", "_seen")

    _KNOWN = (
        "chat-retry", "cpu-throttle", "factor-datetime", "factor-empty",
        "feedback-metrics", "final-decision", "gym-numpy", "json-parse",
        "parquet-read", "qlib-data-profile", "research-prompt", "safe-ic", "ui-metrics",
    )

    def __init__(self, stream):
        self._stream = stream
        self._seen = set()

    def write(self, s):
        if "-patch]" in s:
            tag = s.split("-patch]")[0].strip().strip("[]").strip()
            if tag in self._KNOWN:
                if tag in self._seen:
                    return len(s)  # 吞掉重复项
                self._seen.add(tag)
        # 其余内容（业务日志、traceback、warning）原样透传到真 stderr，不做任何改写
        return self._stream.write(s)

    def flush(self):
        fl = getattr(self._stream, "flush", None)
        if fl:
            fl()

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self._stream, name)


_real_stderr = sys.stderr
sys.stderr = _OnceStderr(_real_stderr)

try:
    from chat_retry_patch import apply as apply_chat_retry_patch
    from cpu_throttle_patch import apply as apply_cpu_throttle_patch
    from factor_datetime_patch import apply as apply_datetime_patch
    from factor_empty_patch import apply as apply_factor_empty_patch
    from feedback_metrics_patch import apply as apply_feedback_metrics_patch
    from final_decision_patch import apply as apply_final_decision_patch
    from gym_numpy_patch import apply as apply_gym_numpy_patch
    from json_parse_patch import apply as apply_json_patch
    from parquet_read_patch import apply as apply_parquet_read_patch
    from qlib_data_profile_patch import apply as apply_qlib_data_profile_patch
    from research_prompt_patch import apply as apply_research_prompt_patch
    from safe_ic_patch import apply as apply_safe_ic_patch
    from ui_metrics_patch import apply as apply_ui_metrics_patch
    from windows_qlib_patch import apply as apply_windows_qlib_patch

    # Thread caps must run before numpy/OpenMP imports in later patches.
    apply_cpu_throttle_patch()
    apply_json_patch()
    apply_datetime_patch()
    apply_parquet_read_patch()
    apply_qlib_data_profile_patch()
    apply_windows_qlib_patch()
    apply_gym_numpy_patch()
    apply_safe_ic_patch()
    apply_final_decision_patch()
    apply_chat_retry_patch()
    apply_factor_empty_patch()
    apply_ui_metrics_patch()
    apply_feedback_metrics_patch()
    apply_research_prompt_patch()
except Exception:
    pass
# 注意：这里 **故意不还原** stderr。第一版在 finally 里 `sys.stderr = _real_stderr` 立即还原，
# 结果包装器活不到业务代码写日志的时候（实测 mining 日志里 patch 噪声一行没少，仍 1205 行重复）。
# 包装器只吞掉已知 patch 的重复提示，其余内容全部转发，等价于真 stderr，不会污染业务输出。
# 保留模块级引用防止被 GC，且不会覆写后来框架自己设置的 sys.stderr。
# 保留实例引用防止被 GC 回收
_STDERR_FILTER = sys.stderr
