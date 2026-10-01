"""Auto-applied because Invoke-RDAgent puts tools/ on PYTHONPATH.

Must no-op if rdagent is not installed (e.g. conda env rdagent4qlib inheriting PYTHONPATH).
"""

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
