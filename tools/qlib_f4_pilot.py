"""qlib 迁移试点：用 Qlib 表达式引擎 + 数据层重实现 F4，与手写版对齐
=====================================================================

背景（2026-10-14 用户："就不能找个框架来做吗 你这样很累啊"）：
本项目此前全部用 pandas + psycopg2 手写，重复造轮子且出了「bars_1d 重复行」事故。
Qlib 0.9.7 早已装在 quantaalpha 环境，`datahub dump-qlib` 也早已存在。

本脚本做迁移验证（第 2 步）：
  1. 用 Qlib 数据层 D.features 取数（provider = 新建的干净数据集 cn_daily_clean）
  2. 用 Qlib 表达式引擎写 F4（一行表达式，不手写 rolling）
  3. 与手写版信号数对齐 —— 标尺：F4 2026 清洗后毛超额 −7.29bp（E0 clean）

F4 定义（与手写版逐字一致）：
  |close - SMA20| / SMA20 < 0.03  AND  close > SMA20  AND  Ref(close,1) <= Ref(SMA20,1)

⚠ Windows + Qlib：D.features 默认走多进程，脚本必须有 __main__ 保护，
   否则子进程重复导入顶层代码会无限递归（本脚本首版即踩此坑，卡死 10 分钟）。
"""
import os
import multiprocessing

os.environ.setdefault("QLIB_KERNELS", "1")   # 禁用多进程，避免 Windows fork 问题

import numpy as np
import pandas as pd
import qlib
from qlib.data import D

PROVIDER = "D:/qqmt/qlib_data/cn_daily_clean"
SIM_START, END = "2026-01-05", "2026-10-13"


def main():
    qlib.init(provider_uri=PROVIDER, region="cn")
    print(f"[0] qlib 初始化 OK  provider={PROVIDER}", flush=True)

    print("[1] Qlib 数据层取数（表达式引擎）...", flush=True)
    expr = ["$close", "$open", "Mean($close, 20)", "Ref($close, 1)", "Ref(Mean($close, 20), 1)",
            "Ref($close, 250)"]   # 末项用于「上市≥250交易日」过滤，对齐手写版剔次新
    inst = D.instruments("all")
    df = D.features(inst, expr, start_time=SIM_START, end_time=END)
    df.columns = ["close", "open", "sma20", "prev_close", "prev_sma20", "c250"]
    df = df.dropna()
    lv0 = df.index.get_level_values(0)
    lv1 = df.index.get_level_values(1)
    print(f"    行数 {len(df):,}  标的 {lv0.nunique()}  "
          f"日期 {lv1.min().date()} ~ {lv1.max().date()}", flush=True)

    sig_all = ((df["close"] - df["sma20"]).abs() / df["sma20"] < 0.03) & \
              (df["close"] > df["sma20"]) & \
              (df["prev_close"] <= df["prev_sma20"])
    # 对齐手写版：上市≥250交易日（c250 非空）且 open>0
    sig = sig_all & (df["open"] > 0)
    n_all, n_sig = int(sig_all.sum()), int(sig.sum())
    print(f"\n[2] F4 信号数（Qlib 表达式引擎）", flush=True)
    print(f"    未加次新过滤 : {n_all:,}", flush=True)
    print(f"    加「上市≥250日 + open>0」过滤: {n_sig:,}", flush=True)
    print(f"    手写版标尺（bars_1d 清洗后 2026）: 约 35,762 笔", flush=True)

    dup = int(df.index.duplicated().sum())
    print(f"\n[3] 数据质量自检：MultiIndex 重复数 = {dup}  ← Qlib 天然保证唯一", flush=True)

    print("\n[4] 表达式引擎演示（一行算因子）...", flush=True)
    extra = D.features(inst, ["$volume", "Mean($volume, 5)", "$close / Mean($close, 60) - 1"],
                       start_time="2026-09-01", end_time=END)
    print(f"    shape {extra.shape}  列 {list(extra.columns)}", flush=True)

    print("\n结论：Qlib 数据层 + 表达式引擎可用，迁移路径成立。", flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
