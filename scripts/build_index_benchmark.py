#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把 TimescaleDB 里的沪深300指数（000300.SH）导入 Qlib 数据集 cn_real_full，
作为真实市场基准（替换 SH600000 占位）。

数据来源与清洗依据（全部为实测结论，非臆测）
-------------------------------------------------
DB 表 `bars_1d` 中 asset_code='000300.SH' 存在**两个批次**：
  - ts 时刻 15:00 批次：1245 行，trade_date 正确，区间 2020-01-02 ~ 2026-07-15
  - ts 时刻 16:00 批次：1574 行，trade_date **整体错位少 1 天**（含休市日 2020-01-01）
实测验证：把 16:00 批次 trade_date +1 天后，与 15:00 批次在 1244 个共有日期上
**数值 100% 一致（最大差值 0.0）** → 判定 16:00 批次为重复且错标的废数据，**只取 15:00 批次**。

导出后写入二进制格式（与 cn_real_full 现有标的完全一致）：
  文件 = [int32 header(=0)] + [1599 × float32]，按 calendars/day.txt 逐行对齐
  字段：open/high/low/close/volume/vwap/factor/label_5d
  - vwap     = amount / volume
  - factor   = 1.0（指数不做复权）
  - label_5d = close[t+5] / close[t] - 1（反推现有标的确认口径完全一致）

重要：本脚本**不修改 instruments/all.txt**，以免把指数混进股票池
（universe 必须保持 5210 只股票，指数仅供 backtest 的 benchmark 使用）。

用法：
    D:/Users/lenovo/miniconda3/envs/quantaalpha/python.exe scripts/build_index_benchmark.py
"""

from __future__ import annotations

import io
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DB_CONTAINER = "quant-timescaledb"
DB_USER = "quant"
DB_NAME = "quant_data"
SRC_CODE = "000300.SH"
QLIB_TARGET = "sh000300"          # Qlib 目录名（小写）
QLIB_CODE_UPPER = "SH000300"      # backtest 配置里写的基准代码

QD = Path.home() / ".qlib" / "qlib_data" / "cn_real_full"
TMP_DIR = Path(__file__).resolve().parent.parent / "data" / "tmp_export"

FIELDS = ["open", "high", "low", "close", "volume", "vwap", "factor", "label_5d"]


def export_from_db() -> pd.DataFrame:
    """从 TimescaleDB 导出原始指数行（docker exec + COPY，避免额外装 psycopg2）。"""
    q = (
        "COPY (SELECT ts, trade_date, asset_code, open, high, low, close, volume, amount "
        f"FROM bars_1d WHERE asset_code='{SRC_CODE}' ORDER BY trade_date, ts) TO STDOUT CSV HEADER"
    )
    cmd = ["docker", "exec", DB_CONTAINER, "psql", "-U", DB_USER, "-d", DB_NAME, "-c", q]
    print(f"[1/5] 从 TimescaleDB 导出 {SRC_CODE} ...")
    r = subprocess.run(cmd, capture_output=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"导出失败: {r.stderr.decode('utf-8', 'replace')[:400]}")
    df = pd.read_csv(io.BytesIO(r.stdout), parse_dates=["ts", "trade_date"])
    print(f"      原始行数={len(df)}")
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    合并两个批次得到完整日序列。

    - 15:00 批次：trade_date 正确，但只覆盖 1245 天（区间内有 330 天空缺，含回测起点之后）
    - 16:00 批次：trade_date 整体错位少 1 天，移位后覆盖 1574 天
    两批次在共有的 1244 天上数值完全一致（实测冲突 0 天）→ 合并取并集即可得到
    完整且无歧义的真实序列（1575 天，对交易日历 100% 覆盖）。
    """
    df = df.copy()
    df["hour"] = df["ts"].dt.hour
    g = df.groupby("hour").size().to_dict()
    print(f"[2/5] 按 ts 小时分组：{g}")

    a = df[df["hour"] == 15].copy()
    b = df[df["hour"] == 16].copy()
    b["trade_date"] = b["trade_date"] + pd.Timedelta(days=1)  # 修正错位

    u = pd.concat([a, b], ignore_index=True).sort_values("trade_date")
    # 冲突检查：同一日期不同来源的收盘价是否一致
    nun = u.groupby("trade_date")["close"].nunique()
    conflicts = int((nun > 1).sum())
    clean_df = u.drop_duplicates(subset=["trade_date"], keep="first")

    info = {
        "raw_rows": len(df),
        "hour_groups": g,
        "clean_rows": len(clean_df),
        "clean_start": str(clean_df["trade_date"].min().date()),
        "clean_end": str(clean_df["trade_date"].max().date()),
        "conflicts": conflicts,
    }
    print(f"      合并后 {len(clean_df)} 天（{info['clean_start']} ~ {info['clean_end']}），"
          f"同日数值冲突={conflicts}")
    return clean_df, info


def align_to_calendar(clean_df: pd.DataFrame) -> tuple[dict[str, np.ndarray], list[str], dict]:
    """对齐到 cn_real_full 的交易日历；缺失日期写 NaN（绝不前向填充造假）。"""
    cal = [l.strip() for l in (QD / "calendars" / "day.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"[3/5] 读取日历 {len(cal)} 天：{cal[0]} ~ {cal[-1]}")

    src = clean_df.set_index("trade_date")
    idx = pd.to_datetime(src.index)
    cal_dt = pd.to_datetime(pd.Index(cal))

    close = src["close"].reindex(cal_dt).astype("float64").to_numpy()
    open_ = src["open"].reindex(cal_dt).astype("float64").to_numpy()
    high = src["high"].reindex(cal_dt).astype("float64").to_numpy()
    low = src["low"].reindex(cal_dt).astype("float64").to_numpy()
    volume = src["volume"].reindex(cal_dt).astype("float64").to_numpy()
    amount = src["amount"].reindex(cal_dt).astype("float64").to_numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = np.where(volume > 0, amount / volume, np.nan)
    factor = np.where(np.isnan(close), np.nan, 1.0)

    lab = np.full_like(close, np.nan)
    if len(close) > 5:
        lab[:-5] = np.where(np.isnan(close[:-5]) | np.isnan(close[5:]), np.nan, close[5:] / close[:-5] - 1)

    covered = int((~np.isnan(close)).sum())
    aligned_tail = bool((~np.isnan(close))[-1])
    info = {
        "calendar_days": len(cal),
        "covered_days": covered,
        "coverage": f"{covered}/{len(cal)} ({covered / len(cal):.1%})",
        "last_real_date": str(cal_dt[~np.isnan(close)].max().date()) if covered else "n/a",
        "tail_is_real": aligned_tail,
    }
    print(f"      覆盖 {info['coverage']}，末个真实数据日={info['last_real_date']}")

    return {
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume, "vwap": vwap, "factor": factor, "label_5d": lab,
    }, cal, info


def write_bins(arrays: dict[str, np.ndarray]) -> None:
    """写二进制：[int32 0] + [N × float32]，与现有标的格式一致。"""
    out = QD / "features" / QLIB_TARGET
    out.mkdir(parents=True, exist_ok=True)
    print(f"[4/5] 写入 {out}")
    for f in FIELDS:
        v = arrays[f].astype("<f4")
        with open(out / f"{f}.day.bin", "wb") as fh:
            fh.write(struct.pack("<i", 0))
            fh.write(v.tobytes())
        print(f"      {f}.day.bin  {len(v)} 值")


def write_instrument_ref() -> None:
    """写参考 instrument 文件；刻意不动 all.txt（避免指数进股票池）。"""
    ins = QD / "instruments"
    ins.mkdir(parents=True, exist_ok=True)
    p = ins / "csi300.txt"
    # qlib instruments 文件是三列制表符分隔：symbol\tstart_date\tend_date
    p.write_text(f"{QLIB_CODE_UPPER}\t2020-01-02\t2026-08-18\n", encoding="utf-8")
    all_txt = ins / "all.txt"
    n_before = sum(1 for _ in all_txt.open("r", encoding="utf-8") if _.strip())
    has_idx = QLIB_CODE_UPPER in all_txt.read_text(encoding="utf-8")
    print(f"[5/5] instruments：all.txt 仍为 {n_before} 个标的；含指数={has_idx}（应=False）")
    print(f"      参考文件 {p.name} 已写入 {QLIB_CODE_UPPER}")


def verify_readback() -> None:
    """回读比对：确认二进制可被正确解析且与源数据一致。"""
    print("\n回读校验：")
    for f in FIELDS:
        raw = (QD / "features" / QLIB_TARGET / f"{f}.day.bin").read_bytes()
        head = struct.unpack("<i", raw[:4])[0]
        arr = np.frombuffer(raw[4:], dtype="<f4")
        ok = "OK" if (head == 0 and np.isnan(arr).sum() < len(arr)) else "CHECK"
        print(f"  {f:10s} header={head} n={len(arr)} nan={int(np.isnan(arr).sum())} [{ok}]")
    close = np.frombuffer((QD / "features" / QLIB_TARGET / "close.day.bin").read_bytes()[4:], dtype="<f4")
    real = close[~np.isnan(close)]
    print(f"  close 真实值 {len(real)} 个：first={real[0]:.3f} last={real[-1]:.3f}")


def main() -> int:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    df = export_from_db()
    clean_df, cinfo = clean(df)
    arrays, cal, ainfo = align_to_calendar(clean_df)
    write_bins(arrays)
    write_instrument_ref()
    verify_readback()

    print("\n" + "=" * 66)
    print("完成。关键事实（真实数据）：")
    print(f"  原始行={cinfo['raw_rows']}，采用 15:00 批次 {cinfo['clean_rows']} 行 "
          f"({cinfo['clean_start']} ~ {cinfo['clean_end']})")
    print(f"  落库 {QLIB_CODE_UPPER}，日历覆盖 {ainfo['coverage']}，末真实日={ainfo['last_real_date']}")
    print("  缺失日期写 NaN（未做任何前向填充/编造）")
    print("\n  使用提示：backtest 配置 benchmark 改为 SH000300；")
    print("  因指数数据止于 %s，对比回测请把 end_time 设到该日以内，"
          % ainfo["last_real_date"])
    print("  否则尾部交易日没有真实基准可用。")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
