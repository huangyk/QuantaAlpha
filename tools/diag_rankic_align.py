"""平台 RankIC 与复验器口径对齐诊断（一次运行完成全部举证）。

背景 / 为什么要做这个
----------------------
平台回测 `backtest_warfare2_multiyear` 报 **Rank_IC = 0.0689**（`data/results/backtest_bench_csi300/
batch_summary.json`），而 `tools/oos_validate.py` 对同一批因子只测出 **0.017~0.028** 的单因子
RankIC，差 2~4 倍，两边无法互相对照。本脚本把差异「钉死」，不再靠猜：

  1) 复现平台的取数环境：QLIB_DATA_DIR > QLIB_PROVIDER_URI > config provider_uri
     （`quantaalpha/backtest/runner.py` 里的取值优先级）
  2) 扫描 `data/results/` 下所有 `sig_analysis/ric.pkl`，算各自 mean，**定位 0.0689 那一次回测
     属于哪个 run 目录**
  3) 读该 run 的 `artifacts/pred.pkl` —— 这是 **LightGBM 对 N 个因子的预测合成分**
     （`runner.py:554` `pred = model.predict(dataset)`），**不是单因子值**。这是两边量纲不同的
     首要嫌疑：平台衡量的是「多因子模型集成」的信号质量，复验器衡量的是「单因子」信号质量。
  4) 用 `configs/backtest.yaml` 的 `dataset.label` 重算 label，对 **pred** 求日 spearman，
     得到「平台口径 RankIC」
  5) 与平台 `ric.pkl` 逐日对照（相关系数 / 均值差），确认能否复现 0.0689
  6) 顺带报告：市场池实际股票数、qlib 实际 provider 路径、label 表达式

输出
----
  data/results/rankic_align.csv   pred 日截面 rank IC 与平台 ric.pkl 的逐日对照
  （同时把口径维度摘要打到 stdout）

用法
----
  python tools/diag_rankic_align.py
  python tools/diag_rankic_align.py --target-rankic 0.0689
  python tools/diag_rankic_align.py --pred <某个 pred.pkl 路径>   # 跳过扫描，直接对齐
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging_cfg = None  # 保持精简，抑制无谓输出
logger = print

REPO = Path(__file__).resolve().parent.parent
DEFAULT_QLIB_DIR = "C:/Users/lenovo/.qlib/qlib_data/cn_real_full"
RESULTS_ROOT = REPO / "data" / "results"
# 与 configs/backtest.yaml dataset.label 保持一致（runner.py 直接取该字段）
LABEL_EXPR = "Ref($close, -2) / Ref($close, -1) - 1"
PROVIDER_FROM_YAML = "~/.qlib/qlib_data/cn_data"  # 见 configs/backtest.yaml:44


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def resolve_provider() -> str:
    """复刻 runner.py 的 provider 取值优先级：env QLIB_DATA_DIR > QLIB_PROVIDER_URI > config。"""
    import os

    env_dir = os.environ.get("QLIB_DATA_DIR")
    if env_dir:
        return f"env:QLIB_DATA_DIR={env_dir}"
    env_uri = os.environ.get("QLIB_PROVIDER_URI")
    if env_uri:
        return f"env:QLIB_PROVIDER_URI={env_uri}"
    return f"yaml:provider_uri={PROVIDER_FROM_YAML}"


def to_daily_panel(df: pd.DataFrame) -> pd.DataFrame:
    """把 D.features / pred 的返回统一成「行=datetime, 列=instrument」的日截面。

    坑（本环境实测）：`D.features` 返回的 MultiIndex 层序是 ['instrument', 'datetime']
    而不是 qlib 官方常见的 ['datetime', 'instrument']，个股在 level 0，
    直接按日期 groupby 会全部落在一个桶里 -> rank IC 恒 NaN。
    """
    p = df.copy()
    if isinstance(p.index, pd.MultiIndex):
        names = [str(x).lower() for x in p.index.names]
        # 坑（本环境实测，两条层序都真实存在，别写死）：
        #   D.features  返回 ['instrument', 'datetime']（个股在 level 0）
        #   SignalRecord 存出的 pred.pkl 返回 ['datetime', 'instrument']（日期在 level 0）
        # 统一规则：**只保证 datetime 落在 level 0**，多出来的 instrument 层 dropp掉。
        if "datetime" in names and names.index("datetime") != 0:
            p.index = p.index.swaplevel(names.index("datetime"), 0)
            p = p.sort_index()
        if "instrument" in names:
            # 坑：上面的 swap 之后如果只 droplevel("instrument")，标的整层会被扔掉，
            # 面板塌成「206 万行 × 1 列」的假截面（实测 rows 保持 2069937 首末日期没变）。
            # 必须 unstack 把 instrument 透到列上，才是真正的「行=日期 列=标的」日截面。
            p = p.unstack("instrument")
            if isinstance(p.columns, pd.MultiIndex):
                p.columns = p.columns.droplevel(0)
            p = p.sort_index()
    p.index = pd.to_datetime(p.index)
    return p.sort_index().astype(float)


def _long_series(df, name: str) -> pd.Series:
    """把「datetime × instrument」宽面板 stack 成 (datetime, instrument) 长表。

    坑（自己踩的）：先用 .iloc[:,0] 取「某一只票的时间序列」再 concat，得到的是
    单只票的历史序列，接着 groupby(date) 每天只剩 1 行，spearman 算不出来 ->
    日 rank IC 全是 nan（实测 inner 后只有 482 行、n_days=0）。
    必须整张面板 stack，groupby(date) 之后每组才是真正的日截面（~4300 行）。
    另外两个面板列名都是 instrument 代码（几乎完全相同），按列名取也会串。
    """
    s = df.stack() if isinstance(df, pd.DataFrame) else df
    return pd.Series(np.asarray(s, dtype=float), index=s.index, name=name)


def daily_rank_ic(pred_daily: pd.DataFrame, lab_daily: pd.DataFrame) -> pd.Series:
    """pred / label 两个「datetime × instrument」单列表 -> 日截面 spearman 序列。"""
    print(f"       [diag] pred 面板={pred_daily.shape}  label 面板={lab_daily.shape}")
    ps = _long_series(pred_daily, "pred")
    ls = _long_series(lab_daily, "label")
    df = pd.concat([ps, ls], axis=1, join="inner").dropna()
    print(f"       [diag] stack 后 inner 行数={len(df)}（每天每组应有 ~4300 行才对）")
    if df.empty:
        return pd.Series(dtype=float, name="rank_ic")
    df.columns = ["pred", "label"]
    out = {}
    for d, g in df.groupby(level=0):
        a = g["pred"].to_numpy(dtype=float)
        b = g["label"].to_numpy(dtype=float)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 3:
            out[d] = np.nan
            continue
        ra = pd.Series(a[ok]).rank().to_numpy()
        rb = pd.Series(b[ok]).rank().to_numpy()
        ra = ra - ra.mean()
        rb = rb - rb.mean()
        denom = np.sqrt((ra**2).sum() * (rb**2).sum())
        out[d] = float((ra * rb).sum() / denom) if denom > 0 else np.nan
    return pd.Series(out, name="rank_ic")


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def scan_runs(target: float, top: int = 15) -> pd.DataFrame:
    """扫描所有名字含 ric 的 pkl（不限目录深度），返回按 mean 排序的表。

    坑：runner.py 只从 `<exp>/mlruns/<runid>/<runid>/artifacts/sig_analysis/ric.pkl` 读，
    但历轮 workspace 目录名不统一（workspace_exp_* / workspace / ...），
    只 rglob("sig_analysis/ric.pkl") 会漏掉一部分 run（实测漏过 0.0689 那次）。
    """
    rows = []
    for ric_path in sorted(p for p in RESULTS_ROOT.rglob("*ric*.pkl") if p.is_file()):
        try:
            ser = pd.read_pickle(ric_path)
            vals = np.asarray(ser, dtype=float)
            mean = float(np.nanmean(vals)) if len(vals) else float("nan")
            rows.append({"mean_rankic": mean, "n_days": len(ser), "ric_path": str(ric_path)})
        except Exception as exc:  # 坏 pickle 不能中断整体扫描
            rows.append({"mean_rankic": np.nan, "n_days": -1, "ric_path": str(ric_path), "err": str(exc)[:80]})
    df = pd.DataFrame(rows).sort_values("mean_rankic", ascending=False).head(top)
    return df.reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-rankic", type=float, default=0.0689, help="要对齐的目标平台 RankIC")
    ap.add_argument("--pred", type=str, default=None, help="直接指定 pred.pkl 路径，跳过扫描")
    ap.add_argument("--ric", type=str, default=None, help="直接指定 ric.pkl 路径")
    ap.add_argument("--market", type=str, default=None, help="覆盖 market（默认读 configs/backtest.yaml 的 csi300）")
    ap.add_argument("--start", type=str, default="2016-01-01")
    ap.add_argument("--end", type=str, default="2025-12-26")
    ap.add_argument("--out", type=str, default=str(RESULTS_ROOT / "rankic_align.csv"))
    args = ap.parse_args()

    print("=" * 78)
    print("平台 RankIC 口径对齐诊断")
    print("=" * 78)
    print(f"[0] provider 取值（复刻 runner.py 优先级）: {resolve_provider()}")
    print(f"[0] label 表达式: {LABEL_EXPR}")
    print(f"[0] qlib 实际初始化目录（若有 import 失败可据此判断）: {DEFAULT_QLIB_DIR}")
    print(f"[0] market: {args.market or 'csi300 (来自 configs/backtest.yaml:46)'}")
    print(f"[0] 回测区间: {args.start} ~ {args.end}")

    # --- 1. 定位目标 run ---------------------------------------------------
    pred_path = Path(args.pred) if args.pred else None
    ric_path = Path(args.ric) if args.ric else None

    if pred_path is None:
        print(f"\n[1] 扫描 {RESULTS_ROOT} 下所有 sig_analysis/ric.pkl，目标 mean={args.target_rankic} ...")
        table = scan_runs(args.target_rankic)
        print(table.to_string(max_colwidth=110))
        hit = table.iloc[0] if len(table) else None
        if hit is None or not np.isfinite(hit["mean_rankic"]):
            print("!! 没找到任何可读的 ric.pkl，终止")
            return 2
        # 优先挑 mean 最接近目标的（而不是单纯最大）
        best = (table.assign(gap=(table["mean_rankic"] - args.target_rankic).abs())
                      .sort_values("gap").iloc[0])
        ric_path = Path(str(best["ric_path"]))
        pred_path = ric_path.parent.parent / "pred.pkl"  # artifacts/pred.pkl
        print(f"[1] 命中 run ric : {ric_path}")
        print(f"[1] 命中 pred   : {pred_path}  (exists={pred_path.exists()})")
        print(f"[1] 命中 mean   : {best['mean_rankic']:.6f} (gap={best['gap']:.6f}, n_days={best['n_days']})")

    if not pred_path.exists():
        print(f"!! pred.pkl 不存在: {pred_path}")
        return 3

    # --- 2. 读 pred / ric --------------------------------------------------
    pred_raw = pd.read_pickle(pred_path)
    print(f"\n[2] pred 类型={type(pred_raw).__name__}")
    if isinstance(pred_raw, pd.DataFrame):
        print(f"    index={pred_raw.index.names} layers={getattr(pred_raw.index, 'nlevels', 1)} "
              f"cols={list(pred_raw.columns)[:6]}(n={pred_raw.shape[1]}) shape={pred_raw.shape}")
        panel = to_daily_panel(pred_raw)
    else:  # Series
        panel = pd.DataFrame({"pred": to_daily_panel(pred_raw)})
        print(f"    Series -> panel shape={panel.shape}")
    print(f"    对齐后 panel: rows={len(panel)} cols={panel.shape[1]} 首日={panel.index[0]} 末日={panel.index[-1]}")

    platform_ric = None
    if ric_path and ric_path.exists():
        platform_ric = pd.read_pickle(ric_path).astype(float)
        platform_ric.index = pd.to_datetime(platform_ric.index)
        print(f"[2] 平台 ric.pkl: n={len(platform_ric)} mean={platform_ric.mean():.6f} "
              f"范围 {platform_ric.index[0]} ~ {platform_ric.index[-1]}")

    # --- 3. 重算 label 并对齐 ---------------------------------------------
    try:
        import qlib
        from qlib.data import D

        qlib.init(provider_uri=DEFAULT_QLIB_DIR, region="cn")
        instruments = args.market or "csi300"
        try:
            raw_pool = D.instruments(instruments)
            stock_list = D.list_instruments(raw_pool, start_time=args.start, end_time=args.end, as_list=True)
            n_market = f"{len(stock_list)} 只（样例 {stock_list[:3]}）"
        except Exception as exc:
            n_market = f"({type(exc).__name__}: {str(exc)[:60]})"
        print(f"\n[3] market='{instruments}' 按 config 取到的股票数 = {n_market}   << 口径差参考")

        # 关键：label 按 pred 实际出现的标的取，避开 market 定义不一致
        # （本环境 instruments/csi300.txt 只有 SH000300 一行，直接按 market 取 label 会撞空）
        ins_list = [c for c in panel.columns if str(c).startswith(("SH", "SZ", "BJ"))] or list(panel.columns)
        print(f"[3] 改用 pred 实际标的取 label: {len(ins_list)} 只（样例 {ins_list[:5]}）")
        label_raw = D.features(ins_list, [LABEL_EXPR], start_time=args.start, end_time=args.end)
        # 坑：D.features 对单个表达式可能返回 **Series**（不是 DataFrame），直接 .columns = [...] 会炸；
        # 另外绝不能对 MultiIndex 表做 .T —— 转置会把列标签顶成 index 层（实测 index 里冒出 'LABEL0'）。
        if isinstance(label_raw, pd.Series):
            label_raw = label_raw.to_frame("LABEL0")
        else:
            label_raw = label_raw.copy()
            if label_raw.shape[1] != 1:
                label_raw.columns = ["LABEL0"]
        label_panel = to_daily_panel(label_raw)
        print(f"[3] label panel: rows={len(label_panel)} cols={label_panel.shape[1]} "
              f"首日={label_panel.index[0]} 末日={label_panel.index[-1]}")

        # 与 pred 取交集（只比两者都有的日期 × 标的）
        pred_ins = [c for c in panel.columns if c in set(label_panel.columns)]
        print(f"[3] pred 与 label 共通标的数 = {len(pred_ins)} / pred {panel.shape[1]}")
        sub_pred = panel[pred_ins]
        sub_lab = label_panel.reindex(columns=pred_ins)

        mine = daily_rank_ic(sub_pred, sub_lab)
        mine.name = "pred_rank_ic"
        print(f"\n[4] 用【pred 合成分】重算的日 RankIC: mean={mine.mean():.6f}, n_days={mine.notna().sum()}")

        if platform_ric is not None:
            cmp = pd.concat(
                [mine.rename("mine_pred"), platform_ric.rename("platform_ric")], axis=1, join="inner"
            ).dropna()
            cmp = cmp[cmp.index.isin(mine.dropna().index)]
            overlap = cmp.dropna()
            corr = overlap["mine_pred"].corr(overlap["platform_ric"]) if len(overlap) > 2 else float("nan")
            print(f"[5] 逐日对照: 重叠 {len(cmp)} 天, 相关系数={corr:.4f}")
            print(f"    mine mean  = {cmp['mine_pred'].mean():.6f}")
            print(f"    platform   = {cmp['platform_ric'].mean():.6f}")
            print(f"    差值(platform-mine) = {cmp['platform_ric'].mean() - cmp['mine_pred'].mean():.6f}")
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cmp.to_csv(out_path)
            print(f"[5] 已写出逐日对照 -> {out_path}")

            verdict = "复现成功" if abs(corr - 1.0) < 1e-6 else "口径仍有差异"
            print(f"\n[6] 结论: {verdict}")
            if abs(corr - 1.0) >= 1e-6:
                print("    平台 RankIC 与「pred 合成分」对 label 的截面 spearman 不一致 ->")
                print("    差异来源需继续排查：标的池 / label 表达式 / qlib provider(cn_data vs cn_real_full) / 训练集切分")
    except Exception as exc:
        import traceback

        traceback.print_exc()
        print(f"\n!! 对齐过程异常: {type(exc).__name__}: {exc}")
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
