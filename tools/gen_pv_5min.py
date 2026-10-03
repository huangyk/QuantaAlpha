"""生成 5 分钟级价量 HDF5，供 LLM 生成的 factor.py 直接读取。

背景（为什么必须新造一份）
--------------------------
因子计算链路是：runner 把 `FACTOR_COSTEER_SETTINGS.data_folder` 里的文件链接进
sub-workspace，然后执行 LLM 生成的 `factor.py`，而 factor.py 第一行是写死的

    df = pd.read_hdf('./daily_pv.h5', key='data')

也就是说：**因子算在什么频率的数据上，完全由 daily_pv.h5 决定**，跟回测用的
qlib 数据集（provider_uri / conf 的 freq）是两条独立的线。

2026-10-04 踩的坑：5min 挖掘那一轮只改了回测侧（conf_5min_nextday.yaml 指向
cn_5m_full），daily_pv.h5 仍是日频的，于是：
  - 因子值是日频的（index 1599 个日期，500 只）；
  - 回测 label 是 5min 的（8688 根带时分秒的 timestamp）；
  - 两者 index 完全对不上，因子根本没法参与训练。
结果 17 条轨迹训出的都是同一个 4 特征基线模型，RankIC 全部相同。

所以要做真正的分钟级挖掘，必须让 daily_pv.h5 本身是 5 分钟数据。本脚本就干这个。

口径
----
- 字段与日频版对齐（$open/$close/$high/$low/$volume），额外加 $vwap 和 $factor：
  5min 数据集里这两个字段都存在，其中 $factor 恒为 1.0（入库是不复权价）。
- `$return` = 按**股票**做时序 pct_change（groupby(level=1)）。
  已用真实数据核验：日频 daily_pv.h5 的 $return 与 close.pct_change() 相关系数 = 1.0，
  说明平台口径就是这样，本脚本保持一致。
  （注意：`data_template/generate.py` 里写的是 groupby(level=0)，那会算出
   "同一时刻相邻股票收盘价之比"，是错的；现有 h5 并非由它生成，别照抄那一行。）

用法
----
    # 先小规模测速
    python tools/gen_pv_5min.py --limit 20 --out /tmp/_probe.h5
    # 正式生成（liquid500，约 500 只 × 8688 根）
    python tools/gen_pv_5min.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# 因子脚本里 `import pandas` 必须先于 qlib 相关的重活；这里只有 pandas，无妨。
import pandas as pd

DEFAULT_PROVIDER = "D:/qqmt/qlib_data/cn_5m_full"
DEFAULT_OUT_DIR = "git_ignore_folder/factor_implementation_source_data_5min"


def build(provider_uri: str, market: str, limit: int, start: str, end: str) -> pd.DataFrame:
    import qlib
    from qlib.data import D

    qlib.init(provider_uri=provider_uri, region="cn")

    # 不能直接 D.instruments(market)：它的默认 freq=day，而本数据集 calendars/ 里
    # 只有 5min.txt，会抛 "does not contain data for day"。直接读 instruments 文件。
    inst_file = Path(provider_uri) / "instruments" / f"{market}.txt"
    if not inst_file.exists():
        raise FileNotFoundError(f"instruments 文件不存在: {inst_file}")
    codes = [ln.split()[0] for ln in inst_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if limit and limit > 0:
        codes = codes[:limit]
    print(f"[gen_pv_5min] market={market} codes={len(codes)} range={start}..{end}", flush=True)

    fields = ["$open", "$close", "$high", "$low", "$volume", "$vwap", "$factor"]
    t0 = time.time()
    data = D.features(codes, fields, start_time=start, end_time=end, freq="5min")
    print(f"[gen_pv_5min] D.features done in {time.time()-t0:.1f}s rawshape={data.shape}", flush=True)

    # D.features 返回 index=(instrument, datetime)；swaplevel -> (datetime, instrument)，
    # 与日频 daily_pv.h5 一致，factor.py / function_lib 的 groupby 才对得上。
    data = data.swaplevel().sort_index()

    # 时间序收益：必须按股票分组，不能按 datetime 分组
    data["$return"] = data.groupby(level=1)["$close"].pct_change().fillna(0.0)
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=os.environ.get("QLIB_5M_DIR", DEFAULT_PROVIDER))
    ap.add_argument("--market", default="liquid500", help="instruments 名称，如 liquid500 / all")
    ap.add_argument("--start", default="2026-01-05")
    ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--limit", type=int, default=0, help="只取前 N 只（0=全部），用于测速")
    ap.add_argument("--out", default="", help="输出 h5 路径；默认写到 5min 数据目录")
    args = ap.parse_args()

    data = build(args.provider, args.market, args.limit, args.start, args.end)

    if args.out:
        out = Path(args.out)
    else:
        root = Path(__file__).resolve().parent.parent
        out = root / DEFAULT_OUT_DIR / "daily_pv.h5"
    out.parent.mkdir(parents=True, exist_ok=True)

    tmp = out.with_suffix(".h5.tmp")
    if tmp.exists():
        tmp.unlink()
    t0 = time.time()
    data.to_hdf(str(tmp), key="data", mode="w", format="table")
    if out.exists():
        out.unlink()
    tmp.rename(out)
    print(f"[gen_pv_5min] written {out} shape={data.shape} in {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
