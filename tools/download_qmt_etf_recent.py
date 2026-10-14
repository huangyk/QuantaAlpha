# -*- coding: utf-8 -*-
"""用 miniQMT 的 xtdata.download_history_data2 把近期 ETF 日线拉进本地 datadir。

背景：本机 QMT 有两套 datadir：
  - D:/中航证券QMT实盘-交易端/datadir          (回填脚本旧读的, 停在 2026-09-29)
  - D:/中航证券QMT实盘-交易端/userdata_mini/datadir  (正在运行的 XtMiniQmt.exe 实时写入)
老的回填脚本读错目录，且 miniQMT 不会主动把 10/9 复市后的日线下到本地。
本脚本用 xtdata(沙箱仅 connect/download_history_data2 可用)把缺口期拉进 userdata_mini/datadir，
再由 backfill_etf_qmt_local.py(已改为优先读 userdata_mini)回填进 TimescaleDB。

用法(必须在 qmt312 环境跑, 且 XtMiniQmt.exe 在运行):
    conda activate qmt312
    python tools/download_qmt_etf_recent.py
"""
import sys
import time

from backfill_etf_qmt_local import POOL, market_of


def main():
    import xtquant.xtdata as xt
    try:
        xt.connect()
    except Exception as e:
        print(f"[错误] xtdata.connect 失败(需 XtMiniQmt.exe 在运行): {e}")
        sys.exit(2)
    print("xtdata 已连接")

    start = "20260901"
    end = "20261013"
    ok, fail = 0, []
    for code, name in POOL.items():
        # xtdata 用完整代码如 '510300.SH'；period='1d' 日线
        try:
            xt.download_history_data2(code, period="1d",
                                      start_time=start, end_time=end,
                                      incrementally=True)
            ok += 1
        except Exception as e:
            fail.append((code, str(e)))
    # 等待后台下载线程把数据落盘
    time.sleep(5)
    print(f"下载请求已发出: 成功 {ok}/{len(POOL)}")
    if fail:
        print("失败:")
        for c, e in fail:
            print(f"  {c}: {e}")
    print("下一步: 运行 python tools/backfill_etf_qmt_local.py --clean-legacy (quantaalpha 环境)")


if __name__ == "__main__":
    main()
