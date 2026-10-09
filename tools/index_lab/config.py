"""全局配置：数据库连接、指数清单、成本与常量。"""
from typing import List, Tuple

# TimescaleDB（QMT datadir 回填，行情至 2026-09-30）
DB = dict(host="127.0.0.1", port=5432, dbname="quant_data",
          user="quant", password="quant123456")

# 6 大指数（name, code）
INDICES: List[Tuple[str, str]] = [
    ("上证指数", "000001.SH"),
    ("沪深300", "000300.SH"),
    ("上证50",  "000016.SH"),
    ("中证500", "000905.SH"),
    ("创业板指", "399006.SZ"),
    ("科创50",  "000688.SH"),
]

# 回测默认窗口起点
DEFAULT_START = "2018-01-01"

# 年交易日 / 成本
TRADING_DAYS = 252
INDEX_COST = 0.0003   # 指数回测：万3（含近似冲击）
ETF_COMM = 0.0001     # ETF 实盘佣金：万1（免印花）

# 趋势门控默认用的指数（B 功能：ETF 等权持有叠加 MA60 门控）
GATE_INDEX: Tuple[str, str] = ("中证500", "000905.SH")
GATE_MA = 60

# 干净 ETF 来源（与七星高照研究同口径）
CLEAN_SOURCE = "qmt_local_86400"
