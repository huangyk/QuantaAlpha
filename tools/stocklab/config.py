"""stocklab 参数配置——所有可调参数集中在此，便于外部研究员复用/复现。

改策略只改本文件；不要把魔法数字散落在各模块。
"""
# 数据源复用 index_lab 的同一份 DB 配置（禁止另起数据源，尤其禁止默认改用 akshare）
from tools.index_lab.config import DB  # noqa: F401

# ---------------- 数据层 ----------------
ASSET_TYPE = "stock"
BARS_TABLE = "bars_1d"
START = "2011-01-01"
END = "2026-09-30"          # 日频库股票数据最新到 2026-09-30

# ---------------- 股票池过滤（时点口径，铁律）----------------
MIN_LIST_DAYS = 250         # 剔次新股：上市满 250 个自然日
EXCLUDE_ST = True           # 剔 ST（仅当前标记，无历史时点 -> 回测近似，见 data.py 文档）
EXCLUDE_DELISTED = True     # 剔已退市（用 delist_date，真实历史）
MIN_LISTED_FOR_UNIVERSE = 250

# ---------------- 周频择时 ----------------
REBALANCE = "W"             # 每周调仓（周五收盘出信号，下周一生效）
LOOKBACK_WEEKS = 52         # 趋势/动量回看窗口（周）
COST_BPS = 1.0              # 单边万1（ETF/股票通用近似）

# ---------------- 评估 ----------------
TEST_START_WEEKLY = "2016-01-01"   # walk-forward 测试窗起点（留足训练历史）
ANNUAL_WEEKS = 52                  # 周频年化因子

# ---------------- ML 择时叠加层 ----------------
ML_MODELS = ("gbm", "xgb", "mlp")   # LightGBM / XGBoost / PyTorch MLP
ML_MIN_TRAIN_WEEKS = 104            # 至少 2 年训练历史再开始预测
ML_PROB_FLOOR = 0.2                 # 概率下限，避免 0 暴露导致长期空仓
ML_PROB_CEIL = 1.0
