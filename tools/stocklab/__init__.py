"""stocklab：股票周频择时研究包（工程化分层，供内部与外部研究员复用）。

分层约定（严禁跨层直接调 SQL / 混算指标）：
    config.py    参数集中管理（改参数只改这里）
    data.py      数据层：时点股票池、日线矩阵、周频面板
    features.py  特征工程：只产出特征，不含标签与交易逻辑
    signal.py    信号层：特征/状态 -> 目标仓位
    backtest.py  评估层：指标 + walk-forward 框架（复用 index_lab.metrics）
    cli.py       命令行入口

复用既有资产：DB 配置与绩效指标直接引用 tools.index_lab，避免重复实现。
"""

__all__ = ["config", "data", "features", "signal", "backtest"]
