"""
index_lab —— 指数技术打法验证统一包
====================================
把"别人说的指数关口/抄跌/趋势到底准不准"整合成一个可运行方案，含：
  - support   圆整数支撑检验
  - bounce    抢反弹"赚几点"回测
  - dip       顺趋势抄跌(干净前向收益)
  - momentum  指数动量/趋势跟踪
  - calltracker  博主判断按时间戳录入 + 自动打分
  - etf_guard    ETF 等权持有的 MA60 降回撤保护实测

统一约定：
  * 数据层 io_db.load_ohlc 返回 DatetimeIndex + [open,high,low,close,volume]
  * 所有回测均"信号用前一日收盘算、次日均(open)进场"——杜绝前视
  * 指标统一由 metrics.metrics(rets) 计算
  * 运行: python -m tools.index_lab            (跑全部方向)
          python -m tools.index_lab support    (单方向)
"""
