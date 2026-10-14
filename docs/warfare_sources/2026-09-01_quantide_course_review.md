# 来源捕获：花 930 美元买一门量化课? GPT 如何评估量化课程值不值

- **原始链接**：https://mp.weixin.qq.com/s/iq0UiyV5CcUa_0OpxEonuA
- **公众号/作者**：Aaron 量化Quantide（原创）
- **发布时间**：2026年9月1日 07:58
- **捕获方式**：WebFetch 提取（非逐字原文，为结构化摘要）
- **捕获日期**：2026-10-13
- **文章性质**：课程评测 / 观点营销文（**非战法/策略教学文**）

---

## 正文摘要

### 1. 评测对象：Quantinsti《llm-trading-strategies》课程（$934）
GPT 打分（满分10）：
- 教学路径 8 / 数据→模型→策略闭环 8 / LLM 基础 6 / 现代 LLM 技术 4 / Quant 严谨性 6 / 工程化 5 / 实战案例 7 / 2026 年先进性 **4** / 中国大陆适用性 **5**

核心批评：
- 现代 LLM 技术仍用 2019 年 FinBERT 架构，只适合情绪分类 → 最多叫 Financial NLP for Event-driven Trading，不是 LLM Trading。
- 2026 年谈 LLM Quant 应是 Claude/GPT/DeepSeek 一类。

模块评价：
- LLM 基础（Generative AI、RNN、LSTM、Attention、Transformer）：太长、偏历史课
- LLM 使用（训练、RLHF、Prompt Engineering）：必要但浅
- 金融 NLP（FinBERT、金融情绪分析）：**课程真正核心**
- 数据工程（FOMC transcript、PDF/文本清洗、音频转文字）：很实用
- 策略（**情绪阈值、rolling sentiment、价格趋势过滤**）：合理但过于单一
- 工程/部署（RAG、fine-tuning、distillation、deployment）：点到为止，无法形成能力

结论：2026 年看 LLM for Quant，这门课"已经可以淘汰了"。

### 2. GPT 反向评测 Quantide 自家课程（blog.quantide.cn，三门）
- 《量化交易中的 Numpy 和 Pandas》：编程基本功 + 量化数据处理
- 《量化二十四课》：全栈入门（数据、策略、统计、因子、回测 → QMT/PTrade 实盘）
- 《因子投资与机器学习策略》：进阶（因子研究、Alphalens、Alpha101、LightGBM、Pair Trading、Walk-Forward）

专业度检测点（《量化二十四课》第22章）：明确讲
**幸存者偏差、Look-ahead bias、PIT 数据、复权导致的前视问题、T+1、涨跌停、冲击成本、不可能成交价格、撮合差异、过拟合**

因子研究链路（被赞"路线对"）：
因子生成 → 异常值 → 缺失值 → 分布调整 → 标准化 → 中性化 → forward return → 回归 → IC → 分层回测 →（Alphalens）→ IC、分层收益、换手率、因子单调性、参数调优、样本外测试、参数高原

不足：更像"3~4年研究员水平"培训，缺期权/高频；Alphalens 投入偏重（2024底设计，若今天重做压缩到 1/5）；现代深度学习/组合管理/风险模型未覆盖；LLM/Agent 未成体系。

综合评分（去掉主动不选部分）：**8.14**（对比 Quantinsti 5.89）。

---

## 对"提炼战法/因子"任务的价值评估
- **本文不是战法教学文**，无具体可回测的买卖信号定义（如"回踩MA20""首阴"等）。
- 唯一被提到的具体策略是 **"情绪阈值 + rolling sentiment + 价格趋势过滤"**，但：
  1. 原文评其为"合理但过于单一"；
  2. 属**情绪/文本驱动**，依赖新闻/财报/社媒文本源 —— 本项目 TimescaleDB（bars_1d/bars_5m）**不含任何文本/新闻数据**，且用户铁律"禁止用 akshare 替代 QMT"，暂无合规文本源 → **无法在本项目真实回测**。
- 文中列出的"偏差清单"（幸存者/Look-ahead/PIT/复权前视/T+1/涨跌停/冲击成本/不可能成交价/撮合差异/过拟合）与本项目既有验证铁律高度一致，属研究规范共识，**非新战法**。
- 因子研究链路（IC→分层→单调性→样本外→参数高原）即本项目 `dixi_fullpanel_backtest.py` 已在用的框架。

**结论**：本篇无可提取的具体战法；建议作为"研究规范参考"归档，不进入 `warfares_catalog.md` 的战法表。
如需情绪类策略，须先解决文本数据源（用户授权后方可引入）。
