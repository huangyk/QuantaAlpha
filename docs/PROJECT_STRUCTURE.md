# 项目结构

> 更新于 2026-10-14。本文件描述**仓库实际结构**；研究规范见 `research_sop.md`，
> 交付流程见 `warfare_intake_sop.md`，脚本索引见 `../tools/README.md`。

```
QuantaAlpha/
├── quantaalpha/                  # 核心 Python 包（RD-Agent 上的 Alpha 挖掘框架）
│   ├── pipeline/                 #   主工作流：factor_mining / loop / planning / evolution
│   ├── factors/                  #   因子定义与评估（coder / runner / library / proposal）
│   ├── backtest/                 #   独立回测模块 V2（run_backtest / runner / factor_loader）
│   ├── strategy/                 #   实盘策略（absmom_qmt / absmom_ptrade / pool_hold）
│   ├── llm/ core/ cli.py         #   LLM 客户端 / 核心抽象 / CLI 入口
│
├── tools/                        # ★ 研究脚本工作区（159 个 .py，见 tools/README.md）
│   ├── README.md                 #   脚本索引与命名约定（必读）
│   ├── backfill_*.py             #   数据回填（QMT datadir → TimescaleDB）
│   ├── exec_layer_*.py           #   执行层 L4 实验
│   ├── validate_*.py / *_backtest.py   # 战法验证
│   ├── factor_*.py / build_*.py  #   因子构建与检验
│   ├── etf_*.py                  #   ETF 研究（17 个）
│   ├── portfolio_*.py / strategy_*.py  # 组合与策略
│   ├── make_order_list.py / make_trading_plan.py  # 实盘下单清单
│   ├── _*.py                     #   诊断脚本（下划线前缀）
│   └── *_patch.py                #   依赖库运行时补丁
│
├── docs/                         # ★ 文档（交接包主体）
│   ├── research_sop.md           #   研究规范（阶段 1 回测细则）
│   ├── warfare_intake_sop.md     #   战法接收流程（六阶段编排）★2026-10-14 新增
│   ├── warfares_catalog.md       #   战法总册（信号定义 + 验证状态）
│   ├── a_share_warfare_landscape.md  # 圈内生态 + 后续三条主线
│   ├── HANDOFF.md                #   交接总说明
│   ├── PROJECT_STRUCTURE.md      #   本文件
│   ├── *_verdict.md              #   各战法终审报告（dixi / fibonacci / stock / etf_portfolio …）
│   └── warfare_sources/          #   战法来源原始材料
│
├── configs/                      # 配置
│   ├── experiment.yaml / backtest*.yaml   # 实验与回测参数
│   ├── .env.example              #   环境变量模板（复制为根目录 .env）
│   └── mcp/                      #   MCP 连接器配置
│
├── frontend-v2/                  # Web 工作台（React + TypeScript + FastAPI）
│   ├── src/  backend/  start.sh
│
├── data/                         # 实验产物（gitignore）
├── log/ mlruns/ pickle_cache/    # 运行日志 / MLflow / 缓存（gitignore）
│
├── run.sh                        # 主实验启动脚本（加载 .env + 隔离 + 校验）
├── launcher.py                   # CLI 入口
└── pyproject.toml / requirements*.txt
```

---

## 仓库外的关联项目（不在本仓库，但强依赖）

| 路径 | 作用 | 环境 |
|:---|:---|:---|
| `D:/qqmt/quant_datahub` | **数据采集管道**：QMT → TimescaleDB → Qlib 数据集 | `qmt312` |
| `D:/qqmt/qlib_data/cn_daily_clean` | **干净 Qlib 数据集**（2011-01-04~2026-10-13，5397 只）★2026-10-14 重建 | — |
| `D:/中航证券QMT实盘-交易端/datadir` | QMT 本地行情文件（兜底数据源） | — |

## 环境

| 环境 | 用途 |
|:---|:---|
| `quantaalpha`（Python 3.11） | **主力**：挖因子 + 回测 + Qlib + 后端 |
| `qmt312` | QMT xtdata 取数 + datahub 采集 |

## 数据底座

`TimescaleDB quant_data`（127.0.0.1:5432 / quant / quant123456）：
`bars_1d`（1245 万行，清洗后）、`bars_5m`（4247 万行，2026 起）、`bars_1m`（5130 万行，2026-07 起）、
`sector_master/membership`（SW1 申万一级）、`asset_master`、`corporate_action_factor`。

> ⚠ **数据质量铁律**：跑任何读 `bars_1d` 的脚本前，先确认 `(asset_code, trade_date)` 唯一。
> 2026-10-14 曾因 42.8 万组重复行导致信号全错（详见 `warfare_intake_sop.md` §9）。
