# QuantaAlpha 项目长期记忆

## 项目定位
QuantaAlpha = LLM 驱动 Alpha 因子挖掘框架（vendoring rdagent==0.8.0）。因子在 `quantaalpha/factors/`，回测在 `quantaalpha/backtest/runner.py`（`BacktestRunner`，QLib 驱动）。真实数据铁律：回测/分析必须基于真实数据，禁止编造；不会的地方直接打断任务。

## 数据（2026-09-30 核实）
- Qlib 数据：`QLIB_DATA_DIR` 已切到 **cn_real_full**（2020-01-02~2026-08-18，全 A ~5210 只，1599 天）。`cn_data` 仅 2025 单年(226天)作 sanity-check；`cn_real_r500` 同区间 500 只（迭代更快）。runner.py 取值优先级：`QLIB_DATA_DIR` env > `QLIB_PROVIDER_URI` > config `provider_uri`。
- 指数基准：**已补齐（2026-10-01 实测更正，旧笔记“无 SH000300”已作废）**。`cn_real_full/features/sh000300/` 有完整 8 字段，实读 1599 行、close 3159~5808（真实沪深300点位）、仅 24 个 NaN；`conf_baseline.yaml` / mining 实际用的 `conf_combined_factors.yaml` 均已 `benchmark: SH000300`，**不再是 SH600000 占位**。
- 字段：cn_real_full 每只票 8 个字段 = `$open/$high/$low/$close/$volume/$vwap` + `factor`(复权) + `label_5d`。**`$vwap` 5211/5211 全齐**（5210 股 + SH000300），可算 Alpha101 的 `adv{d}`。**无基本面字段**（无股本/市值 cap、无 PE/PB）→ 依赖 cap 的公式算不了。
- 频率：`calendars/` 只有 `day.txt` → **仅日频**，无分钟级；分钟数据只存在于 TimescaleDB（非 Qlib）。
- `instruments/csi300.txt` 只列 SH000300 本身 1 行（非 300 成分股）→ 想跑沪深300成分池需自己造成分股列表，现阶段只能用 `market: all`。
- TimescaleDB（旧项目 `D:\qqmt\code` 的 `quant-timescaledb`）：`bars_1d` 658万行 / `bars_5m` 3140万行 / `bars_1m` 空；非 Qlib 格式，分钟级回测需 ETL。

## 通达信战法落地（`reference/tdx_strategies/`）
- 架构：QuantaAlpha **无** jqrs 的 FilterRule/SortRule/SellRule/SizingRule；M2=QLib 表达式因子，M4/M5 非原生（faithful 止损/仓位须 jqrs）。F1–F9 因子表达式已定义并 expr_parser 校验；F9=主力流 L2 近似。
- 战法2（均线回踩低吸）已落地：因子 F2/F3/F4。**多周期验证(cn_real_full, train2020-23/valid2024/test2025-26) IC=0.072/ICIR=1.04/年化34%/回撤-31%/IR=1.08**——信号跨年稳健、幅度偏弱(IC≈0.07)，可作基线；基准已是 SH000300 真实指数（非占位），仅 M4/M5（止损/仓位）非原生。
- 当前执行：**方向② 多周期验证**（cn_real_full，train2020-23/valid2024/test2025-26）。
- 平台审计（2026-10-01）：全链路已跑通（数据→挖掘→评估→展示），无阻断缺陷；遗留=任务并发无限制+状态仅内存(重启丢)、仅日频、无基本面、Alpha101 需算子别名改造、UI 不能直接改 experiment.yaml。见当日日志。
- 后期 deferred（用户确认会做，③已完成）：① 方向①扩 F1/F6/F8/F9 跑战法1/4/6 横向比较+相关性；② 多周期指标定权(风险平价/均值方差)替换先验 35/25/20/20；③ 下载 SH000300 指数特征补基准；④ Phase2 分钟级 TimescaleDB `bars_5m`→Qlib 5min ETL（战法6 faithful）；⑤ jqrs faithful 买卖/止损(M4/M5)；⑥ 战法3/5 延后 Phase2。

## 旧项目 D:\qqmt\code 关键事实
- 数据库：psycopg3+psycopg_pool，无 ORM；连接串由 `config/db_config.json` 同字段在多处拼出；建表全靠 `data_pipeline/schema.py` 内嵌 DDL（无 Alembic/create_hypertable）。
- TimescaleDB 容器沿用旧项目；但 **timescaledb 扩展从未安装**，`bars_5m` 是普通表，无 chunk_time_interval/retention/压缩策略（上一轮审计结论）。

## 数据采集基础设施（2026-10-02 实测核实）
- **DB 连接（可用）**：容器 `quant-timescaledb`（timescale/timescaledb:2.26.2-pg17）在线，`127.0.0.1:5432`，库 `quant_data`，用户 `quant`，密码 `quant123456`。配置文件 `D:\qqmt\code\jq_research_system\config\db_config.json`；compose 在 `D:\qqmt\code\docker-compose.timescaledb.yml`。
- **现有覆盖（2026-10-02 实测，覆盖旧结论：旧数字为 2026-09 快照，已过时）**：`bars_1d` 12,316,609 行 / **2011-01-04~2026-09-30** / 5364 只（10/1-7 国庆休市，故即"最新"）；`bars_5m` 39,084,028 行 / 2026-01-05~2026-09-30 / 5364 只；`bars_1m` 原 25,351,840 行仅 2026-07-17~08-18，正在回补至 2026 全年。
- **⚠️ 硬更正：QMT 分钟数据深度 ≠ 5 年。同月窗口公平实测（600000.SH）**：
  - **5m 只能取到 2026-01-05 之后**（2020/2022/2024/2025-01/2025-06 全部 0 行）→ 现有 bars_5m 已是全部可得范围，**无法再回溯**。
  - **1m 可取到 2025-04-11 之后**（2025-02/03 无，2025-04-11 起有，5 月起完整）→ 比 5m 更深，是唯一还能大幅补采的分钟数据。
  - 旧笔记/画像里"QMT 可取 5 年分钟数据"的说法作废。
  - 判别要点：用**整年**窗口探测会因请求过大返回空而误判，必须用**同月小窗口**公平对比。
  → 日线缺口 = 2026-08-19 之后；分钟 5m 只有 7.5 个月，1m 完全空白。
- **K 线数据源 = QMT xtdata**（不是 akshare；akshare 在旧项目只用于基本面估值/财报）。`xtquant` **只在 conda 环境 `qmt312`**（Py3.12，`D:\Users\lenovo\miniconda3\envs\qmt312`）；该环境同时有 psycopg3 + pandas，可一个环境完成取数+写库。
- **QMT 调用铁律（2026-10-01 误判已更正）**：
  1. **日期参数必须是 `YYYYMMDD`（无横线）**。传 `YYYY-MM-DD` → QMT 报「起始时间错误」，且
     `download_history_data2` 会**长时间阻塞**（表现为"程序卡死"，实测 >240s）。
     改成 `YYYYMMDD` 后同一调用只要 **0.1 秒**。→ 之前误判为"分钟取数必挂、必须子进程隔离"是错的，
     真实原因是参数格式。子进程隔离仍保留作兜底（见下）。
  2. **`import pandas` 必须先于 `import xtdata`**，反序直接 **Segmentation fault**（旧项目 `qmt_child_fetch.py` 也有此注释）。
  3. 先 `download_history_data2()` 再 `get_local_data()`；不要直接调 `get_market_data_ex`。
  4. `get_local_data` 返回的**日线 index 是字符串**（如 `'20240923'`），不是 datetime，需自行转换。
- **QMT 本地数据深度（实测 600000.SH 日线）**：`2016-11-16 ~ 2026-08-18`（2370 行）；
  执行 `download_history_data2('20260819','20261001')` 后可延伸到 **2026-09-30**。
  → 即：历史可从 2016 年起补（比库内 2020 起点更深），最新能拿到上一个交易日。
- **库表字段（复用，勿改）**：`bars_1d`=(ts, asset_code, asset_type, open, high, low, close, volume, amount, turnover_rate, trade_date, source, batch_id, task_id, data_status)；`bars_5m`=(ts, trade_date, asset_code, asset_type, open, high, low, close, volume, amount, source, batch_id, task_id, data_status, extra JSONB)，PK(asset_code, ts)，另有 `stg_bars_5m` UNLOGGED 暂存表。
- **⚠️ UNLOGGED 暂存表被硬关机摧毁 → 采集全量失败（已遇到两次同源问题）**：
  - 现象：`COPY dh_stg_1m` 报 `could not open file "base/16384/<relfilenode>"`，每个 chunk 都失败，行数零增长（主表 bars_* 仍完好）。
  - 根因：`dh_stg_1d/5m/1m` 与旧项目 `stg_bars_*` 均为 UNLOGGED（relpersistence='u'），硬关机后主文件缺失。
  - **修复（关键）**：其 DDL 是 `CREATE UNLOGGED TABLE IF NOT EXISTS`，表在 catalog 里还在，**只跑 init-schema 不生效**；必须先
    `DROP TABLE IF EXISTS dh_stg_1d CASCADE; DROP TABLE IF EXISTS dh_stg_5m CASCADE; DROP TABLE IF EXISTS dh_stg_1m CASCADE;`
    再 `python -m datahub init-schema` 重建，最后用 `count(*)` 验证可读。暂存表写完即清，DROP 无持久数据损失。
- **入库幂等性（已核实，放心重跑）**：`schema.merge_staging` 是 `INSERT ... SELECT DISTINCT ON (asset_code,ts) ... ON CONFLICT (asset_code, ts) DO UPDATE` → 重复采集只会覆盖，不会产生重复行（早期 93.7 万重复是旧路径遗留）。
- **写入吞吐<｜hy_place▁holder▁no▁813｜>**：子进程固定开销约 2.2s 主导，`chunk_size` 默认 30 太保守。实测 30→2,230、200→3,806、**500→≈6,375 rows/s**（1m 的 child_timeout=600s，500 只×10 天≈84 万行≈132~220s，安全）。
- **磁盘**：D: 剩 172GB；**E: 剩 3.3TB（3.7T 仅用 11%）** ← 分钟数据大规模归档应优先考虑 E:。bars_1m 约 **275 B/行**，2026 全年补满约再 +57GB。
- **代码风格（沿用旧项目）**：psycopg3 + psycopg_pool 连接池、`COPY` 协议批量写（`db.copy_dataframe`）、additive-only 幂等 DDL（IF NOT EXISTS）、配置驱动、中文 docstring、`logging.getLogger(__name__)`。
