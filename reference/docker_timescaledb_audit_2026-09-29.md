# Docker / TimescaleDB 配置核查报告
**日期**：2026-09-29
**任务**：确认旧项目（/d/qqmt/code）的 Docker 配置是否完整保留，重点核实 5 分钟行情数据的「保留时长 / 缓存间隔」设置是否仍存在且生效。

---

## 一、结论速览

| 核查项 | 结果 |
|---|---|
| 容器 `quant-timescaledb` 是否还在运行 | ✅ 运行中（Up ~20min，Created 2026-05-19，健康） |
| `quant-qlib-data` 容器是否还在运行 | ✅ 运行中（Up ~20min，Created 2026-08-22，healthy） |
| compose 配置文件是否保留完整 | ✅ `/d/qqmt/code/docker-compose.timescaledb.yml` 与 `/d/qqmt/code/quantaalpha/docker-compose.qlib-data.yml` 均完好 |
| 数据卷是否保留 | ✅ `/d/qqmt/code/db/data` 持久化（末次写入 2026-09-30） |
| **TimescaleDB 扩展是否已安装** | ❌ **未安装**（available 2.26.2，但 `installed_version` 为空） |
| **5 分钟表 `bars_5m` 是否为超表(hypertable)** | ❌ **否，是普通 PostgreSQL 表** |
| **`chunk_time_interval`（缓存间隔）是否生效** | ❌ **不存在 / 未生效** |
| **保留策略(retention / drop_chunks)是否生效** | ❌ **不存在 / 未生效** |
| **压缩策略(compression)是否生效** | ❌ **不存在 / 未生效** |
| 5 分钟数据本身是否还在 | ✅ `bars_5m` 31,385,710 行，5,219 只标的，ts 区间 2026-01-05 → 2026-08-18 |

**核心结论**：旧项目的配置文件（compose / db_config / 容器 / 数据卷）**完整保留**；但用户所关切的「5 分钟数据保留时长 / 缓存间隔」这类 **TimescaleDB 级设置从未被真正配置过，因此现在没有、过去也没有生效**。容器虽使用 `timescale/timescaledb` 镜像，但 `timescaledb` 扩展从未在 `quant_data` 库内 `CREATE EXTENSION`，`bars_5m` 一直是一张普通表。

---

## 二、证据链

### 1. 容器与定义文件（配置文件层面：保留完整）
- `docker ps`：`quant-timescaledb`（timescale/timescaledb:2.26.2-pg17，0.0.0.0:5432→5432）、`quant-qlib-data`（alpine:3.20，sleep infinity）均在运行。
- 两容器均由旧项目 `/d/qqmt/code/` 下的 compose 定义：
  - `docker-compose.timescaledb.yml`：image / container_name `quant-timescaledb` / `restart: unless-stopped` / env（POSTGRES_DB=quant_data, USER=quant, PASSWORD=quant123456）/ 卷 `./db/data:/var/lib/postgresql/data` + `./db/init:/docker-entrypoint-initdb.d`。
  - `quantaalpha/docker-compose.qlib-data.yml`：bind 宿主机 Qlib 缓存 + `data/results`。
- `db_config.json`（jq_research_system/config）密码 `quant123456` 与 compose 一致，文件完好。

### 2. TimescaleDB 扩展未安装（关键事实）
```sql
SELECT name, default_version, installed_version
FROM pg_extension WHERE name='timescaledb';
-- name=timescaledb, default_version=2.26.2, installed_version=<空>
```
- 扩展「可用但未安装」→ `_timescaledb_catalog.*` 与 `timescaledb_information.*` 视图均不存在。
- 因此 `bars_5m` 不是超表，不存在 `chunk_time_interval`、retention job、compression job。

### 3. `bars_5m` 实测结构（普通表，无分区/无触发/无策略）
- 列：`ts timestamptz NOT NULL, trade_date date, asset_code text, ... close/volume/amount ...`
- 索引：`bars_5m_pkey(asset_code, ts)`、`idx_bars_5m_asset_ts_desc`、`idx_bars_5m_trade_date_asset`。
- `pg_trigger` 对 `bars_5m` 无任何触发器（无 chunk 自动路由）。
- 行数：31,385,710；distinct asset_code：5,219；ts 区间 **2026-01-05 09:35 → 2026-08-18 15:00**。

### 4. 旧项目代码从未定义超表 / 保留策略
- `jq_research_system/data_pipeline/schema.py`：`bars_5m` 用 **普通 `CREATE TABLE`**，无 `create_hypertable`。
- 全仓 grep（`*.py` / `*.yaml` / `*.json` / `*.env` / `*.sql`）对 `hypertable | chunk_time | retention | drop_chunks | cache_interval | bars_5m`：**零命中**（无 .sql 文件，无 retention 配置）。
- `db/init/` 目录**为空**（docker-entrypoint-initdb.d 未挂载任何初始化脚本）；`db_config.json` 引用的 `db/init/001_schema.sql` **不存在**。
- 应用以 `auto_init_schema: false` 运行，表结构由 Python 运行时创建，不走 SQL 初始化。

### 5. 真正存在的「5m」相关设置（仅为 ETL 批处理调参，非 DB 保留/缓存）
| 位置 | 参数 | 含义 |
|---|---|---|
| qmt_web/settings.py | `download_chunk_5m = 30` | QMT 下载批大小 |
| qmt_web/config.py | `batch_timeout_sec_5m = 900` | 单批超时 |
| qmt_web/config.py | `download_batch_size_5m = 100` | 下载批大小 |
| qmt_web/config.py | `load_batch_size_5m = 80` | 入库批大小 |
| data_pipeline/fetcher.py | `_DEFAULT_CHUNK_SIZE_5M = 100` | 抓取分块大小 |

> 这些只是「抓取/入库的批量大小」，与 TimescaleDB 意义上的「数据保留时长」或「chunk 缓存间隔」**不是一回事**。若用户记忆中的「缓存间隔」指的就是 `download_chunk_5m`，那它确实仍在代码中保留并生效；但它不属于数据库层设置。

---

## 三、附带发现（非本次核查重点，但值得记录）
1. **`bars_1m` 为空（0 行）**：1 分钟表从未被填充。
2. **`bars_1d` 正常**：6,585,157 行（与记忆中「约 646 万行日线」吻合）。
3. **5 分钟数据停更**：`bars_5m` 最新 ts 为 2026-08-18，相对当前上下文日期 2026-09-29 已约 6 周未写入，疑似 5m 流水线已停更（属数据新鲜度问题，非配置丢失）。

---

## 四、建议（如需让「保留时长 / 缓存间隔」真正生效）
若用户确实希望 5 分钟数据具备 TimescaleDB 的 chunk 分区与自动保留，需要：
1. 在 `quant_data` 执行 `CREATE EXTENSION IF NOT EXISTS timescaledb;`
2. 将 `bars_5m` 转为超表：`SELECT create_hypertable('bars_5m','ts');`（如需保留历史主键需调整）。
3. 配置 `chunk_time_interval`：`SELECT set_chunk_time_interval('bars_5m', INTERVAL '7 day');`
4. 配置保留策略：`SELECT add_retention_policy('bars_5m', INTERVAL '365 day');`
5. （可选）压缩策略 `add_compression_policy`。

> 以上为「若需要」的改造路径；本次仅核查，**未做任何改动**。是否执行取决于用户确认。
