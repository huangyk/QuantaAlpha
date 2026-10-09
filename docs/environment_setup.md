# QuantaAlpha 环境与依赖管理（务必照此执行）

> 适用：本机（lenovo / Windows）。任何"跑不起来 / ModuleNotFoundError"基本都是**环境用错**或**依赖没装**。
> 最后校准：2026-10-09（已为 `quantaalpha` 补齐 xgboost + psycopg2）。

---

## 1. 一句话结论

- **前端**：用 **nvm 管理的 node v24.9.0**，不要直跑 `vite.js`、不要托管 node。
- **后端 / 因子挖掘 / 回测**：全部用 **conda 环境 `quantaalpha`**（Python 3.11）。
- **QMT 取数**：在**你的 QMT 机器**上用 `qmt312` 环境跑 `xtdata`；沙箱里 `xtdata` 取数接口会 segfault，只能直读 `datadir` 或走 datahub。
- **数据源铁律**：QMT 优先，**禁止默认 akshare**（用户硬规则）。

---

## 2. conda 环境总览

| 环境 | 用途 | 关键包 | 何时用 |
|---|---|---|---|
| `base` | conda 根（系统级） | psycopg2 2.9.12 | **不要用于项目**。历史上回测因 `quantaalpha` 缺 psycopg2 临时用过 base，现已统一回 `quantaalpha` |
| **`quantaalpha`** ★ | **主力环境**：因子挖掘(rdagent)+回测+后端 | qlib 0.9.7, rdagent 0.8.0, lightgbm 4.7.0, torch 2.5.1+cu121, **xgboost 2.1.4**, **psycopg2 2.9.13**, scipy 1.17.1, numpy 1.26.4, pandas 2.3.3 | 挖因子、跑回测、起后端 FastAPI |
| `qmt312` | **QMT 312 客户端 API**（xtdata 取数 / datahub 同步） | qlib, xgboost 3.3.0, lightgbm 4.6.0, torch | 仅在你 QMT 机器上拉数据；numpy=2.4，**不可**与 `quantaalpha` 混用包 |
| `rd-agent` | 旧 rdagent 变体 | rdagent 0.8.0, qlib | 不用（已被 `quantaalpha` 取代） |
| `rdagent4qlib` | 空/损坏环境 | （pip list 失败） | 不用 |

> 为什么是 `quantaalpha` 而不是 `rdagent`/`qmt312`：`quantaalpha` 是唯一的 **qlib 0.9.7 + rdagent 0.8.0 + 完整科学栈** 环境，因子挖掘(`quantaalpha/factors/factor_template/conf_*.yaml`)和回测都在这里。
> `qmt312` 是 QMT 行情客户端环境（API 不同、numpy 大版本不同），**不要**在上面跑 rdagent。

---

## 3. 前端（nvm node）

```bash
cd /d/qqmt/QuantaAlpha/frontend-v2
PATH="/d/nvm4w/nodejs:$PATH" npm run dev      # -> http://localhost:3000
```

- 必须用 nvm 的 `node v24.9.0`（npm 11.6.0）。
- **不要**用 WorkBuddy 托管 node，也**不要**直接 `node vite.js`（会缺依赖/配置）。
- 深色主题：已在 `index.html` 置 `<html class="dark">`，Tailwind `darkMode:'class'`。

---

## 4. 后端（conda quantaalpha）

```bash
D:/Users/lenovo/miniconda3/envs/quantaalpha/python.exe frontend-v2/backend/app.py   # -> :8000
```

- 健康检查：`GET http://127.0.0.1:8000/api/health` → `{"status":"healthy"}`
- 注意：`app.py` 用 `subprocess` 派生子进程跑挖掘/回测，**自身不直接连 TimescaleDB**（不 import psycopg2），所以起服务不需要数据库密码，但子进程跑回测时会用 `quantaalpha` 里的 psycopg2 连库。

---

## 5. 依赖安装（已执行，备查）

2026-10-09 在 `quantaalpha` 补齐了缺失包（**适配 numpy<2.0，不能照搬 `qmt312` 的 xgboost 3.x**，因为 `qmt312` numpy=2.4，编译扩展 ABI 不匹配）：

```bash
D:/Users/lenovo/miniconda3/envs/quantaalpha/python.exe -m pip install "xgboost==2.1.4" psycopg2-binary
```

| 包 | 版本 | 解决什么 |
|---|---|---|
| `xgboost` | 2.1.4 | 消除 qlib 的 `ModuleNotFoundError. XGBModel is skipped` 告警；rdagent 模型提议(proposer)可能生成 XGBoost 模型，未装会在那一步真崩 |
| `psycopg2-binary` | 2.9.13 | 让 DB 回测统一在 `quantaalpha`（之前缺 psycopg2 只能用 base 跑，易踩错） |

> 仍会打印一条良性告警 `ModuleNotFoundError. CatBoostModel are skipped` —— catboost 未装、因子挖掘不需要，**可忽略**。
> 若哪天要跑 CatBoost 模型再 `pip install catboost`（同样锁 numpy 兼容版本）。

---

## 6. 关于 "XGBModel is skipped" 告警

- **来源**：`qlib/contrib/model/__init__.py` 在 import 时检测 `xgboost` 缺失就 `print` 一条，属**良性告警**。
- **默认挖掘用 LGBModel（lightgbm）**，所以这条告警本身不致命。
- **但** rdagent 的模型提议步骤(`quantaalpha/contrib/model/proposal.py`)会把模型提议成 `XGBoost` 类型；若 `XGBModel is None`，真正构造模型时会 `AttributeError/TypeError` 崩溃。这就是"挖掘过程中报错了"的根因之一。
- 现已装 xgboost，该隐患消除。

---

## 7. 因子挖掘怎么跑

- 配置模板：`quantaalpha/factors/factor_template/conf_*.yaml`（默认 `class: LGBModel`）。
- 5 分钟频率 + CoSTEER 框架验证因子；因子门槛：日频 |t|≥2 且三段同号。
- 入口：`run.sh` / `quantaalpha/cli.py`，一律在 `quantaalpha` 环境执行。
- 回测验证：见 `tools/backtest_qixing.py`（跨资产 ETF 轮动/组合层），已统一在 `quantaalpha` 跑。

---

## 8. 之前"改代码报错"的常见成因与避免

| 现象 | 成因 | 正确做法 |
|---|---|---|
| `ModuleNotFoundError: qlib` / `rdagent` | 用了 base 跑需要 qlib 的脚本 | 一律 `quantaalpha` |
| `ModuleNotFoundError: psycopg2` | 在 `quantaalpha` 跑回测（旧环境缺 psycopg2） | 现已装好，继续用 `quantaalpha` |
| `XGBModel is skipped` 后崩溃 | 缺 xgboost，且 rdagent 提议了 XGBoost | 已装 xgboost 2.1.4 |
| 前端 `vite` 报缺模块 | 用了错 node / 直跑 vite.js | nvm node + `npm run dev` |
| `xtdata` 取数 segfault | 沙箱跑 QMT 取数接口 | 取数走 QMT 机器(qmt312)或 datahub 同步，沙箱只直读 datadir |
| 整批 `.dmp` 崩溃转储 | 早期 shell 不稳定产生，非项目文件 | 可安全删除，勿入库 |

---

## 9. 提交约定

- 用户规则：**每次代码改动 commit，不 push**。
- 临时扫描脚本（如 `tools/_*.py`）用完即删，不入库。
- `.dmp` 崩溃转储不入库（已在 `.gitignore` 或手动排除）。
