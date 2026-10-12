# QMT 导入与运行指引（.rzrk 限制的解决方案）

> 结论先行：**我已把策略源码直接部署到 QMT 策略目录，你现在只需重启 QMT 就能看到它。**
> 不需要你手工创建 .rzrk —— QMT 会自动识别。

---

## 一、为什么之前不能直接用（两个真实的坑）

### 坑 1：`.rzrk` 是加密二进制，外部无法生成
QMT【模型研究】→右键→导入 只接受 `.rzrk`。但这个格式是迅投专有的**加密二进制**，
本机 `{QMT}/template/*.rzrk` 打开后全是乱码字节（`5e 21 0e 64 7f 15 ...`，无任何明文结构）。

**所以「给你一个 .rzrk 文件」这条路在 QMT 之外走不通** —— 只有 QMT 自己能生成。

### 坑 2：原 `absmom_qmt.py` 用了 Ptrade 的 API，在 QMT 里必然报错
原脚本调的是 `ContextInfo.order_target_value()` 和 `ContextInfo.portfolio.accounts`，
这是 **Ptrade 的接口，QMT 里根本不存在**（已核对本机 `{QMT}/python/_PyContextInfo.py`）。
直接导入会抛 `AttributeError`。

---

## 二、已解决：用「明文 .py 策略目录」绕过

QMT 的 **Python 策略源码目录是明文的**：

```
D:\中航证券QMT实盘-交易端\python\
   ├── _PyContextInfo.py          ← QMT 的 API 引擎（我按它核对的接口）
   ├── bigqmt_signal_trader_strategy.py  ← 你原来那个大QMT策略
   ├── PY模型回测示例.py            ← QMT 官方示例
   └── absmom_qmt_model.py         ← ★ 我已部署好的 absmom 策略
```

把 `.py` 放进去，**QMT 重启后【模型研究】列表里就会出现它**，可直接回测、实盘。
之后如需 `.rzrk` 备份，在 QMT 内右键该策略 →「导出公式」即可。

**已完成（无需你再做）**：

| 项 | 状态 |
|---|---|
| 部署到 `{QMT}/python/absmom_qmt_model.py` | ✅ 已放好 |
| 编码转 GBK + CRLF（QMT 硬要求） | ✅ |
| 首行 `#coding:gbk` | ✅ |
| 用 QMT 真实 API 重写（非 Ptrade） | ✅ |
| py3.6.8 语法兼容检查 | ✅ 无 py3.7+ 语法 |
| 依赖只有 numpy（QMT 自带） | ✅ |
| 与研究端信号一致性验证 | ✅ 持有集合完全一致 |

---

## 三、你现在要做的（3 步）

### 步骤 1：重启 QMT
**完全退出**（含右下角托盘进程）→ 重新启动、正常登录。

### 步骤 2：【模型研究】里找策略
左侧「模型研究」→ 列表里应出现 **`absmom_qmt_model`**。
看不到就检查：文件是否真在 `D:\中航证券QMT实盘-交易端\python\` 下、是否已重启。

### 步骤 3：点【编译】→【回测】
- 点**编译**（QMT 里"编译"= 保存，不做语法校验）
- 设回测参数：**2019-01-01 ~ 2026-09-30**，初始资金 **80000**，周期 **日线**
- 点**回测**，核对结果：

| 指标 | 期望值（对得上才算部署成功） |
|---|---|
| 年化收益 | 约 **+10%** |
| 最大回撤 | 约 **−9.4%** |
| 调仓次数 | 全年约 **12 次** |

> ⚠️ 若 QMT 回测年化明显低于 10%，**先别加仓**，把回测截图/结果发我，
> 我帮你查是不是数据没下载全（QMT 侧需先在【数据下载】补 ETF 日线）。

---

## 四、切档：主用档 ↔ 低回撤档

改一个数即可。两种方式：

**方式 A（不用碰 QMT，推荐）**——在项目里改后重新部署：
```bash
cd D:\qqmt\QuantaAlpha
python tools/index_lab/deploy_qmt_strategy.py --exposure 0.6   # 低回撤档
python tools/index_lab/deploy_qmt_strategy.py --exposure 1.0   # 主用档(满仓)
```
再重启 QMT。

**方式 B**——直接在 QMT 编辑器里改 `EXPOSURE = 0.6`，点【编译】。

| 档位 | EXPOSURE | 年化 | 回撤 |
|---|---|---|---|
| 主用档 | `1.0` | ~10% | −9.4% |
| 低回撤档 | `0.6` | ~5.5% | −4.8% |

---

## 五、实盘前必做（不可跳过）

1. **模型交易 → 运行模式先选「模拟」**：不下真单，只记录信号。
2. 跑 **1~2 个月末调仓周期**，看下方「消息提示 / 策略信号」是否正常打印。
3. 核对：`account` 打印为空 → 说明【模型交易】里没绑定账号，去界面绑。
4. 模拟盘信号与 `tools/index_lab/make_order_list.py` 生成的清单对得上 → 再切「实盘」。

---

## 六、脚本接口速查（`tools/index_lab/deploy_qmt_strategy.py`）

```bash
python tools/index_lab/deploy_qmt_strategy.py            # 部署（默认档）
python tools/index_lab/deploy_qmt_strategy.py --check    # 只查环境不部署
python tools/index_lab/deploy_qmt_strategy.py --exposure 0.6   # 切低回撤档
```
部署时会自动备份旧版，并做「GBK 解码 + 语法编译」双自检。

---

## 七、QMT API 备忘（已按本机 `_PyContextInfo.py` 核对）

| 用途 | 正确写法 | 说明 |
|---|---|---|
| 买入/卖出 | `passorder(23, 1101, account, code, 5, -1, vol, 备注, quickTrade, "absmom", C)` | **全局函数**，不在 ContextInfo 里 |
| 查持仓 | `get_trade_detail_data(account, "STOCK", "POSITION")` | 同为全局函数 |
| 查资金 | `get_trade_detail_data(account, "STOCK", "ACCOUNT")[0].m_dAvailable` | **单位是分，需 /100** |
| 取行情 | `ContextInfo.get_market_data_ex(["close"], POOL, period="1d", count=N)` | 官方推荐；`get_history_data` 是旧接口 |
| 截历史K线 | `if not ContextInfo.is_last_bar(): return` | **漏写会疯狂废单** |
| 设定股票池 | `ContextInfo.set_universe(POOL)` | `get_history_data` 依赖它 |
| ETF 下单 | orderType 必须 `1101` | ETF 申赎只能用此参数 |
| 枚举 | 23=买入 24=卖出；5=最新价；quickTrade 0=K线结束 2=立即 | |

---

## 八、遇到问题怎么办

| 现象 | 原因 / 处理 |
|---|---|
| 列表里没有 `absmom_qmt_model` | 没重启 / 文件没在 `{QMT}/python/` / 中航版QMT 菜单名不同（找「模型研究」或「策略」） |
| `SyntaxError: encoding problem` | 文件被工具转成了 UTF-8 → 用本部署脚本重写回 GBK |
| `NameError: name 'passorder' is not defined` | 在 QMT 外运行了；须在 QMT 内运行 |
| 回测结果为空 / 全部不交易 | QMT 侧 ETF 日线没下载完 → 【数据下载】补 38 只 ETF 的日线 |
| `account` 为空 | 【模型交易】界面里没绑账号 |
| 报某 API 不存在 | 你的 QMT 版本较旧 → 把报错发我，我按该版本改 |