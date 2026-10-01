# 飞龙体系因子挖掘基准（RD-Agent → QuantaAlpha）

> **用途**：本文件是 `log/2026-08-20_10-03-43-266516` 全会话经验的**可迁移基准**。  
> 后续做 **QuantaAlpha**（或任何新挖矿框架）时，以此为：
> 1. **对照基线**（必须先复现/对齐 Loop11 再宣称超越）  
> 2. **禁区与停手规则**（避免重复空转）  
> 3. **研究方向 brief**（entry / exit / rr 体系）  
>
> **会话状态**：已停于 Loop90 假设生成；`fin_factor` 进程已清空。  
> **配套提示词**：`config/rd_research_prompt.md`（RD 注入用；本文件比它更完整、面向跨项目）。

---

## 0. 一句话结论

在 **Qlib 日频 OHLCV（`cn_real_r500`）+ LightGBM 组合** 设定下：

- 飞龙交易体系可以挖出稳定三因子库：**Loop11 = 最终 SOTA**。  
- Loop12–89（约 78 轮）**全部未能 Replace Best**。  
- 瓶颈是 **同构探索 + filter 稀释组合**，不是「再多挂几十轮」。  
- 下一阶段应用 **QuantaAlpha 做新结构/新轨迹探索**，但必须以 Loop11 为硬基线；不要从零随机动量起步。

---

## 1. 实验设定（复现对齐用）

| 项 | 值 |
|----|-----|
| 会话路径 | `log/2026-08-20_10-03-43-266516` |
| 工具 | RD-Agent `fin_factor`（本仓库 `.\dev.ps1 rd-factor` / `rd-resume`） |
| 数据边界 | Qlib provider；**不读 Postgres**；主字段 `$open/$high/$low/$close/$volume/$factor` |
| 研究池 | `cn_real_r500`（非 smoke） |
| 模型 | LightGBM（因子挖掘路径；非 PyTorch 模型挖矿） |
| 评测 | RD 组合回测：IC / 含成本超额年化 / 最大回撤 等 |
| 研究叙事 | 通达信飞龙线 / 选股「买点·卖点·盈亏比」→ 连续可排序分数 |
| 工程入口 | `config/rd_research_prompt.md` + `tools/research_prompt_patch.py` |

### 成功判定（强制，跨框架通用）

必须同时满足才算「击败基线」：

1. **含成本年化超额 ≥ Loop11**（≈ **42.4%**）  
2. **最大回撤不显著恶化**（Loop11 ≈ **-30.9%**；可允许小幅变差，但不得用回撤换虚高 IC）  
3. **IC 仅作辅证**；**禁止**「IC↑ 但年化↓」记为成功（Loop24 铁证：IC 0.026→0.028，年化 42%→36%）

### 停手规则（RD 已验证）

从上次 SOTA 起连续 **≥25 轮** `decision=False`，且假设落入已知死族 → **方向挖矿结束**，转落地或换框架/换方向。  
本会话实际空转远超此阈值，停在 Loop90 是正确决策。

---

## 2. 锁定基线（Loop11 SOTA）

### 2.1 成绩锚点

| 指标 | 约值 |
|------|------|
| IC | ≈ 0.0259 |
| 含成本年化超额 | ≈ 42.42% |
| 最大回撤 | ≈ -30.89% |

对比路径（台阶，勿复活为「新目标」）：

| Loop | 结果 | 备注 |
|------|------|------|
| 2 | True | 早期三因子进库 |
| 3 | True | 年化/回撤上台阶 |
| 6 | True | 引入 `feilong_geo_squeeze_regime`；chip 因子后来未留在最终三件套 |
| **11** | **True（最终）** | entry 保留 L6 + **加性 exit** + **rr_atr_regime** |
| 12–89 | 全 False | 未进库 |
| 90 | 未完成 | 停机时在假设 `filter_cvma60_discount_persistence` |

历史上曾短暂 SOTA、已被 L11 覆盖的组合（**不要复活整组**）：

- L2：`abs_feilong_vol_comp` / `mean_reversion_wash` / `breakout_quality_vol_scaled`  
- L3：`rr_vol_adjusted` / `exit_pressure_score` / `regime_aware_entry`  
- L6：`feilong_geo_squeeze_regime` + `chip_deep_discount_concentration`（chip 未进最终库）

### 2.2 三因子公式（从 Loop11 任务定义抽出）

#### (1) entry — `feilong_geo_squeeze_regime`

直觉：几何飞龙/VWAP 偏离 × 波动压缩强度；只在 squeeze 时放大。

```text
geo_feilong_dist = sqrt( close/EMA(close,89) * close/VWAP(89) ) - 1
BB_width         = std(close,20) / mean(close,20)
squeeze_intensity = mean(BB_width,60) / BB_width
feilong_geo_squeeze_regime = geo_feilong_dist * squeeze_intensity

VWAP(89) = sum(close*volume, 89) / sum(volume, 89)
```

实现注意：`squeeze_intensity` 分母必须 `eps`/clip，防止 BB_width→0 爆炸（早期 NaN 轮次教训）。

#### (2) exit — `exit_pressure_additive`

直觉：区间进度 + VWAP 延伸 + 量能形态；**加性**（乘性在 L3/L4 等路径上易信号互消）。

```text
range_position_20d = (close - min(low,20)) / (max(high,20) - min(low,20) + eps)
vwap_extension     = (close - VWAP(89)) / VWAP(89)
vol_ratio          = volume / mean(volume,20)
volume_regime      = 1 - abs(vol_ratio-1) / (1 + abs(vol_ratio-1))
exit_pressure_additive = (range_position_20d + vwap_extension + volume_regime) / 3
```

#### (3) rr — `rr_atr_regime`

直觉：ATR 自适应支撑/目标的盈亏比，再按波动 regime 调节。

```text
ATR(14)   = mean(high-low, 14)          # 任务定义里的简化 ATR
support   = min(low,20) - ATR(14)
target    = max(high,12) + ATR(14)
raw_rr    = (close - support) / (target - close + eps)
realized_vol = std(close_ret, 60)
vol_z     = (realized_vol - mean(realized_vol,60)) / (std(realized_vol,60) + eps)
rr_atr_regime = raw_rr / vol_z
```

实现注意：后续大量失败尝试证明 **「再把 vol_z 当更激进分母 / 再套一层 z」易翻车**；QuantaAlpha 若改 RR，优先 clip/分位保护，不要叠除法。

### 2.3 体系角色（QuantaAlpha 假设标签建议沿用）

| role | 含义 | 基线代表 |
|------|------|----------|
| `entry` | 买点质量连续分 | `feilong_geo_squeeze_regime` |
| `exit` | 离场压力连续分 | `exit_pressure_additive` |
| `rr` | 盈亏比估计 | `rr_atr_regime` |
| `filter` | 提高纯度的过滤维 | **本会话后期几乎全灭**，默认高门槛 |

业务对齐（仓库内公式源，非臆造）：

- `jq_research_system/config/主图飞龙线`  
- `jq_research_system/config/附图筹码结构`  
- `jq_research_system/config/选股` / `指标观察`  
- 工程近似：`jqrs/factor_engine.py`、`jqrs/factors.py`

无通达信 `COST`/`WINNER` 时，必须用 **OHLCV 代理**（VWAP / day_cost 等），与 RD 相同。

---

## 3. 会话统计（经验强度）

| 项 | 数 |
|----|----|
| 完成反馈轮次 | ≈90（L90 未完成） |
| `decision=True` | **4**（L2/3/6/11） |
| `decision=False` | **86** |
| L12+ 主因 | 回测低于 SOTA（约 70+） |
| 编码失败跳过 `2_running` | L77、L87（及早期若干 NaN / no factor data） |
| 相似度拒收 | 至少 L42、L49 |

**阶段划分**：

| 阶段 | Loop | 特征 |
|------|------|------|
| 建库 | 0–11 | 体系从散到聚，最终锁三因子 |
| 同构撞墙 | 12–32 | squeeze/entry/wash/exit/rr 变体；偶发 IC 虚高 |
| 提示词转向 filter | 33–36 | 开始单/双 filter；双 ZT 回踩砸收益；编码失败 |
| 长跑空转 | 37–89 | 「SOTA+1 filter」成串死亡；偶发再碰 entry_cvma60 |
| 收工 | 90 | 人为停止 |

---

## 4. 完整证伪清单（QuantaAlpha 禁区 / 负样本）

> 原则：**按失败模式禁家族**，不要只禁单个因子名。  
> QuantaAlpha 的 mutation/crossover 若生成落入下表，应直接降权或拒收。

### 4.1 Entry / 飞龙·squeeze 同构（L12–32 及后期回潮）

禁止再发明主 entry 变体，尤其：

- sigmoid / 平滑 / 自适应窗口 squeeze  
- BB 位置×squeeze、双周期 squeeze、midpoint/VWAP+BB merge  
- 量冲击/量动量×squeeze、power-law / asymmetric squeeze  
- percentile geo-squeeze、squeeze 突破加速、通道位置 regime  
- VWAP 突破强度、动量加速×squeeze、回踩×VWAP 当主 entry  
- **`entry_cvma60*`**（L62–64、L89）：用折价/cvma60 抢 SOTA entry，全面砸收益  

代表：L7–10、L12–16、L18–23、L30–32、L62–64、L89。

### 4.2 Wash / meta 聚合

- 等权/排名四信号 meta-entry、IC 加权 wash  
- wash 乘积 / 阈值门 / 放大挂到飞龙核心上  

代表：L17、L24–30。  
说明：L2 的 `mean_reversion_wash` 是**早期进库台阶**，有 L11 后禁止再堆 wash 主 entry。

### 4.3 Exit 家族（相对加性基线未证明更好）

- 乘性 exit、sigmoid 离场  
- 区间速度/动量加速离场、compress→expand 转移  
- 各类 `exit_decay_*` / time_decay / volume_decay、VWAP 延伸衰减、progress_pressure  

代表：L4、L10、L13、L15–17、L20–23、L27–32。

### 4.4 RR / 波动缩放

- 波动 z **再当激进分母**、截面 std「银弹」、硬 gate regime RR  
- 未裁剪 support/target 比、exp_damp / vol_adaptive 反复磨  

代表：L12–17、L19–20、L22–23、L28、L32。  
注：基线 `rr_atr_regime` **本身含 vol_z**；失败的是「再叠一层/写反/不稳实现」，不是否定「要有 regime」。

### 4.5 Chip / BB 旁支 / RSI 窄幅放大

- chip 浓度/放量确认当主戏  
- BB squeeze rebound 当主 entry  
- RSI×飞龙窄幅放大器  

代表：L4–5、L7、L9、L18、L21；chip_skew 在 L33/35 再挂。

### 4.6 Filter 死族（L37–89 全军覆没，QuantaAlpha 默认禁止）

| 死族 | 代表 Loop |
|------|-----------|
| `filter_volume_pressure*` | 38–41 |
| OC / overnight 动量类 | 45–49 |
| `$factor` 调制 / 排名 / 高斯 conviction / ml_regime×cvma60 | 50–52、75、86、88 |
| `filter_intraday_*`（leverage/conviction/range/piecewise） | 53、59、70–74 |
| `filter_obv*` | 54–55 |
| `filter_volume_price_corr*` | 56–57 |
| `filter_bb_*` | 60–61、65–67 |
| squeeze 交互/转移类 filter | 51、68–69 |
| climax / post_momentum | 76–78（含 L77 编码失败） |
| zt_pullback / pullback_rebound 再版 | 34、58、79–80 |
| `filter_gap_*` | 81–83 |
| amihud / cost_basis / RSI-streak / dual-scale / ml_multi_signal | 37、42–44、85 |
| 双同质 filter 同轮（双胞胎） | 34、41、72、73、75、82 |

### 4.7 工程雷区

- 整轮收益/IC **NaN**（L0–1、L15、L23…）→ eps、clip、禁裸除  
- `No valid factor data` / `All tasks are failed` → 缺 `2_running` 仍写 feedback（L36、L44、L77、L87）  
- `rolling.apply` 内嵌重计算（极慢）  
- 一轮提 3 个因子整桌重开（L12 后伤害极大）  
- 双胞胎变体同轮（`*_narrow`/`*_slow`/`*_ewma`/`*_log`）

---

## 5. 方法论经验（给 QuantaAlpha 的操作原则）

### 5.1 挖矿策略

1. **先对齐基线**：用同一数据与相近评测复现 Loop11 三因子组合成绩，再谈进化超越。  
2. **冷启动可 2–3 角色**；**有 SOTA 后只允许 1 个增量**（或轨迹级「改失败步骤」而不是整桌重写）。  
3. **优先改 exit/rr 的一处结构**，不要用新 entry/filter 硬堆。  
4. **禁区要按家族维护**：每 5–8 轮 False 就归类进负样本库；提示词/规划器必须吃到负样本。  
5. **轨迹进化（QuantaAlpha 优势）应用在**：失败步骤定位、交叉复用「假设结构」；**不要**用来无限交叉已死 filter 家族。

### 5.2 RD vs QuantaAlpha（本仓库语境）

| | RD-Agent（已做完这轮） | QuantaAlpha（下一步） |
|--|------------------------|------------------------|
| 强项 | 方向提示词 + 假设→编码→回测闭环；本仓工具链成熟 | 轨迹级变异/交叉、多样化规划、经验复用 |
| 本会话暴露短板 | SOTA 后同构空转；禁区更新滞后会烧轮次 | — |
| 建议用法 | 基线与负样本已产出；可停 | **以本文为 brief + 负样本**；追求新结构，不重复死族 |
| 不建议 | 继续在本会话 resume 到 100+ | 忽略 Loop11、从随机动量冷启动 |

### 5.3 运维教训（若 QuantaAlpha 也长跑）

- **同一会话只跑一个挖矿进程**（曾 240/60 与 280/70 双开抢同一 `__session__`）。  
- 改研究说明 **不需要重生 Qlib/h5**；换 provider/数据才要。  
- 缺某一步（如只有 `1_coding`+`3_feedback`）常是编码失败跳过回测，不是日志损坏。  
- UI「All Loops」可能触发 Streamlit generator 错误；以状态目录 / 反馈 pkl 为准。  
- 过夜可挂，但必须有 **停手规则**；提示词应在长跑前更新，而不是跑完再补。

---

## 6. QuantaAlpha 建议开局 Brief（可直接粘贴）

```text
目标：在日频 OHLCV（对齐本仓库 Qlib cn_real_r500 或同等级 A 股研究池）上，
挖掘可支撑交易体系的连续因子：entry / exit / rr。

硬基线（必须先对齐再超越）：
- entry: feilong_geo_squeeze_regime
- exit:  exit_pressure_additive
- rr:    rr_atr_regime
- 锚点：IC≈0.026，含成本年化≈42.4%，回撤≈-31%

录取标准：含成本年化与回撤同时不差于基线；禁止仅 IC 改善。

禁止：飞龙/squeeze 同构新 entry；wash/meta 聚合；乘性 exit；
激进 vol_z 叠除；L37–89 全部 filter 死族（volume_pressure、OC、
$factor 调制、intraday、obv、bb filter、climax、zt_pullback、gap、
amihud、cvma60 抢 entry、双胞胎变体）。

鼓励：轨迹级改进 —— 对 exit/rr 做可解释的一步修正；或提出与基线
低相关的全新结构（必须论证正交性）。不要整桌重做三因子。

飞龙无 COST 时用 VWAP/EMA(89) 几何代理；窗口优先 5/10/20/60/89。
数值：eps、clip、有限 float；必须产出可回测因子文件。
```

---

## 7. 落地到本仓库交易栈（与 QuantaAlpha 并行）

挖矿框架产出后，验收应进：

1. **`jqrs` 因子注册 / DSL**：`entry`→选股排序，`exit`→卖出压力，`rr`→仓位/过滤辅助（非固定止损常量）。  
2. **PortfolioEngine**：禁止未来列；用真实交易成本与持仓约束，不单信 RD/QuantaAlpha 报表。  
3. **Workbench**：策略编译与信号，不依赖已移除的 Web 回测页。

Loop11 三因子是当前**唯一推荐移植对象**；L12–89 因子默认不移植。

---

## 8. 关键路径索引

| 资源 | 路径 |
|------|------|
| 本会话日志 | `log/2026-08-20_10-03-43-266516/` |
| Loop11 定义 | `.../Loop_11/direct_exp_gen/experiment generation/` |
| RD 提示词（已含部分禁区） | `config/rd_research_prompt.md` |
| 本基准文档 | `quantaalpha/docs/BASELINE_FROM_RD_AGENT.md` |
| 扫描脚本（可复跑） | `tmp/scan_rd_loops2.py` |
| 通达信/工程公式源 | `jq_research_system/config/*飞龙*`、`jqrs/factor_engine.py` |

---

## 9. 版本

| 字段 | 值 |
|------|-----|
| 文档版本 | 2026-08-22 |
| 数据会话 | `2026-08-20_10-03-43-266516` |
| 基线 Loop | **11** |
| 覆盖反馈 | Loop0–89（90 未完成） |
| 维护建议 | QuantaAlpha 每出现新死族，追加到 §4；若出现新 True 且通过 §1 录取标准，更新 §2 |
