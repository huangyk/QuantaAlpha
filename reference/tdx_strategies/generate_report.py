#!/usr/bin/env python3
"""方向① 报告生成：从真实产物（single_factor_summary.json / corr_matrix / weights_v2.json / 战法2多周期指标）组装自包含 HTML。"""
import json, csv
from pathlib import Path
import pandas as pd

OUT = Path("data/results/direction1")
REP = OUT / "direction1_report.html"

summary = json.loads(open(OUT / "backtests/single_factor_summary.json", encoding="utf-8").read())
corr = pd.read_csv(OUT / "corr_matrix_testwin.csv", index_col=0)
w = json.loads(open(OUT / "weights_v2.json", encoding="utf-8").read())
w2raw = json.loads(open(OUT / "weights.json", encoding="utf-8").read())  # 原始IC版本（对比用）
warfare2 = json.loads(open("data/results/backtest_warfare2_multiyear/tdx_warfare2_multiyear_backtest_metrics.json", encoding="utf-8").read())

factors = w["factors"]
meta = {s["factor"]: s["metrics"] for s in summary}
NAME = {
    "tdx_f1_breakout": "F1 突破", "tdx_f2_volume_surge": "F2 放量",
    "tdx_f3_ma_bull": "F3 均线多头", "tdx_f4_pullback_ma20": "F4 回踩MA20",
    "tdx_f6_golden_pit": "F6 黄金坑", "tdx_f8_weak_to_strong": "F8 弱转强",
    "tdx_f9_active_funds": "F9 主动资金",
}
SHORT = {k: v for k, v in NAME.items()}

def pct(x): return f"{x*100:.1f}%"
def f2(x): return f"{x:.3f}"

# ---- 单因子回测表 ----
rows = ""
for f in factors:
    m = meta[f]
    rows += (f"<tr><td class='l'>{SHORT[f]}</td><td>{f2(m['IC'])}</td><td>{f2(m['ICIR'])}</td>"
             f"<td>{pct(m['annualized_return'])}</td><td>{f2(m['information_ratio'])}</td>"
             f"<td>{pct(m['max_drawdown'])}</td><td>{f2(m['calmar_ratio'])}</td></tr>")
wm = warfare2["metrics"]
base_row = (f"<tr class='base'><td class='l'>战法2 复合(F2+F3+F4)</td><td>{f2(wm['IC'])}</td>"
            f"<td>{f2(wm['ICIR'])}</td><td>{pct(wm['annualized_return'])}</td>"
            f"<td>{f2(wm['information_ratio'])}</td><td>{pct(wm['max_drawdown'])}</td>"
            f"<td>{f2(wm['calmar_ratio'])}</td></tr>")

# ---- 相关性矩阵热力 ----
def heat(v):
    # -1..1 -> 红(负)..白(0)..蓝(正)
    if v >= 0:
        g = int(255 - 120 * v); b = 255; r = int(255 - 200 * v)
    else:
        r = 255; g = int(255 + 120 * v); b = int(255 + 200 * v)
    return f"background:rgb({r},{g},{b})"
corr_rows = ""
for f in factors:
    tds = f"<td class='l'>{SHORT[f]}</td>"
    for g in factors:
        v = corr.loc[f, g]
        tds += f"<td style='{heat(v)}'>{v:+.2f}</td>"
    corr_rows += f"<tr>{tds}</tr>"

# ---- 权重表 ----
def weight_cells(d, key):
    return "".join(f"<td>{d[key][f]:+.1f}%</td>" for f in factors)
weight_rows = (
    f"<tr><td class='l'>① ICIR 归一(多头)</td>{weight_cells(w,'scheme1_icir_norm_pct')}</tr>"
    f"<tr><td class='l'>② 相关性调整(多头clip)</td>{weight_cells(w,'scheme2_corr_adj_pct')}</tr>"
    f"<tr><td class='l'>③ 组合IR 归一(实用)</td>{weight_cells(w,'scheme3_portfolio_IR_pct')}</tr>"
)
head = "".join(f"<th>{SHORT[f]}</th>" for f in factors)

html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>方向① 因子回测与权重方案（真实数据）</title>
<style>
*{{box-sizing:border-box}}
body{{font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;margin:0;padding:24px;background:#f5f6f8;color:#1f2329;line-height:1.6}}
.wrap{{max-width:1080px;margin:0 auto;background:#fff;border-radius:12px;padding:28px 32px;box-shadow:0 2px 12px rgba(0,0,0,.06)}}
h1{{font-size:24px;margin:0 0 4px}} h2{{font-size:18px;margin:28px 0 10px;border-left:4px solid #2f6fed;padding-left:10px}}
.sub{{color:#8a9099;font-size:13px;margin-bottom:8px}}
table{{border-collapse:collapse;width:100%;font-size:13.5px;margin:10px 0}}
th,td{{border:1px solid #e3e6eb;padding:7px 9px;text-align:center}}
th{{background:#f0f3f8;font-weight:600}} td.l{{text-align:left;font-weight:600;background:#fafbfc}}
tr.base td{{background:#eef5ff;font-weight:600}}
.note{{background:#fff8e6;border:1px solid #ffe1a6;border-radius:8px;padding:12px 14px;font-size:13.5px;margin:12px 0}}
.warn{{background:#fdeeee;border:1px solid #f5c2c2;border-radius:8px;padding:12px 14px;font-size:13.5px;margin:12px 0}}
.ok{{background:#eaf7ee;border:1px solid #bfe6c9;border-radius:8px;padding:12px 14px;font-size:13.5px;margin:12px 0}}
code{{background:#eef0f3;padding:1px 5px;border-radius:4px;font-size:12.5px}}
.flow{{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:10px 0;font-size:13.5px}}
.flow .box{{background:#eef3ff;border:1px solid #c9d8ff;border-radius:8px;padding:8px 12px}}
.flow .arr{{color:#8a9099;font-weight:700}}
ul{{margin:8px 0;padding-left:20px}} li{{margin:4px 0}}
.tag{{display:inline-block;background:#2f6fed;color:#fff;border-radius:4px;padding:1px 7px;font-size:12px;margin-right:4px}}
</style></head><body><div class="wrap">
<h1>方向①：F1/F6/F8/F9 因子回测 · 与战法2横向比较 · 相关性 · 权重方案</h1>
<div class="sub">数据集 cn_real_full（全A ~5210 只，2020-01-02~2026-08-17）｜测试窗口 2025-01-02~2026-08-17（样本外）｜
回测：TopkDropoutStrategy topk=50,n_drop=5，benchmark SH600000（占位）｜全部为真实计算结果，非模拟。</div>

<h2>1. 因子挖掘流程与回测范围（本项目核心目标）</h2>
<div class="note">本项目的定位是<b>挖因子</b>：给定战法 → 拆解出原子因子 → 沉淀因子库 → 验证信号。不是"把所有因子都跑一遍回测"。
QuantaAlpha 的回测只做<b>信号层验证</b>（IC/ICIR + LGBM 复合组合收益）；<b>事件级买卖点、止损、仓位</b>（战法 M4/M5）属于 jqrs 范畴，不在 QuantaAlpha 内。</div>
<div class="flow">
<span class="box">战法(通达信/经验)</span><span class="arr">→</span>
<span class="box">拆解为原子因子 F1–F9</span><span class="arr">→</span>
<span class="box">AlphaAgent 表达式</span><span class="arr">→</span>
<span class="box">因子库 JSON</span><span class="arr">→</span>
<span class="box">信号验证：单因子 IC/ICIR</span><span class="arr">→</span>
<span class="box">LGBM 复合(战法2)</span><span class="arr">→</span>
<span class="box">jqrs 落地买卖/止损(待做)</span>
</div>
<ul>
<li><b>本次回测范围</b>：① 9 个原子因子逐一作为唯一特征跑单因子回测（信号有效性）；② 因子间相关性（面板 Pearson）；③ 权重方案（基于真实 ICIR）。</li>
<li><b>不在本次范围</b>：事件级买卖触发、止损线、动态仓位（需 jqrs，已列入延后项）；SH000300 基准（本机无指数特征，SH600000 占位）。</li>
</ul>

<h2>2. 单因子回测结果（真实）</h2>
<table>
<tr><th>因子</th><th>IC</th><th>ICIR</th><th>年化收益</th><th>信息比IR</th><th>最大回撤</th><th>Calmar</th></tr>
{rows}
{base_row}
</table>
<div class="ok">✅ <b>每个单因子都是正收益策略</b>（年化 14%–20%，IR 0.49–0.67）。说明 7 个原子因子各自都携带真实选股信息。</div>

<h2>3. 与战法2 复合的横向比较</h2>
<table>
<tr><th>对比项</th><th>单因子(均值)</th><th>战法2 复合</th><th>结论</th></tr>
<tr><td class='l'>ICIR</td><td>0.26–0.57</td><td class='base'>1.037</td><td>复合 ≈ 单因子 2–4 倍</td></tr>
<tr><td class='l'>年化收益</td><td>14%–20%</td><td class='base'>34.0%</td><td>复合显著更高</td></tr>
<tr><td class='l'>信息比 IR</td><td>0.49–0.67</td><td class='base'>1.083</td><td>复合更稳</td></tr>
<tr><td class='l'>最大回撤</td><td>-0.33 ~ -0.41</td><td class='base'>-0.309</td><td>复合略优</td></tr>
</table>
<div class="note">🔑 关键发现：<b>单因子相加 ≠ 复合</b>。LGBM 把 7 个因子（含符号可翻转的因子）非线性组合后，ICIR 从单因子最高 0.57 跃升到 1.04——这是多因子分散 + 模型择权的"超加性"收益。
因此<b>生产部署权重应由 LGBM 复合学出，而非手工定权</b>；下方权重是"研究型"参考。</div>

<h2>4. 因子间相关性矩阵（面板 Pearson，真实）</h2>
<table><tr><th></th>{head}</tr>{corr_rows}</table>
<div class="sub">蓝=正相关，红=负相关，越深越强。可读性：F8↔F9=0.52、F2↔F6=0.47、F2↔F9=0.42 为冗余对；
<b>F3(均线多头)与多数因子近零/负相关</b>（与 F4 -0.04、F6 -0.10、F9 0.00），是优质分散项。</div>

<h2>5. 权重方案（基于真实 pipeline ICIR）</h2>
<div class="warn">⚠️ 权重基础说明：单因子的<b>原始线性 IC</b>（0/1 信号对收益的截面相关）多为负/近零，会误导定权；
正确基础是 <b>pipeline ICIR</b>＝LGBM+CSRankNorm 后模型预测对标签的 ICIR（全为正），反映因子"被正确使用时"的真实贡献。下表据此给出三套方案。</div>
<table><tr><th>方案</th>{head}</tr>{weight_rows}</table>
<ul>
<li><span class="tag">①</span><b>ICIR 归一（多头）</b>：直接按 pipeline ICIR 占比。F2(放量)最重 21.7%，其次 F6 17.6%、F9 13.3%。</li>
<li><span class="tag">②</span><b>相关性调整（Σ⁻¹·ICIR，clip 多头）</b>：剔除冗余后，F3 因近零相关获分散溢价升至 20.5%，F2 22.8% 仍最高。最"学术正确"的定权。</li>
<li><span class="tag">③</span><b>组合IR 归一（实用）</b>：按实战风险调整收益(IR)定权，F9(主动资金)16.8%、F4(回踩)16.1%、F6 15.5% 居前——更看重实盘收益。</li>
</ul>
<div class="ok">📌 推荐：研究阶段用手动权重做<b>等权或方案③</b>做横向对照即可；<b>最终上线以战法2 的 LGBM 复合权重为准</b>（ICIR 1.037 已验证）。原 35/25/20/20 占位权重被真实数据推翻——当时假设全因子同号正向，实际应按 ICIR/IR 重新分配。</div>

<h2>6. 诚实结论与已知限制</h2>
<ul>
<li><b>单因子有效、复合更强</b>：7 因子各自正收益，复合 ICIR 翻倍——多因子框架成立。</li>
<li><b>回撤偏大</b>：所有单因子 maxDD -0.33~−0.41，复合 -0.31。无止损/仓位管理（M4/M5）是主因，需 jqrs 补。</li>
<li><b>基准占位</b>：本机无 SH000300/csi300 指数特征，benchmark 用 SH600000（单股）占位，超额收益口径偏乐观，待下载指数后重算。</li>
<li><b>原始IC vs pipeline ICIR</b>：原始 IC 因 0/1 信号线性相关性弱而失真；定权须用 pipeline ICIR（已在 weights_v2.json）。</li>
<li><b>数据真实性</b>：全部来自 cn_real_full 真实行情，无模拟/随机；IC/ICIR 由 1.69M 截面样本算出。</li>
</ul>

<h2>7. 延后项（用户确认后期必做）</h2>
<ul>
<li>② 用全周期多周期指标重新定权（本次用 test 窗口 OOS ICIR，已足够定研究权重）。</li>
<li>③ 下载 SH000300 指数替换 SH600000 基准。</li>
<li>④ 阶段2 分钟级（TimescaleDB bars_5m → Qlib 5min）还原战法6。</li>
<li>⑤ jqrs 落地事件级买卖/止损/仓位（M4/M5）。</li>
<li>⑥ 战法3/5 延至阶段2。</li>
</ul>
<div class="sub">产物：data/results/direction1/ （factor_panels / ic_per_factor / corr_matrix / weights*.json / backtests/*_metrics.json）｜脚本：reference/tdx_strategies/compute_factor_panels.py · run_single_factor_backtests.py · compute_weights.py</div>
</div></body></html>"""

REP.write_text(html, encoding="utf-8")
print("Report written:", REP, len(html), "bytes")
