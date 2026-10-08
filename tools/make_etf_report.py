# -*- coding: utf-8 -*-
"""生成 8万 ETF 方案 vs 8万自持股票 的回测对比报告（含净值曲线图）。"""
import pandas as pd
import numpy as np

ETF = pd.read_csv("data/results/etf_equity_curve.csv", index_col=0, parse_dates=True)
BASKET = pd.read_csv("data/results/equity_curve_q80000.csv", index_col=0, parse_dates=True)

ETF_RENAME = {
    "000852.SH": "中证1000ETF", "000905.SH": "中证500ETF",
    "399006.SZ": "创业板指ETF", "000300.SH": "沪深300ETF",
    "BLEND_5050": "50/50(中证500+1000)",
}
ETF = ETF.rename(columns=ETF_RENAME)

# 股票篮子取 P3 列（8万/季频/25只）
basket_col = "P3_1001-3000" if "P3_1001-3000" in BASKET.columns else BASKET.columns[0]

ACCT = 8e4

# ---- 汇总表 ----
rows = []
for col in ["中证1000ETF", "中证500ETF", "创业板指ETF", "沪深300ETF", "50/50(中证500+1000)"]:
    s = ETF[col].dropna()
    if len(s) == 0:
        continue
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    ann = (s.iloc[-1] / ACCT) ** (1 / yrs) - 1
    mdd = (s / s.cummax() - 1).min()
    rows.append((col, round(s.iloc[-1] / 1e4, 2), round((s.iloc[-1] / ACCT - 1) * 100, 1),
                 round(ann * 100, 1), round(mdd * 100, 1)))
bs = BASKET[basket_col].dropna()
byrs = (bs.index[-1] - bs.index[0]).days / 365.25
bann = (bs.iloc[-1] / ACCT) ** (1 / byrs) - 1
bmdd = (bs / bs.cummax() - 1).min()
rows.append(("❌ 自持25只股票(季频)", round(bs.iloc[-1] / 1e4, 2),
             round((bs.iloc[-1] / ACCT - 1) * 100, 1), round(bann * 100, 1), round(bmdd * 100, 1)))

# ---- 画图：对齐到共同日期范围 ----
common = ETF.index.union(BASKET.index)
series = {}
for col in ["中证1000ETF", "中证500ETF", "创业板指ETF", "沪深300ETF", "50/50(中证500+1000)"]:
    if col in ETF.columns:
        series[col] = ETF[col].reindex(common).interpolate().ffill().bfill()
series["❌ 自持25只股票"] = BASKET[basket_col].reindex(common).interpolate().ffill().bfill()

W, H = 860, 420
ml, mr, mt, mb = 60, 20, 30, 40
x0, x1 = ml, W - mr
y0, y1 = H - mb, mt
ymax = max([s.max() for s in series.values()]) * 1.05
ymin = 0
xs = common
def px(i):
    return x0 + (x1 - x0) * i / (len(xs) - 1)
def py(v):
    return y0 + (y1 - y0) * (v - ymin) / (ymax - ymin)

colors = {
    "中证1000ETF": "#2563eb", "中证500ETF": "#16a34a", "创业板指ETF": "#db2777",
    "沪深300ETF": "#f59e0b", "50/50(中证500+1000)": "#7c3aed",
    "❌ 自持25只股票": "#dc2626",
}
svg = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">']
# 网格 + Y轴
for g in range(0, 6):
    v = ymin + (ymax - ymin) * g / 5
    yy = py(v)
    svg.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}" stroke="#eee"/>')
    svg.append(f'<text x="{x0-8}" y="{yy+4:.1f}" font-size="11" fill="#666" text-anchor="end">{v/1e4:.0f}万</text>')
# X轴年份
for yr in range(2011, 2027, 3):
    sub = [i for i, d in enumerate(xs) if d.year == yr]
    if sub:
        i = sub[0]
        svg.append(f'<text x="{px(i):.1f}" y="{y0+18}" font-size="11" fill="#666" text-anchor="middle">{yr}</text>')
# 线段
for col, s in series.items():
    pts = " ".join(f"{px(i):.1f},{py(s.iloc[i]):.1f}" for i in range(len(s)))
    svg.append(f'<polyline points="{pts}" fill="none" stroke="{colors[col]}" stroke-width="2.2"/>')
svg.append(f'<text x="{x0}" y="{mt-10}" font-size="13" fill="#333">8万元 净值曲线：买ETF长期持有 vs 自己持25只股票</text>')
# 图例
ly = y1 + 14
for i, col in enumerate(series):
    lx = x0 + (i % 3) * 280
    lyy = ly + (i // 3) * 16
    svg.append(f'<rect x="{lx}" y="{lyy-9}" width="11" height="11" fill="{colors[col]}"/>')
    svg.append(f'<text x="{lx+15}" y="{lyy}" font-size="11" fill="#333">{col}</text>')
svg.append('</svg>')

# ---- 汇总表 HTML ----
tbl = "<tr><th>方案</th><th>期末(8万起)</th><th>总收益</th><th>年化</th><th>最大回撤</th></tr>"
for name, end, tot, ann, mdd in rows:
    cls = "bad" if "❌" in name else "good"
    tbl += f'<tr class="{cls}"><td>{name}</td><td>{end}万</td><td>{tot:+}%</td><td>{ann:+}%</td><td>{mdd:+}%</td></tr>'

html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>8万账户：ETF方案 vs 自持股票 回测对照</title>
<style>
*{{box-sizing:border-box}} body{{font-family:-apple-system,'Microsoft YaHei',sans-serif;margin:0;padding:24px;background:#f7f8fa;color:#1f2937}}
.wrap{{max-width:920px;margin:0 auto;background:#fff;padding:28px 32px;border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,.06)}}
h1{{font-size:21px;margin:0 0 4px}} .sub{{color:#6b7280;font-size:13px;margin-bottom:18px}}
.card{{background:#f0f7ff;border:1px solid #cfe3ff;border-radius:10px;padding:14px 16px;margin:16px 0;font-size:14px;line-height:1.7}}
svg{{width:100%;height:auto;background:#fafafa;border:1px solid #eee;border-radius:8px;margin:8px 0}}
table{{width:100%;border-collapse:collapse;margin-top:10px;font-size:14px}}
th,td{{padding:9px 10px;text-align:center;border-bottom:1px solid #eee}}
th{{background:#f3f4f6;font-weight:600}} tr.good td:first-child{{color:#15803d;font-weight:600}}
tr.bad{{background:#fef2f2}} tr.bad td:first-child{{color:#b91c1c;font-weight:700}}
.kpi{{display:flex;gap:12px;flex-wrap:wrap;margin:14px 0}}
.kpi div{{flex:1;min-width:150px;background:#f9fafb;border:1px solid #eee;border-radius:8px;padding:12px}}
.kpi b{{font-size:20px;display:block}} .warn{{background:#fff7ed;border-color:#fed7aa;color:#9a3412}}
ul{{line-height:1.8;font-size:14px}}
</style></head><body><div class="wrap">
<h1>8万账户：买ETF长期持有 vs 自己挑25只股票</h1>
<div class="sub">回测区间 2011-01 ~ 2026-09（15.7年）｜资金 8万｜成本已含最低5元佣金+印花税｜2026-10-08</div>

<div class="card"><b>一句话结论：</b>8万这笔钱，<b>买一只宽基小盘ETF长期持有</b>（年化 +2%~+6.5%，15年变 11万~21万）能稳稳拿到中盘流动性溢价；
而<b>自己挑25只股票</b>（季频调仓）被地板佣金+无法分散吃光，15年亏到 0.57万（<b>-93%</b>）。
两者用的是同一套"中盘/小盘溢价"逻辑，差别只在<b>资金规模够不够分散、有没有被最低5元佣金抽死</b>。</div>

<div class="kpi">
<div><b style="color:#16a34a">+2.1%</b>中证1000ETF年化</div>
<div><b style="color:#db2777">+6.5%</b>创业板指ETF年化</div>
<div><b style="color:#dc2626">-45.7%</b>自持25只年化(季频)</div>
</div>

{''.join(svg)}

<table>{tbl}</table>

<div class="card warn"><b>⚠️ 诚实预期（别被图骗了）：</b>
以上是指数<b>价格收益</b>。ETF实盘 = 指数价收 + 分红(~2%/年) − 管理费(~0.5%/年)，净约多 +1~1.5%/年；
即中证1000ETF 实盘约 <b>+3%~+3.5%/年</b>、创业板指约 <b>+7%~+8%/年</b>。
这<b>不是一年30-40%的战法</b>，是中盘/小盘市场的长期β敞口——但对你8万，这是唯一不被成本吃光、又能分散的做法。</div>

<h3 style="margin-top:22px">为什么自持股票会亏93%？（两个死穴）</h3>
<ul>
<li><b>死穴① 最低5元佣金：</b>8万买25只，每只约3200元，理论佣金0.8元，券商<span style="color:#b91c1c">最低收5元</span>→每只每轮被抽0.3%+；季频一年4轮，纯摩擦~1.3%，把正期望磨成负。</li>
<li><b>死穴② 无法分散：</b>机构持~2000只抹平个股风险，你25只等于把赌注压少数票，回撤被放大到-96%。盈亏比被成本压到&lt;1（0.92），必亏。</li>
</ul>

<h3>8万该怎么做（落地）</h3>
<ul>
<li><b>最省心：</b>开市买入 <b>创业板ETF(159915) 或 中证1000ETF(512100)</b> 一次，长期持有，别折腾。一年动1次足矣。</li>
<li><b>更稳：</b>中证500ETF(510500) + 中证1000ETF(512100) 各半仓，分散中盘风险（回撤-69% vs 单押-72%）。</li>
<li><b>仓位：</b>能扛住账户腰斩（-70%回撤）再全仓；扛不住先半仓（4万），留4万补仓。</li>
<li><b>何时卖：</b>指数跌破年线/或你判断牛市结束再减；平时持有拿分红和溢价。</li>
<li><b>数据说明：</b>DB里ETF价格未处理份额拆分（会虚增5倍），本报告改用库内拆干净的真实<b>指数序列</b>当ETF收益代理；akshare本次被代理拦截无法交叉校验，已注明。</li>
</ul>

<div class="sub" style="margin-top:18px">生成：tools/etf_backtest.py + tools/make_etf_report.py ｜ 净值数据：data/results/etf_equity_curve.csv</div>
</div></body></html>"""

with open("docs/etf_vs_stock_backtest.html", "w", encoding="utf-8") as f:
    f.write(html)
print("生成 docs/etf_vs_stock_backtest.html")
print("汇总:")
for name, end, tot, ann, mdd in rows:
    print(f"  {name:22s} 期末{end}万 总{tot:+}% 年化{ann:+}% 回撤{mdd:+}%")
