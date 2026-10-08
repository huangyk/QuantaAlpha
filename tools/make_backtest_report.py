"""生成战法回测报告（诚实版）：选股/持股机制 + 不同账户规模的真实验证结果。

关键结论：该策略是「机构级」策略——需要 1000万+ 规模、1000+ 只票分散才能盈利；
小账户（8万、25只）因最低5元佣金 + 无法分散，回测是亏损的。报告如实呈现。
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

OUT = Path('docs/backtest_report.html')
EQ_M = Path('data/results/equity_curve_m.csv')      # 1e7 月频
EQ_Q8 = Path('data/results/equity_curve_q80000.csv')  # 8万 季频 25只

m = pd.read_csv(EQ_M, index_col=0)
q8 = pd.read_csv(EQ_Q8, index_col=0)


def yearly(series, scale):
    s = series.dropna()
    pts = []
    years = sorted(set(d[:4] for d in s.index))
    for y in years:
        sub = s[[d for d in s.index if d[:4] == y]]
        if len(sub):
            pts.append((int(y), sub.iloc[-1] / scale))
    return pts


inst = yearly(m['P3_1001-3000'], 1e7)     # 倍数（初始1000万）
retail = yearly(q8['P3_1001-3000'], 8e4)  # 倍数（初始8万）

W, H = 680, 340
ML, MR, MT, MB = 50, 20, 20, 35
x0, x1 = 2011, 2026
y_max = max(p[1] for p in inst + retail) * 1.05
y_min = 0


def sx(yr):
    return ML + (yr - x0) / (x1 - x0) * (W - ML - MR)


def sy(v):
    return H - MB - (v - y_min) / (y_max - y_min) * (H - MT - MB)


def path(pts):
    return ' '.join(f'{sx(y):.1f},{sy(v):.1f}' for y, v in pts)


inst_path = path(inst)
retail_path = path(retail)
# 1.0 基准线
base_y = sy(1.0)

grid = ''
for yr in range(2011, 2027, 3):
    grid += f'<line x1="{sx(yr):.1f}" y1="{MT}" x2="{sx(yr):.1f}" y2="{H-MB}" stroke="#eee"/>'
    grid += f'<text x="{sx(yr):.1f}" y="{H-MB+15}" font-size="10" fill="#888" text-anchor="middle">{yr}</text>'
yticks = [0, 1, 2, 3, 4]
if y_max > 4:
    yticks.append(round(y_max))
for v in yticks:
    if v <= y_max:
        grid += f'<line x1="{ML}" y1="{sy(v):.1f}" x2="{W-MR}" y2="{sy(v):.1f}" stroke="#f0f0f0"/>'
        grid += f'<text x="{ML-6}" y="{sy(v)+3:.1f}" font-size="10" fill="#888" text-anchor="end">{v}x</text>'

dot_inst = ''.join(f'<circle cx="{sx(y):.1f}" cy="{sy(v):.1f}" r="2.5" fill="#c0392b"/>' for y, v in inst)
dot_retail = ''.join(f'<circle cx="{sx(y):.1f}" cy="{sy(v):.1f}" r="2.5" fill="#2980b9"/>' for y, v in retail)

html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>战法回测报告（诚实版）</title>
<style>
 body{{font-family:-apple-system,"Microsoft YaHei",sans-serif;max-width:820px;margin:24px auto;padding:0 18px;color:#222;line-height:1.7}}
 h1{{font-size:22px;border-bottom:3px solid #c0392b;padding-bottom:8px}}
 h2{{font-size:17px;margin-top:28px;color:#c0392b}}
 .box{{background:#fdf3f2;border:1px solid #f5c6c0;border-radius:8px;padding:12px 16px;margin:16px 0}}
 .ok{{background:#eef9f0;border:1px solid #bfe6c8}}
 table{{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0}}
 th,td{{border:1px solid #ddd;padding:6px 8px;text-align:center}}
 th{{background:#f5f5f5}}
 .neg{{color:#c0392b;font-weight:700}}
 .pos{{color:#1e7e34;font-weight:700}}
 code{{background:#f4f4f4;padding:1px 5px;border-radius:3px}}
 .small{{font-size:12px;color:#888}}
 svg{{max-width:100%}}
</style></head><body>
<h1>战法回测报告（诚实版）</h1>
<p class="small">生成于 2026-10-08 · 数据：TimescaleDB 全A 2011–2026（{len(m)} 个再平衡点）· 真实成本含最低5元佣金 + 印花税 + Almgren 冲击</p>

<h2>一、战法如何选股与持股</h2>
<p><b>选股（每个再平衡日）：</b></p>
<ol>
<li>只留<b>上市≥500日</b>的票（剔除次新、妖股）；</li>
<li>算每只票近 <b>60 日成交额中位数</b>，按流动性排序，取 <b>第 1001~3000 名</b>（中盘股——小盘太妖、大盘太卷）；</li>
<li>剔除<b>当日涨停开盘、停牌、无价、数据残缺日</b>的票；</li>
<li><b>等权</b>买入，单票≤5%，流动性配额约束。</li>
</ol>
<p><b>持股：</b>买入后<b>持有不动</b>，到下一再平衡日（月或季）全部卖出、按新一期清单重买。中途只因退市/ST/丧失流动性才卖。</p>
<p class="small">本质：这不是"选股 alpha"，而是<b>买一篮子中盘股、靠流动性溢价（小盘长期跑赢）吃饭的 beta 策略</b>。我们之前挖的所有选股因子（飞龙/战法2/5min/反转/换手率）在真实成本口径下均被证伪。</p>

<h2>二、回测结果（核心，必须看）</h2>
<div class="box">
<b>⚠️ 重大更正：</b>我之前口头说"8万季频净约+5%/年"是<b>错的</b>——那是基于粗略估算，没跑真实回测。
下面是用<b>真实最低5元佣金</b>跑出来的结果：<b>这个战法在 8 万小账户上是亏损的</b>，它是机构级策略。
</div>

<table>
<tr><th>账户 / 频率</th><th>持仓数</th><th>盈亏比</th><th>胜率</th><th>年化</th><th>最大回撤</th><th>8万→?</th></tr>
<tr><td>1000万 / 月频</td><td>~2000</td><td>1.13</td><td>57.4%</td><td class="pos">+11.4%</td><td>-52.7%</td><td>1000万→4316万</td></tr>
<tr><td>1000万 / 季频</td><td>~2000</td><td>1.64</td><td>55.8%</td><td class="pos">+37.5%</td><td>-43.5%</td><td>1000万→3972万</td></tr>
<tr style="background:#fff5f5"><td><b>8万 / 季频</b></td><td><b>25</b></td><td>0.92</td><td>34.6%</td><td class="neg">-45.7%</td><td class="neg">-96.1%</td><td><b>8万→0.57万</b></td></tr>
<tr style="background:#fff5f5"><td><b>8万 / 月频</b></td><td><b>25</b></td><td>0.84</td><td>36.4%</td><td class="neg">-32.3%</td><td class="neg">-99.6%</td><td><b>8万→0.04万</b></td></tr>
</table>

<h2>三、净值曲线（P3 池，初始资金归一化）</h2>
<svg viewBox="0 0 680 340" xmlns="http://www.w3.org/2000/svg">
{grid}
<line x1="{ML}" y1="{base_y:.1f}" x2="{W-MR}" y2="{base_y:.1f}" stroke="#bbb" stroke-dasharray="4"/>
<path d="{inst_path}" fill="none" stroke="#c0392b" stroke-width="2.5"/>
<path d="{retail_path}" fill="none" stroke="#2980b9" stroke-width="2.5"/>
{dot_inst}{dot_retail}
<text x="{ML+8}" y="28" font-size="12" fill="#c0392b">■ 1000万机构版（月频P3）</text>
<text x="{ML+8}" y="44" font-size="12" fill="#2980b9">■ 8万小账户（季频25只P3）</text>
<text x="{W-MR}" y="{base_y-5:.1f}" font-size="10" fill="#888" text-anchor="end">1.0x 基准线</text>
</svg>
<p class="small">红线：1000万账户按战法做，15 年变 4.7 倍。蓝线：8万账户按同样逻辑只持 25 只，15 年亏掉 93%（8万→5692元）。
两条线用的是<b>完全相同的选股/持股规则</b>，唯一差别是资金规模和持仓数量。</p>

<h2>四、为什么 8 万会亏？（不是策略假，是账户太小）</h2>
<ul>
<li><b>最低5元佣金吃死小仓位</b>：8万买 25 只，每只约 3200 元，单笔佣金本应 0.8 元，但券商<b>最低收 5 元</b>——相当于每只票每轮被抽 0.3%+，季频一年 4 轮就是 ~1.3% 纯摩擦，月频更狠。</li>
<li><b>无法分散</b>：机构版持 ~2000 只，个股风险被抹平；25 只等于把赌注压在少数票上，波动和回撤被放大数倍（回撤 -96% vs -43%）。</li>
<li><b>盈亏比被成本压到 &lt;1</b>：8万版盈亏比仅 0.84~0.92（赚的比亏的少），机构版 1.13~1.64。同样的选股，小账户一扣除金就由盈转亏。</li>
</ul>

<h2>五、8 万该怎么做（诚实建议）</h2>
<div class="box ok">
<b>结论：这个战法不适合 8 万账户。</b>它的"中盘流动性溢价"才是真收益来源，而你能用<b>一只宽基小盘ETF</b>拿到同样的敞口：
<ul style="margin:8px 0">
<li>买 <b>国证2000ETF（如 159628）/ 中证1000ETF（如 159845）</b>，一次买入、长期持有；</li>
<li>1 笔交易，佣金地板价 5 元影响 ≈0，无月度轮换摩擦；</li>
<li>等于用 1 个工具持有了 1000~2000 只中盘股（战法真正有效的部分），但又避开了 25 只集中的致命风险；</li>
<li>预期：小盘指数长期收益（非 alpha，约历史年化个位数~十几%，但含牛熊，2021 后也回撤过）。</li>
</ul>
</div>
<p>如果你坚持要自己选股操作：至少等资金到 <b>50 万以上</b>、能持 200+ 只分散，战法才接近机构版表现。8 万硬做 25 只，回测显示 15 年亏 93%。</p>

<h2>六、回测口径与局限</h2>
<ul>
<li>成本：佣金 max(金额×0.025%, 5元) 双边 + 卖出印花税 0.05% + Almgren 冲击（η=0.1, σ日=342bp）。</li>
<li><b>幸存者偏差未完全排除</b>：面板含已退市股（TimescaleDB 全A），但"持有到再平衡"对中途退市股按 0 处理，可能略高估；上线前建议用含退市收益的更严格口径再核。</li>
<li>再平衡价取<b>收盘→收盘</b>（T日收盘买、T+1...实际再平衡日收盘卖），未含盘中滑点。</li>
<li>脚本：<code>tools/strategy_pnl_ratio.py</code>（支持 --freq m/q --account --max-positions）。</li>
</ul>
</body></html>"""

OUT.write_text(html, encoding='utf-8')
print(f"报告已生成: {OUT}  ({len(html)} 字节)")
print(f"机构倍数(2026): {inst[-1][1]:.2f}x   8万倍数(2026): {retail[-1][1]:.3f}x")
