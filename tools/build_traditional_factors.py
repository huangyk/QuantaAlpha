#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成「传统因子库」—— Alpha101 中纯价量可算的子集 + 经典技术因子（动量/反转/波动/量价/流动性/价格效率）。

约束（与 5min 挖掘一致）：
- 只用 qlib 算子：TS_/Ref/WMA/Mean/Std/Sum/Min/Max/Log/Abs/Sign（归一化后 = Mean/Std/.../Ref/WMA/Rank/Log/Abs/Sign）
- 窗口只取 {12,24,48,96,240}（12=1小时 / 24=2小时 / 48=1交易日 / 96=2天 / 240=1周）
- 禁止三元 ?:、禁止 Corr（不在白名单）、禁止基本面（无 cap/PE/PB）
- 每个因子附一句中文直觉

输出两份：
- <out_json>           回测用：{"factors": {id: {"factor_name","factor_expression"}}}
- <out_pool>           验证用伪轨迹池：每条因子一个 original 轨迹（backtest_metrics.RankIC=null），
                      供 tools/oos_validate.py --all 单因子复验。
"""
import argparse
import json
from pathlib import Path

# 窗口
W = [12, 24, 48, 96, 240]

# 因子定义： (name, expr_template, 中文直觉)
# 模板用 {w} 占位窗口；{r} 表示单根（用于不需窗口的）
defs = []

# ---- 1. 动量 / 反转（过去 N 根累计收益）----
for w in [12, 24, 48, 96, 240]:
    defs.append((f"TRAD_MOM_{w}", f"TS_MEAN($return,{w})",
                 f"过去{w}根(={w//48 if w>=48 else round(w/48,2)}交易日)累计收益，动量"))
for w in [12, 48, 240]:
    defs.append((f"TRAD_REV_{w}", f"-1*TS_MEAN($return,{w})",
                 f"过去{w}根收益取负=反转（近期涨多了卖）"))
for w in [24, 48, 96]:
    defs.append((f"TRAD_DIRCONS_{w}", f"TS_SUM(SIGN($return),{w})",
                 f"过去{w}根中上涨根数-下跌根数=方向一致性（趋势纯度）"))

# ---- 2. 波动率 / 振幅 ----
for w in [12, 48, 96, 240]:
    defs.append((f"TRAD_RVOL_{w}", f"TS_STD($return,{w})",
                 f"过去{w}根收益标准差=已实现波动"))
for w in [48, 96]:
    defs.append((f"TRAD_GARKLINE_{w}", f"TS_MEAN(Log($high/$low),{w})",
                 f"过去{w}根平均 Log(高/低)=Garman-Klass 波动代理"))
for w in [48, 240]:
    defs.append((f"TRAD_RANGERATIO_{w}", f"TS_MEAN(($high-$low)/$close,{w})",
                 f"过去{w}根平均 (高-低)/收盘=相对振幅"))

# ---- 3. 量能 ----
for w in [12, 48, 240]:
    defs.append((f"TRAD_VOLRATIO_{w}", f"$volume/TS_MEAN($volume,{w})",
                 f"当前量 / 过去{w}根均量=量能放大倍数"))
for w in [48, 96]:
    defs.append((f"TRAD_OBV_MOM_{w}", f"TS_SUM(SIGN($return)*$volume,{w})",
                 f"过去{w}根 量加权净方向（涨时放量=吸筹）"))
for w in [48, 240]:
    defs.append((f"TRAD_AMIHUD_{w}", f"TS_MEAN(ABS($return)/($volume+1),{w})",
                 f"过去{w}根 |收益|/量=Amihud 非流动性（值大=薄成交易波动）"))

# ---- 4. 价格效率（传统版，对照我们 F65 簇）----
for w in [48, 96]:
    defs.append((f"TRAD_EFF_CO_{w}", f"TS_MEAN(ABS($close-$open)/($high-$low),{w})",
                 f"过去{w}根 实体/振幅=日内价格效率（贴近 F65 传统版）"))
for w in [48, 96]:
    defs.append((f"TRAD_WILLIAMS_{w}",
                 f"($close-TS_MIN($low,{w}))/(TS_MAX($high,{w})-TS_MIN($low,{w}))",
                 f"过去{w}根 收盘在高低区间的位置（Williams %R 正向）"))
for w in [48]:
    defs.append((f"TRAD_GAP_{w}", f"($close-TS_MEAN($open,{w}))/TS_MEAN($close,{w})",
                 f"收盘相对过去{w}根开盘均值的偏离=跳空延续"))

# ---- 5. 均值回归（价 vs MA）----
for w in [48, 96, 240]:
    defs.append((f"TRAD_MADEV_{w}", f"($close-TS_MEAN($close,{w}))/TS_MEAN($close,{w})",
                 f"收盘偏离{w}根均线幅度=均值回归压力"))
defs.append((f"TRAD_WMADEV_48", f"($close-WMA($close,48))/WMA($close,48)",
            "收盘偏离48根加权均线=趋势偏离"))

# ---- 6. 截面排名组合（Alpha101 风格：Rank 价差）----
for w in [48, 96]:
    defs.append((f"TRAD_RANKCOMBO_{w}",
                 f"TS_MEAN(Rank($close)-Rank($volume),{w})",
                 f"过去{w}根 价格排名-量排名 的时序均值（高价低量=被看好）"))


def build(out_json: str, out_pool: str):
    factors = {}
    traj = {}
    for i, (name, expr, note) in enumerate(defs):
        fid = f"trad_{i:03d}"
        factors[fid] = {
            "factor_name": name,
            "factor_expression": expr,
            "intuition": note,
        }
        traj[fid] = {
            "hypothesis": note,
            "factors": [{"name": name, "expression": expr}],
            "backtest_metrics": {"RankIC": None},
            "phase": "original",
            "direction_id": "trad",
            "created_at": "2026-10-06T15:30:00",
        }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"factors": factors}, open(out_json, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump({"trajectories": traj, "by_direction": {}, "by_phase": {}},
              open(out_pool, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"传统因子库：{len(defs)} 条")
    print(f"  回测 JSON : {out_json}")
    print(f"  验证伪池 : {out_pool}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default="data/results/traditional_factors_library.json")
    ap.add_argument("--out-pool", default="log/_pool_traditional_tmp.json")
    a = ap.parse_args()
    build(a.out_json, a.out_pool)
