"""功能 A：call tracker —— 把博主"关口/抄跌"判断按时间戳录入，自动打分。

目的：让"别人蛮准"从主观印象变成可证伪的数据。
用法：先 add() 录入每条判断（何时、哪个指数、什么方向、什么价位、给多久），
再 score() 按实际指数路径判定命中与否并汇总（命中率 / 平均前向收益）。
支持 JSON 持久化，方便跨会话累积。
"""
import json
import numpy as np
import pandas as pd

from .config import INDICES
from .io_db import load_ohlc, DataError

_NAME2CODE = dict(INDICES)
_CODE2NAME = {c: n for n, c in INDICES}
_DIRECTIONS = {"up", "down", "support", "resistance"}


class CallTracker:
    def __init__(self):
        self.calls = []

    # ---------- 录入 ----------
    def add(self, timestamp, index, direction, level, horizon_days, note=""):
        """录入一条判断。

        timestamp: '2026-09-17' 等可解析日期
        index:     指数名(如'创业板指')或代码(如'399006.SZ')
        direction: up / down / support / resistance
        level:     价位(浮点)
        horizon_days: 给多少交易日验证
        """
        code = _NAME2CODE.get(index, index)
        if code not in _CODE2NAME:
            raise ValueError(f"未知指数：{index}（可选：{', '.join(_NAME2CODE)}）")
        if direction not in _DIRECTIONS:
            raise ValueError(f"direction 须为 {sorted(_DIRECTIONS)}，收到 {direction}")
        if not (isinstance(level, (int, float)) and level > 0):
            raise ValueError("level 须为正浮点价位")
        if not (isinstance(horizon_days, int) and horizon_days > 0):
            raise ValueError("horizon_days 须为正整数")
        ts = pd.Timestamp(timestamp)
        self.calls.append(dict(ts=str(ts.date()), index=code, direction=direction,
                                level=float(level), horizon_days=horizon_days,
                                note=str(note)))

    # ---------- 打分 ----------
    def score(self, loader=load_ohlc):
        """逐条按实际行情判定命中，返回 (汇总dict, 逐条明细list)。"""
        if not self.calls:
            return dict(total=0, hits=0, hit_rate=0.0, avg_fwd_ret=0.0,
                        median_fwd_ret=0.0, by_direction={}), []
        detail = []
        for call in self.calls:
            code = call["index"]
            try:
                df = loader(code)
            except DataError as e:
                detail.append(dict(**call, ok=False, reason=f"数据缺失:{e}",
                                  correct=False, fwd_ret=None, touch=False))
                continue
            s = df["close"]
            # 入场交易日 = 首个 >= 判断日的交易日
            cand = s.index[s.index >= pd.Timestamp(call["ts"])]
            if len(cand) == 0:
                detail.append(dict(**call, ok=False, reason="判断日晚于数据末端",
                                  correct=False, fwd_ret=None, touch=False))
                continue
            entry_i = s.index.get_loc(cand[0])
            end_i = min(entry_i + call["horizon_days"], len(s) - 1)
            if end_i - entry_i < 3:
                detail.append(dict(**call, ok=False, reason="窗口不足",
                                  correct=False, fwd_ret=None, touch=False))
                continue
            win = s.iloc[entry_i:end_i + 1]
            entry_px = float(s.iloc[entry_i]); end_px = float(s.iloc[end_i])
            lv = call["level"]; dire = call["direction"]
            touch = ((win.min() <= lv) if dire in ("support", "down")
                     else (win.max() >= lv))
            fwd_ret = end_px / entry_px - 1
            if dire == "up":
                correct = win.max() >= lv
            elif dire == "down":
                correct = win.min() <= lv
            elif dire == "support":
                correct = (win.min() <= lv) and (end_px >= entry_px)
            else:  # resistance
                correct = (win.max() >= lv) and (end_px <= entry_px)
            detail.append(dict(**call, index_name=_CODE2NAME.get(code, code),
                              ok=True, correct=bool(correct), touch=bool(touch),
                              fwd_ret=fwd_ret, entry_px=entry_px, end_px=end_px))
        valid = [d for d in detail if d["ok"]]
        hits = sum(1 for d in valid if d["correct"])
        fwd = np.array([d["fwd_ret"] for d in valid if d["fwd_ret"] is not None])
        by_dir = {}
        for d in valid:
            by_dir.setdefault(d["direction"], []).append(d["correct"])
        by_dir = {k: dict(n=len(v), hits=sum(v), rate=float(sum(v) / len(v)))
                  for k, v in by_dir.items()}
        summary = dict(
            total=len(self.calls), scored=len(valid),
            hits=hits, hit_rate=float(hits / len(valid)) if valid else 0.0,
            avg_fwd_ret=float(fwd.mean()) if len(fwd) else 0.0,
            median_fwd_ret=float(np.median(fwd)) if len(fwd) else 0.0,
            by_direction=by_dir,
        )
        return summary, detail

    # ---------- 持久化 ----------
    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.calls, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        t = cls()
        with open(path, "r", encoding="utf-8") as f:
            t.calls = json.load(f)
        return t

    def __len__(self):
        return len(self.calls)
