"""数据层：时点股票池、日线矩阵、周频面板。

铁律：
- 只读 TimescaleDB bars_1d（QMT 回填），不引入 akshare 等替代源。
- 股票池为**时点口径**：按 list_date / delist_date 精确过滤，避免幸存者偏差与前视。
- ST 标记说明：asset_master.is_st 仅为**当前**状态，asset_status_history 表为空（无历史时点），
  故 ST 过滤是近似（会把"当年非ST、如今ST"的股票整段剔除），属已知偏差，已在报告中注明。
- 停牌：日线缺失即为停牌，计算指标前对价格做 ffill(limit=5)；收益/成交用原始 OHLC。
"""
import os

import numpy as np
import pandas as pd
import psycopg2

from .config import (DB, ASSET_TYPE, MIN_LIST_DAYS, EXCLUDE_ST, EXCLUDE_DELISTED,
                     START, END)


def load_asset_master():
    """载入股票主表（代码/上市日/退市日/ST/活跃）。"""
    con = psycopg2.connect(**DB)
    cur = con.cursor()
    cur.execute(
        "SELECT asset_code, list_date, delist_date, is_st, is_active "
        "FROM asset_master WHERE asset_type=%s", (ASSET_TYPE,))
    rows = cur.fetchall()
    con.close()
    df = pd.DataFrame(rows, columns=["asset_code", "list_date", "delist_date",
                                     "is_st", "is_active"])
    df["list_date"] = pd.to_datetime(df["list_date"])
    df["delist_date"] = pd.to_datetime(df["delist_date"])
    return df


def universe_at(am, as_of):
    """返回 as_of 时点可交易的股票代码列表（时点池）。

    条件：上市满 MIN_LIST_DAYS、未退市、非 ST（当前标记近似）。
    """
    as_of = pd.Timestamp(as_of)
    mask = am["list_date"].notna() & (am["list_date"] <= as_of - pd.Timedelta(days=MIN_LIST_DAYS))
    if EXCLUDE_DELISTED:
        mask &= am["delist_date"].isna() | (am["delist_date"] > as_of)
    if EXCLUDE_ST:
        mask &= ~am["is_st"].fillna(False).astype(bool)
    return sorted(am.loc[mask, "asset_code"].tolist())


def load_close_matrix(start=START, end=END, use_cache=True):
    """载入全市场股票收盘价宽表：index=trade_date, columns=asset_code。

    带本地 parquet/pickle 缓存（首次约 20s，后续秒开），便于研究员快速迭代。
    """
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "data", "cache")
    cache_path = os.path.join(cache_dir, "stock_close_matrix.pkl")
    if use_cache and os.path.exists(cache_path):
        return pd.read_pickle(cache_path)
    con = psycopg2.connect(**DB)
    cur = con.cursor()
    cur.execute(
        "SELECT trade_date, asset_code, close FROM bars_1d "
        "WHERE asset_type=%s AND trade_date>=%s AND trade_date<=%s",
        (ASSET_TYPE, start, end))
    rows = cur.fetchall()
    con.close()
    if not rows:
        raise ValueError("bars_1d 无股票数据")
    df = pd.DataFrame(rows, columns=["trade_date", "asset_code", "close"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    wide = df.pivot(index="trade_date", columns="asset_code", values="close")
    wide = wide.sort_index()
    try:
        os.makedirs(cache_dir, exist_ok=True)
        wide.to_pickle(cache_path)
    except Exception:
        pass
    return wide


def build_weekly_close(close_daily):
    """日线收盘宽表 -> 周频收盘（取每周最后一个交易日）。

    **关键**：剔除全市场无交易的空洞周（春节/国庆整周休市）。
    否则这些整行 NaN 的周会让 rolling(ma) 因 min_periods 不足而整片失效长达 40 周，
    导致策略被迫长期空仓（在市%剧烈跳变 0%↔100%）。剔除后滚动窗口在"实际交易周"上计算，更正确。
    """
    wk = close_daily.resample("W").last()
    empty = wk.notna().sum(axis=1) == 0
    return wk[~empty]
