"""特征层：只负责从价格面板算出特征矩阵，不含标签与交易逻辑。

全部为「时点可得」特征：第 t 周只用 ≤t 的收盘/收益，天然无前视。
"""
import numpy as np
import pandas as pd

from .config import MIN_LIST_DAYS, EXCLUDE_ST


def weekly_returns(close_wk):
    """周频收益矩阵（第 t 周 = 收盘_t / 收盘_{t-1} - 1）。"""
    return close_wk.pct_change(fill_method=None)


def build_eligible_mask(weeks_idx, am, min_list_days=MIN_LIST_DAYS, exclude_st=EXCLUDE_ST):
    """时点可交易掩码：weeks × codes 布尔矩阵。

    条件：上市满 min_list_days、未退市、非ST（ST 仅当前标记，近似）。
    """
    codes = am["asset_code"].tolist()
    ld = am["list_date"].values.astype("datetime64[ns]")
    dd = am["delist_date"].values.astype("datetime64[ns]")
    st = am["is_st"].fillna(False).astype(bool).values
    W = weeks_idx.values.astype("datetime64[ns]")
    listed = ld[None, :] <= (W[:, None] - np.timedelta64(min_list_days, "D"))
    not_delisted = np.isnat(dd)[None, :] | (dd[None, :] > W[:, None])
    ok = listed & not_delisted
    if exclude_st:
        ok = ok & (~st)[None, :]
    return pd.DataFrame(ok, index=weeks_idx, columns=codes)


def trend_signal(close_wk, ma_weeks=40):
    """趋势信号：收盘 > N 周均线 -> True（对应 ETF 版的 200 日 SMA，40 周≈200 日）。"""
    ma = close_wk.rolling(ma_weeks).mean()
    return (close_wk > ma) & close_wk.notna()


def inverse_vol(rets_wk, vol_weeks=12, floor=0.05):
    """逆波动率权重（年化）= 1/max(σ, floor)，σ=周收益滚动std*sqrt(52)。"""
    vol = rets_wk.rolling(vol_weeks).std() * np.sqrt(52)
    return 1.0 / vol.clip(lower=floor)


def breadth(close_wk, eligible, ma_weeks=40):
    """市场宽度：时点池中「收盘 > N 周均线」的占比（0~1）。

    这是市场级择时信号（区别于个股级）：回答"当前有多少股票处于中期上升趋势"。
    分母只取时点池内且有有效收盘的股票。
    """
    ma = close_wk.rolling(ma_weeks).mean()
    above = (close_wk > ma) & eligible & close_wk.notna()
    denom = eligible & close_wk.notna()
    n = denom.sum(axis=1)
    n = n.where(n > 0)
    return above.sum(axis=1) / n


def equal_weight_index(close_wk, eligible):
    """时点池等权指数（周频净值），作为"长期持股"的市场基准。"""
    rets = weekly_returns(close_wk)
    m = eligible & rets.notna()
    n = m.sum(axis=1)
    n = n.where(n > 0)
    ew_ret = rets.where(m).sum(axis=1) / n
    return (1 + ew_ret.fillna(0.0)).cumprod()
