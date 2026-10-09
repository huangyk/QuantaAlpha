"""数据层：从 TimescaleDB 加载指数 / ETF 行情，统一错误处理与边界清洗。"""
import psycopg2
import pandas as pd

from .config import DB, CLEAN_SOURCE


class DataError(RuntimeError):
    """数据缺失 / 不足等可预期错误，便于上层捕获并给出清晰提示。"""


def get_conn():
    try:
        return psycopg2.connect(**DB)
    except Exception as e:  # 连接失败不应让栈追踪淹没用户
        raise DataError(f"无法连接 TimescaleDB {DB['host']}:{DB['port']}/{DB['dbname']}：{e}")


def load_ohlc(code: str, start: str = "2018-01-01", end: str = None) -> pd.DataFrame:
    """加载单标的 OHLCV。

    返回 DatetimeIndex + [open, high, low, close, volume]；按日期排序、
    去重(同日取末值)、丢弃 NaN 行。数据不足 60 行抛 DataError。
    """
    con = get_conn()
    try:
        if end:
            sql = ("SELECT trade_date, open, high, low, close, volume FROM bars_1d "
                   "WHERE asset_code=%s AND trade_date>=%s AND trade_date<=%s ORDER BY trade_date")
            cur = con.cursor(); cur.execute(sql, (code, start, end))
        else:
            sql = ("SELECT trade_date, open, high, low, close, volume FROM bars_1d "
                   "WHERE asset_code=%s AND trade_date>=%s ORDER BY trade_date")
            cur = con.cursor(); cur.execute(sql, (code, start))
        rows = cur.fetchall()
    finally:
        con.close()

    if not rows:
        raise DataError(f"无数据：{code}（start={start}）")
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")].dropna()
    if len(df) < 60:
        raise DataError(f"数据过少：{code} 仅 {len(df)} 行（需 >=60）")
    return df


def load_etf_universe(start: str = "2019-01-01", end: str = "2026-09-30",
                      clean_only: bool = True) -> list:
    """返回干净跨资产 ETF 代码列表（与七星高照研究同口径）。"""
    con = get_conn()
    try:
        if clean_only:
            sql = ("SELECT asset_code FROM bars_1d WHERE asset_type='etf' "
                   "GROUP BY asset_code HAVING count(distinct source)=1 "
                   "AND max(source)=%s ORDER BY 1")
            cur = con.cursor(); cur.execute(sql, (CLEAN_SOURCE,))
        else:
            sql = "SELECT distinct asset_code FROM bars_1d WHERE asset_type='etf' ORDER BY 1"
            cur = con.cursor(); cur.execute(sql)
        codes = [r[0] for r in cur.fetchall()]
    finally:
        con.close()
    if not codes:
        raise DataError("ETF 池为空，请检查 bars_1d 中 asset_type='etf' 数据")
    return codes


def load_multi(codes, start: str = "2018-01-01"):
    """批量加载多指数/标的，返回 {name_or_code: df}；单只失败不影响其余。"""
    out = {}
    for nm, code in codes if isinstance(codes[0], tuple) else [(c, c) for c in codes]:
        try:
            out[nm] = load_ohlc(code, start)
        except DataError as e:
            print(f"  [跳过] {nm}({code}): {e}")
    return out
