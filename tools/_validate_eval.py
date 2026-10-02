import logging; logging.basicConfig(level=logging.INFO)
import qlib
from qlib.data import D
import pandas as pd
qlib.init(provider_uri="D:/qqmt/qlib_data/cn_5m_full", region="cn")
codes = ["SH600000","SZ000001","SH601398","SH600519","SZ000002"]
raw = D.features(codes, ["$open","$high","$low","$close","$volume","$vwap"],
                start_time="2026-01-05", end_time="2026-09-30", freq="5min")
print("shape:", raw.shape)
print("cols:", list(raw.columns))
print("head:\n", raw.head(3))
print("levels:", raw.index.names)
# test build_factors import
import sys; sys.path.insert(0, r"D:/qqmt/QuantaAlpha/tools")
from eval_factors import build_factors
g = raw.reset_index(); g = g.rename(columns={g.columns[0]:"datetime", g.columns[1]:"instrument"})
g = g[g.instrument=="SH600000"].rename(columns={c:c.lstrip("$") for c in g.columns if c.startswith("$")})
g = g.sort_values("datetime").set_index("datetime")
f = build_factors(g, N=20)
print("factor cols:", list(f.columns))
print("F1 mean(nonzero frac):", f["F1"].mean(), "F3:", f["F3"].mean(), "F9:", f["F9"].mean())
print("OK")
