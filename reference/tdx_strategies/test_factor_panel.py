"""De-risk: confirm AlphaAgent expr -> Qlib D.features translation works for correlation panels."""
import os, sys
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJ, ".env"))
import qlib
from qlib.data import D

qlib.init(provider_uri="~/.qlib/qlib_data/cn_real_full", region="cn")
inst = D.instruments("all")

# AlphaAgent expr translated to Qlib native syntax
exprs = {
    "f3": "Mean($close,20) > Mean($close,60)",
    "f1": "(($close > Max(Ref($close,1),20)) & ($close > Mean($close,60)))",
    "f6": "((Min($close,20) < Mean($close,60)*0.93) & ($close > Mean($close,20)) & ($volume > Mean($volume,5)*1.5))",
    "f8": "(($open < Ref($close,1)) & ($close > Ref($close,1)) & ($volume > Ref($volume,1)*1.2) & ($close > $open))",
    "f9": "(($close > $open) & ($close > Mean($close,5)) & ($volume > Ref($volume,1)*1.2))",
}
for k, e in exprs.items():
    df = D.features(inst, [e], start_time="2025-01-02", end_time="2026-08-17", freq="day")
    print(f"{k}: shape={df.shape} nan%={df.isna().mean().mean():.4f} unique_sample={sorted(df.iloc[:,0].dropna().unique())[:5]}")
print("TRANSLATION_OK")
