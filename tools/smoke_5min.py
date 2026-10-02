import os
os.environ.setdefault("QLIB_DATA_DIR", "D:/qqmt/qlib_data/cn_5m_test")
import sys
sys.path.insert(0, os.getcwd())

from quantaalpha.backtest.runner import BacktestRunner


def main():
    r = BacktestRunner("D:/qqmt/tmp_conf_5min_smoke.yaml")
    print(">>> 解析到的 freq =", r.freq)
    res = r.run(factor_source="alpha158_20")
    print(">>> RESULT KEYS:", list(res.keys()) if isinstance(res, dict) else type(res))
    for k in ("IC", "ICIR", "RankIC", "RankICIR", "annualized_return",
              "max_drawdown", "information_ratio", "sharpe"):
        if isinstance(res, dict) and k in res:
            print(f"    {k}: {res[k]}")


if __name__ == "__main__":
    main()
