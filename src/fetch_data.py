"""Seed bundled CSV data. Tries yfinance for a cointegrated-ish ETF pair;
falls back to synthetic cointegrated series if the network fetch fails.

Run once to (re)generate data/prices.csv. All other code runs OFFLINE from that CSV.
"""
import sys, os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data")
os.makedirs(OUT, exist_ok=True)

TICKERS = ["EWA", "EWC"]
START = "2015-01-01"
END = "2024-12-31"


def try_yfinance():
    import yfinance as yf
    df = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise RuntimeError("empty frame")
    close = df["Close"].dropna()
    if close.shape[0] < 500 or close.shape[1] < 2:
        raise RuntimeError(f"insufficient data: {close.shape}")
    return close


def synthetic():
    """Shared stochastic trend + stationary (mean-reverting) spread => cointegrated."""
    rng = np.random.default_rng(42)
    n = 2500
    common = np.cumsum(rng.normal(0, 1.0, n)) + 50.0
    beta = 0.85
    spread = np.zeros(n)
    theta, sigma = 0.05, 0.6
    for t in range(1, n):
        spread[t] = spread[t - 1] * (1 - theta) + rng.normal(0, sigma)
    y = common + rng.normal(0, 0.3, n) + 10.0
    x = (common - spread) / beta + rng.normal(0, 0.3, n)
    dates = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame({"EWA": y, "EWC": x}, index=dates)


def main():
    source = "real"
    try:
        close = try_yfinance()
        print("yfinance OK:", close.shape)
    except Exception as e:  # noqa: BLE001
        print("yfinance FAILED:", repr(e), file=sys.stderr)
        close = synthetic()
        source = "synthetic"

    close.index.name = "Date"
    close = close[TICKERS]
    close.to_csv(os.path.join(OUT, "prices.csv"))
    with open(os.path.join(OUT, "DATA_SOURCE.txt"), "w") as f:
        f.write(source + "\n")
    print("WROTE", close.shape, "rows -> data/prices.csv ; source =", source)
    print(close.head())
    print(close.tail())


if __name__ == "__main__":
    main()
