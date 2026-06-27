"""Run the full stat-arb pipeline offline from data/prices.csv.

Produces:
  figures/prices.png, figures/spread_zscore.png, figures/equity_curve.png,
  figures/sensitivity_heatmap.png
  results/metrics.json   (all numbers cited in the paper / README)
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statarb as sa

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "prices.csv")
FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)

Y_COL, X_COL = "EWA", "EWC"
WINDOW = 30
ENTRY, EXIT = 2.0, 0.5
COST_BPS = 5.0

plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.autolayout": True})


def main():
    with open(os.path.join(ROOT, "data", "DATA_SOURCE.txt")) as f:
        source = f.read().strip()

    df = sa.load_prices(DATA)
    y, x = df[Y_COL], df[X_COL]

    coint = sa.run_cointegration(df, Y_COL, X_COL)
    beta, alpha = coint.hedge_ratio, coint.alpha

    spread = sa.build_spread(y, x, beta, alpha)
    z = sa.rolling_zscore(spread, WINDOW)
    pos = sa.generate_positions(z, ENTRY, EXIT)
    bt = sa.backtest(y, x, pos, beta, COST_BPS)

    # ---- Figure 1: normalised prices ----
    fig, ax = plt.subplots(figsize=(8, 4))
    (y / y.iloc[0]).plot(ax=ax, label=Y_COL)
    (x / x.iloc[0]).plot(ax=ax, label=X_COL)
    ax.set_title(f"Normalised prices: {Y_COL} vs {X_COL}")
    ax.set_ylabel("Price (indexed to 1.0)")
    ax.legend()
    fig.savefig(os.path.join(FIG, "prices.png"))
    plt.close(fig)

    # ---- Figure 2: spread + z-score ----
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    spread.plot(ax=a1, color="navy")
    a1.set_title(f"Spread  s_t = {Y_COL} - ({alpha:.3f} + {beta:.3f}*{X_COL})")
    a1.set_ylabel("spread")
    z.plot(ax=a2, color="darkgreen")
    a2.axhline(ENTRY, color="red", ls="--", lw=0.8)
    a2.axhline(-ENTRY, color="red", ls="--", lw=0.8)
    a2.axhline(EXIT, color="gray", ls=":", lw=0.8)
    a2.axhline(-EXIT, color="gray", ls=":", lw=0.8)
    a2.set_title(f"Rolling z-score (window={WINDOW}); entry=+/-{ENTRY}, exit=+/-{EXIT}")
    a2.set_ylabel("z-score")
    fig.savefig(os.path.join(FIG, "spread_zscore.png"))
    plt.close(fig)

    # ---- Figure 3: equity curve ----
    fig, ax = plt.subplots(figsize=(8, 4))
    bt.equity.plot(ax=ax, color="black")
    ax.set_title(f"Strategy equity curve (Sharpe={bt.sharpe:.2f}, "
                 f"tot={bt.total_return*100:.1f}%, maxDD={bt.max_drawdown*100:.1f}%)")
    ax.set_ylabel("equity (start=1.0)")
    fig.savefig(os.path.join(FIG, "equity_curve.png"))
    plt.close(fig)

    # ---- Sensitivity analysis: Sharpe over (entry, cost) grid ----
    entries = [1.0, 1.5, 2.0, 2.5, 3.0]
    costs = [0.0, 2.5, 5.0, 10.0, 20.0]
    sharpe_grid = np.zeros((len(entries), len(costs)))
    ret_grid = np.zeros((len(entries), len(costs)))
    for i, e in enumerate(entries):
        p = sa.generate_positions(z, e, EXIT)
        for j, c in enumerate(costs):
            b = sa.backtest(y, x, p, beta, c)
            sharpe_grid[i, j] = b.sharpe
            ret_grid[i, j] = b.total_return

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(sharpe_grid, cmap="RdYlGn", aspect="auto", origin="lower")
    ax.set_xticks(range(len(costs)))
    ax.set_xticklabels([f"{c:g}" for c in costs])
    ax.set_yticks(range(len(entries)))
    ax.set_yticklabels([f"{e:g}" for e in entries])
    ax.set_xlabel("transaction cost (bps/side)")
    ax.set_ylabel("z-score entry threshold")
    ax.set_title("Sensitivity: annualised Sharpe ratio")
    for i in range(len(entries)):
        for j in range(len(costs)):
            ax.text(j, i, f"{sharpe_grid[i,j]:.2f}", ha="center", va="center",
                    color="black", fontsize=8)
    fig.colorbar(im, ax=ax, label="Sharpe")
    fig.savefig(os.path.join(FIG, "sensitivity_heatmap.png"))
    plt.close(fig)

    metrics = {
        "data_source": source,
        "tickers": [Y_COL, X_COL],
        "start": str(df.index[0].date()),
        "end": str(df.index[-1].date()),
        "n_obs": int(len(df)),
        "params": {"window": WINDOW, "entry": ENTRY, "exit": EXIT, "cost_bps": COST_BPS},
        "cointegration": {
            "engle_granger_stat": round(coint.eg_stat, 4),
            "engle_granger_pvalue": round(coint.eg_pvalue, 6),
            "adf_spread_stat": round(coint.adf_stat, 4),
            "adf_spread_pvalue": round(coint.adf_pvalue, 6),
            "johansen_trace_stat": round(coint.johansen_trace_stat, 4),
            "johansen_trace_crit_95": round(coint.johansen_trace_crit_95, 4),
            "johansen_cointegrated": bool(coint.johansen_cointegrated),
            "hedge_ratio_beta": round(beta, 4),
            "ols_alpha": round(alpha, 4),
        },
        "backtest": {
            "sharpe": round(bt.sharpe, 4),
            "ann_return": round(bt.ann_return, 4),
            "total_return": round(bt.total_return, 4),
            "max_drawdown": round(bt.max_drawdown, 4),
            "n_trades": int(bt.n_trades),
            "hit_rate": round(bt.hit_rate, 4),
        },
        "sensitivity": {
            "entries": entries,
            "costs": costs,
            "sharpe_grid": [[round(v, 4) for v in row] for row in sharpe_grid.tolist()],
            "return_grid": [[round(v, 4) for v in row] for row in ret_grid.tolist()],
        },
    }
    with open(os.path.join(RES, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics["cointegration"], indent=2))
    print(json.dumps(metrics["backtest"], indent=2))
    print("source:", source)
    print("figures + results/metrics.json written")


if __name__ == "__main__":
    main()
