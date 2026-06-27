"""Statistical-arbitrage / pairs-trading core library.

Pure, deterministic functions operating on pandas/numpy. No network access.

Pipeline:
  1. cointegration tests (Engle-Granger ADF, Johansen trace)
  2. hedge ratio (OLS) and spread construction
  3. rolling z-score signal generation
  4. event-driven backtest with transaction costs
  5. performance statistics (Sharpe, returns, max drawdown)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

TRADING_DAYS = 252


def load_prices(csv_path):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = df.dropna().sort_index()
    return df


@dataclass
class CointResult:
    eg_pvalue: float
    eg_stat: float
    adf_stat: float
    adf_pvalue: float
    johansen_trace_stat: float
    johansen_trace_crit_95: float
    johansen_cointegrated: bool
    hedge_ratio: float
    alpha: float


def engle_granger(y, x):
    stat, pval, _ = coint(y, x)
    return float(stat), float(pval)


def ols_hedge_ratio(y, x):
    X = sm.add_constant(x.values)
    model = sm.OLS(y.values, X).fit()
    alpha, beta = float(model.params[0]), float(model.params[1])
    spread = y - (alpha + beta * x)
    return beta, alpha, spread


def adf_test(series):
    res = adfuller(series.dropna(), autolag="AIC")
    return float(res[0]), float(res[1])


def johansen(df):
    jres = coint_johansen(df.values, det_order=0, k_ar_diff=1)
    trace_stat = float(jres.lr1[0])
    crit_95 = float(jres.cvt[0, 1])
    return trace_stat, crit_95, trace_stat > crit_95


def run_cointegration(df, y_col, x_col):
    y, x = df[y_col], df[x_col]
    eg_stat, eg_pval = engle_granger(y, x)
    beta, alpha, spread = ols_hedge_ratio(y, x)
    adf_stat, adf_pval = adf_test(spread)
    jt_stat, jt_crit, jt_coint = johansen(df[[y_col, x_col]])
    return CointResult(
        eg_pvalue=eg_pval, eg_stat=eg_stat,
        adf_stat=adf_stat, adf_pvalue=adf_pval,
        johansen_trace_stat=jt_stat, johansen_trace_crit_95=jt_crit,
        johansen_cointegrated=jt_coint,
        hedge_ratio=beta, alpha=alpha,
    )


def build_spread(y, x, beta, alpha=0.0):
    return y - (alpha + beta * x)


def rolling_zscore(spread, window=30):
    mean = spread.rolling(window).mean()
    std = spread.rolling(window).std(ddof=0)
    z = (spread - mean) / std
    return z


def generate_positions(zscore, entry=2.0, exit=0.5):
    """Convert z-score into target position on the SPREAD.

    +1 long spread (long y, short x) when z <= -entry
    -1 short spread when z >= +entry
     0 flat when band re-crossed (|z| <= exit). Stateful between crossings.
    """
    z = zscore.values
    pos = np.zeros(len(z))
    state = 0
    for i in range(len(z)):
        zi = z[i]
        if np.isnan(zi):
            pos[i] = 0
            state = 0
            continue
        if state == 0:
            if zi >= entry:
                state = -1
            elif zi <= -entry:
                state = 1
        elif state == 1:
            if zi >= -exit:
                state = 0
        elif state == -1:
            if zi <= exit:
                state = 0
        pos[i] = state
    return pd.Series(pos, index=zscore.index, name="position")


@dataclass
class BacktestResult:
    returns: pd.Series = field(repr=False)
    equity: pd.Series = field(repr=False)
    positions: pd.Series = field(repr=False)
    sharpe: float = 0.0
    ann_return: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    n_trades: int = 0
    hit_rate: float = 0.0


def backtest(y, x, positions, beta, cost_bps=5.0):
    """Event-driven backtest of a dollar-neutral spread position.

    Strategy return at t = position at t-1 applied to spread log-return t-1->t.
    Costs (cost_bps of traded notional per side) charged when position changes.
    """
    ry = np.log(y).diff()
    rx = np.log(x).diff()
    w = 1.0 / (1.0 + abs(beta))
    spread_ret = w * (ry - beta * rx)

    pos = positions.reindex(spread_ret.index).fillna(0.0)
    pos_lag = pos.shift(1).fillna(0.0)

    gross = pos_lag * spread_ret

    turnover = (pos - pos_lag).abs() * (w * (1.0 + abs(beta)))
    cost = turnover * (cost_bps / 1e4)

    net = (gross - cost).fillna(0.0)
    equity = (1.0 + net).cumprod()

    mu = net.mean()
    sigma = net.std(ddof=0)
    sharpe = float(np.sqrt(TRADING_DAYS) * mu / sigma) if sigma > 0 else 0.0
    total_return = float(equity.iloc[-1] - 1.0)
    n = len(net)
    ann_return = float(equity.iloc[-1] ** (TRADING_DAYS / n) - 1.0) if n > 0 else 0.0
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    max_dd = float(dd.min())

    trade_pnls = []
    cur = 0.0
    in_trade = False
    for p, r in zip(pos_lag.values, net.values):
        if p != 0:
            cur += r
            in_trade = True
        else:
            if in_trade:
                trade_pnls.append(cur)
                cur = 0.0
                in_trade = False
    if in_trade:
        trade_pnls.append(cur)
    n_trades = len(trade_pnls)
    hit = float(np.mean([1.0 if t > 0 else 0.0 for t in trade_pnls])) if trade_pnls else 0.0

    return BacktestResult(
        returns=net, equity=equity, positions=pos,
        sharpe=sharpe, ann_return=ann_return, total_return=total_return,
        max_drawdown=max_dd, n_trades=n_trades, hit_rate=hit,
    )
