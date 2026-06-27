"""Unit tests for the stat-arb library: signals, backtest accounting, cointegration."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import statarb as sa


# --------------------------------------------------------------------------- #
# Signal generation: z-score crossing logic                                   #
# --------------------------------------------------------------------------- #
def test_positions_enter_short_on_high_z():
    z = pd.Series([0.0, 1.0, 2.5, 1.0, 0.4, 0.0])
    pos = sa.generate_positions(z, entry=2.0, exit=0.5)
    # crosses +2 at idx2 -> short spread (-1), held until z<=exit (idx4)
    assert pos.iloc[2] == -1
    assert pos.iloc[3] == -1          # still above exit band
    assert pos.iloc[4] == 0           # z=0.4 <= 0.5 exit -> flat
    assert pos.iloc[0] == 0


def test_positions_enter_long_on_low_z():
    z = pd.Series([0.0, -1.0, -2.5, -1.0, -0.4, 0.0])
    pos = sa.generate_positions(z, entry=2.0, exit=0.5)
    assert pos.iloc[2] == 1           # z<=-2 -> long spread
    assert pos.iloc[3] == 1
    assert pos.iloc[4] == 0           # z=-0.4 within exit band -> flat


def test_position_held_between_entry_and_exit():
    # Never re-crosses exit band -> stays in position the whole time.
    z = pd.Series([0.0, 2.5, 2.4, 2.3, 2.2, 2.1])
    pos = sa.generate_positions(z, entry=2.0, exit=0.5)
    assert (pos.iloc[1:] == -1).all()


def test_no_entry_below_threshold():
    z = pd.Series([0.0, 1.0, 1.5, 1.9, 1.99, 0.0])
    pos = sa.generate_positions(z, entry=2.0, exit=0.5)
    assert (pos == 0).all()


def test_nan_zscore_is_flat():
    z = pd.Series([np.nan, np.nan, 2.5, 0.0])
    pos = sa.generate_positions(z, entry=2.0, exit=0.5)
    assert pos.iloc[0] == 0
    assert pos.iloc[1] == 0
    assert pos.iloc[2] == -1


# --------------------------------------------------------------------------- #
# Backtest P&L accounting                                                      #
# --------------------------------------------------------------------------- #
def test_flat_position_zero_pnl():
    idx = pd.date_range("2020-01-01", periods=10)
    y = pd.Series(np.linspace(10, 20, 10), index=idx)
    x = pd.Series(np.linspace(5, 8, 10), index=idx)
    pos = pd.Series(0.0, index=idx)
    bt = sa.backtest(y, x, pos, beta=1.0, cost_bps=5.0)
    assert np.allclose(bt.returns.values, 0.0)
    assert np.isclose(bt.equity.iloc[-1], 1.0)
    assert bt.n_trades == 0


def test_pnl_uses_lagged_position_no_lookahead():
    # Position only set on the last day -> no return can be earned (lagged by 1).
    idx = pd.date_range("2020-01-01", periods=5)
    y = pd.Series([10, 11, 12, 13, 14], index=idx, dtype=float)
    x = pd.Series([10, 10, 10, 10, 10], index=idx, dtype=float)
    pos = pd.Series([0, 0, 0, 0, 1], index=idx, dtype=float)
    bt = sa.backtest(y, x, pos, beta=0.0, cost_bps=0.0)
    assert np.allclose(bt.returns.values, 0.0)


def test_long_spread_profits_when_y_rises():
    idx = pd.date_range("2020-01-01", periods=4)
    y = pd.Series([100, 100, 110, 110], index=idx, dtype=float)
    x = pd.Series([100, 100, 100, 100], index=idx, dtype=float)
    # Long spread from t=1 onward; y jumps t=2 -> profit realised at t=2 (lagged pos).
    pos = pd.Series([0, 1, 1, 1], index=idx, dtype=float)
    bt = sa.backtest(y, x, pos, beta=0.0, cost_bps=0.0)
    assert bt.returns.iloc[2] > 0
    assert bt.total_return > 0


def test_transaction_costs_reduce_pnl():
    idx = pd.date_range("2020-01-01", periods=20)
    rng = np.random.default_rng(0)
    y = pd.Series(100 + np.cumsum(rng.normal(0, 1, 20)), index=idx)
    x = pd.Series(100 + np.cumsum(rng.normal(0, 1, 20)), index=idx)
    pos = pd.Series(([0, 1] * 10), index=idx, dtype=float)  # churns every day
    free = sa.backtest(y, x, pos, beta=1.0, cost_bps=0.0)
    costly = sa.backtest(y, x, pos, beta=1.0, cost_bps=50.0)
    assert costly.total_return < free.total_return


def test_equity_curve_consistent_with_returns():
    idx = pd.date_range("2020-01-01", periods=30)
    rng = np.random.default_rng(1)
    y = pd.Series(100 + np.cumsum(rng.normal(0, 1, 30)), index=idx)
    x = pd.Series(100 + np.cumsum(rng.normal(0, 1, 30)), index=idx)
    pos = pd.Series(rng.choice([-1.0, 0.0, 1.0], 30), index=idx)
    bt = sa.backtest(y, x, pos, beta=1.0, cost_bps=5.0)
    recon = (1.0 + bt.returns).cumprod()
    assert np.allclose(recon.values, bt.equity.values)


# --------------------------------------------------------------------------- #
# Cointegration on known synthetic series                                     #
# --------------------------------------------------------------------------- #
def _cointegrated_pair(n=1500, seed=7):
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.normal(0, 1, n)) + 100
    noise = rng.normal(0, 0.5, n)          # stationary spread
    idx = pd.date_range("2010-01-01", periods=n)
    y = pd.Series(common + noise + 5, index=idx)
    x = pd.Series(common + rng.normal(0, 0.5, n), index=idx)
    return y, x


def test_cointegration_detected_on_synthetic():
    y, x = _cointegrated_pair()
    stat, pval = sa.engle_granger(y, x)
    assert pval < 0.05                      # should reject no-cointegration


def test_independent_random_walks_not_cointegrated():
    rng = np.random.default_rng(123)
    n = 1500
    idx = pd.date_range("2010-01-01", periods=n)
    y = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 100, index=idx)
    x = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 100, index=idx)
    _, pval = sa.engle_granger(y, x)
    assert pval > 0.05                      # independent walks: fail to reject


def test_ols_hedge_ratio_recovers_known_beta():
    rng = np.random.default_rng(3)
    n = 2000
    idx = pd.date_range("2010-01-01", periods=n)
    x = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 50, index=idx)
    true_beta, true_alpha = 1.7, 3.0
    y = true_alpha + true_beta * x + rng.normal(0, 0.5, n)
    beta, alpha, _ = sa.ols_hedge_ratio(y, x)
    assert abs(beta - true_beta) < 0.05
    assert abs(alpha - true_alpha) < 0.5


def test_johansen_detects_cointegration():
    y, x = _cointegrated_pair()
    df = pd.DataFrame({"y": y, "x": x})
    stat, crit, is_coint = sa.johansen(df)
    assert is_coint
    assert stat > crit


def test_zscore_mean_zero_unit_std_ish():
    rng = np.random.default_rng(9)
    s = pd.Series(rng.normal(0, 1, 500))
    z = sa.rolling_zscore(s, window=30).dropna()
    assert abs(z.mean()) < 0.5
    assert 0.5 < z.std() < 1.6
