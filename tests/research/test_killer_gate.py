"""Killer-gate (plan Task 1.8): the new portfolio engine must agree with the legacy
(causal-fixed) single-symbol engine on funding accounting, and reproduce the
-2.862-bps-class negative result when funding collapses below cost.

The two engines are structurally different (legacy = discrete entry/exit cycles;
new = continuous weights), so the equivalence is on GROSS funding collected within
tolerance, not bit-exact. The exact funding sign/magnitude is independently pinned
by the funding-only known-answer test in test_portfolio_engine.py.
"""
import numpy as np
import pandas as pd
import pytest

from trade4.research.panel import Panel
from trade4.research.portfolio_engine import run_portfolio_backtest, EngineConfig
from trade4.research.strategies.funding_carry import FundingCarry
from trade4.backtest.engine import run_backtest, BacktestConfig


def _scenario():
    """120 8h-bars, flat price; funding +5bps for 100 bars then -10bps (forces exit)."""
    t = pd.date_range("2023-01-01", periods=120, freq="8h", tz="UTC")
    rates = np.concatenate([np.full(100, 0.0005), np.full(20, -0.001)])
    funding_df = pd.DataFrame({"timestamp": t, "funding_rate": rates})
    ohlcv_df = pd.DataFrame({"timestamp": t, "open": 100.0, "high": 100.0,
                             "low": 100.0, "close": 100.0})
    panel = Panel(close=pd.DataFrame({"X": [100.0] * 120}, index=t),
                  funding=pd.DataFrame({"X": rates}, index=t))
    return funding_df, ohlcv_df, panel


def test_cross_engine_funding_equivalence():
    funding_df, ohlcv_df, panel = _scenario()

    legacy = run_backtest(
        funding_df, ohlcv_df, pd.DataFrame(),
        BacktestConfig(entry_threshold=0.0001, persistence_threshold=0.0001,
                       persistence_window=3, max_holding_days=60, exit_threshold=0.0,
                       causal_gate=True, exit_mode="rate"),
    )
    legacy_funding_bps = sum(c.funding_received_bps for c in legacy.cycles)
    assert legacy_funding_bps > 0  # a cycle actually completed

    res = run_portfolio_backtest(
        panel, FundingCarry(entry_threshold=0.0001, persistence_window=3).generate_target_weights(panel),
        EngineConfig(cost_bps=0.0, funding_enabled=True, price_pnl_enabled=False, n_legs=2),
    )
    new_funding_bps = res.funding_pnl.sum() * 10_000

    rel = abs(new_funding_bps - legacy_funding_bps) / abs(legacy_funding_bps)
    assert rel < 0.10, (
        f"cross-engine funding mismatch: new={new_funding_bps:.1f} "
        f"legacy={legacy_funding_bps:.1f} rel={rel:.2%}"
    )


def test_carry_negative_under_churn():
    """The -2.862-bps-class result is a CHURN effect: when funding oscillates around
    the entry threshold, the strategy toggles in/out and pays turnover repeatedly while
    collecting little funding -> costs dominate -> negative. (A continuously-held
    position is cost-efficient and stays positive; that distinction is the honest
    finding the continuous engine surfaces vs the legacy cycle engine.)"""
    t = pd.date_range("2023-01-01", periods=120, freq="8h", tz="UTC")
    # funding in +++/--- blocks: the trailing 3-bar mean crosses the threshold up and
    # down each cycle, so the strategy repeatedly toggles in and out (churn).
    pattern = np.tile([0.0004, 0.0004, 0.0004, -0.0004, -0.0004, -0.0004], 20)
    panel = Panel(close=pd.DataFrame({"X": [100.0] * 120}, index=t),
                  funding=pd.DataFrame({"X": pattern}, index=t))
    w = FundingCarry(entry_threshold=0.0001, persistence_window=3).generate_target_weights(panel)
    turnover = w.fillna(0.0).diff().abs().sum().sum()
    assert turnover > 5, f"scenario must churn; turnover={turnover}"  # sanity: it toggles
    res = run_portfolio_backtest(
        panel, w, EngineConfig(cost_bps=10.0, price_pnl_enabled=False, n_legs=2))
    net_bps = (res.equity.iloc[-1] - 1.0) * 10_000
    assert net_bps < 0, f"expected churn costs to dominate, got {net_bps:.1f} bps"
