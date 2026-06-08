import numpy as np
import pandas as pd
import pytest

from trade4.research.panel import Panel
from trade4.research.strategy import Strategy
from trade4.research.portfolio_engine import (
    run_portfolio_backtest,
    assert_causal,
    EngineConfig,
)


def _const_funding(close):
    return pd.DataFrame(0.0, index=close.index, columns=close.columns)


# ----- known-answer: price PnL -----

def test_known_answer_price_pnl_single_long():
    t = pd.date_range("2023-01-01", periods=2, freq="8h", tz="UTC")
    close = pd.DataFrame({"A": [100.0, 110.0]}, index=t)
    panel = Panel(close=close, funding=_const_funding(close))
    weights = pd.DataFrame({"A": [1.0, 1.0]}, index=t)
    cfg = EngineConfig(cost_bps=0.0, funding_enabled=False)
    res = run_portfolio_backtest(panel, weights, cfg)
    assert abs(res.equity.iloc[-1] / res.equity.iloc[0] - 1.10) < 1e-9


# ----- known-answer: funding only -----

def test_known_answer_funding_only():
    # Flat price, short 1 unit, funding +1% for 3 bars.
    # funding_pnl_t = -w_{t-1} * rate_t ; short(w=-1), rate=+0.01 -> +0.01 per bar * 3 = +0.03
    t = pd.date_range("2023-01-01", periods=4, freq="8h", tz="UTC")
    close = pd.DataFrame({"A": [100.0, 100.0, 100.0, 100.0]}, index=t)
    funding = pd.DataFrame({"A": [0.0, 0.01, 0.01, 0.01]}, index=t)
    panel = Panel(close=close, funding=funding)
    weights = pd.DataFrame({"A": [-1.0, -1.0, -1.0, -1.0]}, index=t)
    cfg = EngineConfig(cost_bps=0.0, funding_enabled=True)
    res = run_portfolio_backtest(panel, weights, cfg)
    assert abs(res.funding_pnl.sum() - 0.03) < 1e-9


def test_delta_neutral_mode_suppresses_price_pnl():
    # +10% price move, short 1 unit, delta-neutral -> price PnL hedged away,
    # only funding (+1% on short = +0.01) remains.
    t = pd.date_range("2023-01-01", periods=2, freq="8h", tz="UTC")
    close = pd.DataFrame({"A": [100.0, 110.0]}, index=t)
    funding = pd.DataFrame({"A": [0.0, 0.01]}, index=t)
    panel = Panel(close=close, funding=funding)
    weights = pd.DataFrame({"A": [-1.0, -1.0]}, index=t)
    cfg = EngineConfig(cost_bps=0.0, funding_enabled=True, price_pnl_enabled=False)
    res = run_portfolio_backtest(panel, weights, cfg)
    # net return = funding only (+0.01), NOT the -10% the naked short would have lost
    assert abs(res.returns.sum() - 0.01) < 1e-9


def test_holding_cost_drags_held_notional():
    # Flat price, no funding, hold gross 1.0 short for 4 bars at 2 bps/bar holding cost.
    # held from bar1..3 (w_prev nonzero) -> 3 bars * 2bps = 6 bps drag.
    t = pd.date_range("2023-01-01", periods=4, freq="8h", tz="UTC")
    close = pd.DataFrame({"A": [100.0] * 4}, index=t)
    funding = pd.DataFrame({"A": [0.0] * 4}, index=t)
    panel = Panel(close=close, funding=funding)
    weights = pd.DataFrame({"A": [-1.0, -1.0, -1.0, -1.0]}, index=t)
    cfg = EngineConfig(cost_bps=0.0, funding_enabled=False, price_pnl_enabled=False,
                       holding_cost_bps_per_bar=2.0)
    res = run_portfolio_backtest(panel, weights, cfg)
    assert abs(res.cost.sum() - 3 * 2.0 / 10_000) < 1e-12


def test_long_pays_positive_funding():
    # long(w=+1), rate=+0.01 -> funding_pnl = -1*0.01 = -0.01 (long PAYS)
    t = pd.date_range("2023-01-01", periods=2, freq="8h", tz="UTC")
    close = pd.DataFrame({"A": [100.0, 100.0]}, index=t)
    funding = pd.DataFrame({"A": [0.0, 0.01]}, index=t)
    panel = Panel(close=close, funding=funding)
    weights = pd.DataFrame({"A": [1.0, 1.0]}, index=t)
    res = run_portfolio_backtest(panel, weights, EngineConfig(cost_bps=0.0))
    assert abs(res.funding_pnl.sum() - (-0.01)) < 1e-9


# ----- PIT-mask assert -----

def test_mask_assert_blocks_untradeable_weight():
    t = pd.date_range("2023-01-01", periods=3, freq="8h", tz="UTC")
    close = pd.DataFrame({"A": [10.0, 11.0, 12.0], "B": [np.nan, np.nan, 5.0]}, index=t)
    panel = Panel(close=close, funding=_const_funding(close.fillna(0.0)))
    # put weight on B at bar0 where it is NOT tradeable
    weights = pd.DataFrame({"A": [0.0, 0.0, 0.0], "B": [1.0, 0.0, 0.0]}, index=t)
    with pytest.raises(AssertionError):
        run_portfolio_backtest(panel, weights, EngineConfig(cost_bps=0.0))


# ----- future-perturbation tripwire -----

class _CheatStrategy(Strategy):
    name = "cheat"

    def generate_target_weights(self, panel):
        fwd = panel.close.shift(-1)
        return (fwd > panel.close).astype(float) - 0.5


class _HonestStrategy(Strategy):
    name = "honest"

    def generate_target_weights(self, panel):
        past = panel.close.pct_change(fill_method=None).fillna(0.0)
        return (past > 0).astype(float) - 0.5


def _toy_panel(n=12):
    t = pd.date_range("2023-01-01", periods=n, freq="8h", tz="UTC")
    rng = np.random.default_rng(0)
    px = 100 * (1 + pd.DataFrame(rng.normal(0, 0.01, (n, 3)),
                                 index=t, columns=["A", "B", "C"])).cumprod()
    fund = pd.DataFrame(0.0, index=t, columns=["A", "B", "C"])
    return Panel(close=px, funding=fund)


def test_tripwire_catches_lookahead():
    with pytest.raises(AssertionError):
        assert_causal(_CheatStrategy(), _toy_panel())


def test_tripwire_passes_honest():
    assert_causal(_HonestStrategy(), _toy_panel())


class _OICheatStrategy(Strategy):
    name = "oi_cheat"

    def generate_target_weights(self, panel):
        # peeks at FUTURE open interest
        fwd_oi = panel.open_interest.shift(-1)
        return (fwd_oi > panel.open_interest).astype(float) - 0.5


def _toy_panel_with_oi(n=12):
    t = pd.date_range("2023-01-01", periods=n, freq="8h", tz="UTC")
    rng = np.random.default_rng(0)
    px = 100 * (1 + pd.DataFrame(rng.normal(0, 0.01, (n, 3)),
                                 index=t, columns=["A", "B", "C"])).cumprod()
    fund = pd.DataFrame(0.0, index=t, columns=["A", "B", "C"])
    oi = pd.DataFrame(rng.uniform(1e6, 5e6, (n, 3)), index=t, columns=["A", "B", "C"])
    return Panel(close=px, funding=fund, open_interest=oi)


def test_tripwire_catches_open_interest_lookahead():
    # The tripwire must perturb open_interest too, or OI-peeking slips through.
    with pytest.raises(AssertionError):
        assert_causal(_OICheatStrategy(), _toy_panel_with_oi())
