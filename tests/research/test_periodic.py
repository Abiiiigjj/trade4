import numpy as np
import pandas as pd

from trade4.research.panel import Panel
from trade4.research.portfolio_engine import assert_causal
from trade4.research.strategies.periodic import PeriodicRebalance
from trade4.research.strategies.xs_momentum import XSMomentum


def _panel(n=80, k=6):
    t = pd.date_range("2023-01-01", periods=n, freq="8h", tz="UTC")
    rng = np.random.default_rng(0)
    cols = [f"S{i}" for i in range(k)]
    close = 100 * (1 + pd.DataFrame(rng.normal(0.0002, 0.01, (n, k)), index=t, columns=cols)).cumprod()
    funding = pd.DataFrame(0.0, index=t, columns=cols)
    return Panel(close=close, funding=funding)


def test_periodic_is_causal():
    assert_causal(PeriodicRebalance(base=XSMomentum(lookback=21), every=7), _panel())


def test_periodic_reduces_turnover():
    panel = _panel()
    base = XSMomentum(lookback=21)
    wrapped = PeriodicRebalance(base=base, every=21)
    base_turnover = base.generate_target_weights(panel).diff().abs().sum().sum()
    wrapped_turnover = wrapped.generate_target_weights(panel).diff().abs().sum().sum()
    assert wrapped_turnover < base_turnover  # holding between rebalances cuts turnover


def test_periodic_holds_weights_between_rebalances():
    panel = _panel(n=10)
    wrapped = PeriodicRebalance(base=XSMomentum(lookback=2), every=5)
    w = wrapped.generate_target_weights(panel)
    # bars 1..4 must equal bar 0 (held), bars 6..9 equal bar 5
    pd.testing.assert_series_equal(w.iloc[1], w.iloc[0], check_names=False)
    pd.testing.assert_series_equal(w.iloc[4], w.iloc[0], check_names=False)
    pd.testing.assert_series_equal(w.iloc[6], w.iloc[5], check_names=False)
