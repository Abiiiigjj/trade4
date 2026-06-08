import numpy as np
import pandas as pd

from trade4.research.panel import Panel
from trade4.research.portfolio_engine import assert_causal
from trade4.research.strategies.xs_momentum import XSMomentum


def _panel(n=80, k=6):
    t = pd.date_range("2023-01-01", periods=n, freq="8h", tz="UTC")
    rng = np.random.default_rng(0)
    cols = [f"S{i}" for i in range(k)]
    close = 100 * (1 + pd.DataFrame(rng.normal(0.0002, 0.01, (n, k)), index=t, columns=cols)).cumprod()
    funding = pd.DataFrame(0.0, index=t, columns=cols)
    return Panel(close=close, funding=funding)


def test_xs_momentum_is_causal():
    assert_causal(XSMomentum(lookback=21, quantile=0.33), _panel())


def test_xs_momentum_is_dollar_neutral():
    w = XSMomentum(lookback=21, quantile=0.33).generate_target_weights(_panel())
    assert abs(w.iloc[-1].sum()) < 1e-9


def test_xs_momentum_longs_winner():
    t = pd.date_range("2023-01-01", periods=30, freq="8h", tz="UTC")
    close = pd.DataFrame({
        "WIN": np.linspace(100, 150, 30),   # strong up
        "FLAT": [100.0] * 30,
        "LOSE": np.linspace(100, 70, 30),   # strong down
    }, index=t)
    funding = pd.DataFrame(0.0, index=t, columns=close.columns)
    w = XSMomentum(lookback=21, quantile=0.34).generate_target_weights(
        Panel(close=close, funding=funding)).iloc[-1]
    assert w["WIN"] > 0 and w["LOSE"] < 0
