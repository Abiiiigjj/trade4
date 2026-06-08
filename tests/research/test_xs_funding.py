import numpy as np
import pandas as pd

from trade4.research.panel import Panel
from trade4.research.portfolio_engine import assert_causal
from trade4.research.strategies.xs_funding import XSFunding


def _panel(n=40, k=6):
    t = pd.date_range("2023-01-01", periods=n, freq="8h", tz="UTC")
    rng = np.random.default_rng(0)
    cols = [f"S{i}" for i in range(k)]
    close = 100 * (1 + pd.DataFrame(rng.normal(0, 0.005, (n, k)), index=t, columns=cols)).cumprod()
    funding = pd.DataFrame(rng.normal(0.0003, 0.0004, (n, k)), index=t, columns=cols)
    return Panel(close=close, funding=funding)


def test_xs_funding_is_causal():
    assert_causal(XSFunding(quantile=0.33, lookback=3), _panel())


def test_xs_funding_is_dollar_neutral():
    w = XSFunding(quantile=0.33, lookback=3).generate_target_weights(_panel())
    assert abs(w.iloc[-1].sum()) < 1e-9


def test_xs_funding_longs_lowest_shorts_highest():
    t = pd.date_range("2023-01-01", periods=5, freq="8h", tz="UTC")
    close = pd.DataFrame(100.0, index=t, columns=["A", "B", "C"])
    funding = pd.DataFrame({"A": 0.001, "B": 0.0, "C": -0.001}, index=t)  # A high, C low
    w = XSFunding(quantile=0.34, lookback=1).generate_target_weights(
        Panel(close=close, funding=funding)).iloc[-1]
    assert w["C"] > 0 and w["A"] < 0  # long lowest funding (C), short highest (A)
