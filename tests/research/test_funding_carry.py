import numpy as np
import pandas as pd

from trade4.research.panel import Panel
from trade4.research.portfolio_engine import assert_causal
from trade4.research.strategies.funding_carry import FundingCarry


def _panel(n=30):
    t = pd.date_range("2023-01-01", periods=n, freq="8h", tz="UTC")
    rng = np.random.default_rng(1)
    px = 100 * (1 + pd.DataFrame(rng.normal(0, 0.005, (n, 2)),
                                 index=t, columns=["A", "B"])).cumprod()
    fund = pd.DataFrame(rng.normal(0.0003, 0.0002, (n, 2)), index=t, columns=["A", "B"])
    return Panel(close=px, funding=fund)


def test_funding_carry_is_causal():
    assert_causal(FundingCarry(entry_threshold=0.0002, persistence_window=3), _panel())


def test_funding_carry_shorts_positive_funding():
    # All funding strongly positive -> strategy should be short (negative weights).
    t = pd.date_range("2023-01-01", periods=5, freq="8h", tz="UTC")
    close = pd.DataFrame({"A": [100.0] * 5, "B": [100.0] * 5}, index=t)
    funding = pd.DataFrame({"A": [0.001] * 5, "B": [0.001] * 5}, index=t)
    w = FundingCarry(entry_threshold=0.0002).generate_target_weights(
        Panel(close=close, funding=funding)
    )
    assert (w.iloc[-1] <= 0).all()
    assert w.iloc[-1].sum() < 0
