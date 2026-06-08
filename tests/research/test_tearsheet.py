import numpy as np
import pandas as pd

from trade4.research.panel import Panel
from trade4.research.portfolio_engine import EngineConfig
from trade4.research.strategies.funding_carry import FundingCarry
from trade4.research.tearsheet import build_tearsheet


def _panel(n=200):
    t = pd.date_range("2023-01-01", periods=n, freq="8h", tz="UTC")
    rng = np.random.default_rng(0)
    close = pd.DataFrame(100.0, index=t, columns=["A", "B"])
    funding = pd.DataFrame(rng.normal(0.0004, 0.0002, (n, 2)), index=t, columns=["A", "B"])
    return Panel(close=close, funding=funding)


def test_tearsheet_has_cost_sweep_and_regime():
    ts = build_tearsheet(
        FundingCarry(), _panel(),
        EngineConfig(cost_bps=2.0, price_pnl_enabled=False),
        n_trials=10, trial_sharpe_var=0.02,
    )
    assert set(ts["cost_sweep"]) == {1.0, 2.0, 3.0}
    assert "dsr" in ts and "sharpe" in ts and "regime" in ts
    assert set(ts["regime"]) == {"high", "low"}
