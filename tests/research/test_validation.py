import numpy as np
import pandas as pd

from trade4.research.panel import Panel
from trade4.research.strategy import Strategy
from trade4.research.portfolio_engine import EngineConfig
from trade4.research.trial_registry import TrialRegistry
from trade4.research.validation import walk_forward, WindowResult


class _FixedStrat(Strategy):
    name = "fixed"

    def __init__(self, w=-1.0):
        self.w = w

    def generate_target_weights(self, panel):
        return pd.DataFrame(self.w, index=panel.times, columns=panel.symbols)


def _panel(n=300):
    t = pd.date_range("2022-01-01", periods=n, freq="8h", tz="UTC")
    rng = np.random.default_rng(0)
    close = pd.DataFrame(100.0, index=t, columns=["A"])
    funding = pd.DataFrame(rng.normal(0.0005, 0.0002, (n, 1)), index=t, columns=["A"])
    return Panel(close=close, funding=funding)


def test_walk_forward_produces_multiple_windows():
    reg = TrialRegistry()
    res = walk_forward(
        _panel(), _FixedStrat, param_grid=[{"w": -1.0}, {"w": -0.5}],
        cfg=EngineConfig(cost_bps=2.0, price_pnl_enabled=False),
        is_bars=120, oos_bars=60, registry=reg,
    )
    assert len(res) >= 2
    assert all(isinstance(r, WindowResult) for r in res)
    # registry saw every param for every window
    assert reg.n_trials == sum(2 for _ in res)
