import numpy as np
import pandas as pd

from trade4.research.panel import Panel
from trade4.research.strategy import Strategy
from trade4.research.portfolio_engine import EngineConfig
from trade4.research.blend import run_blend, BlendComponent


class _ConstWeight(Strategy):
    def __init__(self, w, name):
        self.w = w
        self.name = name

    def generate_target_weights(self, panel):
        return pd.DataFrame(self.w, index=panel.times, columns=panel.symbols)


def _panel(n=20):
    t = pd.date_range("2023-01-01", periods=n, freq="8h", tz="UTC")
    close = pd.DataFrame({"A": np.linspace(100, 110, n)}, index=t)
    funding = pd.DataFrame({"A": [0.0] * n}, index=t)
    return Panel(close=close, funding=funding)


def test_blend_is_allocation_weighted_sum_of_component_returns():
    panel = _panel()
    cfg = EngineConfig(cost_bps=0.0, funding_enabled=False)
    comps = [
        BlendComponent(_ConstWeight(1.0, "long"), cfg, alloc=0.7),
        BlendComponent(_ConstWeight(-1.0, "short"), cfg, alloc=0.3),
    ]
    res = run_blend(panel, comps)
    expected = 0.7 * res.component_returns["long"] + 0.3 * res.component_returns["short"]
    pd.testing.assert_series_equal(res.returns, expected, check_names=False)


def test_blend_normalises_allocations():
    panel = _panel()
    cfg = EngineConfig(cost_bps=0.0, funding_enabled=False)
    # allocs 7 and 3 should behave identically to 0.7 and 0.3
    a = run_blend(panel, [BlendComponent(_ConstWeight(1.0, "x"), cfg, 7),
                          BlendComponent(_ConstWeight(-1.0, "y"), cfg, 3)])
    b = run_blend(panel, [BlendComponent(_ConstWeight(1.0, "x"), cfg, 0.7),
                          BlendComponent(_ConstWeight(-1.0, "y"), cfg, 0.3)])
    pd.testing.assert_series_equal(a.returns, b.returns)
