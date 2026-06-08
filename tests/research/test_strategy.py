import pandas as pd

from trade4.research.strategy import Strategy
from trade4.research.panel import Panel


class _EqualLongShort(Strategy):
    name = "equal_long_short"

    def generate_target_weights(self, panel):
        w = pd.DataFrame(0.0, index=panel.times, columns=panel.symbols)
        if len(panel.symbols) >= 2:
            w.iloc[:, 0] = 0.5
            w.iloc[:, 1] = -0.5
        return w


def test_strategy_emits_dollar_neutral_weights():
    t = pd.date_range("2023-01-01", periods=3, freq="8h", tz="UTC")
    close = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, index=t)
    funding = pd.DataFrame({"A": 0.0, "B": 0.0}, index=t)
    w = _EqualLongShort().generate_target_weights(Panel(close=close, funding=funding))
    assert abs(w.iloc[0].sum()) < 1e-12
