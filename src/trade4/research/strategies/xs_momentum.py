"""Cross-sectional momentum strategy.

Long the trailing winners, short the trailing losers. The most academically robust,
latency-insensitive crypto factor — a natural candidate for regime-independent edge.
"""
from dataclasses import dataclass

import pandas as pd

from trade4.research.strategy import Strategy
from trade4.research.panel import Panel


@dataclass
class XSMomentum(Strategy):
    lookback: int = 21        # bars of trailing return (causal)
    quantile: float = 0.25
    name: str = "xs_momentum"

    def generate_target_weights(self, panel: Panel) -> pd.DataFrame:
        # trailing return over `lookback` bars ending at t (uses close[t] and close[t-lb])
        mom = panel.close / panel.close.shift(self.lookback) - 1.0
        mom = mom.where(panel.tradeable)
        weights = pd.DataFrame(0.0, index=panel.times, columns=panel.symbols)
        for t in panel.times:
            row = mom.loc[t].dropna()
            if len(row) < 2:
                continue
            n_side = max(1, int(len(row) * self.quantile))
            ordered = row.sort_values()
            losers = ordered.index[:n_side]
            winners = ordered.index[-n_side:]
            weights.loc[t, winners] = 0.5 / n_side
            weights.loc[t, losers] = -0.5 / n_side
        return weights
