"""Cross-sectional funding-dispersion strategy.

Long the lowest-funding perps, short the highest-funding perps. This harvests the
*dispersion* of funding across the universe, not its level — so it can survive when
the overall funding level (carry) collapses below break-even. The core hypothesis
the study tests.
"""
from dataclasses import dataclass

import pandas as pd

from trade4.research.strategy import Strategy
from trade4.research.panel import Panel


@dataclass
class XSFunding(Strategy):
    quantile: float = 0.25
    lookback: int = 3
    top_k: int | None = None  # if set, use exactly K names per side (concentrated/tradeable)
    name: str = "xs_funding"

    def generate_target_weights(self, panel: Panel) -> pd.DataFrame:
        signal = panel.funding.rolling(self.lookback, min_periods=1).mean()  # causal
        signal = signal.where(panel.tradeable)
        weights = pd.DataFrame(0.0, index=panel.times, columns=panel.symbols)
        for t in panel.times:
            row = signal.loc[t].dropna()
            if len(row) < 2:
                continue
            if self.top_k is not None:
                n_side = min(self.top_k, len(row) // 2)
            else:
                n_side = max(1, int(len(row) * self.quantile))
            ordered = row.sort_values()
            longs = ordered.index[:n_side]    # lowest funding
            shorts = ordered.index[-n_side:]  # highest funding
            weights.loc[t, longs] = 0.5 / n_side
            weights.loc[t, shorts] = -0.5 / n_side
        return weights
