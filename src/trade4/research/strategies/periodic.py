"""Periodic-rebalance wrapper — the clean, general low-turnover lever.

Wraps any strategy and only adopts its target weights on a fixed rebalance grid
(e.g. weekly = 21 bars on an 8h clock), holding them constant in between. This cuts
turnover by ~`every`x without touching the underlying signal, directly testing whether
a gross edge survives once transaction costs are amortised over a realistic holding
period. Causal: weights at t come from the most recent rebalance bar <= t.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from trade4.research.strategy import Strategy
from trade4.research.panel import Panel


@dataclass
class PeriodicRebalance(Strategy):
    base: Strategy = field(default=None)  # type: ignore[assignment]
    every: int = 21  # bars between rebalances (weekly on an 8h clock)
    name: str = field(default="")

    def __post_init__(self) -> None:
        if self.base is None:
            raise ValueError("PeriodicRebalance requires a base strategy")
        if not self.name:
            self.name = f"{self.base.name}_rb{self.every}"

    def generate_target_weights(self, panel: Panel) -> pd.DataFrame:
        w = self.base.generate_target_weights(panel)
        keep = pd.Series(False, index=panel.times)
        keep.iloc[::self.every] = True  # bar 0, every, 2*every, ... are rebalance bars
        held = w.copy()
        held.loc[~keep] = np.nan         # blank non-rebalance rows
        return held.ffill().fillna(0.0)  # hold last rebalance weights
