"""Causal port of the legacy single-symbol funding-carry strategy.

The legacy engine (``backtest/engine.py:125``) decided entries using the *mean of
future funding rates* — a look-ahead bug. This port uses only the **trailing**
funding average (data ``<= t``): short the perp where funding has been persistently
positive, to harvest the funding it pays.
"""
from dataclasses import dataclass

import pandas as pd

from trade4.research.strategy import Strategy
from trade4.research.panel import Panel


@dataclass
class FundingCarry(Strategy):
    entry_threshold: float = 0.0001
    persistence_window: int = 3
    name: str = "funding_carry"

    def generate_target_weights(self, panel: Panel) -> pd.DataFrame:
        # Trailing mean only — causal. min_periods=1 keeps early bars usable.
        roll = panel.funding.rolling(self.persistence_window, min_periods=1).mean()
        signal = (roll >= self.entry_threshold).astype(float)
        weights = -signal  # short the perp to receive persistently-positive funding
        weights = weights.where(panel.tradeable, 0.0)
        gross = weights.abs().sum(axis=1).replace(0.0, 1.0)
        return weights.div(gross, axis=0)
