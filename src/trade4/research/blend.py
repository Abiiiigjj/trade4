"""Capital-allocation blend of independently-costed sub-strategies.

The components run under DIFFERENT engine configs (xs_funding is perp-only with price
PnL; funding_carry is delta-neutral with 2-leg + holding cost), so a blend cannot be a
sum of weight frames through one engine. Each component is backtested under its OWN
correct cost model, then combined at the RETURNS level by a fixed capital allocation.

Robustness here is structural (complementary weaknesses), not fitted: the allocation is
chosen a priori from the thesis, never optimised on the data.
"""
from dataclasses import dataclass, field

import pandas as pd

from trade4.research.panel import Panel
from trade4.research.strategy import Strategy
from trade4.research.portfolio_engine import run_portfolio_backtest, EngineConfig, BacktestResult


@dataclass
class BlendComponent:
    strategy: Strategy
    cfg: EngineConfig
    alloc: float  # capital share (need not sum to 1; normalised internally)


@dataclass
class BlendResult:
    equity: pd.Series
    returns: pd.Series
    component_returns: dict[str, pd.Series] = field(default_factory=dict)


def run_blend(panel: Panel, components: list[BlendComponent]) -> BlendResult:
    total = sum(c.alloc for c in components)
    if total <= 0:
        raise ValueError("allocations must sum to a positive number")
    blended = pd.Series(0.0, index=panel.times)
    comp_returns: dict[str, pd.Series] = {}
    for c in components:
        res = run_portfolio_backtest(panel, c.strategy.generate_target_weights(panel), c.cfg)
        comp_returns[c.strategy.name] = res.returns
        blended = blended + (c.alloc / total) * res.returns
    equity = (1.0 + blended).cumprod()
    return BlendResult(equity=equity, returns=blended, component_returns=comp_returns)
