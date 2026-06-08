"""Per-strategy tearsheet — the honest report card.

Reports the metrics that make overfitting and cost-fragility visible:
Sharpe, max drawdown, **Deflated Sharpe** (corrected for trial count), a
**cost-sensitivity sweep** (1x/2x/3x), and performance **conditioned on funding
regime** (does the edge survive when carry is dead?).
"""
from dataclasses import replace
from typing import Any

import pandas as pd

from trade4.research.panel import Panel
from trade4.research.strategy import Strategy
from trade4.research.portfolio_engine import run_portfolio_backtest, EngineConfig
from trade4.research.metrics import (
    sharpe_ratio, max_drawdown, deflated_sharpe_ratio, funding_regime,
)


def build_tearsheet(
    strategy: Strategy,
    panel: Panel,
    cfg: EngineConfig,
    n_trials: int,
    trial_sharpe_var: float,
    bar: str = "8h",
) -> dict[str, Any]:
    w = strategy.generate_target_weights(panel)
    base = run_portfolio_backtest(panel, w, cfg)

    cost_sweep = {}
    for mult in (1.0, 2.0, 3.0):
        res = run_portfolio_backtest(panel, w, replace(cfg, cost_multiplier=mult))
        cost_sweep[mult] = float((res.equity.iloc[-1] - 1.0) * 10_000)  # net bps

    regime = funding_regime(panel.funding.mean(axis=1))
    by_regime: dict[str, float] = {}
    for label in ("high", "low"):
        mask = (regime == label).to_numpy()
        r = base.returns[mask]
        by_regime[label] = sharpe_ratio(r, bar) if len(r) > 2 else 0.0

    return {
        "strategy": strategy.name,
        "sharpe": sharpe_ratio(base.returns, bar),
        "max_dd": max_drawdown(base.equity),
        "dsr": deflated_sharpe_ratio(base.returns, trial_sharpe_var, n_trials),
        "net_bps": float((base.equity.iloc[-1] - 1.0) * 10_000),
        "cost_sweep": cost_sweep,
        "regime": by_regime,
    }
