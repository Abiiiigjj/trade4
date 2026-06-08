"""The study — run all market-neutral strategies through the harness and deliver an
honest verdict.

Answers the research question: is there a market-neutral crypto strategy with
out-of-sample edge in the current regime, after costs?

Design notes (from the Phase-2 review):
* The **PBO matrix** is built from a SINGLE common window (every config run over the
  same full-panel slice) so the columns are index-aligned and equal-length — feeding
  the cross-window walk-forward registry would union non-overlapping indices into a
  mostly-NaN matrix and degenerate PBO.
* **DSR scope** is consistent: the trial-Sharpe variance V and the deflated Sharpe are
  both measured over the full-panel horizon.
* The **OOS gate** is judged separately by the multi-window walk-forward.
"""
from typing import Any

import numpy as np
import pandas as pd

from trade4.research.panel import Panel
from trade4.research.portfolio_engine import run_portfolio_backtest, EngineConfig
from trade4.research.trial_registry import TrialRegistry
from trade4.research.validation import walk_forward, gate_passed
from trade4.research.tearsheet import build_tearsheet
from trade4.research.pbo import probability_of_backtest_overfitting
from trade4.research.strategies.funding_carry import FundingCarry
from trade4.research.strategies.xs_funding import XSFunding
from trade4.research.strategies.xs_momentum import XSMomentum

_CAPACITY_CAVEAT = (
    "Results are 'edge at scale after costs'. A 30-40 name dollar-neutral perp "
    "portfolio is infeasible at EUR 2k (min-notional x both legs); a positive XS "
    "result is NOT an immediately deployable signal for this account."
)

# (strategy_factory, param_grid, EngineConfig) per strategy
_SPECS = {
    "funding_carry": (FundingCarry, [{"entry_threshold": 0.0001}, {"entry_threshold": 0.0003}],
                      EngineConfig(cost_bps=5.0, price_pnl_enabled=False, n_legs=2)),
    "xs_funding": (XSFunding, [{"quantile": 0.25, "lookback": 3}, {"quantile": 0.33, "lookback": 6}],
                   EngineConfig(cost_bps=5.0, price_pnl_enabled=True, n_legs=1)),
    "xs_momentum": (XSMomentum, [{"lookback": 21}, {"lookback": 42}],
                    EngineConfig(cost_bps=5.0, price_pnl_enabled=True, n_legs=1)),
}


def _full_panel_config_returns(panel: Panel) -> pd.DataFrame:
    """Run every (strategy x param) config over the full panel -> index-aligned matrix.

    This is the trial set for DSR (count + variance) and PBO (CSCV matrix)."""
    cols: dict[str, pd.Series] = {}
    for name, (factory, grid, cfg) in _SPECS.items():
        for params in grid:
            strat = factory(**params)
            res = run_portfolio_backtest(panel, strat.generate_target_weights(panel), cfg)
            cols[f"{name}|{params}"] = res.returns
    return pd.DataFrame(cols, index=panel.times)


def _per_period_sharpe_variance(matrix: pd.DataFrame) -> float:
    srs = []
    for c in matrix.columns:
        r = matrix[c].dropna()
        if len(r) > 1 and r.std(ddof=1) > 0:
            srs.append(r.mean() / r.std(ddof=1))
    return float(np.var(srs, ddof=1)) if len(srs) > 1 else 0.0


def run_study(panel: Panel, seed: int = 0, is_bars: int = 180, oos_bars: int = 90,
              pbo_splits: int = 8) -> dict[str, Any]:
    # --- trial set over a single common window (full panel): DSR inputs + PBO matrix ---
    matrix = _full_panel_config_returns(panel)
    n_trials = matrix.shape[1]
    trial_var = _per_period_sharpe_variance(matrix)
    pbo = probability_of_backtest_overfitting(matrix, n_splits=pbo_splits)

    # --- per-strategy: OOS gate (walk-forward) + tearsheet (DSR uses full-panel V/N) ---
    registry = TrialRegistry()
    tearsheets: dict[str, Any] = {}
    ranking: list[tuple[str, float]] = []
    for name, (factory, grid, cfg) in _SPECS.items():
        windows = walk_forward(panel, factory, grid, cfg, is_bars, oos_bars, registry)
        ts = build_tearsheet(factory(**grid[0]), panel, cfg,
                             n_trials=n_trials, trial_sharpe_var=trial_var)
        ts["oos_gate_passed"] = gate_passed(windows)
        tearsheets[name] = ts
        ranking.append((name, ts["dsr"]))

    ranking.sort(key=lambda x: x[1], reverse=True)
    survivors = [n for n in _SPECS if tearsheets[n]["oos_gate_passed"]]
    if survivors:
        verdict = (f"Survived OOS gate after costs: {survivors}. "
                   f"DSR ranking {ranking}, PBO={pbo:.2f}.")
    else:
        verdict = (f"No strategy passed the OOS gate after costs. "
                   f"DSR ranking {ranking}, PBO={pbo:.2f}. "
                   f"Consistent with the funding-collapse regime thesis.")

    return {
        "tearsheets": tearsheets,
        "ranking": ranking,
        "pbo": pbo,
        "n_trials": n_trials,
        "verdict": verdict,
        "caveats": {"capacity": _CAPACITY_CAVEAT},
    }
