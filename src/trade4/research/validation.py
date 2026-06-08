"""Multi-window walk-forward validation at portfolio level.

Generalises the single-symbol scalper walk-forward: roll IS/OOS windows across the
panel, optimise by IS Sharpe over a param grid (logging EVERY config to the trial
registry so DSR/PBO see the true N), evaluate on OOS, and apply a portfolio-level
go/no-go gate (Sharpe / drawdown / OOS-degradation — not the trade-level win-rate
and profit-factor gates the scalper used).
"""
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from trade4.research.panel import Panel
from trade4.research.strategy import Strategy
from trade4.research.portfolio_engine import run_portfolio_backtest, EngineConfig
from trade4.research.trial_registry import TrialRegistry
from trade4.research.metrics import sharpe_ratio, max_drawdown


@dataclass
class WindowResult:
    is_start: pd.Timestamp
    oos_start: pd.Timestamp
    best_params: dict[str, Any]
    is_sharpe: float
    oos_sharpe: float
    oos_max_dd: float
    oos_degradation: float


def _slice(panel: Panel, lo: int, hi: int) -> Panel:
    oi = None if panel.open_interest is None else panel.open_interest.iloc[lo:hi]
    return Panel(close=panel.close.iloc[lo:hi], funding=panel.funding.iloc[lo:hi],
                 open_interest=oi)


def walk_forward(
    panel: Panel,
    strategy_factory: Callable[..., Strategy],
    param_grid: list[dict[str, Any]],
    cfg: EngineConfig,
    is_bars: int,
    oos_bars: int,
    registry: TrialRegistry,
    bar: str = "8h",
) -> list[WindowResult]:
    """Rolling IS-optimise (by IS Sharpe) -> OOS-evaluate. Logs every config."""
    n = len(panel.times)
    results: list[WindowResult] = []
    start = 0
    while start + is_bars + oos_bars <= n:
        is_panel = _slice(panel, start, start + is_bars)
        oos_panel = _slice(panel, start + is_bars, start + is_bars + oos_bars)
        best: tuple[dict[str, Any], float] | None = None
        for params in param_grid:
            strat = strategy_factory(**params)
            w = strat.generate_target_weights(is_panel)
            res = run_portfolio_backtest(is_panel, w, cfg)
            sr = sharpe_ratio(res.returns, bar)
            registry.record(f"{strat.name}|{params}|is{start}", res.returns)
            if best is None or sr > best[1]:
                best = (params, sr)
        assert best is not None
        bparams, is_sr = best
        strat = strategy_factory(**bparams)
        ores = run_portfolio_backtest(oos_panel, strat.generate_target_weights(oos_panel), cfg)
        oos_sr = sharpe_ratio(ores.returns, bar)
        deg = (is_sr - oos_sr) / abs(is_sr) if is_sr != 0 else float("inf")
        results.append(WindowResult(
            is_start=panel.times[start], oos_start=panel.times[start + is_bars],
            best_params=bparams, is_sharpe=is_sr, oos_sharpe=oos_sr,
            oos_max_dd=max_drawdown(ores.equity), oos_degradation=deg,
        ))
        start += oos_bars  # roll forward by the OOS window
    return results


# Portfolio-level go/no-go gate (replaces trade-level scalper gates)
_MIN_OOS_SHARPE = 1.5
_MAX_OOS_DD = -0.25
_MAX_OOS_DEGRADATION = 0.30


def gate_passed(windows: list[WindowResult]) -> bool:
    if not windows:
        return False
    mean_oos = float(np.mean([w.oos_sharpe for w in windows]))
    worst_dd = min(w.oos_max_dd for w in windows)
    mean_deg = float(np.mean([w.oos_degradation for w in windows]))
    return bool(mean_oos >= _MIN_OOS_SHARPE and worst_dd >= _MAX_OOS_DD
                and mean_deg <= _MAX_OOS_DEGRADATION)
