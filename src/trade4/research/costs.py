"""Turnover-aware portfolio cost layer.

The legacy ``backtest/cost_model.py`` charges a flat round-trip cost per discrete
trade cycle (single-symbol funding-carry). A portfolio that rebalances a universe
must charge on *turnover* — the sum of absolute weight changes — so that churny
cross-sectional strategies pay for what they trade.
"""
import pandas as pd


def turnover_cost(
    weights: pd.DataFrame,
    cost_bps: float,
    multiplier: float = 1.0,
    n_legs: int = 1,
) -> pd.Series:
    """Per-bar cost = ``sum_symbol |Δweight| * cost_bps/1e4 * multiplier * n_legs``.

    ``cost_bps`` is the one-way cost (fee + slippage) per unit of notional traded.
    The first bar charges the full initial build-up from a flat book.
    ``multiplier`` drives the cost-sensitivity sweep (1x / 2x / 3x).
    ``n_legs`` accounts for multi-leg trades: a delta-neutral carry trades BOTH a
    perp and a spot leg per rebalance, so it costs ~2x a perp-only XS rebalance.

    ADV-aware slippage can later be layered by passing a per-symbol, per-time
    ``cost_bps`` frame; v1 uses a scalar applied uniformly.
    """
    w = weights.fillna(0.0)
    dw = w.diff().abs()
    dw.iloc[0] = w.iloc[0].abs()  # initial build-up from a flat book
    return dw.sum(axis=1) * (cost_bps / 10_000.0) * multiplier * n_legs
