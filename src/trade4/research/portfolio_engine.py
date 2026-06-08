"""Vectorised market-neutral portfolio backtest engine.

This is the core fraud-resistant component. It books, per bar:

* **price PnL** — a position held from ``t-1`` earns the asset return at ``t``;
* **discrete funding PnL** — locked sign convention ``funding_pnl_t = -w_{t-1} * rate_t``
  (a long pays positive funding, a short receives it);
* **turnover cost** — charged on absolute weight changes.

Two structural guards make it hard to lie:

* **PIT-mask assert** — no weight may sit on an untradeable (unlisted) symbol;
* **future-perturbation tripwire** (:func:`assert_causal`) — perturbing data strictly
  after ``t`` must not change any weight at ``<= t``.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from trade4.research.panel import Panel
from trade4.research.strategy import Strategy
from trade4.research.costs import turnover_cost


@dataclass(frozen=True)
class EngineConfig:
    cost_bps: float = 5.0          # one-way fee + slippage per unit notional traded
    funding_enabled: bool = True
    cost_multiplier: float = 1.0   # cost-sensitivity sweep (1x / 2x / 3x)
    price_pnl_enabled: bool = True  # False = delta-neutral mode (carry leg hedged by spot)
    n_legs: int = 1                 # 2 for delta-neutral carry (spot + perp both traded)


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    turnover: pd.Series
    funding_pnl: pd.Series
    cost: pd.Series


def run_portfolio_backtest(
    panel: Panel,
    weights: pd.DataFrame,
    cfg: EngineConfig,
) -> BacktestResult:
    """Run a vectorised backtest of ``weights`` over ``panel``."""
    w = weights.reindex(index=panel.times, columns=panel.symbols).fillna(0.0)

    # PIT-mask assert: a weight on an untradeable symbol is universe-level look-ahead.
    leaked = (w != 0) & (~panel.tradeable)
    if bool(leaked.to_numpy().any()):
        raise AssertionError("weights assigned to untradeable (unlisted) symbols — look-ahead")

    w_prev = w.shift(1).fillna(0.0)

    if cfg.price_pnl_enabled:
        asset_ret = panel.close.pct_change(fill_method=None).fillna(0.0)
        price_pnl = (w_prev * asset_ret).sum(axis=1)
    else:
        # delta-neutral mode: the directional perp leg is hedged by a spot leg of
        # equal notional, so price PnL cancels (residual basis drift lives in costs).
        price_pnl = pd.Series(0.0, index=panel.times)

    if cfg.funding_enabled:
        funding_pnl = (-w_prev * panel.funding.fillna(0.0)).sum(axis=1)
    else:
        funding_pnl = pd.Series(0.0, index=panel.times)

    cost = turnover_cost(w, cfg.cost_bps, cfg.cost_multiplier, cfg.n_legs)
    turnover = w.fillna(0.0).diff().abs()
    turnover.iloc[0] = w.iloc[0].abs()
    turnover_series = turnover.sum(axis=1)

    port_ret = price_pnl + funding_pnl - cost
    equity = (1.0 + port_ret).cumprod()

    return BacktestResult(
        equity=equity,
        returns=port_ret,
        turnover=turnover_series,
        funding_pnl=funding_pnl,
        cost=cost,
    )


def assert_causal(
    strategy: Strategy,
    panel: Panel,
    n_checks: int = 5,
    seed: int = 0,
) -> None:
    """Perturb future data and assert past weights are unchanged.

    For several cut points ``t0``, replace panel data strictly after ``t0`` with
    noise, recompute weights, and require weights at indices ``<= t0`` to be
    bit-identical to the unperturbed run. A strategy that reads the future changes
    those weights and is caught here — the single test that would have flagged the
    phantom-profit bots.
    """
    rng = np.random.default_rng(seed)
    base = strategy.generate_target_weights(panel).reindex(
        index=panel.times, columns=panel.symbols
    )
    times = panel.times
    candidate_cuts = times[len(times) // 3 : -1]
    if len(candidate_cuts) == 0:
        raise ValueError("panel too short for a causality check")
    if len(candidate_cuts) > n_checks:
        idx = np.linspace(0, len(candidate_cuts) - 1, n_checks).astype(int)
        candidate_cuts = candidate_cuts[idx]

    for t0 in candidate_cuts:
        fut = times > t0
        n_fut = int(fut.sum())
        if n_fut == 0:
            continue
        close2 = panel.close.copy()
        fund2 = panel.funding.copy()
        # Decisive perturbation: scramble future prices by a large factor and flip
        # future funding sign/scale. Big enough to flip any future-dependent decision.
        shape = (n_fut, len(panel.symbols))
        close2.loc[fut] = close2.loc[fut].to_numpy() * rng.uniform(0.1, 3.0, shape)
        fund2.loc[fut] = rng.uniform(-0.01, 0.01, shape)
        # The tripwire must perturb EVERY field the Panel exposes, else it gives only
        # partial protection. open_interest is scrambled too where present.
        if panel.open_interest is not None:
            oi2 = panel.open_interest.copy()
            oi2.loc[fut] = oi2.loc[fut].to_numpy() * rng.uniform(0.1, 3.0, shape)
        else:
            oi2 = None
        perturbed = Panel(close=close2, funding=fund2, open_interest=oi2)
        w2 = strategy.generate_target_weights(perturbed).reindex(
            index=panel.times, columns=panel.symbols
        )
        past = times <= t0
        a = base.loc[past].fillna(0.0).to_numpy()
        b = w2.loc[past].fillna(0.0).to_numpy()
        if not np.allclose(a, b, atol=1e-12, rtol=0.0):
            raise AssertionError(
                f"look-ahead: weights at/<= {t0} changed when future data was perturbed"
            )
