"""Capstone: honest OOS validation of the CHOSEN system — vol-targeted xs_funding.

The system was chosen after comparing many configs (concentration sweep, blend allocations,
overlay params, xs-vs-blend). An honest verdict must deflate for that search, not report the
winner's raw Sharpe. This:
  1. assembles the TRIAL SET actually searched, as vol-targeted weekly returns;
  2. computes the chosen system's Deflated Sharpe vs (N trials, their Sharpe variance);
  3. runs PBO (CSCV) over the search matrix — does picking the IS-best generalise OOS?
  4. reports time-split consistency (first half vs second half, and per year).

Run after the cache is warm. Seconds, no downloads.
"""
import logging
from dataclasses import replace

import numpy as np
import pandas as pd

from trade4.research.panel_builder import build_panel
from trade4.research.portfolio_engine import run_portfolio_backtest, EngineConfig
from trade4.research.strategies.xs_funding import XSFunding
from trade4.research.strategies.xs_momentum import XSMomentum
from trade4.research.strategies.funding_carry import FundingCarry
from trade4.research.strategies.periodic import PeriodicRebalance
from trade4.research.overlay import vol_target, releveraging_cost
from trade4.research.metrics import (
    sharpe_ratio, max_drawdown, probabilistic_sharpe_ratio,
    expected_max_sharpe, deflated_sharpe_ratio,
)
from trade4.research.pbo import probability_of_backtest_overfitting
from scripts.run_study import DEFAULT_UNIVERSE, _months, _load

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("final")

CFG_XS = EngineConfig(cost_bps=5.0, n_legs=1, price_pnl_enabled=True)
CFG_CARRY = EngineConfig(cost_bps=5.0, n_legs=2, price_pnl_enabled=False,
                         holding_cost_bps_per_bar=0.25)
RELEV, TARGET, LB, LEV = 10.0, 0.10, 30, 2.0


def _overlaid(panel, strat, cfg, target=TARGET, lb=LB):
    """Full-panel returns of a strategy after the weekly engine AND the vol-target overlay."""
    r = run_portfolio_backtest(panel, strat.generate_target_weights(panel), cfg).returns
    scaled, exp = vol_target(r, target, lb, LEV)
    return scaled - releveraging_cost(exp, RELEV)


def main() -> None:
    months = _months("2022-01", "2025-06")
    funding, klines = {}, {}
    for s in DEFAULT_UNIVERSE:
        f, k = _load(s, months)
        if not f.empty and not k.empty:
            funding[s], klines[s] = f, k
    panel = build_panel(funding, klines, bar="8h")
    logger.info("panel: %d bars x %d symbols", len(panel.times), len(panel.symbols))

    def wk(strat):
        return PeriodicRebalance(base=strat, every=21)

    # --- TRIAL SET: every config genuinely compared on the path to the choice ---
    trials: dict[str, pd.Series] = {}
    # xs_funding family (concentration + quantile)
    for k in (1, 2, 3, 5):
        trials[f"xsf_topk{k}"] = _overlaid(panel, wk(XSFunding(top_k=k, lookback=3)), CFG_XS)
    for q in (0.25, 0.33):
        trials[f"xsf_q{q}"] = _overlaid(panel, wk(XSFunding(quantile=q, lookback=3)), CFG_XS)
    # other base strategies
    trials["xs_momentum"] = _overlaid(panel, wk(XSMomentum(lookback=21)), CFG_XS)
    trials["funding_carry"] = _overlaid(panel, wk(FundingCarry(entry_threshold=0.0001)), CFG_CARRY)
    # overlay-parameter grid on the chosen base (these were searched too)
    chosen_base = wk(XSFunding(quantile=0.25, lookback=3))
    for tv in (0.08, 0.10, 0.12):
        for lb in (20, 30, 45):
            trials[f"ov_t{tv}_lb{lb}"] = _overlaid(panel, chosen_base, CFG_XS, target=tv, lb=lb)

    matrix = pd.DataFrame(trials)
    n_trials = matrix.shape[1]
    srs = [matrix[c].dropna().pipe(lambda r: r.mean() / r.std(ddof=1))
           for c in matrix.columns]
    trial_var = float(np.var(srs, ddof=1))
    pbo = probability_of_backtest_overfitting(matrix, n_splits=10)

    # --- CHOSEN system ---
    chosen = _overlaid(panel, chosen_base, CFG_XS)  # vol-targeted xs_funding q0.25
    eq = (1.0 + chosen).cumprod()
    sr = sharpe_ratio(chosen)
    sr0 = expected_max_sharpe(trial_var, n_trials)
    dsr = deflated_sharpe_ratio(chosen, trial_var, n_trials)
    psr0 = probabilistic_sharpe_ratio(chosen, 0.0)

    print("\n" + "=" * 68)
    print("FINAL VALIDATION — vol-targeted xs_funding (q0.25, weekly, perp-only)")
    print("=" * 68)
    print(f"  Trials searched (N):        {n_trials}")
    print(f"  Trial Sharpe variance (V):  {trial_var:.4f}")
    print(f"  Annualised Sharpe:          {sr:+.2f}")
    print(f"  Max drawdown:               {max_drawdown(eq):.1%}")
    print(f"  Net (3.5y):                 {(eq.iloc[-1]-1)*1e4:+.0f} bp")
    print(f"  PSR (vs 0):                 {psr0:.3f}")
    print(f"  Expected-max Sharpe SR0:    {sr0:.3f}  (per-period, null over {n_trials} trials)")
    print(f"  DEFLATED Sharpe (DSR):      {dsr:.3f}   <- prob. edge is real after the search")
    print(f"  PBO (search overfitting):   {pbo:.2f}   <- <0.5 good, >0.5 overfit")

    # --- time-split consistency (no optimisation; pure OOS measurement) ---
    half = len(chosen) // 2
    sr_h1 = sharpe_ratio(chosen.iloc[:half])
    sr_h2 = sharpe_ratio(chosen.iloc[half:])
    print(f"\n  First-half Sharpe:  {sr_h1:+.2f}   Second-half Sharpe: {sr_h2:+.2f}")
    by = chosen.groupby(chosen.index.year).apply(lambda r: sharpe_ratio(r))
    print("  Per-year Sharpe:    " + "  ".join(f"{y}:{s:+.1f}" for y, s in by.items()))
    print("=" * 68)


if __name__ == "__main__":
    main()
