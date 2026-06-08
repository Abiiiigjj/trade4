"""Step 2: the robust blend — xs_funding core + funding_carry enhancer.

Compares standalone components against a FIXED, a-priori 70/30 blend on the metrics that
matter for robustness: Sharpe, drawdown, cost-robustness, and per-year consistency. No
parameter is optimised on the data; the carry leg is honestly costed (basis drift).

Run after the cache is warm. Seconds, no downloads.
"""
import logging
from dataclasses import replace

import numpy as np
import pandas as pd

from trade4.research.panel_builder import build_panel
from trade4.research.portfolio_engine import run_portfolio_backtest, EngineConfig
from trade4.research.strategies.xs_funding import XSFunding
from trade4.research.strategies.funding_carry import FundingCarry
from trade4.research.strategies.periodic import PeriodicRebalance
from trade4.research.blend import run_blend, BlendComponent
from trade4.research.metrics import sharpe_ratio, max_drawdown
from scripts.run_study import DEFAULT_UNIVERSE, _months, _load

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("blend")

# Fixed, a-priori components (weekly rebalance). NOTHING optimised on the data.
XS = PeriodicRebalance(base=XSFunding(quantile=0.25, lookback=3), every=21)
CARRY = PeriodicRebalance(base=FundingCarry(entry_threshold=0.0001), every=21)
CFG_XS = EngineConfig(cost_bps=5.0, n_legs=1, price_pnl_enabled=True)
# carry honestly costed: 0.25 bp/bar basis drift (the conservative middle of the sweep)
CFG_CARRY = EngineConfig(cost_bps=5.0, n_legs=2, price_pnl_enabled=False,
                         holding_cost_bps_per_bar=0.25)
ALLOC_XS, ALLOC_CARRY = 0.70, 0.30  # a-priori: xs_funding is the robust core


def _row(label, returns, equity):
    by_year = returns.groupby(returns.index.year).apply(lambda r: sharpe_ratio(r))
    yr = "  ".join(f"{y}:{s:+.1f}" for y, s in by_year.items())
    print(f"{label:<22} Sharpe={sharpe_ratio(returns):+6.2f}  maxDD={max_drawdown(equity):6.1%}  "
          f"net={ (equity.iloc[-1]-1)*1e4:+7.0f}bp   per-year[{yr}]")


def main() -> None:
    months = _months("2022-01", "2025-06")
    funding, klines = {}, {}
    for s in DEFAULT_UNIVERSE:
        f, k = _load(s, months)
        if not f.empty and not k.empty:
            funding[s], klines[s] = f, k
    panel = build_panel(funding, klines, bar="8h")
    logger.info("panel: %d bars x %d symbols", len(panel.times), len(panel.symbols))

    def blend(mult=1.0, a_xs=ALLOC_XS, a_carry=ALLOC_CARRY):
        comps = [
            BlendComponent(XS, replace(CFG_XS, cost_multiplier=mult), a_xs),
            BlendComponent(CARRY, replace(CFG_CARRY, cost_multiplier=mult), a_carry),
        ]
        return run_blend(panel, comps)

    print("\n=== Standalone vs Blend (weekly, honestly costed, 1x cost) ===")
    xs_res = run_portfolio_backtest(panel, XS.generate_target_weights(panel), CFG_XS)
    ca_res = run_portfolio_backtest(panel, CARRY.generate_target_weights(panel), CFG_CARRY)
    bl = blend()
    _row("xs_funding (core)", xs_res.returns, xs_res.equity)
    _row("funding_carry @0.25", ca_res.returns, ca_res.equity)
    _row("BLEND 70/30", bl.returns, bl.equity)

    print("\n=== Cost robustness (blend net bps) ===")
    for m in (1.0, 2.0, 3.0):
        b = blend(mult=m)
        print(f"  {m:.0f}x cost: net={ (b.equity.iloc[-1]-1)*1e4:+7.0f}bp  "
              f"Sharpe={sharpe_ratio(b.returns):+.2f}  maxDD={max_drawdown(b.equity):.1%}")

    print("\n=== Allocation sensitivity (diagnostic, NOT optimised) ===")
    for a in (1.0, 0.7, 0.5):
        b = blend(a_xs=a, a_carry=1 - a)
        print(f"  {int(a*100)}/{int((1-a)*100)} xs/carry: "
              f"Sharpe={sharpe_ratio(b.returns):+.2f}  maxDD={max_drawdown(b.equity):.1%}  "
              f"net={ (b.equity.iloc[-1]-1)*1e4:+6.0f}bp")


if __name__ == "__main__":
    main()
