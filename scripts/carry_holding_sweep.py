"""Honest funding_carry: sweep the basis-drift / holding cost the gross run set to zero.

A delta-neutral carry's smooth gross curve assumes a perfect spot hedge. Real hedging
carries a per-bar drag (basis convergence noise, hedge maintenance). Rather than pick one
magic number, sweep it and report the break-even: 'carry survives if basis drift < X bp/bar'.

Run after the cache is warm (scripts/run_study.py populates it). Seconds, no downloads.
"""
import logging

import pandas as pd

from trade4.research.panel_builder import build_panel
from trade4.research.portfolio_engine import run_portfolio_backtest, EngineConfig
from trade4.research.strategies.funding_carry import FundingCarry
from trade4.research.strategies.periodic import PeriodicRebalance
from trade4.research.metrics import sharpe_ratio, max_drawdown
from scripts.run_study import DEFAULT_UNIVERSE, _months, _load

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("carry_sweep")


def main() -> None:
    months = _months("2022-01", "2025-06")
    funding, klines = {}, {}
    for s in DEFAULT_UNIVERSE:
        f, k = _load(s, months)
        if not f.empty and not k.empty:
            funding[s], klines[s] = f, k
    panel = build_panel(funding, klines, bar="8h")
    logger.info("panel: %d bars x %d symbols", len(panel.times), len(panel.symbols))

    # weekly-rebalanced carry (the low-turnover variant that was net-positive)
    strat = PeriodicRebalance(base=FundingCarry(entry_threshold=0.0001), every=21)
    w = strat.generate_target_weights(panel)

    print("\nfunding_carry (weekly rebalance), basis-drift / holding-cost sweep:")
    print(f"{'hc bp/bar':>10} {'hc bp/day':>10} {'net bps':>10} {'Sharpe':>8} {'maxDD':>8}")
    for hc in (0.0, 0.1, 0.25, 0.5, 1.0):
        cfg = EngineConfig(cost_bps=5.0, price_pnl_enabled=False, n_legs=2,
                           holding_cost_bps_per_bar=hc)
        res = run_portfolio_backtest(panel, w, cfg)
        net = (res.equity.iloc[-1] - 1.0) * 10_000
        print(f"{hc:>10.2f} {hc*3:>10.2f} {net:>10.0f} "
              f"{sharpe_ratio(res.returns):>8.2f} {max_drawdown(res.equity):>8.1%}")


if __name__ == "__main__":
    main()
