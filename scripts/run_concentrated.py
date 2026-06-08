"""The EUR 2k-tradeable concentrated variant: top-K xs_funding (perp-only, no hedge).

The 23-name research book is diversified but capital-hungry. xs_funding is perp-only, so a
concentrated top-K version (long K lowest- / short K highest-funding perps) is tradeable with
EUR 2k. This quantifies the concentration trade-off (fewer names -> higher drawdown) AND checks
real tradeability (positions x per-position notional vs Binance min-notional).

Run after the cache is warm. Seconds, no downloads.
"""
import logging
from dataclasses import replace

import pandas as pd

from trade4.research.panel_builder import build_panel
from trade4.research.portfolio_engine import run_portfolio_backtest, EngineConfig
from trade4.research.strategies.xs_funding import XSFunding
from trade4.research.strategies.periodic import PeriodicRebalance
from trade4.research.overlay import vol_target, releveraging_cost
from trade4.research.metrics import sharpe_ratio, max_drawdown
from scripts.run_study import DEFAULT_UNIVERSE, _months, _load

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("concentrated")

CFG = EngineConfig(cost_bps=5.0, n_legs=1, price_pnl_enabled=True)  # perp-only, no hedge
CAPITAL_USD = 2150.0       # ~EUR 2k
MIN_NOTIONAL_USD = 5.0     # Binance USDT-M futures floor (practical comfort ~20)
TARGET_VOL, VOL_LB, MAX_LEV, RELEV_BPS = 0.10, 30, 2.0, 5.0


def _per_year(returns):
    by = returns.groupby(returns.index.year).apply(lambda r: sharpe_ratio(r))
    return "  ".join(f"{y}:{s:+.1f}" for y, s in by.items())


def main() -> None:
    months = _months("2022-01", "2025-06")
    funding, klines = {}, {}
    for s in DEFAULT_UNIVERSE:
        f, k = _load(s, months)
        if not f.empty and not k.empty:
            funding[s], klines[s] = f, k
    panel = build_panel(funding, klines, bar="8h")
    logger.info("panel: %d bars x %d symbols", len(panel.times), len(panel.symbols))

    print(f"\nConcentrated xs_funding (weekly, perp-only, vol-targeted) — tradeability at "
          f"${CAPITAL_USD:.0f}:")
    print(f"{'config':<16}{'Sharpe':>7}{'maxDD':>8}{'net':>9}  {'pos':>4}{'$/pos':>8}  trade?  per-year")
    configs = [("top_k=1", dict(top_k=1)), ("top_k=2", dict(top_k=2)),
               ("top_k=3", dict(top_k=3)), ("top_k=5", dict(top_k=5)),
               ("quantile=0.25", dict(quantile=0.25))]
    for label, params in configs:
        strat = PeriodicRebalance(base=XSFunding(lookback=3, **params), every=21)
        w = strat.generate_target_weights(panel)
        res = run_portfolio_backtest(panel, w, CFG)
        scaled, exp = vol_target(res.returns, TARGET_VOL, VOL_LB, MAX_LEV)
        net = scaled - releveraging_cost(exp, RELEV_BPS)
        eq = (1.0 + net).cumprod()

        # capacity: avg positions held and per-position notional at this capital
        names_per_bar = (w != 0).sum(axis=1)
        avg_names = names_per_bar[names_per_bar > 0].mean()
        avg_exp = exp[exp > 0].mean()
        gross_usd = avg_exp * CAPITAL_USD
        per_pos = gross_usd / avg_names if avg_names else 0.0
        tradeable = "YES" if per_pos >= MIN_NOTIONAL_USD else "no"

        print(f"{label:<16}{sharpe_ratio(net):>7.2f}{max_drawdown(eq):>8.1%}"
              f"{(eq.iloc[-1]-1)*1e4:>+8.0f}bp  {avg_names:>4.0f}{per_pos:>8.0f}  {tradeable:>5}   "
              f"{_per_year(net)}")


if __name__ == "__main__":
    main()
