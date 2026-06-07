"""Position lifecycle: decide which positions to open, close, or rebalance."""
import logging
from dataclasses import dataclass

import pandas as pd

from trade4.live.state import OpenPosition
from trade4.backtest.engine import REBALANCE_DELTA_THRESHOLD

logger = logging.getLogger(__name__)

# Exit when funding drops below this (rate per 8h interval)
EXIT_FUNDING_THRESHOLD: float = 0.0
# Exit after this many days regardless
MAX_HOLDING_DAYS: int = 30


@dataclass
class PositionDecision:
    symbol: str
    action: str  # "open" | "close" | "rebalance" | "hold"
    reason: str
    target_spot_qty: float = 0.0    # for rebalance
    target_perp_qty: float = 0.0


def should_exit(
    position: OpenPosition,
    latest_funding_rate: float,
    current_price: float,
) -> tuple[bool, str]:
    """Return (exit, reason) based on funding rate and holding period."""
    holding_days = (
        pd.Timestamp.now(tz="UTC") - pd.Timestamp(position.opened_at)
    ).days

    if latest_funding_rate <= EXIT_FUNDING_THRESHOLD:
        return True, f"funding_below_threshold ({latest_funding_rate:.5f})"
    if holding_days >= MAX_HOLDING_DAYS:
        return True, f"max_holding_days ({holding_days}d)"
    return False, ""


def needs_rebalance(
    position: OpenPosition,
    current_price: float,
) -> tuple[bool, float]:
    """Return (rebalance_needed, target_perp_qty) based on delta drift.

    Delta drift occurs when the spot value changes but the perp hedge
    remains at the original notional. We rebalance when drift > threshold.
    """
    original_notional = position.spot_qty * position.spot_entry_price
    current_notional  = position.spot_qty * current_price
    drift = abs(current_notional - original_notional) / original_notional

    if drift >= REBALANCE_DELTA_THRESHOLD:
        # Adjust perp qty so it matches current spot value
        target_perp_qty = position.spot_qty  # qty stays same, price moved
        return True, target_perp_qty

    return False, position.perp_qty


def compute_entries(
    screener_df: pd.DataFrame,
    open_symbols: set[str],
    max_positions: int,
    notional_per_position_eur: float,
) -> list[dict]:
    """Return list of new positions to open.

    Picks top screened symbols not already open, up to fill available slots.
    """
    available_slots = max_positions - len(open_symbols)
    if available_slots <= 0:
        return []

    candidates = screener_df[~screener_df["symbol"].isin(open_symbols)].copy()
    candidates = candidates.sort_values("avg_funding_30d", ascending=False)
    to_open = candidates.head(available_slots)

    return [
        {
            "symbol": row["symbol"],
            "notional_eur": notional_per_position_eur,
            "fdusd_eligible": bool(row.get("fdusd_zero_fee", False)),
        }
        for _, row in to_open.iterrows()
    ]


def make_decisions(
    open_positions: list[OpenPosition],
    screener_df: pd.DataFrame,
    funding_data: dict[str, pd.DataFrame],
    current_prices: dict[str, float],
    max_positions: int,
    notional_per_position_eur: float,
) -> list[PositionDecision]:
    """Full cycle decision-making: exit checks → rebalance checks → new entries."""
    decisions: list[PositionDecision] = []
    symbols_after_exits: set[str] = set()

    for pos in open_positions:
        sym = pos.symbol
        price = current_prices.get(sym, pos.spot_entry_price)

        # Latest funding rate for this symbol
        fd = funding_data.get(sym, pd.DataFrame())
        if fd.empty:
            latest_rate = 0.0
        else:
            latest_rate = float(fd.sort_values("timestamp").iloc[-1]["funding_rate"])

        exit_flag, exit_reason = should_exit(pos, latest_rate, price)
        if exit_flag:
            decisions.append(PositionDecision(sym, "close", exit_reason))
        else:
            rebal, target_perp = needs_rebalance(pos, price)
            if rebal:
                decisions.append(PositionDecision(
                    sym, "rebalance",
                    f"delta_drift>{REBALANCE_DELTA_THRESHOLD:.0%}",
                    target_spot_qty=pos.spot_qty,
                    target_perp_qty=target_perp,
                ))
            else:
                decisions.append(PositionDecision(sym, "hold", ""))
            symbols_after_exits.add(sym)

    # Compute new entries for available slots
    entries = compute_entries(
        screener_df, symbols_after_exits, max_positions, notional_per_position_eur
    )
    for e in entries:
        sym = e["symbol"]
        fd = funding_data.get(sym, pd.DataFrame())
        if fd.empty:
            latest_rate = 0.0
        else:
            latest_rate = float(fd.sort_values("timestamp").iloc[-1]["funding_rate"])
        if latest_rate <= EXIT_FUNDING_THRESHOLD:
            logger.info("Skipping open %s — current funding %.5f below threshold", sym, latest_rate)
            continue
        decisions.append(PositionDecision(
            sym, "open",
            f"screener_top (fdusd={e['fdusd_eligible']})",
        ))

    return decisions
