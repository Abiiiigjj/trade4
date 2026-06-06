import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from trade4.data.binance import FDUSD_ZERO_FEE_BASES

logger = logging.getLogger(__name__)

FUNDING_INTERVALS_PER_DAY = 3  # 8h intervals


# Maker-only FDUSD round-trip cost in bps (2× perp_maker entry/exit + slippage + basis + depeg)
_MAKER_FDUSD_RT_BPS: float = 16.5

# Minimum hold days used for stress-gate calculation
_STRESS_HOLD_DAYS: int = 7


@dataclass
class ScreenerConfig:
    entry_threshold_per_interval: float = 0.00005   # 0.005% per 8h
    max_slippage_bps: float = 50.0
    position_size_eur: float = 500.0
    min_pct_positive: float = 0.6
    volume_fraction_cap: float = 0.005  # max fraction of 24h volume
    min_intervals: float = 270.0        # 90 days × 3 intervals/day
    require_positive_90d: bool = True   # avg_funding_90d must also be positive
    stress_gate: bool = True            # flip_1x net_edge must be > 0


def compute_funding_stats(funding_df: pd.DataFrame) -> dict[str, float]:
    now = funding_df["timestamp"].max()
    d30 = now - pd.Timedelta(days=30)
    d90 = now - pd.Timedelta(days=90)
    df30 = funding_df[funding_df["timestamp"] >= d30]
    df90 = funding_df[funding_df["timestamp"] >= d90]
    return {
        "avg_funding_30d": float(df30["funding_rate"].mean()) if len(df30) else 0.0,
        "avg_funding_90d": float(df90["funding_rate"].mean()) if len(df90) else 0.0,
        "pct_positive_intervals": float((funding_df["funding_rate"] > 0).mean()),
        "n_intervals": float(len(funding_df)),
    }


def estimate_slippage_bps(
    orderbook_df: pd.DataFrame,
    notional_eur: float,
    price_eur: float,
    side: str,
) -> float:
    """Walk the orderbook and compute average fill price vs mid-price.

    ``side`` accepts ``"ask"``/``"bid"`` (orderbook column values) or
    ``"buy"``/``"sell"`` as aliases.
    """
    side_map = {"buy": "ask", "sell": "bid"}
    ob_side = side_map.get(side, side)

    if price_eur <= 0:
        return 50.0
    qty_needed = notional_eur / price_eur
    levels = orderbook_df[orderbook_df["side"] == ob_side].sort_values(
        "price", ascending=(ob_side == "ask")
    )
    if levels.empty:
        return 50.0

    filled_qty = 0.0
    filled_cost = 0.0
    for _, row in levels.iterrows():
        take = min(row["qty"], qty_needed - filled_qty)
        filled_qty += take
        filled_cost += take * row["price"]
        if filled_qty >= qty_needed:
            break

    if filled_qty == 0:
        return 50.0

    avg_fill = filled_cost / filled_qty
    # Compute mid-price from orderbook for a neutral reference
    best_ask = orderbook_df[orderbook_df["side"] == "ask"]["price"].min()
    best_bid = orderbook_df[orderbook_df["side"] == "bid"]["price"].max()
    mid_price = (best_ask + best_bid) / 2.0 if (best_ask > 0 and best_bid > 0) else price_eur
    # Slippage = deviation of avg fill from mid, in basis points
    slippage = abs(avg_fill - mid_price) / mid_price * 10_000
    return round(slippage, 4)


def screen_coins(
    symbols: list[str],
    funding_data: dict[str, pd.DataFrame],
    orderbook_data: dict[str, pd.DataFrame],
    ohlcv_data: dict[str, pd.DataFrame],
    config: ScreenerConfig,
) -> pd.DataFrame:
    rows = []
    for symbol in symbols:
        if symbol not in funding_data or funding_data[symbol].empty:
            continue
        stats = compute_funding_stats(funding_data[symbol])

        # Hard filter 1: minimum history
        if stats["n_intervals"] < config.min_intervals:
            continue
        # Hard filter 2: both 30d AND 90d must be above threshold
        if stats["avg_funding_30d"] < config.entry_threshold_per_interval:
            continue
        if config.require_positive_90d and stats["avg_funding_90d"] < config.entry_threshold_per_interval:
            continue
        if stats["pct_positive_intervals"] < config.min_pct_positive:
            continue
        # Hard filter 3: stress gate — 30% of funding over 7 days must cover maker FDUSD costs
        if config.stress_gate:
            flip_funding_bps = (
                stats["avg_funding_30d"] * 3 * _STRESS_HOLD_DAYS * 0.3 * 10_000
            )
            if flip_funding_bps <= _MAKER_FDUSD_RT_BPS:
                continue

        slippage_bps = 0.0
        if symbol in orderbook_data and not orderbook_data[symbol].empty:
            ob = orderbook_data[symbol]
            best_ask = ob[ob["side"] == "ask"]["price"].min()
            slippage_bps = estimate_slippage_bps(ob, config.position_size_eur, best_ask, "ask")

        if slippage_bps > config.max_slippage_bps:
            continue

        if symbol in ohlcv_data and not ohlcv_data[symbol].empty:
            ohlcv = ohlcv_data[symbol]
            avg_volume = float(ohlcv["volume"].tail(30).mean())
            best_ask = (
                orderbook_data[symbol][orderbook_data[symbol]["side"] == "ask"]["price"].min()
                if symbol in orderbook_data and not orderbook_data[symbol].empty
                else 1.0
            )
            position_qty = config.position_size_eur / best_ask if best_ask > 0 else 0.0
            if avg_volume > 0 and (position_qty / avg_volume) > config.volume_fraction_cap:
                continue

        base = symbol.replace("USDT", "").replace("FDUSD", "")
        fdusd_eligible = bool(base in FDUSD_ZERO_FEE_BASES)

        rows.append({
            "symbol": symbol,
            "avg_funding_30d": stats["avg_funding_30d"],
            "avg_funding_90d": stats["avg_funding_90d"],
            "pct_positive_intervals": stats["pct_positive_intervals"],
            "n_intervals": stats["n_intervals"],
            "slippage_est_bps": slippage_bps,
            "fdusd_zero_fee": fdusd_eligible,
            "gate_candidate": True,
        })

    if not rows:
        return pd.DataFrame()

    df = (
        pd.DataFrame(rows)
        .sort_values("avg_funding_30d", ascending=False)
        .reset_index(drop=True)
    )
    # Ensure Python bool columns so `is True` checks work in tests/callers
    for col in ("fdusd_zero_fee", "gate_candidate"):
        if col in df.columns:
            df[col] = df[col].astype(object)
    return df
