from dataclasses import dataclass
from typing import Literal

BINANCE_MAINT_MARGIN_RATE: float = 0.005  # 0.5% for ≤20x leverage on USDT-M


@dataclass
class PositionParams:
    qty: float
    leverage: int
    notional: float
    margin: float
    actual_risk_eur: float
    sl_price: float
    liq_price: float
    liq_buffer_ok: bool


def _liquidation_price(
    entry: float,
    leverage: int,
    side: Literal["long", "short"],
) -> float:
    if side == "long":
        return entry * (1 - 1 / leverage + BINANCE_MAINT_MARGIN_RATE)
    return entry * (1 + 1 / leverage - BINANCE_MAINT_MARGIN_RATE)


def size_position(
    balance: float,
    entry_price: float,
    sl_price: float,
    side: Literal["long", "short"],
    max_risk_fraction: float = 0.02,
    max_leverage: int = 20,
    min_liq_buffer: float = 0.20,
) -> PositionParams:
    """Compute position size for a trade with fixed risk fraction.

    If the max_leverage cap binds, qty is reduced so risk < max_risk_fraction.
    leverage is the pure ratio notional/margin and is not the profit lever
    (profit is determined by qty).
    """
    sl_distance = abs(entry_price - sl_price)
    if sl_distance == 0:
        raise ValueError("sl_price must differ from entry_price")

    risk_eur = balance * max_risk_fraction
    qty = risk_eur / sl_distance
    notional = qty * entry_price

    raw_leverage = notional / balance
    leverage = min(max(1, int(raw_leverage) + 1), max_leverage)

    if raw_leverage > max_leverage:
        qty = (balance * max_leverage) / entry_price
        notional = qty * entry_price
        risk_eur = qty * sl_distance

    liq_price = _liquidation_price(entry_price, max_leverage, side)

    # Directional buffer: SL must be between entry and liq_price (not beyond liq).
    # For a long: entry > sl > liq_price (good). If sl < liq_price, sl_to_liq < 0 → False.
    if side == "long":
        entry_to_liq = entry_price - liq_price
        sl_to_liq = sl_price - liq_price
    else:
        entry_to_liq = liq_price - entry_price
        sl_to_liq = liq_price - sl_price
    liq_buffer_ok = entry_to_liq > 0 and (sl_to_liq / entry_to_liq) >= min_liq_buffer

    return PositionParams(
        qty=qty,
        leverage=leverage,
        notional=notional,
        margin=balance,
        actual_risk_eur=risk_eur,
        sl_price=sl_price,
        liq_price=liq_price,
        liq_buffer_ok=liq_buffer_ok,
    )


def circuit_breaker_triggered(
    realized_pnl_today: float,
    day_start_balance: float,
    daily_loss_limit: float = 0.03,
) -> bool:
    """Returns True when realized PnL today exceeds the daily loss limit.

    Uses realized PnL only (not unrealized) to avoid premature firing
    during temporary drawdowns in open positions.
    """
    return realized_pnl_today < -(day_start_balance * daily_loss_limit)
