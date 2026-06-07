"""Telegram alert sender — fire-and-forget, never raises."""
import logging
import os

import requests

logger = logging.getLogger(__name__)

_BASE = "https://api.telegram.org"


def _send(text: str) -> None:
    token   = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        logger.debug("Telegram not configured, skipping alert")
        return
    try:
        requests.post(
            f"{_BASE}/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception as exc:
        logger.warning("Telegram send failed: %s", exc)


def alert_opened(symbol: str, notional_eur: float, spot_entry: float, perp_entry: float) -> None:
    _send(
        f"✅ *OPENED* `{symbol}`\n"
        f"Notional: {notional_eur:.0f} EUR\n"
        f"Spot entry: {spot_entry:.2f}  Perp entry: {perp_entry:.2f}"
    )


def alert_closed(symbol: str, net_pnl_eur: float, reason: str) -> None:
    emoji = "💰" if net_pnl_eur >= 0 else "🔴"
    _send(
        f"{emoji} *CLOSED* `{symbol}`\n"
        f"Net PnL: {net_pnl_eur:+.2f} EUR\n"
        f"Reason: {reason}"
    )


def alert_rebalanced(symbol: str, drift_pct: float) -> None:
    _send(f"⚖️ *REBALANCED* `{symbol}` (drift {drift_pct:.1%})")


def alert_error(context: str, error: str) -> None:
    _send(f"🚨 *ERROR* in `{context}`\n`{error}`")


def daily_summary(
    n_open: int,
    total_eur: float,
    daily_pnl_eur: float,
    open_symbols: list[str],
) -> None:
    symbols = ", ".join(f"`{s}`" for s in open_symbols) or "—"
    _send(
        f"📊 *Daily Summary*\n"
        f"Open positions: {n_open}  ({symbols})\n"
        f"Daily PnL: {daily_pnl_eur:+.2f} EUR\n"
        f"Total balance: {total_eur:.0f} EUR"
    )
