"""Paper trading executor: real market data, simulated order fills."""
import logging

import requests

logger = logging.getLogger(__name__)

_SPOT_BASE = "https://api.binance.com"
_PERP_BASE = "https://fapi.binance.com"

_SESSION = requests.Session()


def _get(base: str, path: str, **params) -> dict | list:
    resp = _SESSION.get(f"{base}{path}", params=params or None, timeout=10)
    resp.raise_for_status()
    return resp.json()


class PaperExecutor:
    """Simulates order execution using real market prices.

    No API keys required — uses public endpoints only.
    Fills are assumed at the current best bid/ask (no partial fills, no real slippage).
    """

    def __init__(self, start_balance_eur: float = 2000.0, eur_to_usdt: float = 1.08) -> None:
        self._balance_eur = start_balance_eur
        self._eur_to_usdt = eur_to_usdt
        # Simulated per-asset holdings: {symbol: {"spot_qty": float, "perp_qty": float}}
        self._holdings: dict[str, dict] = {}
        logger.info("PaperExecutor started — balance %.0f EUR (%.0f USDT)",
                    start_balance_eur, start_balance_eur * eur_to_usdt)

    # ── Price feeds (real public API) ─────────────────────────────────────────

    def get_spot_price(self, symbol: str) -> float:
        """Best bid price for spot symbol."""
        data = _get(_SPOT_BASE, "/api/v3/ticker/bookTicker", symbol=symbol)
        return float(data["bidPrice"])

    def get_perp_price(self, symbol: str) -> float:
        """Best ask price for perp symbol (entry = buying the ask)."""
        data = _get(_PERP_BASE, "/fapi/v1/ticker/bookTicker", symbol=symbol)
        return float(data["askPrice"])

    def get_perp_mark_price(self, symbol: str) -> float:
        """Mark price for PnL estimation."""
        data = _get(_PERP_BASE, "/fapi/v1/premiumIndex", symbol=symbol)
        return float(data["markPrice"])

    # ── Simulated balance (public-API spot price × simulated qty) ─────────────

    def spot_balance(self, asset: str) -> float:
        """Returns simulated free balance as USDT-equivalent."""
        if asset in ("USDT", "FDUSD"):
            # Committed capital is tracked outside; return remaining
            committed = sum(
                h["spot_qty"] * self.get_spot_price(sym + "USDT")
                for sym, h in self._holdings.items()
                if h.get("spot_qty", 0) > 0
            )
            total_usdt = self._balance_eur * self._eur_to_usdt
            return max(0.0, total_usdt - committed)
        return 0.0

    def futures_usdt_balance(self) -> float:
        """Simulated futures account balance."""
        return self._balance_eur * self._eur_to_usdt * 0.5  # half allocated to futures margin

    # ── Simulated order execution ─────────────────────────────────────────────

    def open_delta_neutral(
        self,
        symbol: str,
        notional_eur: float,
        fdusd_eligible: bool,
        perp_leverage: int = 3,
        eur_to_usdt: float = 1.08,
    ) -> tuple[float, float, float, float]:
        """Simulate opening: fill at current bid (spot) and ask (perp)."""
        notional_usdt = notional_eur * eur_to_usdt

        # Use USDT spot pair for price (FDUSD not always on public spot API)
        spot_price = self.get_spot_price(symbol)
        perp_price = self.get_perp_price(symbol)

        qty = notional_usdt / spot_price

        # Track simulated position
        self._holdings[symbol] = {"spot_qty": qty, "perp_qty": qty}

        logger.info("[PAPER] Opened %s: qty=%.6f spot=%.2f perp=%.2f notional=%.0f EUR",
                    symbol, qty, spot_price, perp_price, notional_eur)
        return qty, spot_price, qty, perp_price

    def close_delta_neutral(
        self,
        symbol: str,
        spot_qty: float,
        perp_qty: float,
        fdusd_eligible: bool,
    ) -> tuple[float, float]:
        """Simulate closing: fill at current ask (spot) and bid (perp)."""
        # Exit spot at best ask (opposite of entry bid)
        data = _get(_SPOT_BASE, "/api/v3/ticker/bookTicker", symbol=symbol)
        spot_exit = float(data["askPrice"])

        # Exit perp short by buying at best bid
        data_p = _get(_PERP_BASE, "/fapi/v1/ticker/bookTicker", symbol=symbol)
        perp_exit = float(data_p["bidPrice"])

        self._holdings.pop(symbol, None)

        logger.info("[PAPER] Closed %s: spot_exit=%.2f perp_exit=%.2f", symbol, spot_exit, perp_exit)
        return spot_exit, perp_exit

    def set_perp_hedge_qty(
        self,
        symbol: str,
        target_qty: float,
        current_qty: float,
    ) -> None:
        """Simulate rebalancing the perp hedge."""
        if symbol in self._holdings:
            self._holdings[symbol]["perp_qty"] = target_qty
        logger.info("[PAPER] Rebalanced %s perp: %.6f → %.6f", symbol, current_qty, target_qty)
