"""Binance REST executor: spot buy/sell + perp short/close via signed API."""
import hashlib
import hmac
import logging
import time
from typing import Literal
from urllib.parse import urlencode

import requests

logger = logging.getLogger(__name__)

_SPOT_BASE  = "https://api.binance.com"
_PERP_BASE  = "https://fapi.binance.com"
_TNET_SPOT  = "https://testnet.binance.vision"
_TNET_PERP  = "https://testnet.binancefuture.com"

# Maker order fill timeout: cancel and retry as market if not filled after N seconds
MAKER_TIMEOUT_SEC = 30


class BinanceExecutor:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False) -> None:
        self._key    = api_key
        self._secret = api_secret.encode()
        self._spot_base = _TNET_SPOT if testnet else _SPOT_BASE
        self._perp_base = _TNET_PERP if testnet else _PERP_BASE
        self._session = requests.Session()
        self._session.headers["X-MBX-APIKEY"] = api_key

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _sign(self, params: dict) -> dict:
        params["timestamp"] = int(time.time() * 1000)
        qs = urlencode(params)
        sig = hmac.new(self._secret, qs.encode(), hashlib.sha256).hexdigest()
        params["signature"] = sig
        return params

    def _spot(self, method: str, path: str, **params) -> dict:
        params = self._sign(params)
        resp = self._session.request(
            method, f"{self._spot_base}{path}", params=params, timeout=15
        )
        resp.raise_for_status()
        return resp.json()

    def _perp(self, method: str, path: str, **params) -> dict:
        params = self._sign(params)
        resp = self._session.request(
            method, f"{self._perp_base}{path}", params=params, timeout=15
        )
        resp.raise_for_status()
        return resp.json()

    # ── Account info ──────────────────────────────────────────────────────────

    def spot_balance(self, asset: str) -> float:
        """Free balance of an asset in spot account."""
        data = self._spot("GET", "/api/v3/account")
        for b in data["balances"]:
            if b["asset"] == asset:
                return float(b["free"])
        return 0.0

    def futures_usdt_balance(self) -> float:
        """Available USDT in futures account."""
        data = self._perp("GET", "/fapi/v2/account")
        for asset in data["assets"]:
            if asset["asset"] == "USDT":
                return float(asset["availableBalance"])
        return 0.0

    def perp_position(self, symbol: str) -> dict:
        """Current perp position for symbol. Returns positionAmt, entryPrice, etc."""
        data = self._perp("GET", "/fapi/v2/positionRisk", symbol=symbol)
        for p in data:
            if p["symbol"] == symbol:
                return p
        return {}

    def get_spot_price(self, symbol: str) -> float:
        """Best bid price for spot symbol."""
        data = self._spot("GET", "/api/v3/ticker/bookTicker", symbol=symbol)
        return float(data["bidPrice"])

    def get_perp_price(self, symbol: str) -> float:
        """Best ask price for perp symbol."""
        data = self._perp("GET", "/fapi/v1/ticker/bookTicker", symbol=symbol)
        return float(data["askPrice"])

    # ── Order placement ───────────────────────────────────────────────────────

    def _spot_lot_size(self, symbol: str) -> tuple[float, int]:
        """Returns (min_qty, qty_precision) for a spot symbol."""
        info = self._spot("GET", "/api/v3/exchangeInfo")
        for s in info["symbols"]:
            if s["symbol"] == symbol:
                for f in s["filters"]:
                    if f["filterType"] == "LOT_SIZE":
                        step = float(f["stepSize"])
                        precision = len(f["stepSize"].rstrip("0").split(".")[-1]) if "." in f["stepSize"] else 0
                        return float(f["minQty"]), precision
        return 0.0, 8

    def _round_qty(self, qty: float, precision: int) -> float:
        factor = 10 ** precision
        return int(qty * factor) / factor

    def _place_spot_limit_buy(
        self, symbol: str, qty: float, price: float
    ) -> dict:
        """Place limit buy on spot. Returns order dict."""
        _, prec = self._spot_lot_size(symbol)
        qty = self._round_qty(qty, prec)
        return self._spot(
            "POST", "/api/v3/order",
            symbol=symbol, side="BUY", type="LIMIT",
            timeInForce="GTX",  # post-only (maker)
            quantity=qty, price=f"{price:.8f}",
        )

    def _place_spot_market_buy(self, symbol: str, quote_qty: float) -> dict:
        """Market buy using quoteOrderQty (USDT amount)."""
        return self._spot(
            "POST", "/api/v3/order",
            symbol=symbol, side="BUY", type="MARKET",
            quoteOrderQty=f"{quote_qty:.2f}",
        )

    def _place_perp_limit_short(
        self, symbol: str, qty: float, price: float, leverage: int
    ) -> dict:
        """Set leverage then place limit short on perp."""
        self._perp("POST", "/fapi/v1/leverage", symbol=symbol, leverage=leverage)
        return self._perp(
            "POST", "/fapi/v1/order",
            symbol=symbol, side="SELL", positionSide="SHORT",
            type="LIMIT", timeInForce="GTX",
            quantity=f"{qty:.8f}", price=f"{price:.2f}",
        )

    def _cancel_order(self, symbol: str, order_id: int, is_perp: bool) -> None:
        if is_perp:
            self._perp("DELETE", "/fapi/v1/order", symbol=symbol, orderId=order_id)
        else:
            self._spot("DELETE", "/api/v3/order", symbol=symbol, orderId=order_id)

    def _wait_fill(
        self, symbol: str, order_id: int, is_perp: bool
    ) -> dict:
        """Poll until order is FILLED or timeout. Returns filled order."""
        deadline = time.time() + MAKER_TIMEOUT_SEC
        while time.time() < deadline:
            if is_perp:
                o = self._perp("GET", "/fapi/v1/order", symbol=symbol, orderId=order_id)
            else:
                o = self._spot("GET", "/api/v3/order", symbol=symbol, orderId=order_id)
            if o["status"] == "FILLED":
                return o
            time.sleep(1)
        raise TimeoutError(f"Order {order_id} not filled after {MAKER_TIMEOUT_SEC}s")

    # ── High-level entry / exit ───────────────────────────────────────────────

    def open_delta_neutral(
        self,
        symbol: str,
        notional_eur: float,
        fdusd_eligible: bool,
        perp_leverage: int = 3,
        eur_to_usdt: float = 1.0,
    ) -> tuple[float, float, float, float]:
        """Open delta-neutral position: spot long + perp short.

        Returns (spot_qty, spot_entry, perp_qty, perp_entry).
        Raises on any order failure.
        """
        notional_usdt = notional_eur * eur_to_usdt
        spot_symbol  = symbol.replace("USDT", "FDUSD") if fdusd_eligible else symbol
        perp_symbol  = symbol  # always USDT-M perp

        # Current prices
        spot_ask = self.get_spot_price(spot_symbol)
        perp_bid = self.get_perp_price(perp_symbol)

        qty = notional_usdt / spot_ask
        _, prec = self._spot_lot_size(spot_symbol)
        qty = self._round_qty(qty, prec)

        logger.info("Opening %s: qty=%.6f spot_price=%.2f perp_price=%.2f",
                    symbol, qty, spot_ask, perp_bid)

        # Place spot limit buy (post-only maker)
        spot_order = self._place_spot_limit_buy(spot_symbol, qty, spot_ask)
        try:
            spot_filled = self._wait_fill(spot_symbol, spot_order["orderId"], is_perp=False)
        except TimeoutError:
            self._cancel_order(spot_symbol, spot_order["orderId"], is_perp=False)
            # Fallback: market buy
            logger.warning("%s spot maker timeout — falling back to market", symbol)
            spot_filled = self._place_spot_market_buy(spot_symbol, notional_usdt)

        spot_qty   = float(spot_filled.get("executedQty", qty))
        spot_entry = float(spot_filled.get("price") or spot_filled.get("fills", [{}])[0].get("price", spot_ask))

        # Place perp limit short
        perp_order = self._place_perp_limit_short(perp_symbol, spot_qty, perp_bid, perp_leverage)
        try:
            perp_filled = self._wait_fill(perp_symbol, perp_order["orderId"], is_perp=True)
        except TimeoutError:
            self._cancel_order(perp_symbol, perp_order["orderId"], is_perp=True)
            logger.warning("%s perp maker timeout — falling back to market", symbol)
            self._perp("POST", "/fapi/v1/order",
                       symbol=perp_symbol, side="SELL", positionSide="SHORT",
                       type="MARKET", quantity=f"{spot_qty:.8f}")
            perp_filled = self.perp_position(perp_symbol)

        perp_qty   = float(perp_filled.get("executedQty") or abs(float(perp_filled.get("positionAmt", spot_qty))))
        perp_entry = float(perp_filled.get("avgPrice") or perp_filled.get("entryPrice", perp_bid))

        logger.info("Opened %s: spot_qty=%.6f@%.2f  perp_qty=%.6f@%.2f",
                    symbol, spot_qty, spot_entry, perp_qty, perp_entry)
        return spot_qty, spot_entry, perp_qty, perp_entry

    def close_delta_neutral(
        self,
        symbol: str,
        spot_qty: float,
        perp_qty: float,
        fdusd_eligible: bool,
    ) -> tuple[float, float]:
        """Close both legs with market orders. Returns (spot_exit, perp_exit)."""
        spot_symbol = symbol.replace("USDT", "FDUSD") if fdusd_eligible else symbol

        # Market sell spot
        spot_result = self._spot(
            "POST", "/api/v3/order",
            symbol=spot_symbol, side="SELL", type="MARKET",
            quantity=f"{spot_qty:.8f}",
        )
        spot_exit = float(
            spot_result.get("fills", [{}])[0].get("price", 0)
            or spot_result.get("price", 0)
        )

        # Market buy perp to close short
        perp_result = self._perp(
            "POST", "/fapi/v1/order",
            symbol=symbol, side="BUY", positionSide="SHORT",
            type="MARKET", quantity=f"{perp_qty:.8f}",
        )
        perp_exit = float(perp_result.get("avgPrice", 0))

        logger.info("Closed %s: spot_exit=%.2f  perp_exit=%.2f", symbol, spot_exit, perp_exit)
        return spot_exit, perp_exit

    def set_perp_hedge_qty(
        self,
        symbol: str,
        target_qty: float,
        current_qty: float,
    ) -> None:
        """Rebalance perp short to target_qty by adjusting delta."""
        diff = abs(target_qty - current_qty)
        if diff < 1e-6:
            return
        if target_qty > current_qty:
            # Need more short
            self._perp("POST", "/fapi/v1/order",
                       symbol=symbol, side="SELL", positionSide="SHORT",
                       type="MARKET", quantity=f"{diff:.8f}")
        else:
            # Reduce short
            self._perp("POST", "/fapi/v1/order",
                       symbol=symbol, side="BUY", positionSide="SHORT",
                       type="MARKET", quantity=f"{diff:.8f}")
        logger.info("Rebalanced %s perp: %.6f → %.6f", symbol, current_qty, target_qty)
