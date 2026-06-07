import logging
import time
import requests
import pandas as pd

logger = logging.getLogger(__name__)

FDUSD_ZERO_FEE_BASES: frozenset[str] = frozenset(
    {"BTC", "ETH", "SOL", "DOGE", "LINK", "BNB", "XRP"}
)

_BASE = "https://fapi.binance.com"
_SESSION = requests.Session()


def _get(path: str, params: dict) -> dict | list:
    resp = _SESSION.get(f"{_BASE}{path}", params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_funding_history(
    symbol: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Fetch funding rate history — public endpoint, no auth needed."""
    since = int(start_ts.timestamp() * 1000)
    until = int(end_ts.timestamp() * 1000) if end_ts else None

    rows = []
    while True:
        params: dict = {"symbol": symbol, "startTime": since, "limit": 1000}
        if until:
            params["endTime"] = until
        batch = _get("/fapi/v1/fundingRate", params)
        if not batch:
            break
        rows.extend(batch)
        last_ts = int(batch[-1]["fundingTime"])
        if len(batch) < 1000:
            break
        if until and last_ts >= until:
            break
        since = last_ts + 1
        time.sleep(0.1)

    if not rows:
        return pd.DataFrame(columns=["timestamp", "symbol", "funding_rate"])

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(
            [int(r["fundingTime"]) for r in rows], unit="ms", utc=True
        ).astype("datetime64[ns, UTC]"),
        "symbol": symbol,
        "funding_rate": [float(r["fundingRate"]) for r in rows],
    })
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def fetch_ohlcv(
    symbol: str,
    interval: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Fetch OHLCV candles — public endpoint, no auth needed."""
    since = int(start_ts.timestamp() * 1000)
    until = int(end_ts.timestamp() * 1000) if end_ts else None

    rows = []
    while True:
        params: dict = {"symbol": symbol, "interval": interval, "startTime": since, "limit": 1500}
        if until:
            params["endTime"] = until
        batch = _get("/fapi/v1/klines", params)
        if not batch:
            break
        rows.extend(batch)
        last_ts = int(batch[-1][0])
        if len(batch) < 1500:
            break
        if until and last_ts >= until:
            break
        since = last_ts + 1
        time.sleep(0.1)

    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(
        [[int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])] for r in rows],
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).astype("datetime64[ns, UTC]")
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def fetch_orderbook(symbol: str, limit: int = 20) -> pd.DataFrame:
    """Fetch current orderbook snapshot — public endpoint, no auth needed."""
    data = _get("/fapi/v1/depth", {"symbol": symbol, "limit": limit})
    bids = pd.DataFrame(data["bids"], columns=["price", "qty"]).astype(float)
    bids["side"] = "bid"
    asks = pd.DataFrame(data["asks"], columns=["price", "qty"]).astype(float)
    asks["side"] = "ask"
    return pd.concat([bids, asks], ignore_index=True)


def list_perp_symbols() -> list[str]:
    """Return all USDT-margined perpetual symbols — public endpoint, no auth needed."""
    data = _get("/fapi/v1/exchangeInfo", {})
    return [
        s["symbol"] for s in data["symbols"]
        if s.get("contractType") == "PERPETUAL"
        and s.get("quoteAsset") == "USDT"
        and s.get("status") == "TRADING"
    ]


def list_perp_symbols_with_onboard() -> dict[str, pd.Timestamp]:
    """Returns {symbol: onboard_timestamp} for all active USDT-M perpetuals.

    onboardDate from exchangeInfo is used for point-in-time universe filtering
    in backtests — prevents survivorship bias and look-ahead bias.
    """
    data = _get("/fapi/v1/exchangeInfo", {})
    result: dict[str, pd.Timestamp] = {}
    for s in data["symbols"]:
        if (
            s.get("contractType") == "PERPETUAL"
            and s.get("quoteAsset") == "USDT"
            and s.get("status") == "TRADING"
        ):
            if "onboardDate" not in s:
                logger.warning("symbol %s has no onboardDate — skipped from universe", s["symbol"])
                continue
            onboard_ms = int(s["onboardDate"])
            result[s["symbol"]] = pd.Timestamp(onboard_ms, unit="ms", tz="UTC")
    return result


def filter_symbols_at_date(
    symbol_onboard: dict[str, pd.Timestamp],
    as_of: pd.Timestamp,
) -> list[str]:
    """Return symbols that were listed on or before as_of (inclusive)."""
    return [sym for sym, ts in symbol_onboard.items() if ts <= as_of]
