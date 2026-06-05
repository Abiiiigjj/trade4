import logging
import os
from functools import lru_cache
import ccxt
import pandas as pd

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_exchange() -> ccxt.okx:
    return ccxt.okx({
        "apiKey": os.getenv("OKX_API_KEY", ""),
        "secret": os.getenv("OKX_API_SECRET", ""),
        "password": os.getenv("OKX_PASSPHRASE", ""),
    })


def fetch_funding_history(
    symbol: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Fetch funding rate history. symbol: OKX format e.g. 'DOGE-USDT-SWAP'."""
    ex = _get_exchange()
    ccxt_symbol = _to_ccxt(symbol)
    since = int(start_ts.timestamp() * 1000)
    until = int(end_ts.timestamp() * 1000) if end_ts else None

    rows = []
    while True:
        batch = ex.fetch_funding_rate_history(ccxt_symbol, since=since, limit=100)
        if not batch:
            break
        rows.extend(batch)
        last_ts = batch[-1]["timestamp"]
        if until and last_ts >= until:
            break
        if len(batch) < 100:
            break
        since = last_ts + 1

    if not rows:
        return pd.DataFrame(columns=["timestamp", "symbol", "funding_rate"])

    df = pd.DataFrame({
        "timestamp": pd.to_datetime([r["timestamp"] for r in rows], unit="ms", utc=True).astype("datetime64[ns, UTC]"),
        "symbol": symbol,
        "funding_rate": [float(r["fundingRate"]) for r in rows],
    })
    if until:
        df = df[df["timestamp"] <= pd.Timestamp(until, unit="ms", tz="UTC")]
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def fetch_ohlcv(
    symbol: str,
    interval: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    ex = _get_exchange()
    ccxt_symbol = _to_ccxt(symbol)
    since = int(start_ts.timestamp() * 1000)
    until = int(end_ts.timestamp() * 1000) if end_ts else None

    rows = []
    while True:
        batch = ex.fetch_ohlcv(ccxt_symbol, timeframe=interval, since=since, limit=300)
        if not batch:
            break
        rows.extend(batch)
        last_ts = batch[-1][0]
        if until and last_ts >= until:
            break
        if len(batch) < 300:
            break
        since = last_ts + 1

    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).astype("datetime64[ns, UTC]")
    if until:
        df = df[df["timestamp"] <= pd.Timestamp(until, unit="ms", tz="UTC")]
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def fetch_orderbook(symbol: str, limit: int = 20) -> pd.DataFrame:
    ex = _get_exchange()
    ccxt_symbol = _to_ccxt(symbol)
    book = ex.fetch_order_book(ccxt_symbol, limit=limit)
    bids = pd.DataFrame(book["bids"], columns=["price", "qty"])
    bids["side"] = "bid"
    asks = pd.DataFrame(book["asks"], columns=["price", "qty"])
    asks["side"] = "ask"
    return pd.concat([bids, asks], ignore_index=True)


def list_perp_symbols() -> list[str]:
    ex = _get_exchange()
    markets = ex.load_markets()
    return [
        m["id"] for m in markets.values()
        if m.get("type") == "swap" and m.get("quote") == "USDT" and m.get("active")
    ]


def _to_ccxt(symbol: str) -> str:
    """OKX symbols like 'DOGE-USDT-SWAP' -> ccxt format 'DOGE/USDT:USDT'."""
    if "-SWAP" in symbol:
        parts = symbol.replace("-SWAP", "").split("-")
        return f"{parts[0]}/{parts[1]}:{parts[1]}"
    return symbol
