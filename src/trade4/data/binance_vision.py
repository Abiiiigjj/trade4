"""Point-in-time historical fetch from the public data.binance.vision dumps.

These monthly archives include **delisted** USDT-M perpetual symbols (e.g. SRMUSDT),
which the live REST API no longer serves. They are the mechanism for a true
point-in-time universe with no survivorship bias.

URL patterns (USDT-M futures):
* funding: ``.../monthly/fundingRate/{SYM}/{SYM}-fundingRate-{YYYY-MM}.zip``
* klines:  ``.../monthly/klines/{SYM}/{INTERVAL}/{SYM}-{INTERVAL}-{YYYY-MM}.zip``
"""
import io
import logging
import urllib.error
import urllib.request
import zipfile

import pandas as pd

logger = logging.getLogger(__name__)

_BASE = "https://data.binance.vision/data/futures/um/monthly"
_TIMEOUT = 30

# kline column order in the vision dumps
_KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]


def _download_csv(url: str) -> pd.DataFrame | None:
    """Download a zipped CSV; return raw DataFrame or None on 404."""
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            logger.debug("vision 404: %s", url)
            return None
        raise
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as fh:
            first = fh.readline()
        has_header = not first.split(b",")[0].strip().replace(b"-", b"").isdigit()
        with zf.open(name) as fh:
            return pd.read_csv(fh, header=0 if has_header else None)


def fetch_funding_month(symbol: str, year: int, month: int) -> pd.DataFrame:
    """Funding rates for one symbol-month → ``[timestamp, funding_rate, interval_hours]``.

    Empty frame if the dump does not exist (symbol not listed that month).
    """
    url = f"{_BASE}/fundingRate/{symbol}/{symbol}-fundingRate-{year}-{month:02d}.zip"
    raw = _download_csv(url)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["timestamp", "funding_rate", "interval_hours"])
    # columns: calc_time, funding_interval_hours, last_funding_rate
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(raw.iloc[:, 0].astype("int64"), unit="ms", utc=True),
        "interval_hours": raw.iloc[:, 1].astype("int64"),
        "funding_rate": raw.iloc[:, -1].astype(float),
    })
    return df[["timestamp", "funding_rate", "interval_hours"]].sort_values(
        "timestamp"
    ).reset_index(drop=True)


def fetch_klines_month(
    symbol: str, interval: str, year: int, month: int
) -> pd.DataFrame:
    """OHLCV for one symbol-month → ``[timestamp, open, high, low, close, volume]``.

    Empty frame if the dump does not exist.
    """
    url = f"{_BASE}/klines/{symbol}/{interval}/{symbol}-{interval}-{year}-{month:02d}.zip"
    raw = _download_csv(url)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    raw = raw.iloc[:, : len(_KLINE_COLS)].copy()
    raw.columns = _KLINE_COLS
    out = pd.DataFrame({
        "timestamp": pd.to_datetime(raw["open_time"].astype("int64"), unit="ms", utc=True),
        "open": raw["open"].astype(float),
        "high": raw["high"].astype(float),
        "low": raw["low"].astype(float),
        "close": raw["close"].astype(float),
        "volume": raw["volume"].astype(float),
    })
    return out.sort_values("timestamp").reset_index(drop=True)


def infer_listing_window(
    symbol: str, start_year: int = 2020, end_year: int = 2026
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Scan funding dumps to find a symbol's first and last available month.

    Returns ``(first_ts, last_ts)`` — the point-in-time tradeable window — or
    ``(None, None)`` if no data exists.
    """
    first_ts: pd.Timestamp | None = None
    last_ts: pd.Timestamp | None = None
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            df = fetch_funding_month(symbol, year, month)
            if df.empty:
                continue
            if first_ts is None:
                first_ts = df["timestamp"].iloc[0]
            last_ts = df["timestamp"].iloc[-1]
    return first_ts, last_ts
