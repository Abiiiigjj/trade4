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
import os
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_BASE = "https://data.binance.vision/data/futures/um/monthly"
_TIMEOUT = 30

# Parquet cache: monthly vision dumps are immutable history, so cache them (incl. the
# empty 404 results) to avoid re-downloading ~1900 files on every study run.
_CACHE_DIR = Path(os.environ.get(
    "TRADE4_VISION_CACHE", str(Path.home() / ".cache" / "trade4_vision")))


def _cache_get(key: str) -> pd.DataFrame | None:
    p = _CACHE_DIR / f"{key}.parquet"
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception:  # corrupt cache file -> re-fetch
            logger.warning("corrupt cache, refetching: %s", p)
    return None


def _cache_put(key: str, df: pd.DataFrame) -> None:
    p = _CACHE_DIR / f"{key}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)

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
    key = f"funding/{symbol}-{year}-{month:02d}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    url = f"{_BASE}/fundingRate/{symbol}/{symbol}-fundingRate-{year}-{month:02d}.zip"
    raw = _download_csv(url)
    if raw is None or raw.empty:
        result = pd.DataFrame({
            "timestamp": pd.Series([], dtype="datetime64[ns, UTC]"),
            "funding_rate": pd.Series([], dtype="float64"),
            "interval_hours": pd.Series([], dtype="int64"),
        })
    else:
        # columns: calc_time, funding_interval_hours, last_funding_rate
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(raw.iloc[:, 0].astype("int64"), unit="ms", utc=True),
            "interval_hours": raw.iloc[:, 1].astype("int64"),
            "funding_rate": raw.iloc[:, -1].astype(float),
        })
        result = df[["timestamp", "funding_rate", "interval_hours"]].sort_values(
            "timestamp").reset_index(drop=True)
    _cache_put(key, result)
    return result


def fetch_klines_month(
    symbol: str, interval: str, year: int, month: int
) -> pd.DataFrame:
    """OHLCV for one symbol-month → ``[timestamp, open, high, low, close, volume]``.

    Empty frame if the dump does not exist.
    """
    key = f"klines_{interval}/{symbol}-{year}-{month:02d}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    url = f"{_BASE}/klines/{symbol}/{interval}/{symbol}-{interval}-{year}-{month:02d}.zip"
    raw = _download_csv(url)
    if raw is None or raw.empty:
        cols = {"timestamp": pd.Series([], dtype="datetime64[ns, UTC]")}
        cols.update({c: pd.Series([], dtype="float64")
                     for c in ("open", "high", "low", "close", "volume")})
        result = pd.DataFrame(cols)
    else:
        raw = raw.iloc[:, : len(_KLINE_COLS)].copy()
        raw.columns = _KLINE_COLS
        result = pd.DataFrame({
            "timestamp": pd.to_datetime(raw["open_time"].astype("int64"), unit="ms", utc=True),
            "open": raw["open"].astype(float),
            "high": raw["high"].astype(float),
            "low": raw["low"].astype(float),
            "close": raw["close"].astype(float),
            "volume": raw["volume"].astype(float),
        }).sort_values("timestamp").reset_index(drop=True)
    _cache_put(key, result)
    return result


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
