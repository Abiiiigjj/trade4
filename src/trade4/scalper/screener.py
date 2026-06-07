import pandas as pd
import numpy as np


def compute_atr_score(ohlcv_1h: pd.DataFrame, period: int = 14) -> float:
    """ATR as fraction of current price. Returns 0.0 if insufficient data."""
    df = ohlcv_1h.sort_values("timestamp") if "timestamp" in ohlcv_1h.columns else ohlcv_1h
    if len(df) < period + 1:
        return 0.0

    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(com=period - 1, adjust=False).mean().iloc[-1]
    price = df["close"].iloc[-1]
    return float(atr / price) if price > 0 else 0.0


def screen_by_volatility(
    ohlcv_data: dict[str, pd.DataFrame],
    volume_data_24h: dict[str, float],
    symbol_onboard: dict[str, pd.Timestamp] | None = None,
    as_of: pd.Timestamp | None = None,
    min_volume_usdt: float = 200_000_000.0,
    top_n: int = 20,
    atr_period: int = 14,
) -> list[str]:
    """Return top-N symbols by ATR volatility score.

    Filters:
    - 24h volume >= min_volume_usdt
    - Point-in-time: only symbols listed on or before as_of (if provided)

    Symbols are ranked highest ATR-score first.
    """
    scores: dict[str, float] = {}
    for sym, df in ohlcv_data.items():
        if df is None or df.empty:
            continue
        if symbol_onboard is not None and as_of is not None:
            onboard_ts = symbol_onboard.get(sym, pd.Timestamp.max.tz_localize("UTC"))
            if onboard_ts > as_of:
                continue
        vol = volume_data_24h.get(sym, 0.0)
        if vol < min_volume_usdt:
            continue
        scores[sym] = compute_atr_score(df, atr_period)

    ranked = sorted(scores, key=lambda s: scores[s], reverse=True)
    return ranked[:top_n]
