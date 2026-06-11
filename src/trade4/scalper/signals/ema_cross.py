import pandas as pd
import numpy as np


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    return 100 - 100 / (1 + rs)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def generate_signals(
    df_1m: pd.DataFrame,
    df_15m: pd.DataFrame,
    ema_fast: int = 9,
    ema_slow: int = 21,
    ema_trend: int = 200,
    rsi_period: int = 14,
    atr_period: int = 14,
    atr_min_pct: float = 0.0005,
    sl_atr_multiplier: float = 1.5,
) -> pd.DataFrame:
    """Generate EMA-cross momentum signals.

    Returns DataFrame with columns:
        timestamp, signal (1=long, -1=short),
        atr_1m, sl_distance, entry_price,
        sl_price_long, sl_price_short

    Only rows with non-zero signals are returned.
    Both DataFrames require columns: timestamp, open, high, low, close, volume.
    Timestamps must be UTC-aware datetime64.
    """
    df1 = df_1m.copy().set_index("timestamp").sort_index()
    df15 = df_15m.copy().set_index("timestamp").sort_index()

    df1["ema_fast"] = _ema(df1["close"], ema_fast)
    df1["ema_slow"] = _ema(df1["close"], ema_slow)
    df1["rsi"] = _rsi(df1["close"], rsi_period)
    df1["atr"] = _atr(df1, atr_period)

    df1["cross_long"] = (
        (df1["ema_fast"] > df1["ema_slow"])
        & (df1["ema_fast"].shift(1) <= df1["ema_slow"].shift(1))
    )
    df1["cross_short"] = (
        (df1["ema_fast"] < df1["ema_slow"])
        & (df1["ema_fast"].shift(1) >= df1["ema_slow"].shift(1))
    )

    df15["ema200"] = _ema(df15["close"], ema_trend)
    df15["trend_up"] = df15["close"] > df15["ema200"]
    trend_up = df15["trend_up"].reindex(df1.index, method="ffill").fillna(False)

    atr_ok = df1["atr"] > (df1["close"] * atr_min_pct)

    signal = pd.Series(0, index=df1.index, dtype=int)
    signal[df1["cross_long"] & (df1["rsi"] > 50) & trend_up & atr_ok] = 1
    signal[df1["cross_short"] & (df1["rsi"] < 50) & ~trend_up & atr_ok] = -1

    sl_dist = df1["atr"] * sl_atr_multiplier
    result = pd.DataFrame({
        "timestamp": df1.index,
        "signal": signal.values,
        "atr_1m": df1["atr"].values,
        "sl_distance": sl_dist.values,
        "entry_price": df1["close"].values,
    })
    result["sl_price_long"] = result["entry_price"] - result["sl_distance"]
    result["sl_price_short"] = result["entry_price"] + result["sl_distance"]

    return result[result["signal"] != 0].reset_index(drop=True)
