import pandas as pd
import numpy as np


def generate_signals(
    df_1m: pd.DataFrame,
    volume_multiplier: float = 3.0,
    price_move_pct: float = 0.015,
    vol_ma_period: int = 20,
    sl_pct: float = 0.008,
    tp_pct: float = 0.012,
) -> pd.DataFrame:
    """Detect pump/dump spikes from volume and price acceleration.

    Returns DataFrame with columns:
        timestamp, signal (1=long, -1=short),
        entry_price, sl_price, tp_price

    Entry is assumed to fill at the close of the trigger candle
    (market order). sl_pct and tp_pct are applied to entry_price.

    Note: Taker market orders into spikes carry 0.1–0.5% slippage —
    the ScalperCostModel stressed mode accounts for this in backtest.
    """
    df = df_1m.copy().set_index("timestamp").sort_index()

    df["vol_ma"] = df["volume"].rolling(vol_ma_period, min_periods=vol_ma_period).mean()
    df["vol_spike"] = df["volume"] > (df["vol_ma"] * volume_multiplier)

    candle_return = (df["close"] - df["open"]) / df["open"].replace(0, float("nan"))

    pump = df["vol_spike"] & (candle_return > price_move_pct)
    dump = df["vol_spike"] & (candle_return < -price_move_pct)

    signal = pd.Series(0, index=df.index, dtype=int)
    signal[pump] = 1
    signal[dump] = -1

    result = pd.DataFrame({
        "timestamp": df.index,
        "signal": signal.values,
        "entry_price": df["close"].values,
    })
    result = result[result["signal"] != 0].copy()

    result["sl_price"] = result.apply(
        lambda r: r["entry_price"] * (1 - sl_pct)
        if r["signal"] == 1
        else r["entry_price"] * (1 + sl_pct),
        axis=1,
    )
    result["tp_price"] = result.apply(
        lambda r: r["entry_price"] * (1 + tp_pct)
        if r["signal"] == 1
        else r["entry_price"] * (1 - tp_pct),
        axis=1,
    )
    return result.reset_index(drop=True)
