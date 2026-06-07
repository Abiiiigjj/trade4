import pandas as pd
import numpy as np
import pytest
from trade4.scalper.signals.pump_scanner import generate_signals


def _make_df(closes: list[float], volumes: list[float], start: pd.Timestamp) -> pd.DataFrame:
    n = len(closes)
    timestamps = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    opens = [c * 0.998 for c in closes]
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": [c * 1.003 for c in closes],
        "low": [c * 0.997 for c in closes],
        "close": closes,
        "volume": volumes,
    })


def _flat_prices(n: int, price: float = 100.0) -> list[float]:
    return [price] * n


def _baseline_volumes(n: int, base: float = 1_000_000.0) -> list[float]:
    return [base] * n


def test_long_signal_on_volume_spike_with_pump():
    start = pd.Timestamp("2024-01-01", tz="UTC")
    # 25 normal candles, then 1 candle with 4× volume and +2% price move
    prices = _flat_prices(25) + [100.0, 102.0]  # last candle: +2%
    volumes = _baseline_volumes(25) + [1_000_000.0, 4_000_000.0]
    # Override open for pump candle so return = (102-100)/100 = +2%
    df = _make_df(prices, volumes, start)
    df.iloc[-1, df.columns.get_loc("open")] = 100.0

    signals = generate_signals(df, volume_multiplier=3.0, price_move_pct=0.015)
    assert len(signals) == 1
    assert signals.iloc[0]["signal"] == 1


def test_short_signal_on_volume_spike_with_dump():
    start = pd.Timestamp("2024-01-01", tz="UTC")
    prices = _flat_prices(25) + [100.0, 98.0]  # last candle: -2%
    volumes = _baseline_volumes(25) + [1_000_000.0, 4_000_000.0]
    df = _make_df(prices, volumes, start)
    df.iloc[-1, df.columns.get_loc("open")] = 100.0

    signals = generate_signals(df, volume_multiplier=3.0, price_move_pct=0.015)
    assert len(signals) == 1
    assert signals.iloc[0]["signal"] == -1


def test_no_signal_volume_below_multiplier():
    start = pd.Timestamp("2024-01-01", tz="UTC")
    prices = _flat_prices(25) + [100.0, 102.0]
    volumes = _baseline_volumes(25) + [1_000_000.0, 2_000_000.0]  # only 2× not 3×
    df = _make_df(prices, volumes, start)
    df.iloc[-1, df.columns.get_loc("open")] = 100.0

    signals = generate_signals(df, volume_multiplier=3.0, price_move_pct=0.015)
    assert len(signals) == 0


def test_no_signal_price_move_below_threshold():
    start = pd.Timestamp("2024-01-01", tz="UTC")
    prices = _flat_prices(25) + [100.0, 100.5]  # only +0.5%
    volumes = _baseline_volumes(25) + [1_000_000.0, 5_000_000.0]
    df = _make_df(prices, volumes, start)
    df.iloc[-1, df.columns.get_loc("open")] = 100.0

    signals = generate_signals(df, volume_multiplier=3.0, price_move_pct=0.015)
    assert len(signals) == 0


def test_sl_and_tp_computed_correctly_for_long():
    start = pd.Timestamp("2024-01-01", tz="UTC")
    prices = _flat_prices(25) + [100.0, 102.0]
    volumes = _baseline_volumes(25) + [1_000_000.0, 4_000_000.0]
    df = _make_df(prices, volumes, start)
    df.iloc[-1, df.columns.get_loc("open")] = 100.0

    signals = generate_signals(df, sl_pct=0.008, tp_pct=0.012)
    row = signals.iloc[0]
    assert row["sl_price"] == pytest.approx(row["entry_price"] * (1 - 0.008), rel=1e-4)
    assert row["tp_price"] == pytest.approx(row["entry_price"] * (1 + 0.012), rel=1e-4)
