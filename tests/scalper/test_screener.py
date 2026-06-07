import pandas as pd
import numpy as np
import pytest
from trade4.scalper.screener import compute_atr_score, screen_by_volatility


def _make_1h_ohlcv(atr_pct: float, n: int = 50, price: float = 100.0) -> pd.DataFrame:
    """Synthetic 1h OHLCV with controlled ATR as % of price."""
    start = pd.Timestamp("2024-01-01", tz="UTC")
    timestamps = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    half_range = price * atr_pct / 2
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": [price] * n,
        "high": [price + half_range] * n,
        "low": [price - half_range] * n,
        "close": [price] * n,
        "volume": [1_000_000.0] * n,
    })


def test_atr_score_higher_for_volatile_asset():
    df_volatile = _make_1h_ohlcv(atr_pct=0.02)  # 2% ATR
    df_calm = _make_1h_ohlcv(atr_pct=0.005)     # 0.5% ATR
    assert compute_atr_score(df_volatile) > compute_atr_score(df_calm)


def test_atr_score_returns_zero_for_insufficient_data():
    df = _make_1h_ohlcv(atr_pct=0.01, n=5)  # fewer than period=14
    assert compute_atr_score(df, period=14) == 0.0


def test_screen_returns_top_n_by_volatility():
    ohlcv = {
        "ALUSDT": _make_1h_ohlcv(atr_pct=0.05),   # highest volatility
        "BLUSDT": _make_1h_ohlcv(atr_pct=0.02),
        "CLUSTDT": _make_1h_ohlcv(atr_pct=0.01),
        "DLUSDT": _make_1h_ohlcv(atr_pct=0.001),  # lowest
    }
    volumes = {"ALUSDT": 500e6, "BLUSDT": 500e6, "CLUSTDT": 500e6, "DLUSDT": 500e6}
    result = screen_by_volatility(ohlcv, volumes, top_n=2)
    assert result == ["ALUSDT", "BLUSDT"]


def test_screen_excludes_below_volume_threshold():
    ohlcv = {
        "HIGHVOL": _make_1h_ohlcv(atr_pct=0.05),
        "LOWVOL": _make_1h_ohlcv(atr_pct=0.10),  # more volatile but low volume
    }
    volumes = {"HIGHVOL": 500e6, "LOWVOL": 50e6}  # LOWVOL below 200M threshold
    result = screen_by_volatility(ohlcv, volumes, min_volume_usdt=200e6, top_n=5)
    assert "LOWVOL" not in result
    assert "HIGHVOL" in result


def test_screen_point_in_time_excludes_future_listings():
    ohlcv = {
        "OLD": _make_1h_ohlcv(atr_pct=0.01),
        "NEW": _make_1h_ohlcv(atr_pct=0.05),  # more volatile but listed later
    }
    volumes = {"OLD": 500e6, "NEW": 500e6}
    onboard = {
        "OLD": pd.Timestamp("2020-01-01", tz="UTC"),
        "NEW": pd.Timestamp("2025-01-01", tz="UTC"),
    }
    as_of = pd.Timestamp("2024-01-01", tz="UTC")
    result = screen_by_volatility(ohlcv, volumes, symbol_onboard=onboard, as_of=as_of)
    assert "NEW" not in result
    assert "OLD" in result
