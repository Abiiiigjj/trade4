import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


@pytest.fixture
def sample_funding_df() -> pd.DataFrame:
    """3 days of 8h funding intervals for DOGEUSDT, always positive."""
    timestamps = pd.date_range("2024-01-01", periods=9, freq="8h", tz="UTC")
    return pd.DataFrame({
        "timestamp": timestamps,
        "symbol": "DOGEUSDT",
        "funding_rate": [0.0001, 0.0002, 0.00015, 0.0001, 0.00025, 0.0002, 0.0003, 0.0001, 0.00018],
    })


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Daily OHLCV for DOGE, 90 days."""
    dates = pd.date_range("2024-01-01", periods=90, freq="1d", tz="UTC")
    rng = np.random.default_rng(42)
    close = 0.10 + rng.normal(0, 0.005, 90).cumsum()
    close = np.clip(close, 0.05, 0.30)
    return pd.DataFrame({
        "timestamp": dates,
        "open": close * 0.999,
        "high": close * 1.005,
        "low": close * 0.995,
        "close": close,
        "volume": rng.uniform(1e8, 5e8, 90),
    })


@pytest.fixture
def sample_orderbook() -> pd.DataFrame:
    """Simulated orderbook for DOGEUSDT at price ~0.10."""
    levels = 10
    base_price = 0.10
    bids = pd.DataFrame({
        "price": [base_price - i * 0.0001 for i in range(levels)],
        "qty": [500_000.0] * levels,
        "side": ["bid"] * levels,
    })
    asks = pd.DataFrame({
        "price": [base_price + 0.0001 + i * 0.0001 for i in range(levels)],
        "qty": [500_000.0] * levels,
        "side": ["ask"] * levels,
    })
    return pd.concat([bids, asks], ignore_index=True)
