import pandas as pd
import pytest

from trade4.data.binance_vision import fetch_funding_month, fetch_klines_month


@pytest.mark.integration
def test_delisted_symbol_returns_funding():
    # SRMUSDT was delisted; its history must still be retrievable from the vision dumps.
    df = fetch_funding_month("SRMUSDT", 2022, 9)
    assert not df.empty
    assert {"timestamp", "funding_rate"}.issubset(df.columns)
    assert df["timestamp"].is_monotonic_increasing
    # header row must NOT leak into the data
    assert pd.api.types.is_float_dtype(df["funding_rate"])
    assert df["funding_rate"].abs().max() < 1.0  # sane funding-rate magnitude


@pytest.mark.integration
def test_missing_month_returns_empty():
    df = fetch_funding_month("SRMUSDT", 2019, 1)  # before listing
    assert df.empty


@pytest.mark.integration
def test_klines_fetch_live_symbol():
    df = fetch_klines_month("BTCUSDT", "1h", 2023, 1)
    assert not df.empty
    assert {"timestamp", "open", "high", "low", "close", "volume"}.issubset(df.columns)
    assert (df["high"] >= df["low"]).all()
