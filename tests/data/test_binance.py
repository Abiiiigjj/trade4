import pandas as pd
import pytest
from unittest.mock import patch
from trade4.data.binance import (
    fetch_funding_history,
    fetch_ohlcv,
    fetch_orderbook,
    list_perp_symbols,
    FDUSD_ZERO_FEE_BASES,
)


def test_fetch_funding_history_returns_dataframe():
    mock_batch = [
        {"fundingTime": 1704067200000, "symbol": "DOGEUSDT", "fundingRate": "0.0001"},
        {"fundingTime": 1704096000000, "symbol": "DOGEUSDT", "fundingRate": "0.0002"},
    ]
    with patch("trade4.data.binance._get", return_value=mock_batch):
        df = fetch_funding_history("DOGEUSDT", start_ts=pd.Timestamp("2024-01-01", tz="UTC"))
    assert isinstance(df, pd.DataFrame)
    assert "timestamp" in df.columns
    assert "funding_rate" in df.columns
    assert df["timestamp"].dtype == "datetime64[ns, UTC]"
    assert len(df) == 2


def test_fetch_funding_history_empty_returns_empty_df():
    with patch("trade4.data.binance._get", return_value=[]):
        df = fetch_funding_history("DOGEUSDT", start_ts=pd.Timestamp("2024-01-01", tz="UTC"))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_fetch_orderbook_returns_bids_and_asks():
    mock_book = {
        "bids": [["0.0999", "500000"], ["0.0998", "400000"]],
        "asks": [["0.1001", "500000"], ["0.1002", "400000"]],
    }
    with patch("trade4.data.binance._get", return_value=mock_book):
        df = fetch_orderbook("DOGEUSDT")
    assert set(df["side"].unique()) == {"bid", "ask"}
    assert "price" in df.columns
    assert "qty" in df.columns


def test_fdusd_zero_fee_bases_contains_expected_coins():
    assert "DOGE" in FDUSD_ZERO_FEE_BASES
    assert "BTC" in FDUSD_ZERO_FEE_BASES
    assert "SOL" in FDUSD_ZERO_FEE_BASES


def test_fetch_ohlcv_returns_dataframe():
    mock_klines = [
        [1704067200000, "0.09", "0.095", "0.085", "0.092", "1000000",
         1704153599999, "92000", "5000", "500000", "46000", "0"],
    ]
    with patch("trade4.data.binance._get", return_value=mock_klines):
        df = fetch_ohlcv("DOGEUSDT", interval="1d", start_ts=pd.Timestamp("2024-01-01", tz="UTC"))
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert df["timestamp"].dtype == "datetime64[ns, UTC]"
