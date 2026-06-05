import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from trade4.data.binance import (
    fetch_funding_history,
    fetch_ohlcv,
    fetch_orderbook,
    list_perp_symbols,
    FDUSD_ZERO_FEE_BASES,
)


def test_fetch_funding_history_returns_dataframe():
    with patch("trade4.data.binance._get_exchange") as mock_ex:
        inst = MagicMock()
        inst.fetch_funding_rate_history.return_value = [
            {"timestamp": 1704067200000, "symbol": "DOGE/USDT:USDT", "fundingRate": 0.0001},
            {"timestamp": 1704096000000, "symbol": "DOGE/USDT:USDT", "fundingRate": 0.0002},
        ]
        mock_ex.return_value = inst
        df = fetch_funding_history("DOGEUSDT", start_ts=pd.Timestamp("2024-01-01", tz="UTC"))
    assert isinstance(df, pd.DataFrame)
    assert "timestamp" in df.columns
    assert "funding_rate" in df.columns
    assert df["timestamp"].dtype == "datetime64[ns, UTC]"
    assert len(df) == 2


def test_fetch_funding_history_empty_returns_empty_df():
    with patch("trade4.data.binance._get_exchange") as mock_ex:
        inst = MagicMock()
        inst.fetch_funding_rate_history.return_value = []
        mock_ex.return_value = inst
        df = fetch_funding_history("DOGEUSDT", start_ts=pd.Timestamp("2024-01-01", tz="UTC"))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_fetch_orderbook_returns_bids_and_asks():
    with patch("trade4.data.binance._get_exchange") as mock_ex:
        inst = MagicMock()
        inst.fetch_order_book.return_value = {
            "bids": [[0.0999, 500000.0], [0.0998, 400000.0]],
            "asks": [[0.1001, 500000.0], [0.1002, 400000.0]],
        }
        mock_ex.return_value = inst
        df = fetch_orderbook("DOGEUSDT")
    assert set(df["side"].unique()) == {"bid", "ask"}
    assert "price" in df.columns
    assert "qty" in df.columns


def test_fdusd_zero_fee_bases_contains_expected_coins():
    assert "DOGE" in FDUSD_ZERO_FEE_BASES
    assert "BTC" in FDUSD_ZERO_FEE_BASES
    assert "SOL" in FDUSD_ZERO_FEE_BASES


def test_fetch_ohlcv_returns_dataframe():
    with patch("trade4.data.binance._get_exchange") as mock_ex:
        inst = MagicMock()
        inst.fetch_ohlcv.return_value = [
            [1704067200000, 0.09, 0.095, 0.085, 0.092, 1_000_000.0],
        ]
        mock_ex.return_value = inst
        df = fetch_ohlcv("DOGEUSDT", interval="1d", start_ts=pd.Timestamp("2024-01-01", tz="UTC"))
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert df["timestamp"].dtype == "datetime64[ns, UTC]"
