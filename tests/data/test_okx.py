import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from trade4.data.okx import fetch_funding_history, fetch_ohlcv, fetch_orderbook, list_perp_symbols


def test_fetch_funding_history_returns_dataframe():
    with patch("trade4.data.okx._get_exchange") as mock_ex:
        inst = MagicMock()
        inst.fetch_funding_rate_history.return_value = [
            {"timestamp": 1704067200000, "symbol": "DOGE/USDT:USDT", "fundingRate": 0.0001},
            {"timestamp": 1704096000000, "symbol": "DOGE/USDT:USDT", "fundingRate": 0.00015},
        ]
        mock_ex.return_value = inst
        df = fetch_funding_history("DOGE-USDT-SWAP", start_ts=pd.Timestamp("2024-01-01", tz="UTC"))
    assert isinstance(df, pd.DataFrame)
    assert "timestamp" in df.columns
    assert "funding_rate" in df.columns
    assert len(df) == 2


def test_fetch_orderbook_structure():
    with patch("trade4.data.okx._get_exchange") as mock_ex:
        inst = MagicMock()
        inst.fetch_order_book.return_value = {
            "bids": [[0.0999, 500000.0]],
            "asks": [[0.1001, 500000.0]],
        }
        mock_ex.return_value = inst
        df = fetch_orderbook("DOGE-USDT-SWAP")
    assert set(df["side"].unique()) == {"bid", "ask"}


def test_fetch_ohlcv_columns():
    with patch("trade4.data.okx._get_exchange") as mock_ex:
        inst = MagicMock()
        inst.fetch_ohlcv.return_value = [
            [1704067200000, 0.09, 0.095, 0.085, 0.092, 1_000_000.0],
        ]
        mock_ex.return_value = inst
        df = fetch_ohlcv("DOGE-USDT-SWAP", interval="1d", start_ts=pd.Timestamp("2024-01-01", tz="UTC"))
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
