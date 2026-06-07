import pandas as pd
import pytest
from unittest.mock import patch
from trade4.data.binance import (
    fetch_funding_history,
    fetch_ohlcv,
    fetch_orderbook,
    list_perp_symbols,
    list_perp_symbols_with_onboard,
    filter_symbols_at_date,
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


def _mock_exchange_info() -> dict:
    return {
        "symbols": [
            {
                "symbol": "BTCUSDT",
                "contractType": "PERPETUAL",
                "quoteAsset": "USDT",
                "status": "TRADING",
                "onboardDate": 1569888000000,  # 2019-10-01
            },
            {
                "symbol": "SOLUSDT",
                "contractType": "PERPETUAL",
                "quoteAsset": "USDT",
                "status": "TRADING",
                "onboardDate": 1623024000000,  # 2021-06-07
            },
            {
                "symbol": "NEWUSDT",
                "contractType": "PERPETUAL",
                "quoteAsset": "USDT",
                "status": "TRADING",
                "onboardDate": 1748822400000,  # 2025-06-02 (future in backtest)
            },
            {
                "symbol": "BTCDOMUSDT",
                "contractType": "PERPETUAL",
                "quoteAsset": "USDT",
                "status": "SETTLING",  # not TRADING
                "onboardDate": 1569888000000,
            },
        ]
    }


def test_list_perp_symbols_with_onboard_filters_non_trading():
    with patch("trade4.data.binance._get", return_value=_mock_exchange_info()):
        result = list_perp_symbols_with_onboard()
    assert "BTCDOMUSDT" not in result
    assert "BTCUSDT" in result
    assert "SOLUSDT" in result


def test_list_perp_symbols_with_onboard_returns_timestamps():
    with patch("trade4.data.binance._get", return_value=_mock_exchange_info()):
        result = list_perp_symbols_with_onboard()
    assert isinstance(result["BTCUSDT"], pd.Timestamp)
    assert result["BTCUSDT"].tzinfo is not None  # UTC-aware


def test_filter_symbols_at_date_excludes_future_listings():
    onboard = {
        "BTCUSDT": pd.Timestamp("2019-10-01", tz="UTC"),
        "SOLUSDT": pd.Timestamp("2021-06-07", tz="UTC"),
        "NEWUSDT": pd.Timestamp("2025-06-02", tz="UTC"),
    }
    as_of = pd.Timestamp("2024-01-01", tz="UTC")
    result = filter_symbols_at_date(onboard, as_of)
    assert "BTCUSDT" in result
    assert "SOLUSDT" in result
    assert "NEWUSDT" not in result


def test_filter_symbols_at_date_includes_same_day():
    onboard = {"SOLUSDT": pd.Timestamp("2024-01-01", tz="UTC")}
    result = filter_symbols_at_date(onboard, pd.Timestamp("2024-01-01", tz="UTC"))
    assert "SOLUSDT" in result
