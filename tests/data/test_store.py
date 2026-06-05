import pandas as pd
import pytest
from pathlib import Path
from trade4.data.store import save_df, load_df, get_last_timestamp


def test_save_and_load_roundtrip(tmp_data_dir):
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
        "funding_rate": [0.0001, 0.0002],
    })
    save_df(df, "binance", "funding", "DOGEUSDT", base_dir=tmp_data_dir)
    result = load_df("binance", "funding", "DOGEUSDT", base_dir=tmp_data_dir)
    pd.testing.assert_frame_equal(df, result)


def test_load_returns_none_when_missing(tmp_data_dir):
    result = load_df("binance", "funding", "NONEXISTENT", base_dir=tmp_data_dir)
    assert result is None


def test_get_last_timestamp_none_when_missing(tmp_data_dir):
    result = get_last_timestamp("binance", "funding", "NONEXISTENT", base_dir=tmp_data_dir)
    assert result is None


def test_get_last_timestamp_returns_max(tmp_data_dir):
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-02"], utc=True),
        "funding_rate": [0.0001, 0.0003, 0.0002],
    })
    save_df(df, "binance", "funding", "DOGEUSDT", base_dir=tmp_data_dir)
    last = get_last_timestamp("binance", "funding", "DOGEUSDT", base_dir=tmp_data_dir)
    assert last == pd.Timestamp("2024-01-03", tz="UTC")


def test_save_overwrites_existing(tmp_data_dir):
    df1 = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01"], utc=True),
        "funding_rate": [0.0001],
    })
    df2 = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01"], utc=True),
        "funding_rate": [0.9999],
    })
    save_df(df1, "binance", "funding", "DOGEUSDT", base_dir=tmp_data_dir)
    save_df(df2, "binance", "funding", "DOGEUSDT", base_dir=tmp_data_dir)
    result = load_df("binance", "funding", "DOGEUSDT", base_dir=tmp_data_dir)
    assert result["funding_rate"].iloc[0] == 0.9999
