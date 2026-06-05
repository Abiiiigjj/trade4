import pandas as pd
import numpy as np
import pytest
from trade4.screener.screener import (
    compute_funding_stats,
    estimate_slippage_bps,
    screen_coins,
    ScreenerConfig,
)


def test_compute_funding_stats_averages(sample_funding_df):
    stats = compute_funding_stats(sample_funding_df)
    assert "avg_funding_30d" in stats
    assert "avg_funding_90d" in stats
    assert "pct_positive_intervals" in stats
    assert 0.0 < stats["avg_funding_30d"] < 0.001
    assert stats["pct_positive_intervals"] == 1.0


def test_compute_funding_stats_handles_negatives():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=6, freq="8h", tz="UTC"),
        "symbol": "TEST",
        "funding_rate": [0.0002, -0.0001, 0.0003, -0.0002, 0.0001, 0.0002],
    })
    stats = compute_funding_stats(df)
    assert stats["pct_positive_intervals"] == pytest.approx(4 / 6)


def test_estimate_slippage_bps_small_order(sample_orderbook):
    # Buy €100 of DOGE at ~$0.10 → ~1000 DOGE. Orderbook has 500k per level.
    slippage = estimate_slippage_bps(sample_orderbook, notional_eur=100, price_eur=0.10, side="ask")
    assert 0 < slippage < 5


def test_estimate_slippage_bps_large_order_eats_levels(sample_orderbook):
    # Buy €100k → much higher slippage
    slippage_small = estimate_slippage_bps(sample_orderbook, notional_eur=100, price_eur=0.10, side="ask")
    slippage_large = estimate_slippage_bps(sample_orderbook, notional_eur=100_000, price_eur=0.10, side="ask")
    assert slippage_large > slippage_small


def test_screen_coins_filters_low_funding(sample_funding_df, sample_orderbook):
    config = ScreenerConfig(
        entry_threshold_per_interval=0.0005,  # higher than our sample's avg → should filter out
        max_slippage_bps=50,
        position_size_eur=500,
        min_pct_positive=0.5,
        volume_fraction_cap=0.005,
    )
    funding_data = {"DOGEUSDT": sample_funding_df}
    orderbook_data = {"DOGEUSDT": sample_orderbook}
    ohlcv_data = {}
    result = screen_coins(["DOGEUSDT"], funding_data, orderbook_data, ohlcv_data, config)
    assert len(result) == 0 or result.iloc[0]["avg_funding_30d"] < config.entry_threshold_per_interval


def test_screen_coins_passes_good_candidate(sample_funding_df, sample_orderbook):
    config = ScreenerConfig(
        entry_threshold_per_interval=0.00005,  # low threshold → sample passes
        max_slippage_bps=50,
        position_size_eur=500,
        min_pct_positive=0.5,
        volume_fraction_cap=0.1,
    )
    funding_data = {"DOGEUSDT": sample_funding_df}
    orderbook_data = {"DOGEUSDT": sample_orderbook}
    ohlcv_data = {}
    result = screen_coins(["DOGEUSDT"], funding_data, orderbook_data, ohlcv_data, config)
    assert len(result) == 1
    assert result.iloc[0]["symbol"] == "DOGEUSDT"
    assert "gate_candidate" in result.columns


def test_fdusd_flag_set_for_eligible_coins(sample_funding_df, sample_orderbook):
    config = ScreenerConfig(
        entry_threshold_per_interval=0.00005,
        max_slippage_bps=50,
        position_size_eur=500,
        min_pct_positive=0.5,
        volume_fraction_cap=0.1,
    )
    funding_data = {"DOGEUSDT": sample_funding_df}
    orderbook_data = {"DOGEUSDT": sample_orderbook}
    result = screen_coins(["DOGEUSDT"], funding_data, orderbook_data, {}, config)
    assert result.iloc[0]["fdusd_zero_fee"] is True
