import pandas as pd
import numpy as np
import pytest
from trade4.backtest.engine import (
    BacktestConfig,
    BacktestResult,
    CycleResult,
    run_backtest,
    count_funding_intervals,
    estimate_spread_bps,
    maker_fill_simulated,
    split_walk_forward,
)
from trade4.backtest.cost_model import CostModel, DEFAULT_FEE_SCHEDULE


DEFAULT_CONFIG = BacktestConfig(
    entry_threshold=0.00005,
    exit_threshold=0.0,
    persistence_threshold=0.00003,
    persistence_window=5,
    max_holding_days=30,
    position_size_eur=500.0,
    cost_model=CostModel(
        fee_schedule=DEFAULT_FEE_SCHEDULE,
        slippage_entry_bps=5.0, slippage_exit_bps=5.0,
        basis_drift_bps=2.0, fdusd_depeg_bps=0.0,
        use_fdusd=False, use_maker_spot=False, use_maker_perp=False,
    ),
)


def test_count_funding_intervals_exact_timing():
    # Entry 04:00 UTC, exit 04:00 UTC next day
    # Intervals strictly between entry(04:00) and exit(04:00+1d): 08:00, 16:00, 00:00+1d -> 3 intervals
    entry = pd.Timestamp("2024-01-01 04:00:00", tz="UTC")
    exit_ = pd.Timestamp("2024-01-02 04:00:00", tz="UTC")
    funding_df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=12, freq="8h", tz="UTC"),
        "funding_rate": [0.0001] * 12,
    })
    collected = count_funding_intervals(entry, exit_, funding_df)
    assert len(collected) == 3


def test_count_funding_intervals_entry_on_interval_not_collected():
    # If entry IS exactly at 08:00, that interval should NOT be collected (half-open: (entry, exit])
    entry = pd.Timestamp("2024-01-01 08:00:00", tz="UTC")
    exit_ = pd.Timestamp("2024-01-01 20:00:00", tz="UTC")
    funding_df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01 08:00", "2024-01-01 16:00"], utc=True),
        "funding_rate": [0.0001, 0.0002],
    })
    collected = count_funding_intervals(entry, exit_, funding_df)
    assert len(collected) == 1
    assert collected.iloc[0]["timestamp"] == pd.Timestamp("2024-01-01 16:00", tz="UTC")


def test_estimate_spread_bps_normal_candle():
    # HL range = 0.0005, close = 0.10 → spread = (0.0005/0.10)*0.5*10000 = 25 bps, well below 100 cap
    row = pd.Series({"open": 0.10, "high": 0.10025, "low": 0.09975, "close": 0.10, "volume": 1e8})
    spread = estimate_spread_bps(row)
    assert 0 < spread < 100


def test_estimate_spread_bps_caps_at_max():
    row = pd.Series({"open": 0.10, "high": 1.0, "low": 0.001, "close": 0.10, "volume": 1e8})
    spread = estimate_spread_bps(row)
    assert spread <= 100.0


def test_maker_fill_buy_fills_when_low_crosses():
    candle = pd.Series({"low": 0.098, "high": 0.102, "close": 0.10})
    assert maker_fill_simulated(candle, limit_price=0.099, side="buy") is True


def test_maker_fill_buy_no_fill_when_low_above_limit():
    candle = pd.Series({"low": 0.10, "high": 0.105, "close": 0.102})
    assert maker_fill_simulated(candle, limit_price=0.099, side="buy") is False


def test_maker_fill_sell_fills_when_high_crosses():
    candle = pd.Series({"low": 0.098, "high": 0.102, "close": 0.10})
    assert maker_fill_simulated(candle, limit_price=0.101, side="sell") is True


def test_split_walk_forward_correct_split():
    # periods=4000 × 8h ≈ 4.1 years → spans from 2023-01-01 to ~2027, crossing 2024-12-31
    funding_df = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=4000, freq="8h", tz="UTC"),
        "funding_rate": [0.0001] * 4000,
    })
    in_sample, out_of_sample = split_walk_forward(
        funding_df, in_sample_end=pd.Timestamp("2024-12-31", tz="UTC")
    )
    assert in_sample["timestamp"].max() <= pd.Timestamp("2024-12-31", tz="UTC")
    assert out_of_sample["timestamp"].min() > pd.Timestamp("2024-12-31", tz="UTC")


def test_run_backtest_returns_result(sample_orderbook):
    # Generate 2 years of consistently high funding
    timestamps = pd.date_range("2023-01-01", periods=365*3, freq="8h", tz="UTC")
    funding_df = pd.DataFrame({
        "timestamp": timestamps,
        "symbol": "DOGEUSDT",
        "funding_rate": [0.0002] * len(timestamps),
    })
    ohlcv_df = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=365*2, freq="D", tz="UTC"),
        "open": [0.10] * 365*2,
        "high": [0.105] * 365*2,
        "low": [0.095] * 365*2,
        "close": [0.10] * 365*2,
        "volume": [1e9] * 365*2,
    })
    result = run_backtest(funding_df, ohlcv_df, sample_orderbook, DEFAULT_CONFIG)
    assert isinstance(result, BacktestResult)
    assert isinstance(result.cycles, list)
    assert isinstance(result.max_drawdown_bps, float)
    assert isinstance(result.net_pnl_bps, float)


def test_run_backtest_no_cycles_when_funding_too_low(sample_ohlcv_df, sample_orderbook):
    timestamps = pd.date_range("2023-01-01", periods=100, freq="8h", tz="UTC")
    funding_df = pd.DataFrame({
        "timestamp": timestamps,
        "symbol": "DOGEUSDT",
        "funding_rate": [-0.0001] * 100,  # always negative -> no entry
    })
    result = run_backtest(funding_df, sample_ohlcv_df, sample_orderbook, DEFAULT_CONFIG)
    assert len(result.cycles) == 0
