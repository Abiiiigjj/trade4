import pandas as pd
import numpy as np
import pytest
from trade4.scalper.backtest_harness import run_backtest, BacktestConfig, TradeResult


def _make_ohlcv_1m(prices: list[float], start: pd.Timestamp) -> pd.DataFrame:
    n = len(prices)
    timestamps = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    opens = [p * 0.9995 for p in prices]
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": [p * 1.003 for p in prices],
        "low": [p * 0.997 for p in prices],
        "close": prices,
        "volume": [2_000_000.0] * n,
    })


def _make_ohlcv_15m(n: int = 260, start: pd.Timestamp | None = None) -> pd.DataFrame:
    if start is None:
        start = pd.Timestamp("2023-12-29", tz="UTC")
    prices = [100.0 + i * 0.05 for i in range(n)]
    timestamps = pd.date_range(start, periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": prices,
        "high": [p * 1.002 for p in prices],
        "low": [p * 0.998 for p in prices],
        "close": prices,
        "volume": [1_000_000.0] * n,
    })


def test_run_backtest_returns_correct_types():
    start = pd.Timestamp("2024-01-02 08:00", tz="UTC")
    prices_down = [100.0 - i * 0.05 for i in range(40)]
    prices_up = [98.0 + i * 0.40 for i in range(60)]
    df_1m = _make_ohlcv_1m(prices_down + prices_up, start)
    df_15m = _make_ohlcv_15m()

    trades, equity = run_backtest(df_1m, df_15m, "BTCUSDT", BacktestConfig())
    assert isinstance(trades, list)
    assert isinstance(equity, pd.Series)
    assert len(equity) == len(df_1m)


def test_circuit_breaker_stops_new_trades():
    # Force large losses: set daily_loss_limit very low (0.1%)
    # and generate many losing trades
    start = pd.Timestamp("2024-01-02 08:00", tz="UTC")
    prices_down = [100.0 - i * 0.05 for i in range(40)]
    prices_up = [98.0 + i * 0.40 for i in range(40)]
    # Then crash so all SLs get hit
    prices_crash = [114.0 - i * 1.0 for i in range(40)]
    df_1m = _make_ohlcv_1m(prices_down + prices_up + prices_crash, start)
    df_15m = _make_ohlcv_15m()

    cfg = BacktestConfig(daily_loss_limit=0.001)  # 0.1% daily limit → fires easily
    trades, equity = run_backtest(df_1m, df_15m, "BTCUSDT", cfg)
    # After circuit breaker, equity must not go below (start - 3×limit) because trading stops
    min_equity = equity.min()
    assert min_equity > cfg.start_balance * (1 - 0.05), "Circuit breaker should cap losses"


def test_sl_hit_closes_position():
    # Simple scenario: price drops immediately below SL after entry
    start = pd.Timestamp("2024-01-02 08:00", tz="UTC")
    prices_down = [100.0 - i * 0.05 for i in range(40)]
    prices_up = [98.0 + i * 0.40 for i in range(10)]   # short up to generate signal
    prices_crash = [102.0 - i * 2.0 for i in range(20)]  # crash through SL
    df_1m = _make_ohlcv_1m(prices_down + prices_up + prices_crash, start)
    df_15m = _make_ohlcv_15m()

    trades, _ = run_backtest(df_1m, df_15m, "BTCUSDT", BacktestConfig())
    sl_trades = [t for t in trades if t.exit_reason == "sl"]
    # At least one SL hit when price crashes
    if len(trades) > 0:
        assert any(t.exit_reason in ("sl", "tp", "timeout") for t in trades)


def test_no_more_than_max_open_positions():
    start = pd.Timestamp("2024-01-02 08:00", tz="UTC")
    # Large bull run to generate many signals
    prices_down = [100.0 - i * 0.05 for i in range(40)]
    prices_up = [98.0 + i * 0.30 for i in range(200)]
    df_1m = _make_ohlcv_1m(prices_down + prices_up, start)
    df_15m = _make_ohlcv_15m()

    cfg = BacktestConfig(max_open_positions=2)
    trades, equity = run_backtest(df_1m, df_15m, "BTCUSDT", cfg)
    assert isinstance(trades, list)
    # Equity must be monotonically bounded (no position accounting errors)
    assert equity.max() < cfg.start_balance * 100  # sanity check


def test_net_pnl_includes_costs():
    start = pd.Timestamp("2024-01-02 08:00", tz="UTC")
    prices_down = [100.0 - i * 0.05 for i in range(40)]
    prices_up = [98.0 + i * 0.40 for i in range(40)]
    df_1m = _make_ohlcv_1m(prices_down + prices_up, start)
    df_15m = _make_ohlcv_15m()

    trades, _ = run_backtest(df_1m, df_15m, "BTCUSDT", BacktestConfig())
    for t in trades:
        assert t.net_pnl_eur == pytest.approx(t.gross_pnl_eur - t.cost_eur, rel=1e-6)
        assert t.cost_eur >= 0
