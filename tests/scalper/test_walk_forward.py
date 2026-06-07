import pandas as pd
import numpy as np
import pytest
from trade4.scalper.walk_forward import compute_metrics, run_walk_forward, WalkForwardMetrics, WalkForwardResult
from trade4.scalper.backtest_harness import TradeResult, BacktestConfig


def _make_trades(pnls: list[float]) -> list[TradeResult]:
    ts = pd.Timestamp("2024-01-01", tz="UTC")
    return [
        TradeResult(
            entry_ts=ts + pd.Timedelta(hours=i),
            exit_ts=ts + pd.Timedelta(hours=i, minutes=5),
            symbol="BTCUSDT",
            strategy="ema_cross",
            side="long",
            entry_price=100.0,
            exit_price=101.0 if p > 0 else 99.0,
            qty=abs(p),
            gross_pnl_eur=p + 1.0,
            cost_eur=1.0,
            net_pnl_eur=p,
            exit_reason="tp" if p > 0 else "sl",
        )
        for i, p in enumerate(pnls)
    ]


def _make_equity(pnls: list[float], start_balance: float = 2000.0) -> pd.Series:
    ts = pd.date_range("2024-01-01", periods=len(pnls) + 1, freq="h", tz="UTC")
    values = [start_balance + sum(pnls[:i]) for i in range(len(pnls) + 1)]
    return pd.Series(values, index=ts)


def test_compute_metrics_win_rate():
    trades = _make_trades([10.0, -5.0, 10.0, 10.0, -5.0])  # 3 wins, 2 losses
    equity = _make_equity([10.0, -5.0, 10.0, 10.0, -5.0])
    metrics = compute_metrics(trades, equity, 2000.0)
    assert metrics.win_rate == pytest.approx(0.6, rel=1e-4)
    assert metrics.n_trades == 5


def test_compute_metrics_profit_factor():
    trades = _make_trades([10.0, -5.0, 10.0])  # wins=20, losses=5
    equity = _make_equity([10.0, -5.0, 10.0])
    metrics = compute_metrics(trades, equity, 2000.0)
    assert metrics.profit_factor == pytest.approx(4.0, rel=1e-4)


def test_compute_metrics_no_trades_returns_zeros():
    metrics = compute_metrics([], pd.Series(dtype=float), 2000.0)
    assert metrics.n_trades == 0
    assert metrics.sharpe == 0.0
    assert metrics.profit_factor == 0.0


def test_gate_passed_requires_oos_metrics():
    result = WalkForwardResult(
        best_params={},
        in_sample=WalkForwardMetrics(sharpe=2.0, max_drawdown=-0.10, profit_factor=1.8, win_rate=0.55, n_trades=100, final_balance=2400.0),
        out_of_sample=WalkForwardMetrics(sharpe=1.6, max_drawdown=-0.15, profit_factor=1.5, win_rate=0.50, n_trades=30, final_balance=2200.0),
        oos_degradation=0.20,
        gate_passed=True,
    )
    assert result.gate_passed is True


def test_gate_fails_when_oos_sharpe_below_threshold():
    result = WalkForwardResult(
        best_params={},
        in_sample=WalkForwardMetrics(sharpe=2.5, max_drawdown=-0.08, profit_factor=2.0, win_rate=0.60, n_trades=200, final_balance=2800.0),
        out_of_sample=WalkForwardMetrics(sharpe=0.8, max_drawdown=-0.25, profit_factor=1.1, win_rate=0.42, n_trades=50, final_balance=1900.0),
        oos_degradation=0.68,
        gate_passed=False,
    )
    assert result.gate_passed is False
