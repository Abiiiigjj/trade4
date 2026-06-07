from dataclasses import dataclass, replace
from typing import Any
import pandas as pd

from trade4.scalper.backtest_harness import run_backtest, BacktestConfig, TradeResult


@dataclass
class WalkForwardMetrics:
    sharpe: float
    max_drawdown: float
    profit_factor: float
    win_rate: float
    n_trades: int
    final_balance: float


@dataclass
class WalkForwardResult:
    best_params: dict[str, Any]
    in_sample: WalkForwardMetrics
    out_of_sample: WalkForwardMetrics
    oos_degradation: float  # (IS_sharpe - OOS_sharpe) / abs(IS_sharpe)
    gate_passed: bool


# Phase-2 go/no-go thresholds (from spec section 8)
_MIN_SHARPE: float = 1.5
_MAX_DRAWDOWN: float = -0.20
_MIN_PROFIT_FACTOR: float = 1.4
_MIN_WIN_RATE: float = 0.45
_MAX_OOS_DEGRADATION: float = 0.30


def compute_metrics(
    trades: list[TradeResult],
    equity_curve: pd.Series,
    start_balance: float,
) -> WalkForwardMetrics:
    """Compute performance metrics from a completed backtest run."""
    if not trades or equity_curve.empty:
        return WalkForwardMetrics(
            sharpe=0.0, max_drawdown=0.0, profit_factor=0.0,
            win_rate=0.0, n_trades=0, final_balance=start_balance,
        )

    pnls = [t.net_pnl_eur for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    profit_factor = sum(wins) / abs(sum(losses)) if losses and abs(sum(losses)) > 0 else float("inf")
    win_rate = len(wins) / len(pnls) if pnls else 0.0

    eq = equity_curve.sort_index()
    daily = eq.resample("D").last().dropna().pct_change().dropna()
    sharpe = float((daily.mean() / daily.std()) * (252 ** 0.5)) if len(daily) > 1 and daily.std() > 0 else 0.0

    running_max = eq.cummax()
    dd = (eq - running_max) / running_max
    max_dd = float(dd.min()) if not dd.empty else 0.0

    return WalkForwardMetrics(
        sharpe=sharpe,
        max_drawdown=max_dd,
        profit_factor=profit_factor,
        win_rate=win_rate,
        n_trades=len(trades),
        final_balance=float(eq.iloc[-1]),
    )


def _gate_check(oos: WalkForwardMetrics, degradation: float) -> bool:
    return (
        oos.sharpe >= _MIN_SHARPE
        and oos.max_drawdown >= _MAX_DRAWDOWN
        and oos.profit_factor >= _MIN_PROFIT_FACTOR
        and oos.win_rate >= _MIN_WIN_RATE
        and degradation <= _MAX_OOS_DEGRADATION
    )


def run_walk_forward(
    df_1m: pd.DataFrame,
    df_15m: pd.DataFrame,
    symbol: str,
    is_months: int = 9,
    oos_months: int = 3,
    param_grid: list[dict[str, Any]] | None = None,
    base_config: BacktestConfig | None = None,
) -> WalkForwardResult:
    """Walk-forward validation with in-sample optimization + OOS gate.

    param_grid: list of dicts, each dict overrides BacktestConfig fields.
    Best params are chosen by IS Sharpe, then validated on OOS.
    """
    if base_config is None:
        base_config = BacktestConfig()
    if param_grid is None:
        param_grid = [{}]

    df_1m = df_1m.sort_values("timestamp")
    start_ts = df_1m["timestamp"].min()
    is_end = start_ts + pd.DateOffset(months=is_months)
    oos_end = is_end + pd.DateOffset(months=oos_months)

    is_1m = df_1m[df_1m["timestamp"] < is_end].copy()
    oos_1m = df_1m[(df_1m["timestamp"] >= is_end) & (df_1m["timestamp"] < oos_end)].copy()
    is_15m = df_15m[df_15m["timestamp"] < is_end].copy()
    oos_15m = df_15m[(df_15m["timestamp"] >= is_end) & (df_15m["timestamp"] < oos_end)].copy()

    best_sharpe = float("-inf")
    best_params: dict[str, Any] = {}
    best_is_trades: list[TradeResult] = []
    best_is_curve = pd.Series(dtype=float)

    for params in param_grid:
        cfg = replace(base_config, **params)
        trades, curve = run_backtest(is_1m, is_15m, symbol, cfg)
        m = compute_metrics(trades, curve, cfg.start_balance)
        if m.sharpe > best_sharpe:
            best_sharpe = m.sharpe
            best_params = params
            best_is_trades, best_is_curve = trades, curve

    best_cfg = replace(base_config, **best_params)
    oos_trades, oos_curve = run_backtest(oos_1m, oos_15m, symbol, best_cfg)

    is_metrics = compute_metrics(best_is_trades, best_is_curve, base_config.start_balance)
    oos_metrics = compute_metrics(oos_trades, oos_curve, base_config.start_balance)

    oos_deg = (
        (is_metrics.sharpe - oos_metrics.sharpe) / abs(is_metrics.sharpe)
        if is_metrics.sharpe != 0
        else float("inf")
    )

    return WalkForwardResult(
        best_params=best_params,
        in_sample=is_metrics,
        out_of_sample=oos_metrics,
        oos_degradation=oos_deg,
        gate_passed=_gate_check(oos_metrics, oos_deg),
    )
