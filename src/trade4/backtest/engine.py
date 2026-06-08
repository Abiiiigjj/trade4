import logging
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from trade4.backtest.cost_model import CostModel, DEFAULT_FEE_SCHEDULE, compute_round_trip_cost_bps, gate_passed

logger = logging.getLogger(__name__)

REBALANCE_DELTA_THRESHOLD = 0.02  # 2% drift triggers rebalance cost


@dataclass
class BacktestConfig:
    entry_threshold: float = 0.00005
    exit_threshold: float = 0.0
    persistence_threshold: float = 0.00003
    persistence_window: int = 5
    max_holding_days: int = 30
    causal_gate: bool = True  # True=trailing(honest), False=look-ahead(optimistic ceiling)
    exit_mode: str = "rate"        # "rate"=instant on single neg settlement, "rolling"=hysteresis on trailing avg
    min_hold_intervals: int = 0    # minimum funding intervals to hold before funding-exit allowed (~3/day)
    position_size_eur: float = 500.0
    cost_model: CostModel = field(default_factory=lambda: CostModel(
        fee_schedule=DEFAULT_FEE_SCHEDULE,
        slippage_entry_bps=5.0,
        slippage_exit_bps=5.0,
        basis_drift_bps=2.0,
        fdusd_depeg_bps=0.0,
        use_fdusd=False,
        use_maker_spot=False,
        use_maker_perp=False,
    ))


@dataclass
class CycleResult:
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    intervals_collected: int
    funding_received_bps: float
    round_trip_cost_bps: float
    net_pnl_bps: float
    gate_passed: bool
    rebalance_count: int
    exit_reason: str


@dataclass
class BacktestResult:
    cycles: list[CycleResult]
    equity_curve: pd.Series      # cumulative net_pnl_bps indexed by timestamp
    max_drawdown_bps: float
    net_pnl_bps: float
    pct_gate_passed: float
    n_intervals_total: int


def count_funding_intervals(
    entry_ts: pd.Timestamp,
    exit_ts: pd.Timestamp,
    funding_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return funding rows where entry_ts < timestamp <= exit_ts (half-open on left)."""
    mask = (funding_df["timestamp"] > entry_ts) & (funding_df["timestamp"] <= exit_ts)
    return funding_df[mask].copy()


def estimate_spread_bps(ohlcv_row: pd.Series) -> float:
    """Estimate half-spread in bps from OHLCV candle high/low."""
    if ohlcv_row["close"] <= 0:
        return 10.0
    hl_range = ohlcv_row["high"] - ohlcv_row["low"]
    spread = (hl_range / ohlcv_row["close"]) * 0.5 * 10_000
    return min(float(spread), 100.0)


def maker_fill_simulated(candle: pd.Series, limit_price: float, side: str) -> bool:
    """Returns True if a maker limit order at limit_price would fill in this candle."""
    if side == "buy":
        return float(candle["low"]) <= limit_price
    return float(candle["high"]) >= limit_price


def split_walk_forward(
    df: pd.DataFrame,
    in_sample_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    in_sample = df[df["timestamp"] <= in_sample_end].copy()
    out_of_sample = df[df["timestamp"] > in_sample_end].copy()
    return in_sample, out_of_sample


def run_backtest(
    funding_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    orderbook_df: pd.DataFrame,
    config: BacktestConfig,
) -> BacktestResult:
    funding_df = funding_df.sort_values("timestamp").reset_index(drop=True)
    ohlcv_df = ohlcv_df.sort_values("timestamp").reset_index(drop=True)

    funding_df["rolling_avg"] = (
        funding_df["funding_rate"]
        .rolling(window=config.persistence_window, min_periods=1)
        .mean()
    )

    cycles: list[CycleResult] = []
    in_position = False
    entry_ts: pd.Timestamp | None = None
    entry_price: float = 0.0

    for i, row in funding_df.iterrows():
        ts = row["timestamp"]
        rate = row["funding_rate"]
        rolling = row["rolling_avg"]

        ohlcv_row = _get_ohlcv_row(ohlcv_df, ts)
        if ohlcv_row is None:
            continue

        mid_price = float(ohlcv_row["close"])
        spread_bps = estimate_spread_bps(ohlcv_row)

        if not in_position:
            if rate >= config.entry_threshold and rolling >= config.persistence_threshold:
                horizon_days = min(config.max_holding_days, 14)
                horizon_intervals = horizon_days * 3
                if config.causal_gate:
                    # CAUSAL: trailing rolling funding (known at decision time), assumed to persist over horizon.
                    expected_funding_bps = float(rolling) * horizon_intervals * 10_000
                else:
                    # OPTIMISTIC (look-ahead): peeks at realized future funding -> upper bound only.
                    future_funding = funding_df[funding_df["timestamp"] > ts].head(horizon_intervals)
                    remaining_intervals = len(future_funding)
                    avg_future_rate = float(future_funding["funding_rate"].mean()) if remaining_intervals > 0 else 0.0
                    expected_funding_bps = avg_future_rate * remaining_intervals * 10_000

                if gate_passed(expected_funding_bps, config.cost_model):
                    in_position = True
                    entry_ts = ts
                    entry_price = mid_price * (1 + spread_bps / 10_000)

        else:
            assert entry_ts is not None
            days_held = (ts - entry_ts).total_seconds() / 86_400
            exit_reason = None

            # Funding-drop signal: instant single-rate vs hysteresis on trailing rolling avg
            if config.exit_mode == "rolling":
                funding_dropped = rolling < config.exit_threshold
            else:
                funding_dropped = rate < config.exit_threshold
            min_hold_ok = (days_held * 3.0) >= config.min_hold_intervals

            if funding_dropped and min_hold_ok:
                exit_reason = "funding_flip"
            elif days_held >= config.max_holding_days:
                exit_reason = "max_holding"

            if exit_reason:
                collected = count_funding_intervals(entry_ts, ts, funding_df)
                funding_received_bps = float(collected["funding_rate"].sum()) * 10_000

                rebalances = _count_rebalances(entry_ts, ts, ohlcv_df, entry_price)
                rebalance_cost = rebalances * compute_round_trip_cost_bps(config.cost_model) * 0.5
                cost = compute_round_trip_cost_bps(config.cost_model) + rebalance_cost
                net_pnl = funding_received_bps - cost

                cycles.append(CycleResult(
                    entry_ts=entry_ts,
                    exit_ts=ts,
                    intervals_collected=len(collected),
                    funding_received_bps=funding_received_bps,
                    round_trip_cost_bps=cost,
                    net_pnl_bps=net_pnl,
                    gate_passed=net_pnl >= 0,
                    rebalance_count=rebalances,
                    exit_reason=exit_reason,
                ))
                in_position = False
                entry_ts = None

    # NOTE: open position at end of data is silently dropped (upward bias on truncated periods).
    # For multi-year backtests the effect is negligible; make explicit in Phase-1 with forced close.
    return _build_result(cycles, funding_df)


def _get_ohlcv_row(ohlcv_df: pd.DataFrame, ts: pd.Timestamp) -> pd.Series | None:
    if ohlcv_df.empty:
        return None
    idx = ohlcv_df["timestamp"].searchsorted(ts, side="right") - 1
    if idx < 0 or idx >= len(ohlcv_df):
        return None
    return ohlcv_df.iloc[idx]


def _count_rebalances(
    entry_ts: pd.Timestamp,
    exit_ts: pd.Timestamp,
    ohlcv_df: pd.DataFrame,
    entry_price: float,
) -> int:
    if entry_price <= 0:
        return 0
    period = ohlcv_df[
        (ohlcv_df["timestamp"] >= entry_ts) & (ohlcv_df["timestamp"] <= exit_ts)
    ]
    if period.empty:
        return 0
    drift = (period["close"] - entry_price).abs() / entry_price
    above = drift > REBALANCE_DELTA_THRESHOLD
    # Count leading-edge crossings (below→above threshold), not candles above
    return int((above & ~above.shift(1, fill_value=False)).sum())


def _build_result(cycles: list[CycleResult], funding_df: pd.DataFrame) -> BacktestResult:
    if not cycles:
        return BacktestResult(
            cycles=[],
            equity_curve=pd.Series(dtype=float),
            max_drawdown_bps=0.0,
            net_pnl_bps=0.0,
            pct_gate_passed=0.0,
            n_intervals_total=len(funding_df),
        )

    pnls = [c.net_pnl_bps for c in cycles]
    equity = pd.Series(
        data=pd.Series(pnls).cumsum().values,
        index=[c.exit_ts for c in cycles],
        name="equity_bps",
    )
    running_max = equity.cummax()
    drawdown = equity - running_max
    max_dd = float(drawdown.min())

    return BacktestResult(
        cycles=cycles,
        equity_curve=equity,
        max_drawdown_bps=max_dd,
        net_pnl_bps=float(sum(pnls)),
        pct_gate_passed=float(sum(1 for c in cycles if c.gate_passed) / len(cycles)),
        n_intervals_total=len(funding_df),
    )
