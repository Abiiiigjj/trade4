from dataclasses import dataclass, field
from typing import Literal
import pandas as pd
import numpy as np

from trade4.scalper.signals.ema_cross import generate_signals as _ema_signals
from trade4.scalper.signals.pump_scanner import generate_signals as _pump_signals
from trade4.scalper.risk_manager import size_position, circuit_breaker_triggered, PositionParams
from trade4.backtest.cost_model import ScalperCostModel, scalper_round_trip_bps


@dataclass
class TradeResult:
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    symbol: str
    strategy: Literal["ema_cross", "pump_scanner"]
    side: Literal["long", "short"]
    entry_price: float
    exit_price: float
    qty: float
    gross_pnl_eur: float
    cost_eur: float
    net_pnl_eur: float
    exit_reason: Literal["sl", "tp", "timeout", "circuit_breaker"]


@dataclass
class BacktestConfig:
    start_balance: float = 2000.0
    max_risk_fraction: float = 0.02
    max_leverage: int = 20
    daily_loss_limit: float = 0.03
    max_open_positions: int = 2
    # EMA-Cross params
    ema_fast: int = 9
    ema_slow: int = 21
    ema_trend: int = 200
    rsi_period: int = 14
    atr_period: int = 14
    atr_min_pct: float = 0.0005
    sl_atr_multiplier: float = 1.5
    tp_atr_multiplier: float = 2.5  # fixed TP (approximates trailing stop for backtest)
    max_hold_minutes_ema: int = 30
    # Pump Scanner params
    pump_volume_multiplier: float = 3.0
    pump_price_move_pct: float = 0.015
    pump_vol_ma_period: int = 20
    pump_sl_pct: float = 0.008
    pump_tp_pct: float = 0.012
    pump_max_leverage: int = 7
    max_hold_minutes_pump: int = 5
    # Cost
    cost_model: ScalperCostModel = field(default_factory=ScalperCostModel)
    # Session filter (UTC hours, inclusive start exclusive end)
    session_windows: tuple[tuple[int, int], ...] = ((7, 11), (13, 21))


def _in_session(ts: pd.Timestamp, windows: tuple[tuple[int, int], ...]) -> bool:
    h = ts.hour
    return any(start <= h < end for start, end in windows)


def _finalize_trade(
    pos: dict,
    symbol: str,
    exit_ts: pd.Timestamp,
    exit_price: float,
    exit_reason: Literal["sl", "tp", "timeout", "circuit_breaker"],
    cost_model: ScalperCostModel,
) -> TradeResult:
    qty: float = pos["qty"]
    entry_price: float = pos["entry_price"]
    side: str = pos["side"]
    params: PositionParams = pos["params"]

    gross_pnl = (exit_price - entry_price) * qty if side == "long" else (entry_price - exit_price) * qty
    stressed = pos["strategy"] == "pump_scanner"
    cost_bps = scalper_round_trip_bps(cost_model, stressed=stressed)
    cost_eur = params.notional * cost_bps / 10_000
    net_pnl = gross_pnl - cost_eur

    return TradeResult(
        entry_ts=pos["entry_ts"],
        exit_ts=exit_ts,
        symbol=symbol,
        strategy=pos["strategy"],
        side=side,
        entry_price=entry_price,
        exit_price=exit_price,
        qty=qty,
        gross_pnl_eur=gross_pnl,
        cost_eur=cost_eur,
        net_pnl_eur=net_pnl,
        exit_reason=exit_reason,
    )


def run_backtest(
    df_1m: pd.DataFrame,
    df_15m: pd.DataFrame,
    symbol: str,
    config: BacktestConfig,
) -> tuple[list[TradeResult], pd.Series]:
    """Event-loop backtest for one symbol. Returns (trades, equity_curve).

    equity_curve is a pd.Series indexed by 1m timestamps with running balance.
    SL/TP are checked against candle high/low (assumes worst-case intrabar fill).
    """
    df_1m = df_1m.sort_values("timestamp").reset_index(drop=True)
    df_15m = df_15m.sort_values("timestamp").reset_index(drop=True)

    ema_sigs = _ema_signals(
        df_1m, df_15m,
        ema_fast=config.ema_fast,
        ema_slow=config.ema_slow,
        ema_trend=config.ema_trend,
        rsi_period=config.rsi_period,
        atr_period=config.atr_period,
        atr_min_pct=config.atr_min_pct,
        sl_atr_multiplier=config.sl_atr_multiplier,
    )
    pump_sigs = _pump_signals(
        df_1m,
        volume_multiplier=config.pump_volume_multiplier,
        price_move_pct=config.pump_price_move_pct,
        vol_ma_period=config.pump_vol_ma_period,
        sl_pct=config.pump_sl_pct,
        tp_pct=config.pump_tp_pct,
    )

    ema_map: dict[pd.Timestamp, object] = {row.timestamp: row for row in ema_sigs.itertuples(index=False)}
    pump_map: dict[pd.Timestamp, object] = {row.timestamp: row for row in pump_sigs.itertuples(index=False)}

    balance = config.start_balance
    open_positions: list[dict] = []
    trades: list[TradeResult] = []
    equity_values: list[float] = []
    equity_index: list[pd.Timestamp] = []

    day_start_balance = balance
    realized_pnl_today = 0.0
    current_day: object = None

    for _, candle in df_1m.iterrows():
        ts: pd.Timestamp = candle["timestamp"]
        close: float = float(candle["close"])
        high: float = float(candle["high"])
        low: float = float(candle["low"])

        candle_day = ts.date()
        if candle_day != current_day:
            current_day = candle_day
            day_start_balance = balance
            realized_pnl_today = 0.0

        breaker = circuit_breaker_triggered(realized_pnl_today, day_start_balance, config.daily_loss_limit)
        if breaker and open_positions:
            for pos in open_positions:
                t = _finalize_trade(pos, symbol, ts, close, "circuit_breaker", config.cost_model)
                realized_pnl_today += t.net_pnl_eur
                balance += t.net_pnl_eur
                trades.append(t)
            open_positions = []

        if not breaker:
            still_open: list[dict] = []
            for pos in open_positions:
                sl = pos["sl"]
                tp = pos["tp"]
                side = pos["side"]
                timeout_ts: pd.Timestamp = pos["timeout_ts"]

                hit_sl = (side == "long" and low <= sl) or (side == "short" and high >= sl)
                hit_tp = (side == "long" and high >= tp) or (side == "short" and low <= tp)
                timed_out = ts >= timeout_ts

                if hit_sl or hit_tp or timed_out:
                    # SL takes priority over TP if both triggered
                    exit_price = sl if hit_sl else (tp if hit_tp else close)
                    reason: Literal["sl", "tp", "timeout"] = "sl" if hit_sl else ("tp" if hit_tp else "timeout")
                    t = _finalize_trade(pos, symbol, ts, exit_price, reason, config.cost_model)
                    realized_pnl_today += t.net_pnl_eur
                    balance += t.net_pnl_eur
                    trades.append(t)
                else:
                    still_open.append(pos)
            open_positions = still_open

            if len(open_positions) < config.max_open_positions and _in_session(ts, config.session_windows):
                if ts in ema_map:
                    sig = ema_map[ts]
                    side = "long" if sig.signal == 1 else "short"
                    sl_price = sig.sl_price_long if side == "long" else sig.sl_price_short
                    tp_price = (
                        close + sig.atr_1m * config.tp_atr_multiplier
                        if side == "long"
                        else close - sig.atr_1m * config.tp_atr_multiplier
                    )
                    params = size_position(balance, close, sl_price, side,
                                           config.max_risk_fraction, config.max_leverage)
                    if params.liq_buffer_ok:
                        open_positions.append({
                            "entry_ts": ts,
                            "side": side,
                            "entry_price": close,
                            "sl": sl_price,
                            "tp": tp_price,
                            "timeout_ts": ts + pd.Timedelta(minutes=config.max_hold_minutes_ema),
                            "qty": params.qty,
                            "strategy": "ema_cross",
                            "params": params,
                        })

                if len(open_positions) < config.max_open_positions and ts in pump_map:
                    sig = pump_map[ts]
                    side = "long" if sig.signal == 1 else "short"
                    params = size_position(balance, close, sig.sl_price, side,
                                           config.max_risk_fraction, config.pump_max_leverage)
                    if params.liq_buffer_ok:
                        open_positions.append({
                            "entry_ts": ts,
                            "side": side,
                            "entry_price": close,
                            "sl": sig.sl_price,
                            "tp": sig.tp_price,
                            "timeout_ts": ts + pd.Timedelta(minutes=config.max_hold_minutes_pump),
                            "qty": params.qty,
                            "strategy": "pump_scanner",
                            "params": params,
                        })

        equity_values.append(balance)
        equity_index.append(ts)

    last_close = float(df_1m.iloc[-1]["close"])
    last_ts = df_1m.iloc[-1]["timestamp"]
    for pos in open_positions:
        t = _finalize_trade(pos, symbol, last_ts, last_close, "timeout", config.cost_model)
        balance += t.net_pnl_eur
        trades.append(t)

    return trades, pd.Series(equity_values, index=equity_index)
