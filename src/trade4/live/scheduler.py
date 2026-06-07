"""
Main live loop for funding-rate capture.

Runs every 8h after Binance funding settlement (00:05, 08:05, 16:05 UTC).
Usage:
    python -m trade4.live.scheduler              # live
    python -m trade4.live.scheduler --testnet    # testnet (paper)
    python -m trade4.live.scheduler --once       # single cycle then exit
"""
import argparse
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from trade4.data import binance as bn
from trade4.screener.screener import screen_coins, ScreenerConfig
from trade4.live.executor import BinanceExecutor
from trade4.live.position_manager import make_decisions, compute_entries
from trade4.live.state import (
    init_db, get_open_positions, open_symbols, open_position,
    close_position, log_balance, get_trade_history, DB_PATH,
)
from trade4.live import telegram as tg

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
MAX_POSITIONS          = int(os.getenv("MAX_POSITIONS", "4"))
NOTIONAL_PER_POS_EUR   = float(os.getenv("NOTIONAL_PER_POS_EUR", "500"))
PERP_LEVERAGE          = int(os.getenv("PERP_LEVERAGE", "3"))
EUR_TO_USDT            = float(os.getenv("EUR_TO_USDT", "1.08"))   # update manually or fetch
HISTORY_START          = pd.Timestamp("2023-01-01", tz="UTC")

_TESTNET_MODE = os.getenv("TRADE4_TESTNET_THRESHOLDS", "0") == "1"

SCREENER_CONFIG = ScreenerConfig(
    # In testnet mode use relaxed threshold to verify full order flow.
    # Live: keep at 5e-5 (funding must cover round-trip cost).
    entry_threshold_per_interval=0.00003 if _TESTNET_MODE else 0.00005,
    max_slippage_bps=50.0,
    position_size_eur=NOTIONAL_PER_POS_EUR,
    min_pct_positive=0.60,
    volume_fraction_cap=0.005,
    min_intervals=270.0,
    require_positive_90d=False if _TESTNET_MODE else True,
    stress_gate=not _TESTNET_MODE,
)

# Funding-capture universe — FDUSD symbols first (zero spot fee), rest standard
UNIVERSE: list[str] = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "LINKUSDT", "BNBUSDT", "XRPUSDT",
    "AVAXUSDT", "ADAUSDT", "DOTUSDT", "POLUSDT", "LTCUSDT",
]


def _fetch_universe() -> tuple[dict, dict, dict]:
    """Fetch funding, orderbook, ohlcv for each symbol in universe."""
    funding_data, orderbook_data, ohlcv_data = {}, {}, {}
    for sym in UNIVERSE:
        try:
            funding_data[sym]   = bn.fetch_funding_history(sym, HISTORY_START)
            orderbook_data[sym] = bn.fetch_orderbook(sym)
            ohlcv_data[sym]     = bn.fetch_ohlcv(sym, "1d",
                                     pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=100))
        except Exception as exc:
            logger.warning("Fetch failed for %s: %s", sym, exc)
    return funding_data, orderbook_data, ohlcv_data


def _current_prices(symbols: set[str], executor: BinanceExecutor) -> dict[str, float]:
    prices = {}
    for sym in symbols:
        try:
            prices[sym] = executor.get_spot_price(sym)
        except Exception as exc:
            logger.warning("Price fetch failed for %s: %s", sym, exc)
    return prices


def _net_pnl_for_position(
    position,
    spot_exit: float,
    perp_exit: float,
    funding_data: dict,
) -> tuple[float, float, float]:
    """Estimate net PnL for a closed position.
    Returns (funding_bps, cost_bps, net_pnl_eur).
    """
    # Funding collected: sum of funding payments received while short perp
    fd = funding_data.get(position.symbol, pd.DataFrame())
    if not fd.empty:
        mask = fd["timestamp"] >= pd.Timestamp(position.opened_at)
        collected_rates = fd[mask]["funding_rate"].sum()
        funding_bps = collected_rates * position.perp_qty * position.perp_entry_price / 100
    else:
        funding_bps = 0.0

    # Round-trip cost (taker close, maker open approximation)
    notional_usdt = position.notional_eur * EUR_TO_USDT
    cost_bps = notional_usdt * 0.0015  # ~15 bps round trip for FDUSD maker
    cost_eur = cost_bps / EUR_TO_USDT

    # Price PnL (should be near zero for delta-neutral, small slippage)
    spot_pnl = (spot_exit - position.spot_entry_price) * position.spot_qty
    perp_pnl = (position.perp_entry_price - perp_exit) * position.perp_qty
    price_pnl_eur = (spot_pnl + perp_pnl) / EUR_TO_USDT

    net_pnl_eur = price_pnl_eur + funding_bps / EUR_TO_USDT - cost_eur
    return funding_bps, cost_bps, net_pnl_eur


def run_cycle(executor: BinanceExecutor) -> None:
    logger.info("=== Cycle start %s ===", datetime.now(timezone.utc).isoformat())

    # 1. Fetch market data
    funding_data, orderbook_data, ohlcv_data = _fetch_universe()

    # 2. Screen
    screener_df = screen_coins(
        UNIVERSE, funding_data, orderbook_data, ohlcv_data, SCREENER_CONFIG
    )
    logger.info("Screener passed: %d symbols", len(screener_df))

    # 3. Load current positions
    positions = get_open_positions()
    open_syms  = {p.symbol for p in positions}
    screener_symbols = set(screener_df["symbol"].tolist()) if not screener_df.empty else set()
    prices     = _current_prices(open_syms | screener_symbols, executor)

    # 4. Make decisions
    decisions = make_decisions(
        positions, screener_df, funding_data, prices,
        MAX_POSITIONS, NOTIONAL_PER_POS_EUR,
    )

    # 5. Execute
    for dec in decisions:
        try:
            if dec.action == "close":
                pos = next(p for p in positions if p.symbol == dec.symbol)
                spot_exit, perp_exit = executor.close_delta_neutral(
                    pos.symbol, pos.spot_qty, pos.perp_qty, pos.fdusd_eligible
                )
                funding_bps, cost_bps, net_pnl = _net_pnl_for_position(
                    pos, spot_exit, perp_exit, funding_data
                )
                close_position(pos.id, funding_bps, cost_bps, net_pnl, dec.reason)
                tg.alert_closed(dec.symbol, net_pnl, dec.reason)
                logger.info("Closed %s  net_pnl=%.2f EUR  reason=%s",
                            dec.symbol, net_pnl, dec.reason)

            elif dec.action == "open":
                row = screener_df[screener_df["symbol"] == dec.symbol]
                if row.empty:
                    continue
                fdusd_ok = bool(row.iloc[0].get("fdusd_zero_fee", False))
                s_qty, s_entry, p_qty, p_entry = executor.open_delta_neutral(
                    dec.symbol, NOTIONAL_PER_POS_EUR, fdusd_ok, PERP_LEVERAGE, EUR_TO_USDT
                )
                pos_id = open_position(
                    dec.symbol, s_qty, p_qty, s_entry, p_entry,
                    NOTIONAL_PER_POS_EUR, fdusd_ok, PERP_LEVERAGE,
                )
                tg.alert_opened(dec.symbol, NOTIONAL_PER_POS_EUR, s_entry, p_entry)
                logger.info("Opened %s id=%d  spot=%.6f@%.2f  perp=%.6f@%.2f",
                            dec.symbol, pos_id, s_qty, s_entry, p_qty, p_entry)

            elif dec.action == "rebalance":
                pos = next(p for p in positions if p.symbol == dec.symbol)
                price = prices.get(dec.symbol, pos.spot_entry_price)
                drift = abs(price - pos.spot_entry_price) / pos.spot_entry_price
                executor.set_perp_hedge_qty(
                    pos.symbol, dec.target_perp_qty, pos.perp_qty
                )
                tg.alert_rebalanced(dec.symbol, drift)
                logger.info("Rebalanced %s drift=%.2f%%", dec.symbol, drift * 100)

        except Exception as exc:
            logger.error("Action %s %s failed: %s", dec.action, dec.symbol, exc)
            tg.alert_error(f"{dec.action} {dec.symbol}", str(exc))

    # 6. Log balance snapshot
    try:
        spot_usdt  = executor.spot_balance("USDT")
        spot_fdusd = executor.spot_balance("FDUSD")
        fut_usdt   = executor.futures_usdt_balance()
        total_eur  = (spot_usdt + spot_fdusd + fut_usdt) / EUR_TO_USDT
        log_balance(spot_usdt, spot_fdusd, fut_usdt, total_eur)
        logger.info("Balance: spot_USDT=%.2f  spot_FDUSD=%.2f  futures_USDT=%.2f  total=%.2f EUR",
                    spot_usdt, spot_fdusd, fut_usdt, total_eur)
    except Exception as exc:
        logger.warning("Balance log failed: %s", exc)

    # 7. Daily summary (only at 00:05 UTC cycle)
    if datetime.now(timezone.utc).hour == 0:
        open_positions_now = get_open_positions()
        try:
            spot_usdt  = executor.spot_balance("USDT")
            spot_fdusd = executor.spot_balance("FDUSD")
            fut_usdt   = executor.futures_usdt_balance()
            total_eur  = (spot_usdt + spot_fdusd + fut_usdt) / EUR_TO_USDT
        except Exception:
            total_eur = 0.0
        trades_today = get_trade_history()
        daily_pnl = (
            trades_today["net_pnl_eur"]
            .tail(len(decisions))
            .sum()
        )
        tg.daily_summary(
            len(open_positions_now), total_eur, daily_pnl,
            [p.symbol for p in open_positions_now],
        )

    logger.info("=== Cycle complete ===")


def _next_settlement_sleep() -> float:
    """Seconds until 5 minutes after the next 8h funding settlement (UTC)."""
    now  = datetime.now(timezone.utc)
    hour = now.hour
    next_settlement_hour = ((hour // 8) + 1) * 8 % 24
    next_dt = now.replace(
        hour=next_settlement_hour, minute=5, second=0, microsecond=0
    )
    if next_dt <= now:
        next_dt = next_dt.replace(day=next_dt.day + 1)
    return (next_dt - now).total_seconds()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testnet", action="store_true",
                        help="Use Binance testnet (no real orders)")
    parser.add_argument("--once", action="store_true",
                        help="Run one cycle then exit")
    args = parser.parse_args()

    api_key    = os.environ["BINANCE_API_KEY"]
    api_secret = os.environ["BINANCE_API_SECRET"]

    init_db()
    executor = BinanceExecutor(api_key, api_secret, testnet=args.testnet)

    if args.testnet:
        logger.info("*** TESTNET MODE — no real orders will be placed ***")

    if args.once:
        run_cycle(executor)
        return

    logger.info("Starting live scheduler (MAX_POSITIONS=%d, NOTIONAL=%.0f EUR)",
                MAX_POSITIONS, NOTIONAL_PER_POS_EUR)
    while True:
        try:
            run_cycle(executor)
        except Exception as exc:
            logger.exception("Cycle failed: %s", exc)
            tg.alert_error("scheduler.run_cycle", str(exc))

        sleep_secs = _next_settlement_sleep()
        logger.info("Next cycle in %.0f minutes", sleep_secs / 60)
        time.sleep(sleep_secs)


if __name__ == "__main__":
    main()
