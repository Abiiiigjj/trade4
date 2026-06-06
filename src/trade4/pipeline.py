"""
Phase-0 pipeline: fetch → screen → backtest → report.
Run: python -m trade4.pipeline
"""
import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from trade4.data import binance as bn
from trade4.data import okx as ox
from trade4.data.store import load_df, save_df, get_last_timestamp
from trade4.screener.screener import screen_coins, ScreenerConfig
from trade4.backtest.cost_model import CostModel, DEFAULT_FEE_SCHEDULE
from trade4.backtest.engine import BacktestConfig, run_backtest, split_walk_forward
from trade4.report.report import generate_report

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
REPORT_PATH = Path("output/phase0_report.html")
IN_SAMPLE_END = pd.Timestamp("2024-12-31", tz="UTC")
HISTORY_START = pd.Timestamp("2023-01-01", tz="UTC")
TOP_N_SYMBOLS = 100


def _fetch_with_cache(exchange: str, symbol: str, fetcher_fn, data_type: str) -> pd.DataFrame:
    last = get_last_timestamp(exchange, data_type, symbol, base_dir=DATA_DIR)
    start = last + pd.Timedelta(hours=1) if last else HISTORY_START
    fresh = fetcher_fn(symbol, start_ts=start)
    if fresh.empty:
        existing = load_df(exchange, data_type, symbol, base_dir=DATA_DIR)
        return existing if existing is not None else pd.DataFrame()
    existing = load_df(exchange, data_type, symbol, base_dir=DATA_DIR)
    combined = pd.concat([existing, fresh] if existing is not None else [fresh])
    combined = combined.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    save_df(combined, exchange, data_type, symbol, base_dir=DATA_DIR)
    return combined


def main() -> None:
    logger.info("=== Phase-0 Pipeline Start ===")

    logger.info("Loading Binance perp symbols...")
    symbols = bn.list_perp_symbols()[:TOP_N_SYMBOLS]
    logger.info("Screening %d symbols", len(symbols))

    funding_data: dict[str, pd.DataFrame] = {}
    orderbook_data: dict[str, pd.DataFrame] = {}
    ohlcv_data: dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        logger.info("Fetching %s...", symbol)
        funding_data[symbol] = _fetch_with_cache(
            "binance", symbol, bn.fetch_funding_history, "funding"
        )
        ohlcv_data[symbol] = _fetch_with_cache(
            "binance", symbol,
            lambda s, start_ts: bn.fetch_ohlcv(s, "1d", start_ts),
            "ohlcv",
        )
        orderbook_data[symbol] = bn.fetch_orderbook(symbol)

    config = ScreenerConfig(
        entry_threshold_per_interval=0.00005,
        max_slippage_bps=50.0,
        position_size_eur=500.0,
        min_pct_positive=0.55,
        volume_fraction_cap=0.005,
    )
    screener_df = screen_coins(symbols, funding_data, orderbook_data, ohlcv_data, config)
    logger.info("Screener passed: %d coins", len(screener_df))

    if screener_df.empty:
        logger.warning("No coins passed screening. Adjust thresholds or fetch more symbols.")
        return

    backtest_results = {}
    for _, row in screener_df.iterrows():
        symbol = row["symbol"]
        if symbol not in funding_data or funding_data[symbol].empty:
            continue
        use_fdusd = bool(row["fdusd_zero_fee"])
        bt_model = CostModel(
            fee_schedule=DEFAULT_FEE_SCHEDULE,
            slippage_entry_bps=5.0,
            slippage_exit_bps=5.0,
            basis_drift_bps=2.0,
            fdusd_depeg_bps=0.5 if use_fdusd else 0.0,
            use_fdusd=use_fdusd,
            use_maker_spot=False,
            use_maker_perp=False,
        )
        bt_config = BacktestConfig(
            entry_threshold=0.00005,
            exit_threshold=0.0,
            persistence_threshold=0.00003,
            persistence_window=5,
            max_holding_days=30,
            position_size_eur=500.0,
            cost_model=bt_model,
        )
        ob = orderbook_data.get(symbol, pd.DataFrame())
        result = run_backtest(funding_data[symbol], ohlcv_data.get(symbol, pd.DataFrame()), ob, bt_config)
        backtest_results[symbol] = result
        logger.info("%s: %d cycles, net P&L %.1f bps", symbol, len(result.cycles), result.net_pnl_bps)

    generate_report(screener_df, backtest_results, output_path=REPORT_PATH)
    logger.info("Report saved to %s", REPORT_PATH)
    logger.info("=== Phase-0 Pipeline Complete ===")


if __name__ == "__main__":
    main()
