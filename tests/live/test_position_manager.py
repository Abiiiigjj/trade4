import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta
from trade4.live.state import OpenPosition
from trade4.live.position_manager import (
    should_exit, needs_rebalance, compute_entries, make_decisions,
    EXIT_FUNDING_THRESHOLD, MAX_HOLDING_DAYS,
)


def _pos(
    symbol="BTCUSDT",
    spot_qty=0.01,
    spot_entry=50000.0,
    perp_entry=50010.0,
    days_old=0,
    notional=500.0,
) -> OpenPosition:
    opened = datetime.now(timezone.utc) - timedelta(days=days_old)
    return OpenPosition(
        id=1, symbol=symbol,
        spot_qty=spot_qty, perp_qty=spot_qty,
        spot_entry_price=spot_entry, perp_entry_price=perp_entry,
        notional_eur=notional, opened_at=opened,
        fdusd_eligible=True, perp_leverage=3,
    )


def _screener(symbols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "symbol": symbols,
        "avg_funding_30d": [0.0001 * (i + 1) for i in range(len(symbols))],
        "fdusd_zero_fee": [True] * len(symbols),
    })


# ── should_exit ───────────────────────────────────────────────────────────────

def test_exit_on_low_funding():
    pos = _pos()
    flag, reason = should_exit(pos, EXIT_FUNDING_THRESHOLD - 0.00001, 50000.0)
    assert flag is True
    assert "funding" in reason


def test_no_exit_positive_funding():
    pos = _pos()
    flag, _ = should_exit(pos, 0.0001, 50000.0)
    assert flag is False


def test_exit_on_max_holding_days():
    pos = _pos(days_old=MAX_HOLDING_DAYS + 1)
    flag, reason = should_exit(pos, 0.0001, 50000.0)
    assert flag is True
    assert "max_holding" in reason


def test_no_exit_within_holding_days():
    pos = _pos(days_old=MAX_HOLDING_DAYS - 1)
    flag, _ = should_exit(pos, 0.0001, 50000.0)
    assert flag is False


# ── needs_rebalance ───────────────────────────────────────────────────────────

def test_no_rebalance_within_threshold():
    pos = _pos(spot_entry=50000.0)
    needed, _ = needs_rebalance(pos, 50500.0)  # 1% move — below 2% threshold
    assert needed is False


def test_rebalance_needed_on_large_move():
    pos = _pos(spot_entry=50000.0)
    needed, target = needs_rebalance(pos, 52000.0)  # 4% move
    assert needed is True
    assert target == pytest.approx(pos.spot_qty)


# ── compute_entries ───────────────────────────────────────────────────────────

def test_compute_entries_respects_open_symbols():
    screener = _screener(["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    entries = compute_entries(screener, {"BTCUSDT"}, max_positions=3, notional_per_position_eur=500.0)
    symbols = [e["symbol"] for e in entries]
    assert "BTCUSDT" not in symbols
    assert len(entries) == 2


def test_compute_entries_respects_max_positions():
    screener = _screener(["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"])
    entries = compute_entries(screener, set(), max_positions=2, notional_per_position_eur=500.0)
    assert len(entries) == 2


def test_compute_entries_empty_when_slots_full():
    screener = _screener(["ETHUSDT"])
    entries = compute_entries(screener, {"BTCUSDT", "ETHUSDT"}, max_positions=2, notional_per_position_eur=500.0)
    assert entries == []


# ── make_decisions ────────────────────────────────────────────────────────────

def test_make_decisions_closes_low_funding():
    pos = _pos("BTCUSDT")
    funding_data = {"BTCUSDT": pd.DataFrame({
        "timestamp": [pd.Timestamp.now(tz="UTC")],
        "funding_rate": [-0.0001],  # below threshold
    })}
    decisions = make_decisions(
        [pos], _screener([]), funding_data, {"BTCUSDT": 50000.0}, 3, 500.0
    )
    close_dec = [d for d in decisions if d.action == "close"]
    assert len(close_dec) == 1
    assert close_dec[0].symbol == "BTCUSDT"


def test_make_decisions_opens_new_when_slot_available():
    funding_data = {"ETHUSDT": pd.DataFrame({
        "timestamp": [pd.Timestamp.now(tz="UTC")],
        "funding_rate": [0.0001],
    })}
    decisions = make_decisions(
        [], _screener(["ETHUSDT"]), funding_data, {}, 3, 500.0
    )
    open_dec = [d for d in decisions if d.action == "open"]
    assert any(d.symbol == "ETHUSDT" for d in open_dec)
