import pytest
from pathlib import Path
from datetime import timezone, datetime
from trade4.live.state import (
    init_db, open_position, close_position,
    get_open_positions, open_symbols, log_balance, get_trade_history,
)


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    path = tmp_path / "test.db"
    init_db(path)
    return path


def test_open_and_retrieve_position(tmp_db):
    pos_id = open_position(
        symbol="BTCUSDT", spot_qty=0.01, perp_qty=0.01,
        spot_entry=50000.0, perp_entry=50010.0,
        notional_eur=500.0, fdusd_eligible=True, perp_leverage=3,
        path=tmp_db,
    )
    assert pos_id == 1
    positions = get_open_positions(tmp_db)
    assert len(positions) == 1
    p = positions[0]
    assert p.symbol == "BTCUSDT"
    assert p.spot_qty == pytest.approx(0.01)
    assert p.fdusd_eligible is True


def test_open_symbols_returns_set(tmp_db):
    open_position("BTCUSDT", 0.01, 0.01, 50000.0, 50010.0, 500.0, True, 3, tmp_db)
    open_position("ETHUSDT", 0.1,  0.1,  3000.0,  3001.0,  300.0, True, 3, tmp_db)
    syms = open_symbols(tmp_db)
    assert syms == {"BTCUSDT", "ETHUSDT"}


def test_close_position_removes_from_open(tmp_db):
    pos_id = open_position(
        "SOLUSDT", 1.0, 1.0, 150.0, 151.0, 150.0, False, 3, tmp_db
    )
    close_position(pos_id, 5.0, 2.0, 3.0, "funding_below_threshold", tmp_db)
    assert get_open_positions(tmp_db) == []


def test_close_position_records_trade(tmp_db):
    pos_id = open_position(
        "DOGEUSDT", 1000.0, 1000.0, 0.10, 0.101, 100.0, True, 3, tmp_db
    )
    close_position(pos_id, 10.0, 3.0, 7.0, "funding_below_threshold", tmp_db)
    hist = get_trade_history(tmp_db)
    assert len(hist) == 1
    assert hist.iloc[0]["symbol"] == "DOGEUSDT"
    assert hist.iloc[0]["net_pnl_eur"] == pytest.approx(7.0)


def test_close_nonexistent_raises(tmp_db):
    with pytest.raises(ValueError, match="No open position"):
        close_position(999, 1.0, 0.5, 0.5, "test", tmp_db)


def test_log_balance(tmp_db):
    log_balance(500.0, 300.0, 200.0, 925.0, tmp_db)
    import sqlite3
    con = sqlite3.connect(tmp_db)
    rows = con.execute("SELECT * FROM balance_log").fetchall()
    con.close()
    assert len(rows) == 1
    assert rows[0][4] == pytest.approx(925.0)  # total_eur
