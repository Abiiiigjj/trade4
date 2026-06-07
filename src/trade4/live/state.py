"""SQLite persistence for live positions, trades, and balance log."""
import sqlite3
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

import pandas as pd

logger = logging.getLogger(__name__)

DB_PATH = Path("data/live.db")


@dataclass
class OpenPosition:
    id: int
    symbol: str
    spot_qty: float
    perp_qty: float
    spot_entry_price: float
    perp_entry_price: float
    notional_eur: float
    opened_at: datetime
    fdusd_eligible: bool
    perp_leverage: int


@contextmanager
def _conn(path: Path = DB_PATH) -> Generator[sqlite3.Connection, None, None]:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db(path: Path = DB_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _conn(path) as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS positions (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol           TEXT    NOT NULL,
                spot_qty         REAL    NOT NULL,
                perp_qty         REAL    NOT NULL,
                spot_entry_price REAL    NOT NULL,
                perp_entry_price REAL    NOT NULL,
                notional_eur     REAL    NOT NULL,
                opened_at        TEXT    NOT NULL,
                fdusd_eligible   INTEGER NOT NULL DEFAULT 0,
                perp_leverage    INTEGER NOT NULL DEFAULT 3,
                status           TEXT    NOT NULL DEFAULT 'open'
            );
            CREATE TABLE IF NOT EXISTS trades (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol           TEXT NOT NULL,
                opened_at        TEXT NOT NULL,
                closed_at        TEXT NOT NULL,
                notional_eur     REAL NOT NULL,
                funding_collected_bps REAL NOT NULL,
                cost_bps         REAL NOT NULL,
                net_pnl_eur      REAL NOT NULL,
                exit_reason      TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS balance_log (
                ts               TEXT NOT NULL,
                spot_usdt        REAL NOT NULL,
                spot_fdusd       REAL NOT NULL,
                futures_usdt     REAL NOT NULL,
                total_eur        REAL NOT NULL
            );
        """)
    logger.info("DB initialised at %s", path)


def open_position(
    symbol: str,
    spot_qty: float,
    perp_qty: float,
    spot_entry: float,
    perp_entry: float,
    notional_eur: float,
    fdusd_eligible: bool,
    perp_leverage: int = 3,
    path: Path = DB_PATH,
) -> int:
    with _conn(path) as con:
        cur = con.execute(
            """INSERT INTO positions
               (symbol, spot_qty, perp_qty, spot_entry_price, perp_entry_price,
                notional_eur, opened_at, fdusd_eligible, perp_leverage)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (symbol, spot_qty, perp_qty, spot_entry, perp_entry, notional_eur,
             datetime.now(timezone.utc).isoformat(), int(fdusd_eligible), perp_leverage),
        )
        return cur.lastrowid


def close_position(
    position_id: int,
    funding_collected_bps: float,
    cost_bps: float,
    net_pnl_eur: float,
    exit_reason: str,
    path: Path = DB_PATH,
) -> None:
    with _conn(path) as con:
        row = con.execute(
            "SELECT * FROM positions WHERE id=? AND status='open'", (position_id,)
        ).fetchone()
        if not row:
            raise ValueError(f"No open position with id={position_id}")
        con.execute(
            "UPDATE positions SET status='closed' WHERE id=?", (position_id,)
        )
        con.execute(
            """INSERT INTO trades
               (symbol, opened_at, closed_at, notional_eur,
                funding_collected_bps, cost_bps, net_pnl_eur, exit_reason)
               VALUES (?,?,?,?,?,?,?,?)""",
            (row["symbol"], row["opened_at"],
             datetime.now(timezone.utc).isoformat(),
             row["notional_eur"], funding_collected_bps, cost_bps, net_pnl_eur,
             exit_reason),
        )


def get_open_positions(path: Path = DB_PATH) -> list[OpenPosition]:
    with _conn(path) as con:
        rows = con.execute(
            "SELECT * FROM positions WHERE status='open' ORDER BY opened_at"
        ).fetchall()
    return [
        OpenPosition(
            id=r["id"], symbol=r["symbol"],
            spot_qty=r["spot_qty"], perp_qty=r["perp_qty"],
            spot_entry_price=r["spot_entry_price"],
            perp_entry_price=r["perp_entry_price"],
            notional_eur=r["notional_eur"],
            opened_at=datetime.fromisoformat(r["opened_at"]),
            fdusd_eligible=bool(r["fdusd_eligible"]),
            perp_leverage=r["perp_leverage"],
        )
        for r in rows
    ]


def open_symbols(path: Path = DB_PATH) -> set[str]:
    with _conn(path) as con:
        rows = con.execute(
            "SELECT symbol FROM positions WHERE status='open'"
        ).fetchall()
    return {r["symbol"] for r in rows}


def log_balance(
    spot_usdt: float,
    spot_fdusd: float,
    futures_usdt: float,
    total_eur: float,
    path: Path = DB_PATH,
) -> None:
    with _conn(path) as con:
        con.execute(
            "INSERT INTO balance_log VALUES (?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), spot_usdt, spot_fdusd,
             futures_usdt, total_eur),
        )


def get_trade_history(path: Path = DB_PATH) -> pd.DataFrame:
    with _conn(path) as con:
        return pd.read_sql("SELECT * FROM trades ORDER BY closed_at", con)
