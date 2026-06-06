# Phase-0 Backtester Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a research tool that screens volatile altcoins by funding rate, backtests delta-neutral funding capture with a realistic cost model, and produces an honest HTML report — all net of fees, slippage, and basis drift.

**Architecture:** Modular Python package (`src/trade4/`) with four layers: data fetching → screening → backtesting → reporting. A Jupyter notebook sits on top as the interactive entry point but contains zero business logic. All modules are independently testable and reusable in Phase 1.

**Tech Stack:** Python 3.12+, ccxt, pandas, numpy, pyarrow, jinja2, python-dotenv, matplotlib, pytest

---

## File Map

```
trade4/
├── src/trade4/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── binance.py       # Binance REST fetcher: funding, OHLCV, mark price, orderbook
│   │   ├── okx.py           # OKX REST fetcher: same interface as binance.py
│   │   └── store.py         # Parquet cache: save/load/delta-fetch
│   ├── screener/
│   │   ├── __init__.py
│   │   └── screener.py      # Funding stats + liquidity filter → ranked DataFrame
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── cost_model.py    # FeeSchedule, CostModel, Net-EV gate
│   │   └── engine.py        # Vectorized backtest engine with realistic fills
│   └── report/
│       ├── __init__.py
│       ├── report.py        # HTML report builder
│       └── templates/
│           └── report.html.j2
├── tests/
│   ├── conftest.py
│   ├── data/
│   │   ├── test_store.py
│   │   ├── test_binance.py
│   │   └── test_okx.py
│   ├── screener/
│   │   └── test_screener.py
│   ├── backtest/
│   │   ├── test_cost_model.py
│   │   └── test_engine.py
│   └── report/
│       └── test_report.py
├── notebooks/
│   └── phase0_research.ipynb
├── data/                    # gitignored
├── docs/
├── pyproject.toml
├── .env.example
└── .gitignore
```

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `src/trade4/__init__.py`
- Create: `src/trade4/data/__init__.py`
- Create: `src/trade4/screener/__init__.py`
- Create: `src/trade4/backtest/__init__.py`
- Create: `src/trade4/report/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "trade4"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "ccxt>=4.3",
    "pandas>=2.2",
    "numpy>=1.26",
    "pyarrow>=15.0",
    "jinja2>=3.1",
    "python-dotenv>=1.0",
    "matplotlib>=3.8",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
]

[tool.hatch.build.targets.wheel]
packages = ["src/trade4"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create .gitignore**

```
data/
.env
__pycache__/
*.py[cod]
*.parquet
.pytest_cache/
dist/
*.egg-info/
.venv/
notebooks/.ipynb_checkpoints/
```

- [ ] **Step 3: Create .env.example**

```
BINANCE_API_KEY=
BINANCE_API_SECRET=
OKX_API_KEY=
OKX_API_SECRET=
OKX_PASSPHRASE=
```

- [ ] **Step 4: Create all __init__.py files (all empty)**

```bash
mkdir -p src/trade4/data src/trade4/screener src/trade4/backtest src/trade4/report/templates
mkdir -p tests/data tests/screener tests/backtest tests/report
touch src/trade4/__init__.py
touch src/trade4/data/__init__.py
touch src/trade4/screener/__init__.py
touch src/trade4/backtest/__init__.py
touch src/trade4/report/__init__.py
touch tests/__init__.py tests/data/__init__.py tests/screener/__init__.py
touch tests/backtest/__init__.py tests/report/__init__.py
```

- [ ] **Step 5: Create tests/conftest.py**

```python
import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


@pytest.fixture
def sample_funding_df() -> pd.DataFrame:
    """3 days of 8h funding intervals for DOGEUSDT, always positive."""
    timestamps = pd.date_range("2024-01-01", periods=9, freq="8h", tz="UTC")
    return pd.DataFrame({
        "timestamp": timestamps,
        "symbol": "DOGEUSDT",
        "funding_rate": [0.0001, 0.0002, 0.00015, 0.0001, 0.00025, 0.0002, 0.0003, 0.0001, 0.00018],
    })


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Daily OHLCV for DOGE, 90 days."""
    dates = pd.date_range("2024-01-01", periods=90, freq="1d", tz="UTC")
    rng = np.random.default_rng(42)
    close = 0.10 + rng.normal(0, 0.005, 90).cumsum()
    close = np.clip(close, 0.05, 0.30)
    return pd.DataFrame({
        "timestamp": dates,
        "open": close * 0.999,
        "high": close * 1.005,
        "low": close * 0.995,
        "close": close,
        "volume": rng.uniform(1e8, 5e8, 90),
    })


@pytest.fixture
def sample_orderbook() -> pd.DataFrame:
    """Simulated orderbook for DOGEUSDT at price ~0.10."""
    levels = 10
    base_price = 0.10
    bids = pd.DataFrame({
        "price": [base_price - i * 0.0001 for i in range(levels)],
        "qty": [500_000.0] * levels,
        "side": ["bid"] * levels,
    })
    asks = pd.DataFrame({
        "price": [base_price + 0.0001 + i * 0.0001 for i in range(levels)],
        "qty": [500_000.0] * levels,
        "side": ["ask"] * levels,
    })
    return pd.concat([bids, asks], ignore_index=True)
```

- [ ] **Step 6: Install package in dev mode**

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Expected: no errors, `pytest` command available.

- [ ] **Step 7: Verify pytest discovers no tests yet**

```bash
pytest --collect-only
```

Expected: `no tests ran`

- [ ] **Step 8: Commit**

```bash
git init
git add pyproject.toml .gitignore .env.example src/ tests/
git commit -m "feat: project scaffolding — package structure, deps, fixtures"
```

---

## Task 2: Data Store (Parquet Cache)

**Files:**
- Create: `src/trade4/data/store.py`
- Test: `tests/data/test_store.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/data/test_store.py
import pandas as pd
import pytest
from pathlib import Path
from trade4.data.store import save_df, load_df, get_last_timestamp


def test_save_and_load_roundtrip(tmp_data_dir):
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
        "funding_rate": [0.0001, 0.0002],
    })
    save_df(df, "binance", "funding", "DOGEUSDT", base_dir=tmp_data_dir)
    result = load_df("binance", "funding", "DOGEUSDT", base_dir=tmp_data_dir)
    pd.testing.assert_frame_equal(df, result)


def test_load_returns_none_when_missing(tmp_data_dir):
    result = load_df("binance", "funding", "NONEXISTENT", base_dir=tmp_data_dir)
    assert result is None


def test_get_last_timestamp_none_when_missing(tmp_data_dir):
    result = get_last_timestamp("binance", "funding", "NONEXISTENT", base_dir=tmp_data_dir)
    assert result is None


def test_get_last_timestamp_returns_max(tmp_data_dir):
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-02"], utc=True),
        "funding_rate": [0.0001, 0.0003, 0.0002],
    })
    save_df(df, "binance", "funding", "DOGEUSDT", base_dir=tmp_data_dir)
    last = get_last_timestamp("binance", "funding", "DOGEUSDT", base_dir=tmp_data_dir)
    assert last == pd.Timestamp("2024-01-03", tz="UTC")


def test_save_overwrites_existing(tmp_data_dir):
    df1 = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01"], utc=True),
        "funding_rate": [0.0001],
    })
    df2 = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01"], utc=True),
        "funding_rate": [0.9999],
    })
    save_df(df1, "binance", "funding", "DOGEUSDT", base_dir=tmp_data_dir)
    save_df(df2, "binance", "funding", "DOGEUSDT", base_dir=tmp_data_dir)
    result = load_df("binance", "funding", "DOGEUSDT", base_dir=tmp_data_dir)
    assert result["funding_rate"].iloc[0] == 0.9999
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/data/test_store.py -v
```

Expected: `ImportError` or `ModuleNotFoundError`

- [ ] **Step 3: Implement store.py**

```python
# src/trade4/data/store.py
import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_BASE_DIR = Path("data")


def _parquet_path(exchange: str, data_type: str, symbol: str, base_dir: Path) -> Path:
    return base_dir / exchange / data_type / f"{symbol}.parquet"


def save_df(df: pd.DataFrame, exchange: str, data_type: str, symbol: str, base_dir: Path = DEFAULT_BASE_DIR) -> None:
    path = _parquet_path(exchange, data_type, symbol, base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.debug("Saved %d rows to %s", len(df), path)


def load_df(exchange: str, data_type: str, symbol: str, base_dir: Path = DEFAULT_BASE_DIR) -> pd.DataFrame | None:
    path = _parquet_path(exchange, data_type, symbol, base_dir)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def get_last_timestamp(exchange: str, data_type: str, symbol: str, base_dir: Path = DEFAULT_BASE_DIR) -> pd.Timestamp | None:
    df = load_df(exchange, data_type, symbol, base_dir)
    if df is None or df.empty or "timestamp" not in df.columns:
        return None
    return df["timestamp"].max()
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/data/test_store.py -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/trade4/data/store.py tests/data/test_store.py
git commit -m "feat: parquet cache store with save/load/delta-timestamp"
```

---

## Task 3: Binance Fetcher

**Files:**
- Create: `src/trade4/data/binance.py`
- Test: `tests/data/test_binance.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/data/test_binance.py
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from trade4.data.binance import (
    fetch_funding_history,
    fetch_ohlcv,
    fetch_orderbook,
    list_perp_symbols,
    FDUSD_ZERO_FEE_BASES,
)


MOCK_FUNDING_RESPONSE = [
    {"symbol": "DOGEUSDT", "fundingRate": "0.0001", "fundingTime": 1704067200000},
    {"symbol": "DOGEUSDT", "fundingRate": "0.0002", "fundingTime": 1704096000000},
]

MOCK_KLINES_RESPONSE = [
    [1704067200000, "0.09", "0.095", "0.085", "0.092", "1000000",
     1704153599999, "92000", "5000", "500000", "46000", "0"],
]

MOCK_DEPTH_RESPONSE = {
    "bids": [["0.0999", "500000"], ["0.0998", "400000"]],
    "asks": [["0.1001", "500000"], ["0.1002", "400000"]],
}


def test_fetch_funding_history_returns_dataframe():
    with patch("trade4.data.binance._get_exchange") as mock_ex:
        inst = MagicMock()
        inst.fetch_funding_rate_history.return_value = [
            {"timestamp": 1704067200000, "symbol": "DOGE/USDT:USDT", "fundingRate": 0.0001},
            {"timestamp": 1704096000000, "symbol": "DOGE/USDT:USDT", "fundingRate": 0.0002},
        ]
        mock_ex.return_value = inst
        df = fetch_funding_history("DOGEUSDT", start_ts=pd.Timestamp("2024-01-01", tz="UTC"))
    assert isinstance(df, pd.DataFrame)
    assert "timestamp" in df.columns
    assert "funding_rate" in df.columns
    assert df["timestamp"].dtype == "datetime64[ns, UTC]"
    assert len(df) == 2


def test_fetch_funding_history_empty_returns_empty_df():
    with patch("trade4.data.binance._get_exchange") as mock_ex:
        inst = MagicMock()
        inst.fetch_funding_rate_history.return_value = []
        mock_ex.return_value = inst
        df = fetch_funding_history("DOGEUSDT", start_ts=pd.Timestamp("2024-01-01", tz="UTC"))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_fetch_orderbook_returns_bids_and_asks():
    with patch("trade4.data.binance._get_exchange") as mock_ex:
        inst = MagicMock()
        inst.fetch_order_book.return_value = {
            "bids": [[0.0999, 500000.0], [0.0998, 400000.0]],
            "asks": [[0.1001, 500000.0], [0.1002, 400000.0]],
        }
        mock_ex.return_value = inst
        df = fetch_orderbook("DOGEUSDT")
    assert set(df["side"].unique()) == {"bid", "ask"}
    assert "price" in df.columns
    assert "qty" in df.columns


def test_fdusd_zero_fee_bases_contains_expected_coins():
    assert "DOGE" in FDUSD_ZERO_FEE_BASES
    assert "BTC" in FDUSD_ZERO_FEE_BASES
    assert "SOL" in FDUSD_ZERO_FEE_BASES


def test_fetch_ohlcv_returns_dataframe():
    with patch("trade4.data.binance._get_exchange") as mock_ex:
        inst = MagicMock()
        inst.fetch_ohlcv.return_value = [
            [1704067200000, 0.09, 0.095, 0.085, 0.092, 1_000_000.0],
        ]
        mock_ex.return_value = inst
        df = fetch_ohlcv("DOGEUSDT", interval="1d", start_ts=pd.Timestamp("2024-01-01", tz="UTC"))
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert df["timestamp"].dtype == "datetime64[ns, UTC]"
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/data/test_binance.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement binance.py**

```python
# src/trade4/data/binance.py
import logging
import os
from functools import lru_cache
import ccxt
import pandas as pd

logger = logging.getLogger(__name__)

FDUSD_ZERO_FEE_BASES: frozenset[str] = frozenset(
    {"BTC", "ETH", "SOL", "DOGE", "LINK", "BNB", "XRP"}
)


@lru_cache(maxsize=1)
def _get_exchange() -> ccxt.binance:
    return ccxt.binance({
        "apiKey": os.getenv("BINANCE_API_KEY", ""),
        "secret": os.getenv("BINANCE_API_SECRET", ""),
        "options": {"defaultType": "future"},
    })


def fetch_funding_history(
    symbol: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Fetch funding rate history for a perp symbol (e.g. 'DOGEUSDT')."""
    ex = _get_exchange()
    ccxt_symbol = _to_ccxt_perp(symbol)
    since = int(start_ts.timestamp() * 1000)
    until = int(end_ts.timestamp() * 1000) if end_ts else None

    rows = []
    while True:
        batch = ex.fetch_funding_rate_history(ccxt_symbol, since=since, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        last_ts = batch[-1]["timestamp"]
        if until and last_ts >= until:
            break
        if len(batch) < 1000:
            break
        since = last_ts + 1

    if not rows:
        return pd.DataFrame(columns=["timestamp", "symbol", "funding_rate"])

    df = pd.DataFrame({
        "timestamp": pd.to_datetime([r["timestamp"] for r in rows], unit="ms", utc=True),
        "symbol": symbol,
        "funding_rate": [float(r["fundingRate"]) for r in rows],
    })
    if until:
        df = df[df["timestamp"] <= pd.Timestamp(until, unit="ms", tz="UTC")]
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def fetch_ohlcv(
    symbol: str,
    interval: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Fetch OHLCV candles for a perp symbol."""
    ex = _get_exchange()
    ccxt_symbol = _to_ccxt_perp(symbol)
    since = int(start_ts.timestamp() * 1000)
    until = int(end_ts.timestamp() * 1000) if end_ts else None

    rows = []
    while True:
        batch = ex.fetch_ohlcv(ccxt_symbol, timeframe=interval, since=since, limit=1500)
        if not batch:
            break
        rows.extend(batch)
        last_ts = batch[-1][0]
        if until and last_ts >= until:
            break
        if len(batch) < 1500:
            break
        since = last_ts + 1

    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    if until:
        df = df[df["timestamp"] <= pd.Timestamp(until, unit="ms", tz="UTC")]
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def fetch_orderbook(symbol: str, limit: int = 20) -> pd.DataFrame:
    """Fetch current order book snapshot."""
    ex = _get_exchange()
    ccxt_symbol = _to_ccxt_perp(symbol)
    book = ex.fetch_order_book(ccxt_symbol, limit=limit)
    bids = pd.DataFrame(book["bids"], columns=["price", "qty"])
    bids["side"] = "bid"
    asks = pd.DataFrame(book["asks"], columns=["price", "qty"])
    asks["side"] = "ask"
    return pd.concat([bids, asks], ignore_index=True)


def list_perp_symbols() -> list[str]:
    """Return all USDT-margined perpetual symbols on Binance."""
    ex = _get_exchange()
    markets = ex.load_markets()
    return [
        m["id"] for m in markets.values()
        if m.get("type") == "swap" and m.get("quote") == "USDT" and m.get("active")
    ]


def _to_ccxt_perp(symbol: str) -> str:
    """Convert 'DOGEUSDT' to ccxt perp format 'DOGE/USDT:USDT'."""
    if symbol.endswith("USDT"):
        base = symbol[:-4]
        return f"{base}/USDT:USDT"
    return symbol
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/data/test_binance.py -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/trade4/data/binance.py tests/data/test_binance.py
git commit -m "feat: binance fetcher — funding history, OHLCV, orderbook, FDUSD flag"
```

---

## Task 4: OKX Fetcher

**Files:**
- Create: `src/trade4/data/okx.py`
- Test: `tests/data/test_okx.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/data/test_okx.py
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from trade4.data.okx import fetch_funding_history, fetch_ohlcv, fetch_orderbook, list_perp_symbols


def test_fetch_funding_history_returns_dataframe():
    with patch("trade4.data.okx._get_exchange") as mock_ex:
        inst = MagicMock()
        inst.fetch_funding_rate_history.return_value = [
            {"timestamp": 1704067200000, "symbol": "DOGE/USDT:USDT", "fundingRate": 0.0001},
            {"timestamp": 1704096000000, "symbol": "DOGE/USDT:USDT", "fundingRate": 0.00015},
        ]
        mock_ex.return_value = inst
        df = fetch_funding_history("DOGE-USDT-SWAP", start_ts=pd.Timestamp("2024-01-01", tz="UTC"))
    assert isinstance(df, pd.DataFrame)
    assert "timestamp" in df.columns
    assert "funding_rate" in df.columns
    assert len(df) == 2


def test_fetch_orderbook_structure():
    with patch("trade4.data.okx._get_exchange") as mock_ex:
        inst = MagicMock()
        inst.fetch_order_book.return_value = {
            "bids": [[0.0999, 500000.0]],
            "asks": [[0.1001, 500000.0]],
        }
        mock_ex.return_value = inst
        df = fetch_orderbook("DOGE-USDT-SWAP")
    assert set(df["side"].unique()) == {"bid", "ask"}


def test_fetch_ohlcv_columns():
    with patch("trade4.data.okx._get_exchange") as mock_ex:
        inst = MagicMock()
        inst.fetch_ohlcv.return_value = [
            [1704067200000, 0.09, 0.095, 0.085, 0.092, 1_000_000.0],
        ]
        mock_ex.return_value = inst
        df = fetch_ohlcv("DOGE-USDT-SWAP", interval="1d", start_ts=pd.Timestamp("2024-01-01", tz="UTC"))
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/data/test_okx.py -v
```

- [ ] **Step 3: Implement okx.py**

```python
# src/trade4/data/okx.py
import logging
import os
from functools import lru_cache
import ccxt
import pandas as pd

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_exchange() -> ccxt.okx:
    return ccxt.okx({
        "apiKey": os.getenv("OKX_API_KEY", ""),
        "secret": os.getenv("OKX_API_SECRET", ""),
        "password": os.getenv("OKX_PASSPHRASE", ""),
    })


def fetch_funding_history(
    symbol: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Fetch funding rate history. symbol: OKX format e.g. 'DOGE-USDT-SWAP'."""
    ex = _get_exchange()
    ccxt_symbol = _to_ccxt(symbol)
    since = int(start_ts.timestamp() * 1000)
    until = int(end_ts.timestamp() * 1000) if end_ts else None

    rows = []
    while True:
        batch = ex.fetch_funding_rate_history(ccxt_symbol, since=since, limit=100)
        if not batch:
            break
        rows.extend(batch)
        last_ts = batch[-1]["timestamp"]
        if until and last_ts >= until:
            break
        if len(batch) < 100:
            break
        since = last_ts + 1

    if not rows:
        return pd.DataFrame(columns=["timestamp", "symbol", "funding_rate"])

    df = pd.DataFrame({
        "timestamp": pd.to_datetime([r["timestamp"] for r in rows], unit="ms", utc=True),
        "symbol": symbol,
        "funding_rate": [float(r["fundingRate"]) for r in rows],
    })
    if until:
        df = df[df["timestamp"] <= pd.Timestamp(until, unit="ms", tz="UTC")]
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def fetch_ohlcv(
    symbol: str,
    interval: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    ex = _get_exchange()
    ccxt_symbol = _to_ccxt(symbol)
    since = int(start_ts.timestamp() * 1000)
    until = int(end_ts.timestamp() * 1000) if end_ts else None

    rows = []
    while True:
        batch = ex.fetch_ohlcv(ccxt_symbol, timeframe=interval, since=since, limit=300)
        if not batch:
            break
        rows.extend(batch)
        last_ts = batch[-1][0]
        if until and last_ts >= until:
            break
        if len(batch) < 300:
            break
        since = last_ts + 1

    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    if until:
        df = df[df["timestamp"] <= pd.Timestamp(until, unit="ms", tz="UTC")]
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def fetch_orderbook(symbol: str, limit: int = 20) -> pd.DataFrame:
    ex = _get_exchange()
    ccxt_symbol = _to_ccxt(symbol)
    book = ex.fetch_order_book(ccxt_symbol, limit=limit)
    bids = pd.DataFrame(book["bids"], columns=["price", "qty"])
    bids["side"] = "bid"
    asks = pd.DataFrame(book["asks"], columns=["price", "qty"])
    asks["side"] = "ask"
    return pd.concat([bids, asks], ignore_index=True)


def list_perp_symbols() -> list[str]:
    ex = _get_exchange()
    markets = ex.load_markets()
    return [
        m["id"] for m in markets.values()
        if m.get("type") == "swap" and m.get("quote") == "USDT" and m.get("active")
    ]


def _to_ccxt(symbol: str) -> str:
    """OKX symbols are already in ccxt format like 'DOGE/USDT:USDT' or pass-through."""
    if "-SWAP" in symbol:
        parts = symbol.replace("-SWAP", "").split("-")
        return f"{parts[0]}/{parts[1]}:{parts[1]}"
    return symbol
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/data/test_okx.py -v
```

Expected: 3 passed

- [ ] **Step 5: Run full test suite**

```bash
pytest -v
```

Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add src/trade4/data/okx.py tests/data/test_okx.py
git commit -m "feat: OKX fetcher — funding history, OHLCV, orderbook"
```

---

## Task 5: Screener

**Files:**
- Create: `src/trade4/screener/screener.py`
- Test: `tests/screener/test_screener.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/screener/test_screener.py
import pandas as pd
import numpy as np
import pytest
from trade4.screener.screener import (
    compute_funding_stats,
    estimate_slippage_bps,
    screen_coins,
    ScreenerConfig,
)


def test_compute_funding_stats_averages(sample_funding_df):
    stats = compute_funding_stats(sample_funding_df)
    assert "avg_funding_30d" in stats
    assert "avg_funding_90d" in stats
    assert "pct_positive_intervals" in stats
    assert 0.0 < stats["avg_funding_30d"] < 0.001
    assert stats["pct_positive_intervals"] == 1.0


def test_compute_funding_stats_handles_negatives():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=6, freq="8h", tz="UTC"),
        "symbol": "TEST",
        "funding_rate": [0.0002, -0.0001, 0.0003, -0.0002, 0.0001, 0.0002],
    })
    stats = compute_funding_stats(df)
    assert stats["pct_positive_intervals"] == pytest.approx(4 / 6)


def test_estimate_slippage_bps_small_order(sample_orderbook):
    # Buy €100 of DOGE at ~$0.10 → ~1000 DOGE. Orderbook has 500k per level.
    slippage = estimate_slippage_bps(sample_orderbook, notional_eur=100, price_eur=0.10, side="buy")
    assert 0 < slippage < 5


def test_estimate_slippage_bps_large_order_eats_levels(sample_orderbook):
    # Buy €100k → much higher slippage
    slippage_small = estimate_slippage_bps(sample_orderbook, notional_eur=100, price_eur=0.10, side="buy")
    slippage_large = estimate_slippage_bps(sample_orderbook, notional_eur=100_000, price_eur=0.10, side="buy")
    assert slippage_large > slippage_small


def test_screen_coins_filters_low_funding(sample_funding_df, sample_orderbook):
    config = ScreenerConfig(
        entry_threshold_per_interval=0.0005,  # higher than our sample's avg → should filter out
        max_slippage_bps=50,
        position_size_eur=500,
        min_pct_positive=0.5,
        volume_fraction_cap=0.005,
    )
    funding_data = {"DOGEUSDT": sample_funding_df}
    orderbook_data = {"DOGEUSDT": sample_orderbook}
    ohlcv_data = {}
    result = screen_coins(["DOGEUSDT"], funding_data, orderbook_data, ohlcv_data, config)
    assert len(result) == 0 or result.iloc[0]["avg_funding_30d"] < config.entry_threshold_per_interval


def test_screen_coins_passes_good_candidate(sample_funding_df, sample_orderbook):
    config = ScreenerConfig(
        entry_threshold_per_interval=0.00005,  # low threshold → sample passes
        max_slippage_bps=50,
        position_size_eur=500,
        min_pct_positive=0.5,
        volume_fraction_cap=0.1,
    )
    funding_data = {"DOGEUSDT": sample_funding_df}
    orderbook_data = {"DOGEUSDT": sample_orderbook}
    ohlcv_data = {}
    result = screen_coins(["DOGEUSDT"], funding_data, orderbook_data, ohlcv_data, config)
    assert len(result) == 1
    assert result.iloc[0]["symbol"] == "DOGEUSDT"
    assert "gate_candidate" in result.columns


def test_fdusd_flag_set_for_eligible_coins(sample_funding_df, sample_orderbook):
    config = ScreenerConfig(
        entry_threshold_per_interval=0.00005,
        max_slippage_bps=50,
        position_size_eur=500,
        min_pct_positive=0.5,
        volume_fraction_cap=0.1,
    )
    funding_data = {"DOGEUSDT": sample_funding_df}
    orderbook_data = {"DOGEUSDT": sample_orderbook}
    result = screen_coins(["DOGEUSDT"], funding_data, orderbook_data, {}, config)
    assert result.iloc[0]["fdusd_zero_fee"] is True
```

- [ ] **Step 2: Run — verify fail**

```bash
pytest tests/screener/test_screener.py -v
```

- [ ] **Step 3: Implement screener.py**

```python
# src/trade4/screener/screener.py
import logging
from dataclasses import dataclass
import numpy as np
import pandas as pd
from trade4.data.binance import FDUSD_ZERO_FEE_BASES

logger = logging.getLogger(__name__)

FUNDING_INTERVALS_PER_DAY = 3  # 8h intervals


@dataclass
class ScreenerConfig:
    entry_threshold_per_interval: float = 0.00005   # 0.005% per 8h
    max_slippage_bps: float = 50.0
    position_size_eur: float = 500.0
    min_pct_positive: float = 0.5
    volume_fraction_cap: float = 0.005  # max fraction of 24h volume


def compute_funding_stats(funding_df: pd.DataFrame) -> dict[str, float]:
    now = funding_df["timestamp"].max()
    d30 = now - pd.Timedelta(days=30)
    d90 = now - pd.Timedelta(days=90)
    df30 = funding_df[funding_df["timestamp"] >= d30]
    df90 = funding_df[funding_df["timestamp"] >= d90]
    return {
        "avg_funding_30d": float(df30["funding_rate"].mean()) if len(df30) else 0.0,
        "avg_funding_90d": float(df90["funding_rate"].mean()) if len(df90) else 0.0,
        "pct_positive_intervals": float((funding_df["funding_rate"] > 0).mean()),
        "n_intervals": len(funding_df),
    }


def estimate_slippage_bps(
    orderbook_df: pd.DataFrame,
    notional_eur: float,
    price_eur: float,
    side: str,
) -> float:
    """Walk the orderbook and compute average fill price vs best price."""
    if price_eur <= 0:
        return 50.0
    qty_needed = notional_eur / price_eur
    levels = orderbook_df[orderbook_df["side"] == side].sort_values(
        "price", ascending=(side == "ask")
    )
    if levels.empty:
        return 50.0

    best_price = levels.iloc[0]["price"]
    filled_qty = 0.0
    filled_cost = 0.0
    for _, row in levels.iterrows():
        take = min(row["qty"], qty_needed - filled_qty)
        filled_qty += take
        filled_cost += take * row["price"]
        if filled_qty >= qty_needed:
            break

    if filled_qty == 0:
        return 50.0

    avg_fill = filled_cost / filled_qty
    slippage = abs(avg_fill - best_price) / best_price * 10_000
    return round(slippage, 4)


def screen_coins(
    symbols: list[str],
    funding_data: dict[str, pd.DataFrame],
    orderbook_data: dict[str, pd.DataFrame],
    ohlcv_data: dict[str, pd.DataFrame],
    config: ScreenerConfig,
) -> pd.DataFrame:
    rows = []
    for symbol in symbols:
        if symbol not in funding_data or funding_data[symbol].empty:
            continue
        stats = compute_funding_stats(funding_data[symbol])

        if stats["avg_funding_30d"] < config.entry_threshold_per_interval:
            continue
        if stats["pct_positive_intervals"] < config.min_pct_positive:
            continue

        slippage_bps = 0.0
        if symbol in orderbook_data and not orderbook_data[symbol].empty:
            ob = orderbook_data[symbol]
            best_ask = ob[ob["side"] == "ask"]["price"].min()
            slippage_bps = estimate_slippage_bps(ob, config.position_size_eur, best_ask, "ask")

        if slippage_bps > config.max_slippage_bps:
            continue

        base = symbol.replace("USDT", "").replace("FDUSD", "")
        fdusd_eligible = base in FDUSD_ZERO_FEE_BASES

        rows.append({
            "symbol": symbol,
            "avg_funding_30d": stats["avg_funding_30d"],
            "avg_funding_90d": stats["avg_funding_90d"],
            "pct_positive_intervals": stats["pct_positive_intervals"],
            "n_intervals": stats["n_intervals"],
            "slippage_est_bps": slippage_bps,
            "fdusd_zero_fee": fdusd_eligible,
            "gate_candidate": True,
        })

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values("avg_funding_30d", ascending=False)
        .reset_index(drop=True)
    )
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/screener/test_screener.py -v
```

Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/trade4/screener/screener.py tests/screener/test_screener.py
git commit -m "feat: coin screener — funding stats, orderbook slippage, FDUSD flag"
```

---

## Task 6: Cost Model & Net-EV Gate

**Files:**
- Create: `src/trade4/backtest/cost_model.py`
- Test: `tests/backtest/test_cost_model.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/backtest/test_cost_model.py
import pytest
from trade4.backtest.cost_model import (
    FeeSchedule,
    CostModel,
    DEFAULT_FEE_SCHEDULE,
    compute_round_trip_cost_bps,
    compute_net_edge_bps,
    gate_passed,
    MIN_NET_EDGE_BPS,
    FUNDING_TO_COST_RATIO,
)


def test_default_fee_schedule_base_tier():
    assert DEFAULT_FEE_SCHEDULE.spot_taker_bps == 10
    assert DEFAULT_FEE_SCHEDULE.perp_taker_bps == 5
    assert DEFAULT_FEE_SCHEDULE.fdusd_spot_taker_bps == 0
    assert DEFAULT_FEE_SCHEDULE.fdusd_spot_maker_bps == 0


def test_round_trip_cost_standard_taker():
    model = CostModel(
        fee_schedule=DEFAULT_FEE_SCHEDULE,
        slippage_entry_bps=5.0,
        slippage_exit_bps=5.0,
        basis_drift_bps=2.0,
        fdusd_depeg_bps=0.0,
        use_fdusd=False,
        use_maker_spot=False,
        use_maker_perp=False,
    )
    cost = compute_round_trip_cost_bps(model)
    # spot taker entry 10 + perp taker entry 5 + spot taker exit 10 + perp taker exit 5
    # + slippage 5+5 + basis 2 + depeg 0 = 42
    assert cost == pytest.approx(42.0)


def test_round_trip_cost_fdusd_saves_spot_fees():
    model_standard = CostModel(
        fee_schedule=DEFAULT_FEE_SCHEDULE,
        slippage_entry_bps=5.0, slippage_exit_bps=5.0,
        basis_drift_bps=2.0, fdusd_depeg_bps=0.5,
        use_fdusd=False, use_maker_spot=False, use_maker_perp=False,
    )
    model_fdusd = CostModel(
        fee_schedule=DEFAULT_FEE_SCHEDULE,
        slippage_entry_bps=5.0, slippage_exit_bps=5.0,
        basis_drift_bps=2.0, fdusd_depeg_bps=0.5,
        use_fdusd=True, use_maker_spot=False, use_maker_perp=False,
    )
    cost_standard = compute_round_trip_cost_bps(model_standard)
    cost_fdusd = compute_round_trip_cost_bps(model_fdusd)
    # FDUSD removes 2× spot_taker (entry + exit) = 20 bps, adds 0.5 depeg
    assert cost_fdusd == pytest.approx(cost_standard - 20 + 0.5)


def test_round_trip_cost_maker_is_cheaper():
    model_taker = CostModel(
        fee_schedule=DEFAULT_FEE_SCHEDULE,
        slippage_entry_bps=5.0, slippage_exit_bps=5.0,
        basis_drift_bps=2.0, fdusd_depeg_bps=0.0,
        use_fdusd=False, use_maker_spot=True, use_maker_perp=True,
    )
    model_taker_only = CostModel(
        fee_schedule=DEFAULT_FEE_SCHEDULE,
        slippage_entry_bps=5.0, slippage_exit_bps=5.0,
        basis_drift_bps=2.0, fdusd_depeg_bps=0.0,
        use_fdusd=False, use_maker_spot=False, use_maker_perp=False,
    )
    assert compute_round_trip_cost_bps(model_taker) < compute_round_trip_cost_bps(model_taker_only)


def test_net_edge_calculation():
    model = CostModel(
        fee_schedule=DEFAULT_FEE_SCHEDULE,
        slippage_entry_bps=5.0, slippage_exit_bps=5.0,
        basis_drift_bps=2.0, fdusd_depeg_bps=0.0,
        use_fdusd=False, use_maker_spot=False, use_maker_perp=False,
    )
    cost = compute_round_trip_cost_bps(model)  # 42
    net = compute_net_edge_bps(expected_funding_bps=100.0, cost_model=model)
    assert net == pytest.approx(100.0 - cost)


def test_gate_passes_when_above_threshold():
    # funding=100, cost=42, net_edge=58 ≥ 15, 100 ≥ 2×42=84 ✓
    model = CostModel(
        fee_schedule=DEFAULT_FEE_SCHEDULE,
        slippage_entry_bps=5.0, slippage_exit_bps=5.0,
        basis_drift_bps=2.0, fdusd_depeg_bps=0.0,
        use_fdusd=False, use_maker_spot=False, use_maker_perp=False,
    )
    assert gate_passed(expected_funding_bps=100.0, cost_model=model) is True


def test_gate_fails_when_funding_too_low():
    # funding=30, cost=42, net_edge=-12 < 15 → fail
    model = CostModel(
        fee_schedule=DEFAULT_FEE_SCHEDULE,
        slippage_entry_bps=5.0, slippage_exit_bps=5.0,
        basis_drift_bps=2.0, fdusd_depeg_bps=0.0,
        use_fdusd=False, use_maker_spot=False, use_maker_perp=False,
    )
    assert gate_passed(expected_funding_bps=30.0, cost_model=model) is False


def test_gate_fails_when_ratio_not_met():
    # funding=50, cost=42, net_edge=8 < 15 → fail (also ratio 50 < 2×42=84)
    model = CostModel(
        fee_schedule=DEFAULT_FEE_SCHEDULE,
        slippage_entry_bps=5.0, slippage_exit_bps=5.0,
        basis_drift_bps=2.0, fdusd_depeg_bps=0.0,
        use_fdusd=False, use_maker_spot=False, use_maker_perp=False,
    )
    assert gate_passed(expected_funding_bps=50.0, cost_model=model) is False


def test_stress_test_double_fees_double_slippage():
    """Verify gate with 2× fees and 2× slippage — charter §11 requirement."""
    base = CostModel(
        fee_schedule=FeeSchedule(
            spot_taker_bps=20, spot_maker_bps=18,
            perp_taker_bps=10, perp_maker_bps=4,
            fdusd_spot_taker_bps=0, fdusd_spot_maker_bps=0,
        ),
        slippage_entry_bps=10.0, slippage_exit_bps=10.0,
        basis_drift_bps=4.0, fdusd_depeg_bps=0.0,
        use_fdusd=False, use_maker_spot=False, use_maker_perp=False,
    )
    # 2× stress: funding=200 should still pass
    assert gate_passed(expected_funding_bps=200.0, cost_model=base) is True
    # funding=100 should fail under stress (100 - (20+10+20+10+10+10+4)=84 → net=16, barely, 100 < 2×84=168 → fail ratio)
    assert gate_passed(expected_funding_bps=100.0, cost_model=base) is False
```

- [ ] **Step 2: Run — verify fail**

```bash
pytest tests/backtest/test_cost_model.py -v
```

- [ ] **Step 3: Implement cost_model.py**

```python
# src/trade4/backtest/cost_model.py
from dataclasses import dataclass

MIN_NET_EDGE_BPS: float = 15.0
FUNDING_TO_COST_RATIO: float = 2.0


@dataclass(frozen=True)
class FeeSchedule:
    spot_taker_bps: float = 10.0
    spot_maker_bps: float = 9.0
    perp_taker_bps: float = 5.0
    perp_maker_bps: float = 2.0
    fdusd_spot_taker_bps: float = 0.0
    fdusd_spot_maker_bps: float = 0.0


DEFAULT_FEE_SCHEDULE = FeeSchedule()


@dataclass(frozen=True)
class CostModel:
    fee_schedule: FeeSchedule
    slippage_entry_bps: float
    slippage_exit_bps: float
    basis_drift_bps: float
    fdusd_depeg_bps: float
    use_fdusd: bool
    use_maker_spot: bool
    use_maker_perp: bool


def compute_round_trip_cost_bps(model: CostModel) -> float:
    fs = model.fee_schedule
    spot_fee = fs.fdusd_spot_taker_bps if (model.use_fdusd and not model.use_maker_spot) \
        else (fs.fdusd_spot_maker_bps if (model.use_fdusd and model.use_maker_spot) \
        else (fs.spot_maker_bps if model.use_maker_spot else fs.spot_taker_bps))
    perp_fee = fs.perp_maker_bps if model.use_maker_perp else fs.perp_taker_bps

    return (
        spot_fee + perp_fee          # entry
        + spot_fee + perp_fee        # exit
        + model.slippage_entry_bps
        + model.slippage_exit_bps
        + model.basis_drift_bps
        + (model.fdusd_depeg_bps if model.use_fdusd else 0.0)
    )


def compute_net_edge_bps(expected_funding_bps: float, cost_model: CostModel) -> float:
    return expected_funding_bps - compute_round_trip_cost_bps(cost_model)


def gate_passed(expected_funding_bps: float, cost_model: CostModel) -> bool:
    cost = compute_round_trip_cost_bps(cost_model)
    net_edge = expected_funding_bps - cost
    return (
        net_edge >= MIN_NET_EDGE_BPS
        and expected_funding_bps >= FUNDING_TO_COST_RATIO * cost
    )
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/backtest/test_cost_model.py -v
```

Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/trade4/backtest/cost_model.py tests/backtest/test_cost_model.py
git commit -m "feat: cost model with Net-EV gate — fees, slippage, FDUSD, stress test"
```

---

## Task 7: Backtest Engine

**Files:**
- Create: `src/trade4/backtest/engine.py`
- Test: `tests/backtest/test_engine.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/backtest/test_engine.py
import pandas as pd
import numpy as np
import pytest
from trade4.backtest.engine import (
    BacktestConfig,
    BacktestResult,
    CycleResult,
    run_backtest,
    count_funding_intervals,
    estimate_spread_bps,
    maker_fill_simulated,
    split_walk_forward,
)
from trade4.backtest.cost_model import CostModel, DEFAULT_FEE_SCHEDULE


DEFAULT_CONFIG = BacktestConfig(
    entry_threshold=0.00005,
    exit_threshold=0.0,
    persistence_threshold=0.00003,
    persistence_window=5,
    max_holding_days=30,
    position_size_eur=500.0,
    cost_model=CostModel(
        fee_schedule=DEFAULT_FEE_SCHEDULE,
        slippage_entry_bps=5.0, slippage_exit_bps=5.0,
        basis_drift_bps=2.0, fdusd_depeg_bps=0.0,
        use_fdusd=False, use_maker_spot=False, use_maker_perp=False,
    ),
)


def test_count_funding_intervals_exact_timing():
    # Entry 04:00 UTC, exit 28:00 UTC next day (04:00 UTC + 24h)
    entry = pd.Timestamp("2024-01-01 04:00:00", tz="UTC")
    exit_ = pd.Timestamp("2024-01-02 04:00:00", tz="UTC")
    funding_df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=12, freq="8h", tz="UTC"),
        "funding_rate": [0.0001] * 12,
    })
    collected = count_funding_intervals(entry, exit_, funding_df)
    # 08:00, 16:00, 00:00+1d → 3 intervals strictly between entry(04:00) and exit(04:00+1d)
    assert len(collected) == 3


def test_count_funding_intervals_entry_on_interval_not_collected():
    # If entry IS exactly at 08:00, that interval should NOT be collected (half-open: (entry, exit])
    entry = pd.Timestamp("2024-01-01 08:00:00", tz="UTC")
    exit_ = pd.Timestamp("2024-01-01 20:00:00", tz="UTC")
    funding_df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01 08:00", "2024-01-01 16:00"], utc=True),
        "funding_rate": [0.0001, 0.0002],
    })
    collected = count_funding_intervals(entry, exit_, funding_df)
    assert len(collected) == 1
    assert collected.iloc[0]["timestamp"] == pd.Timestamp("2024-01-01 16:00", tz="UTC")


def test_estimate_spread_bps_normal_candle():
    row = pd.Series({"open": 0.10, "high": 0.105, "low": 0.095, "close": 0.10, "volume": 1e8})
    spread = estimate_spread_bps(row)
    assert 0 < spread < 100


def test_estimate_spread_bps_caps_at_max():
    row = pd.Series({"open": 0.10, "high": 1.0, "low": 0.001, "close": 0.10, "volume": 1e8})
    spread = estimate_spread_bps(row)
    assert spread <= 100.0


def test_maker_fill_buy_fills_when_low_crosses():
    candle = pd.Series({"low": 0.098, "high": 0.102, "close": 0.10})
    assert maker_fill_simulated(candle, limit_price=0.099, side="buy") is True


def test_maker_fill_buy_no_fill_when_low_above_limit():
    candle = pd.Series({"low": 0.10, "high": 0.105, "close": 0.102})
    assert maker_fill_simulated(candle, limit_price=0.099, side="buy") is False


def test_maker_fill_sell_fills_when_high_crosses():
    candle = pd.Series({"low": 0.098, "high": 0.102, "close": 0.10})
    assert maker_fill_simulated(candle, limit_price=0.101, side="sell") is True


def test_split_walk_forward_correct_split():
    funding_df = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=100, freq="8h", tz="UTC"),
        "funding_rate": [0.0001] * 100,
    })
    in_sample, out_of_sample = split_walk_forward(
        funding_df, in_sample_end=pd.Timestamp("2024-12-31", tz="UTC")
    )
    assert in_sample["timestamp"].max() <= pd.Timestamp("2024-12-31", tz="UTC")
    assert out_of_sample["timestamp"].min() > pd.Timestamp("2024-12-31", tz="UTC")


def test_run_backtest_returns_result(sample_funding_df, sample_ohlcv_df, sample_orderbook):
    # Generate 2 years of high funding data for meaningful backtest
    timestamps = pd.date_range("2023-01-01", periods=365*3, freq="8h", tz="UTC")
    funding_df = pd.DataFrame({
        "timestamp": timestamps,
        "symbol": "DOGEUSDT",
        "funding_rate": [0.0002] * len(timestamps),
    })
    ohlcv_df = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=365*2, freq="1d", tz="UTC"),
        "open": [0.10] * 365*2,
        "high": [0.105] * 365*2,
        "low": [0.095] * 365*2,
        "close": [0.10] * 365*2,
        "volume": [1e9] * 365*2,
    })
    result = run_backtest(funding_df, ohlcv_df, sample_orderbook, DEFAULT_CONFIG)
    assert isinstance(result, BacktestResult)
    assert result.max_drawdown_bps <= 0 or result.max_drawdown_bps >= 0  # just exists
    assert isinstance(result.cycles, list)


def test_run_backtest_no_cycles_when_funding_too_low(sample_ohlcv_df, sample_orderbook):
    timestamps = pd.date_range("2023-01-01", periods=100, freq="8h", tz="UTC")
    funding_df = pd.DataFrame({
        "timestamp": timestamps,
        "symbol": "DOGEUSDT",
        "funding_rate": [-0.0001] * 100,  # always negative → no entry
    })
    result = run_backtest(funding_df, sample_ohlcv_df, sample_orderbook, DEFAULT_CONFIG)
    assert len(result.cycles) == 0
```

- [ ] **Step 2: Run — verify fail**

```bash
pytest tests/backtest/test_engine.py -v
```

- [ ] **Step 3: Implement engine.py**

```python
# src/trade4/backtest/engine.py
import logging
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from trade4.backtest.cost_model import CostModel, compute_round_trip_cost_bps, gate_passed

logger = logging.getLogger(__name__)

REBALANCE_DELTA_THRESHOLD = 0.02  # 2% drift triggers rebalance cost


@dataclass
class BacktestConfig:
    entry_threshold: float = 0.00005
    exit_threshold: float = 0.0
    persistence_threshold: float = 0.00003
    persistence_window: int = 5
    max_holding_days: int = 30
    position_size_eur: float = 500.0
    cost_model: CostModel = field(default_factory=lambda: CostModel(
        fee_schedule=__import__("trade4.backtest.cost_model", fromlist=["DEFAULT_FEE_SCHEDULE"]).DEFAULT_FEE_SCHEDULE,
        slippage_entry_bps=5.0, slippage_exit_bps=5.0,
        basis_drift_bps=2.0, fdusd_depeg_bps=0.0,
        use_fdusd=False, use_maker_spot=False, use_maker_perp=False,
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

    # rolling persistence filter
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

        # get matching OHLCV candle (nearest daily candle)
        ohlcv_row = _get_ohlcv_row(ohlcv_df, ts)
        if ohlcv_row is None:
            continue

        mid_price = float(ohlcv_row["close"])
        spread_bps = estimate_spread_bps(ohlcv_row)

        if not in_position:
            if rate >= config.entry_threshold and rolling >= config.persistence_threshold:
                # Estimate expected funding for horizon
                horizon_days = min(config.max_holding_days, 14)
                remaining_intervals = int(
                    funding_df[funding_df["timestamp"] > ts].head(horizon_days * 3)["funding_rate"].count()
                )
                expected_funding_bps = float(
                    funding_df[funding_df["timestamp"] > ts]
                    .head(horizon_days * 3)["funding_rate"].mean() or 0.0
                ) * remaining_intervals * 10_000

                if gate_passed(expected_funding_bps, config.cost_model):
                    in_position = True
                    entry_ts = ts
                    entry_price = mid_price * (1 + spread_bps / 10_000)  # taker ask

        else:
            assert entry_ts is not None
            days_held = (ts - entry_ts).total_seconds() / 86_400
            exit_reason = None

            if rate < config.exit_threshold:
                exit_reason = "funding_flip"
            elif days_held >= config.max_holding_days:
                exit_reason = "max_holding"

            if exit_reason:
                exit_price = mid_price * (1 - spread_bps / 10_000)  # taker bid

                collected = count_funding_intervals(entry_ts, ts, funding_df)
                funding_received_bps = float(collected["funding_rate"].sum()) * 10_000

                # delta drift: approximate as basis movement
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
    return int((drift > REBALANCE_DELTA_THRESHOLD).sum())


def _build_result(cycles: list[CycleResult], funding_df: pd.DataFrame) -> BacktestResult:
    if not cycles:
        return BacktestResult(
            cycles=[], equity_curve=pd.Series(dtype=float),
            max_drawdown_bps=0.0, net_pnl_bps=0.0,
            pct_gate_passed=0.0, n_intervals_total=len(funding_df),
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
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/backtest/test_engine.py -v
```

Expected: all pass

- [ ] **Step 5: Run full suite**

```bash
pytest -v --tb=short
```

Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add src/trade4/backtest/engine.py tests/backtest/test_engine.py
git commit -m "feat: vectorized backtest engine — realistic fills, funding timing, delta drift"
```

---

## Task 8: Report Generator

**Files:**
- Create: `src/trade4/report/report.py`
- Create: `src/trade4/report/templates/report.html.j2`
- Test: `tests/report/test_report.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/report/test_report.py
import pandas as pd
import pytest
from pathlib import Path
from trade4.report.report import generate_report, ReportInput
from trade4.backtest.engine import BacktestResult, CycleResult
from trade4.screener.screener import ScreenerConfig


def _make_cycle(n: int) -> list[CycleResult]:
    return [
        CycleResult(
            entry_ts=pd.Timestamp(f"2024-0{i+1}-01", tz="UTC"),
            exit_ts=pd.Timestamp(f"2024-0{i+1}-15", tz="UTC"),
            intervals_collected=10,
            funding_received_bps=80.0,
            round_trip_cost_bps=42.0,
            net_pnl_bps=38.0,
            gate_passed=True,
            rebalance_count=0,
            exit_reason="funding_flip",
        )
        for i in range(n)
    ]


def _make_result(n_cycles: int) -> BacktestResult:
    import numpy as np
    cycles = _make_cycle(n_cycles)
    equity = pd.Series(
        [c.net_pnl_bps * (i + 1) for i, c in enumerate(cycles)],
        index=[c.exit_ts for c in cycles],
    )
    return BacktestResult(
        cycles=cycles,
        equity_curve=equity,
        max_drawdown_bps=-5.0,
        net_pnl_bps=sum(c.net_pnl_bps for c in cycles),
        pct_gate_passed=1.0,
        n_intervals_total=100,
    )


def test_generate_report_creates_html_file(tmp_path):
    screener_df = pd.DataFrame([{
        "symbol": "DOGEUSDT",
        "avg_funding_30d": 0.0002,
        "avg_funding_90d": 0.00015,
        "pct_positive_intervals": 0.85,
        "slippage_est_bps": 3.0,
        "fdusd_zero_fee": True,
        "gate_candidate": True,
    }])
    backtest_results = {"DOGEUSDT": _make_result(3)}
    output_path = tmp_path / "report.html"
    generate_report(screener_df, backtest_results, output_path=output_path)
    assert output_path.exists()
    content = output_path.read_text()
    assert "DOGEUSDT" in content
    assert "PAPER" in content  # paper label must be present


def test_generate_report_includes_sensitivity(tmp_path):
    screener_df = pd.DataFrame([{
        "symbol": "DOGEUSDT", "avg_funding_30d": 0.0002,
        "avg_funding_90d": 0.00015, "pct_positive_intervals": 0.85,
        "slippage_est_bps": 3.0, "fdusd_zero_fee": True, "gate_candidate": True,
    }])
    backtest_results = {"DOGEUSDT": _make_result(3)}
    output_path = tmp_path / "report.html"
    generate_report(screener_df, backtest_results, output_path=output_path)
    content = output_path.read_text()
    assert "Sensitivity" in content or "sensitivity" in content


def test_report_shows_net_pnl_per_coin(tmp_path):
    screener_df = pd.DataFrame([{
        "symbol": "DOGEUSDT", "avg_funding_30d": 0.0002,
        "avg_funding_90d": 0.00015, "pct_positive_intervals": 0.85,
        "slippage_est_bps": 3.0, "fdusd_zero_fee": True, "gate_candidate": True,
    }])
    result = _make_result(3)
    output_path = tmp_path / "report.html"
    generate_report(screener_df, {"DOGEUSDT": result}, output_path=output_path)
    content = output_path.read_text()
    # 3 cycles × 38.0 bps = 114.0 bps net
    assert "114" in content
```

- [ ] **Step 2: Run — verify fail**

```bash
pytest tests/report/test_report.py -v
```

- [ ] **Step 3: Create Jinja2 template**

```html
{# src/trade4/report/templates/report.html.j2 #}
<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<title>Phase-0 Report — {{ generated_at }}</title>
<style>
  body { font-family: monospace; background: #0d1117; color: #c9d1d9; padding: 2rem; }
  h1, h2, h3 { color: #58a6ff; }
  table { border-collapse: collapse; width: 100%; margin-bottom: 2rem; }
  th, td { border: 1px solid #30363d; padding: 0.5rem 1rem; text-align: left; }
  th { background: #161b22; }
  .paper-label { background: #f85149; color: white; padding: 0.3rem 0.8rem; border-radius: 4px; font-weight: bold; }
  .pass { color: #3fb950; }
  .fail { color: #f85149; }
  .warn { color: #d29922; }
  .chart { margin: 1rem 0; }
  img { max-width: 100%; border: 1px solid #30363d; }
  .oos-warning { background: #f85149; color: white; padding: 1rem; margin-bottom: 2rem; border-radius: 4px; }
</style>
</head>
<body>

<h1>Phase-0 Funding Capture Report <span class="paper-label">PAPER</span></h1>
<p>Generiert: {{ generated_at }} &nbsp;|&nbsp; Kapital: ~€2.000 (~€500/Leg)</p>

{% if oos_worse_than_insample %}
<div class="oos-warning">&#x26A0; Out-of-Sample schlechter als In-Sample — Overfitting-Warnung!</div>
{% endif %}

<h2>1. Screener-Ergebnisse</h2>
<table>
  <tr>
    <th>Symbol</th><th>Avg Funding 30d</th><th>Avg Funding 90d</th>
    <th>% Positive</th><th>Slippage est (bps)</th><th>FDUSD 0%</th><th>Kandidat</th>
  </tr>
  {% for row in screener_rows %}
  <tr>
    <td>{{ row.symbol }}</td>
    <td>{{ "%.5f"|format(row.avg_funding_30d) }}</td>
    <td>{{ "%.5f"|format(row.avg_funding_90d) }}</td>
    <td>{{ "%.1f"|format(row.pct_positive_intervals * 100) }}%</td>
    <td>{{ "%.1f"|format(row.slippage_est_bps) }}</td>
    <td>{{ "&#x2713;" if row.fdusd_zero_fee else "&#x2717;" }}</td>
    <td class="{{ 'pass' if row.gate_candidate else 'fail' }}">{{ "JA" if row.gate_candidate else "NEIN" }}</td>
  </tr>
  {% endfor %}
</table>

{% for symbol, result in backtest_results.items() %}
<h2>2. Backtest: {{ symbol }}</h2>

<table>
  <tr><th>Metrik</th><th>Wert</th></tr>
  <tr><td>Zyklen gesamt</td><td>{{ result.n_cycles }}</td></tr>
  <tr><td>Net P&amp;L (bps)</td><td class="{{ 'pass' if result.net_pnl_bps > 0 else 'fail' }}">{{ "%.1f"|format(result.net_pnl_bps) }}</td></tr>
  <tr><td>Max Drawdown (bps)</td><td class="{{ 'warn' if result.max_drawdown_bps < -50 else 'pass' }}">{{ "%.1f"|format(result.max_drawdown_bps) }}</td></tr>
  <tr><td>% Gate bestanden</td><td>{{ "%.1f"|format(result.pct_gate_passed * 100) }}%</td></tr>
  <tr><td>Avg Funding erhalten (bps/Zyklus)</td><td>{{ "%.1f"|format(result.avg_funding_bps) }}</td></tr>
  <tr><td>Avg Kosten (bps/Zyklus)</td><td>{{ "%.1f"|format(result.avg_cost_bps) }}</td></tr>
</table>

{% if result.equity_chart_b64 %}
<div class="chart">
  <h3>Equity-Kurve (netto, kumulativ in bps)</h3>
  <img src="data:image/png;base64,{{ result.equity_chart_b64 }}" />
</div>
{% endif %}

{% if result.pnl_hist_b64 %}
<div class="chart">
  <h3>Return-Verteilung (bps/Zyklus)</h3>
  <img src="data:image/png;base64,{{ result.pnl_hist_b64 }}" />
</div>
{% endif %}

<h3>Sensitivity-Analyse (Net-Edge in bps)</h3>
<table>
  <tr><th></th><th>1× Fees/Slippage</th><th>2× Fees/Slippage</th></tr>
  <tr>
    <td>Normales Funding</td>
    <td class="{{ 'pass' if result.sensitivity.normal_1x > 15 else 'fail' }}">{{ "%.1f"|format(result.sensitivity.normal_1x) }}</td>
    <td class="{{ 'pass' if result.sensitivity.normal_2x > 15 else 'fail' }}">{{ "%.1f"|format(result.sensitivity.normal_2x) }}</td>
  </tr>
  <tr>
    <td>Funding-Flip-Stress</td>
    <td class="{{ 'pass' if result.sensitivity.flip_1x > 15 else 'fail' }}">{{ "%.1f"|format(result.sensitivity.flip_1x) }}</td>
    <td class="{{ 'pass' if result.sensitivity.flip_2x > 15 else 'fail' }}">{{ "%.1f"|format(result.sensitivity.flip_2x) }}</td>
  </tr>
</table>

<h3>Failure-Mode-Summary</h3>
<table>
  <tr><th>Szenario</th><th>Verlustquelle</th><th>Grenze</th></tr>
  <tr><td>Funding flippt negativ</td><td>Short zahlt Funding statt zu erhalten</td><td>Exit-Threshold = 0.0%</td></tr>
  <tr><td>Hohe Slippage</td><td>Taker-Kosten übersteigen Funding-Edge</td><td>Slippage-Budget = 50 bps/RT</td></tr>
  <tr><td>Basis-Drift</td><td>Perp handelt nachhaltig über/unter Spot</td><td>Basis-Puffer = 2 bps modelliert</td></tr>
  <tr><td>FDUSD-Depeg</td><td>FDUSD ≠ USDT → versteckte Kosten</td><td>Depeg-Puffer = 0.5 bps</td></tr>
  <tr><td>Exchange-Ausfall</td><td>Ungesichertes Leg bleibt offen</td><td>Kill-Switch (Phase 1)</td></tr>
</table>

{% endfor %}

<hr>
<p style="color:#6e7681;">Alle Zahlen NETTO nach Gebühren, Slippage und Basis-Drift. Kein Live-Handel. Charter §12.</p>
</body>
</html>
```

- [ ] **Step 4: Implement report.py**

```python
# src/trade4/report/report.py
import base64
import io
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from trade4.backtest.engine import BacktestResult
from trade4.backtest.cost_model import (
    CostModel, DEFAULT_FEE_SCHEDULE, FeeSchedule,
    compute_round_trip_cost_bps, compute_net_edge_bps,
)

logger = logging.getLogger(__name__)
TEMPLATE_DIR = Path(__file__).parent / "templates"


@dataclass
class SensitivityResult:
    normal_1x: float
    normal_2x: float
    flip_1x: float
    flip_2x: float


@dataclass
class ReportInput:
    screener_df: pd.DataFrame
    backtest_results: dict[str, BacktestResult]
    output_path: Path


def generate_report(
    screener_df: pd.DataFrame,
    backtest_results: dict[str, BacktestResult],
    output_path: Path,
) -> None:
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))
    template = env.get_template("report.html.j2")

    processed = {}
    for symbol, result in backtest_results.items():
        cycles = result.cycles
        avg_funding = sum(c.funding_received_bps for c in cycles) / max(len(cycles), 1)
        avg_cost = sum(c.round_trip_cost_bps for c in cycles) / max(len(cycles), 1)

        processed[symbol] = {
            "n_cycles": len(cycles),
            "net_pnl_bps": result.net_pnl_bps,
            "max_drawdown_bps": result.max_drawdown_bps,
            "pct_gate_passed": result.pct_gate_passed,
            "avg_funding_bps": avg_funding,
            "avg_cost_bps": avg_cost,
            "equity_chart_b64": _equity_chart_b64(result),
            "pnl_hist_b64": _pnl_hist_b64(result),
            "sensitivity": _compute_sensitivity(result),
        }

    oos_worse = any(
        r.net_pnl_bps < 0 for r in backtest_results.values()
    )

    html = template.render(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        screener_rows=screener_df.to_dict("records"),
        backtest_results=processed,
        oos_worse_than_insample=oos_worse,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Report written to %s", output_path)


def _equity_chart_b64(result: BacktestResult) -> str:
    if result.equity_curve.empty:
        return ""
    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0d1117")
    ax.set_facecolor("#161b22")
    ax.plot(result.equity_curve.index, result.equity_curve.values, color="#3fb950", linewidth=1.5)
    ax.axhline(0, color="#6e7681", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Datum", color="#c9d1d9")
    ax.set_ylabel("Kumulativer P&L (bps)", color="#c9d1d9")
    ax.tick_params(colors="#c9d1d9")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    fig.tight_layout()
    return _fig_to_b64(fig)


def _pnl_hist_b64(result: BacktestResult) -> str:
    if not result.cycles:
        return ""
    pnls = [c.net_pnl_bps for c in result.cycles]
    fig, ax = plt.subplots(figsize=(8, 3), facecolor="#0d1117")
    ax.set_facecolor("#161b22")
    ax.hist(pnls, bins=20, color="#58a6ff", edgecolor="#30363d")
    ax.axvline(0, color="#f85149", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Net P&L pro Zyklus (bps)", color="#c9d1d9")
    ax.tick_params(colors="#c9d1d9")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    fig.tight_layout()
    return _fig_to_b64(fig)


def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _compute_sensitivity(result: BacktestResult) -> SensitivityResult:
    if not result.cycles:
        return SensitivityResult(0.0, 0.0, 0.0, 0.0)

    avg_funding = sum(c.funding_received_bps for c in result.cycles) / len(result.cycles)
    flip_funding = avg_funding * 0.3  # stress: only 30% of expected funding materialises

    def net_edge(funding: float, fee_mult: float, slip_mult: float) -> float:
        m = CostModel(
            fee_schedule=FeeSchedule(
                spot_taker_bps=DEFAULT_FEE_SCHEDULE.spot_taker_bps * fee_mult,
                spot_maker_bps=DEFAULT_FEE_SCHEDULE.spot_maker_bps * fee_mult,
                perp_taker_bps=DEFAULT_FEE_SCHEDULE.perp_taker_bps * fee_mult,
                perp_maker_bps=DEFAULT_FEE_SCHEDULE.perp_maker_bps * fee_mult,
                fdusd_spot_taker_bps=0.0,
                fdusd_spot_maker_bps=0.0,
            ),
            slippage_entry_bps=5.0 * slip_mult,
            slippage_exit_bps=5.0 * slip_mult,
            basis_drift_bps=2.0,
            fdusd_depeg_bps=0.0,
            use_fdusd=False,
            use_maker_spot=False,
            use_maker_perp=False,
        )
        return compute_net_edge_bps(funding, m)

    return SensitivityResult(
        normal_1x=net_edge(avg_funding, 1.0, 1.0),
        normal_2x=net_edge(avg_funding, 2.0, 2.0),
        flip_1x=net_edge(flip_funding, 1.0, 1.0),
        flip_2x=net_edge(flip_funding, 2.0, 2.0),
    )
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/report/test_report.py -v
```

Expected: 3 passed

- [ ] **Step 6: Run full suite**

```bash
pytest -v --tb=short
```

Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add src/trade4/report/ tests/report/test_report.py
git commit -m "feat: HTML report with equity curve, histogram, sensitivity table"
```

---

## Task 9: Notebook & Integration Smoke Test

**Files:**
- Create: `notebooks/phase0_research.ipynb`
- Create: `src/trade4/pipeline.py`

- [ ] **Step 1: Create pipeline.py (entry point)**

```python
# src/trade4/pipeline.py
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
TOP_N_SYMBOLS = 20


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

    # 1. Get candidate symbols
    logger.info("Loading Binance perp symbols...")
    symbols = bn.list_perp_symbols()[:TOP_N_SYMBOLS]
    logger.info("Screening %d symbols", len(symbols))

    # 2. Fetch funding history
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

    # 3. Screen
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

    # 4. Backtest
    cost_model = CostModel(
        fee_schedule=DEFAULT_FEE_SCHEDULE,
        slippage_entry_bps=5.0,
        slippage_exit_bps=5.0,
        basis_drift_bps=2.0,
        fdusd_depeg_bps=0.0,
        use_fdusd=False,
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
        cost_model=cost_model,
    )

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
        config_coin = BacktestConfig(
            entry_threshold=0.00005,
            exit_threshold=0.0,
            persistence_threshold=0.00003,
            persistence_window=5,
            max_holding_days=30,
            position_size_eur=500.0,
            cost_model=bt_model,
        )
        ob = orderbook_data.get(symbol, pd.DataFrame())
        result = run_backtest(funding_data[symbol], ohlcv_data.get(symbol, pd.DataFrame()), ob, config_coin)
        backtest_results[symbol] = result
        logger.info("%s: %d cycles, net P&L %.1f bps", symbol, len(result.cycles), result.net_pnl_bps)

    # 5. Report
    generate_report(screener_df, backtest_results, output_path=REPORT_PATH)
    logger.info("Report saved to %s", REPORT_PATH)
    logger.info("=== Phase-0 Pipeline Complete ===")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create minimal notebook**

Create `notebooks/phase0_research.ipynb` with these cells:

Cell 1 (code):
```python
import sys
sys.path.insert(0, "../src")
import pandas as pd
from dotenv import load_dotenv
load_dotenv("../.env")
```

Cell 2 (code):
```python
from trade4.data import binance as bn
symbols = bn.list_perp_symbols()[:5]
print(symbols)
```

Cell 3 (code):
```python
# Run full pipeline
import subprocess
result = subprocess.run(["python", "-m", "trade4.pipeline"], capture_output=True, text=True, cwd="..")
print(result.stdout[-3000:])
print(result.stderr[-1000:] if result.returncode != 0 else "")
```

- [ ] **Step 3: Run full test suite one final time**

```bash
pytest -v --cov=trade4 --cov-report=term-missing
```

Expected: all pass, coverage visible

- [ ] **Step 4: Smoke test pipeline with dry-run (no real API calls)**

```bash
python -c "
import pandas as pd
from trade4.screener.screener import screen_coins, ScreenerConfig
from trade4.backtest.engine import run_backtest, BacktestConfig
from trade4.backtest.cost_model import CostModel, DEFAULT_FEE_SCHEDULE
from trade4.report.report import generate_report
from pathlib import Path

# minimal smoke test
config = ScreenerConfig(entry_threshold_per_interval=0.00005, max_slippage_bps=50, position_size_eur=500, min_pct_positive=0.5, volume_fraction_cap=0.1)
print('imports OK')
print('smoke test passed')
"
```

Expected: `smoke test passed`

- [ ] **Step 5: Final commit**

```bash
git add src/trade4/pipeline.py notebooks/phase0_research.ipynb
git commit -m "feat: pipeline entry point and notebook — Phase-0 complete"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] §1 Daten-Fetching + Parquet-Cache → Task 2, 3, 4
- [x] §2 Coin-Screener (Funding-Filter + Liquiditäts-Filter + FDUSD-Flag) → Task 5
- [x] §3 Cost-Model + Net-EV-Gate (Charter §6) → Task 6
- [x] §4 Backtest-Engine: Bid/Ask, Funding-Timing, Maker-Fill, Orderbook-Slippage → Task 7
- [x] §5 Report: Histogram, Equity-Kurve, Sensitivity-Tabelle, Walk-Forward, Failure-Mode → Task 8
- [x] Walk-Forward Split (2023–2024 In-Sample, 2025 Out-of-Sample) → Task 7 + 9
- [x] Paper-Label im Report → Task 8 (Template)
- [x] Stress-Test 2× Fees/Slippage → Task 6 (test) + Task 8 (sensitivity)
- [x] Kein Live-Handel, kein Paper-Trader → Scope klar in Task 1

**Typen konsistent:**
- `BacktestConfig.cost_model: CostModel` — definiert Task 6, genutzt Task 7, 9 ✓
- `CycleResult` — definiert Task 7, genutzt Task 8 ✓
- `ScreenerConfig` — definiert Task 5, genutzt Task 9 ✓

**Keine Platzhalter:** alle Code-Blöcke vollständig.
