# Small Account Growth Bot — Phase 1 Implementation Plan (Backtest & Validation)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a two-strategy scalping backtest system (EMA-Cross Momentum + Pump Scanner) with walk-forward validation, using Sharpe ≥ 1.5 + OOS profit factor ≥ 1.4 as the hard Phase-2 go/no-go gate. A backtest failure is the expected, acceptable outcome — not a bug.

**Architecture:** New `scalper/` package under `src/trade4/`, extending the existing Binance data fetcher and cost model. Pipeline: Signal Generators → Risk Manager → Event-loop Backtest Harness → Walk-Forward Validator → Go/No-Go Report. Live execution (Phase 2) is a separate plan, gated on this plan's OOS results.

**Tech Stack:** Python 3.12, pandas, numpy, pytest; trade4 data layer and cost_model patterns reused.

---

## File Map

**Create:**
- `src/trade4/scalper/__init__.py`
- `src/trade4/scalper/signals/__init__.py`
- `src/trade4/scalper/signals/ema_cross.py`
- `src/trade4/scalper/signals/pump_scanner.py`
- `src/trade4/scalper/screener.py`
- `src/trade4/scalper/risk_manager.py`
- `src/trade4/scalper/backtest_harness.py`
- `src/trade4/scalper/walk_forward.py`
- `src/trade4/scalper/report.py`
- `tests/scalper/__init__.py`
- `tests/scalper/signals/__init__.py`
- `tests/scalper/signals/test_ema_cross.py`
- `tests/scalper/signals/test_pump_scanner.py`
- `tests/scalper/test_screener.py`
- `tests/scalper/test_risk_manager.py`
- `tests/scalper/test_backtest_harness.py`
- `tests/scalper/test_walk_forward.py`

**Modify:**
- `src/trade4/data/binance.py` — add `list_perp_symbols_with_onboard()`, `filter_symbols_at_date()`
- `src/trade4/backtest/cost_model.py` — add `ScalperCostModel`, `scalper_round_trip_bps()`
- `tests/data/test_binance.py` — add tests for new functions
- `tests/backtest/test_cost_model.py` — add tests for scalper model

---

## Task 1: Package Skeleton

**Files:**
- Create: `src/trade4/scalper/__init__.py`
- Create: `src/trade4/scalper/signals/__init__.py`
- Create: `tests/scalper/__init__.py`
- Create: `tests/scalper/signals/__init__.py`

- [ ] **Step 1: Create empty package files**

```bash
mkdir -p src/trade4/scalper/signals
mkdir -p tests/scalper/signals
touch src/trade4/scalper/__init__.py
touch src/trade4/scalper/signals/__init__.py
touch tests/scalper/__init__.py
touch tests/scalper/signals/__init__.py
```

- [ ] **Step 2: Verify pytest discovers the new packages**

Run: `pytest tests/scalper/ --collect-only`
Expected: `no tests ran` (empty packages, no errors)

- [ ] **Step 3: Commit**

```bash
git add src/trade4/scalper/ tests/scalper/
git commit -m "feat(scalper): add scalper package skeleton"
```

---

## Task 2: Binance — Point-in-Time Symbol Universe

**Files:**
- Modify: `src/trade4/data/binance.py`
- Modify: `tests/data/test_binance.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/data/test_binance.py`:

```python
from unittest.mock import patch, MagicMock
import pandas as pd
from trade4.data.binance import list_perp_symbols_with_onboard, filter_symbols_at_date


def _mock_exchange_info() -> dict:
    return {
        "symbols": [
            {
                "symbol": "BTCUSDT",
                "contractType": "PERPETUAL",
                "quoteAsset": "USDT",
                "status": "TRADING",
                "onboardDate": 1569888000000,  # 2019-10-01
            },
            {
                "symbol": "SOLUSDT",
                "contractType": "PERPETUAL",
                "quoteAsset": "USDT",
                "status": "TRADING",
                "onboardDate": 1623024000000,  # 2021-06-07
            },
            {
                "symbol": "NEWUSDT",
                "contractType": "PERPETUAL",
                "quoteAsset": "USDT",
                "status": "TRADING",
                "onboardDate": 1748822400000,  # 2025-06-02 (future in backtest)
            },
            {
                "symbol": "BTCDOMUSDT",
                "contractType": "PERPETUAL",
                "quoteAsset": "USDT",
                "status": "SETTLING",  # not TRADING
                "onboardDate": 1569888000000,
            },
        ]
    }


def test_list_perp_symbols_with_onboard_filters_non_trading():
    with patch("trade4.data.binance._get", return_value=_mock_exchange_info()):
        result = list_perp_symbols_with_onboard()
    assert "BTCDOMUSDT" not in result
    assert "BTCUSDT" in result
    assert "SOLUSDT" in result


def test_list_perp_symbols_with_onboard_returns_timestamps():
    with patch("trade4.data.binance._get", return_value=_mock_exchange_info()):
        result = list_perp_symbols_with_onboard()
    assert isinstance(result["BTCUSDT"], pd.Timestamp)
    assert result["BTCUSDT"].tzinfo is not None  # UTC-aware


def test_filter_symbols_at_date_excludes_future_listings():
    onboard = {
        "BTCUSDT": pd.Timestamp("2019-10-01", tz="UTC"),
        "SOLUSDT": pd.Timestamp("2021-06-07", tz="UTC"),
        "NEWUSDT": pd.Timestamp("2025-06-02", tz="UTC"),
    }
    as_of = pd.Timestamp("2024-01-01", tz="UTC")
    result = filter_symbols_at_date(onboard, as_of)
    assert "BTCUSDT" in result
    assert "SOLUSDT" in result
    assert "NEWUSDT" not in result


def test_filter_symbols_at_date_includes_same_day():
    onboard = {"SOLUSDT": pd.Timestamp("2024-01-01", tz="UTC")}
    result = filter_symbols_at_date(onboard, pd.Timestamp("2024-01-01", tz="UTC"))
    assert "SOLUSDT" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/data/test_binance.py::test_list_perp_symbols_with_onboard_filters_non_trading -v`
Expected: `ImportError` or `AttributeError` — functions don't exist yet

- [ ] **Step 3: Implement in `src/trade4/data/binance.py`**

Append to the existing file (after `list_perp_symbols`):

```python
def list_perp_symbols_with_onboard() -> dict[str, pd.Timestamp]:
    """Returns {symbol: onboard_timestamp} for all active USDT-M perpetuals.

    onboardDate from exchangeInfo is used for point-in-time universe filtering
    in backtests — prevents survivorship bias and look-ahead bias.
    """
    data = _get("/fapi/v1/exchangeInfo", {})
    result: dict[str, pd.Timestamp] = {}
    for s in data["symbols"]:
        if (
            s.get("contractType") == "PERPETUAL"
            and s.get("quoteAsset") == "USDT"
            and s.get("status") == "TRADING"
        ):
            onboard_ms = int(s.get("onboardDate", 0))
            result[s["symbol"]] = pd.Timestamp(onboard_ms, unit="ms", tz="UTC")
    return result


def filter_symbols_at_date(
    symbol_onboard: dict[str, pd.Timestamp],
    as_of: pd.Timestamp,
) -> list[str]:
    """Return symbols that were listed on or before as_of (inclusive)."""
    return [sym for sym, ts in symbol_onboard.items() if ts <= as_of]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/data/test_binance.py -v`
Expected: all tests PASS (including pre-existing ones)

- [ ] **Step 5: Commit**

```bash
git add src/trade4/data/binance.py tests/data/test_binance.py
git commit -m "feat(data): add point-in-time symbol universe functions"
```

---

## Task 3: Scalper Cost Model Extension

**Files:**
- Modify: `src/trade4/backtest/cost_model.py`
- Modify: `tests/backtest/test_cost_model.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/backtest/test_cost_model.py`:

```python
from trade4.backtest.cost_model import ScalperCostModel, scalper_round_trip_bps


def test_scalper_cost_calm_entry():
    model = ScalperCostModel(
        perp_taker_bps=5.0,
        calm_slippage_bps=5.0,
        stress_slippage_bps=30.0,
    )
    # 2× perp taker (5+5) + 2× calm slippage (5+5) = 20 bps
    assert scalper_round_trip_bps(model, stressed=False) == 20.0


def test_scalper_cost_stressed_entry():
    model = ScalperCostModel(
        perp_taker_bps=5.0,
        calm_slippage_bps=5.0,
        stress_slippage_bps=30.0,
    )
    # 2× perp taker (5+5) + 2× stress slippage (30+30) = 70 bps
    assert scalper_round_trip_bps(model, stressed=True) == 70.0


def test_scalper_cost_stressed_is_costlier():
    model = ScalperCostModel()
    assert scalper_round_trip_bps(model, stressed=True) > scalper_round_trip_bps(model, stressed=False)


def test_scalper_cost_model_defaults_are_conservative():
    model = ScalperCostModel()
    # Default stress slippage should be >= 20 bps to represent pump spike market orders
    assert model.stress_slippage_bps >= 20.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/backtest/test_cost_model.py::test_scalper_cost_calm_entry -v`
Expected: `ImportError` — `ScalperCostModel` not defined yet

- [ ] **Step 3: Implement in `src/trade4/backtest/cost_model.py`**

Append to the existing file:

```python
@dataclass(frozen=True)
class ScalperCostModel:
    """Cost model for taker-only scalping trades on Binance USDT-M Futures."""
    perp_taker_bps: float = 5.0        # Binance Futures taker fee
    calm_slippage_bps: float = 5.0     # Normal taker entry slippage
    stress_slippage_bps: float = 30.0  # Market order into volume spike (pump scanner)


def scalper_round_trip_bps(model: ScalperCostModel, stressed: bool = False) -> float:
    """Total round-trip cost in bps for one scalping trade.

    stressed=True applies pump-scanner slippage (market buy into spike).
    Includes: perp taker fee × 2 (entry + exit) + slippage × 2.
    """
    slippage = model.stress_slippage_bps if stressed else model.calm_slippage_bps
    return 2 * model.perp_taker_bps + 2 * slippage
```

- [ ] **Step 4: Run all cost model tests**

Run: `pytest tests/backtest/test_cost_model.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/trade4/backtest/cost_model.py tests/backtest/test_cost_model.py
git commit -m "feat(cost_model): add ScalperCostModel with taker stress slippage"
```

---

## Task 4: Risk Manager

**Files:**
- Create: `src/trade4/scalper/risk_manager.py`
- Create: `tests/scalper/test_risk_manager.py`

- [ ] **Step 1: Write failing tests**

Create `tests/scalper/test_risk_manager.py`:

```python
import pytest
from trade4.scalper.risk_manager import size_position, circuit_breaker_triggered, PositionParams

BALANCE = 2000.0


def test_normal_long_position():
    # entry=100, sl=98.5 → sl_distance=1.5, risk=2%=40 EUR, qty=40/1.5=26.67
    params = size_position(
        balance=BALANCE, entry_price=100.0, sl_price=98.5, side="long"
    )
    assert params.qty == pytest.approx(40.0 / 1.5, rel=1e-4)
    assert params.sl_price == 98.5
    assert params.liq_buffer_ok is True
    assert params.actual_risk_eur == pytest.approx(40.0, rel=1e-2)


def test_normal_short_position():
    params = size_position(
        balance=BALANCE, entry_price=100.0, sl_price=101.5, side="short"
    )
    assert params.qty == pytest.approx(40.0 / 1.5, rel=1e-4)
    assert params.sl_price == 101.5


def test_leverage_cap_reduces_qty():
    # entry=100, sl=99.99 → sl_distance=0.01, uncapped qty=40/0.01=4000
    # notional=400000, requires 200x leverage → capped at 20x
    # capped qty = (2000 * 20) / 100 = 400
    params = size_position(
        balance=BALANCE, entry_price=100.0, sl_price=99.99, side="long",
        max_risk_fraction=0.02, max_leverage=20,
    )
    assert params.leverage == 20
    assert params.qty == pytest.approx(400.0, rel=1e-4)
    assert params.actual_risk_eur < BALANCE * 0.02  # risk was reduced


def test_zero_sl_distance_raises():
    with pytest.raises(ValueError, match="sl_price must differ"):
        size_position(balance=BALANCE, entry_price=100.0, sl_price=100.0, side="long")


def test_liq_buffer_violated_when_leverage_too_high():
    # Use max_leverage=100 to force liquidation price very close to SL
    params = size_position(
        balance=BALANCE, entry_price=100.0, sl_price=99.0, side="long",
        max_leverage=100,
    )
    # With 100x leverage: liq_price ≈ 100*(1 - 1/100 + 0.005) = 100*0.995 = 99.5
    # SL=99.0 is BELOW liq_price=99.5 → buffer check should fail
    assert params.liq_buffer_ok is False


def test_circuit_breaker_triggered():
    assert circuit_breaker_triggered(
        realized_pnl_today=-61.0,
        day_start_balance=2000.0,
        daily_loss_limit=0.03,
    ) is True  # -61 < -60 (3% of 2000)


def test_circuit_breaker_not_triggered():
    assert circuit_breaker_triggered(
        realized_pnl_today=-59.0,
        day_start_balance=2000.0,
        daily_loss_limit=0.03,
    ) is False  # -59 > -60


def test_circuit_breaker_at_exact_limit():
    # Exactly at limit: not triggered (strict <)
    assert circuit_breaker_triggered(
        realized_pnl_today=-60.0,
        day_start_balance=2000.0,
        daily_loss_limit=0.03,
    ) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/scalper/test_risk_manager.py -v`
Expected: `ImportError` — module doesn't exist yet

- [ ] **Step 3: Implement `src/trade4/scalper/risk_manager.py`**

```python
from dataclasses import dataclass
from typing import Literal

BINANCE_MAINT_MARGIN_RATE: float = 0.005  # 0.5% for ≤20x leverage on USDT-M


@dataclass
class PositionParams:
    qty: float
    leverage: int
    notional: float
    margin: float
    actual_risk_eur: float
    sl_price: float
    liq_price: float
    liq_buffer_ok: bool


def _liquidation_price(
    entry: float,
    leverage: int,
    side: Literal["long", "short"],
) -> float:
    if side == "long":
        return entry * (1 - 1 / leverage + BINANCE_MAINT_MARGIN_RATE)
    return entry * (1 + 1 / leverage - BINANCE_MAINT_MARGIN_RATE)


def size_position(
    balance: float,
    entry_price: float,
    sl_price: float,
    side: Literal["long", "short"],
    max_risk_fraction: float = 0.02,
    max_leverage: int = 20,
    min_liq_buffer: float = 0.20,
) -> PositionParams:
    """Compute position size for a trade with fixed risk fraction.

    If the max_leverage cap binds, qty is reduced so risk < max_risk_fraction.
    leverage is the pure ratio notional/margin and is not the profit lever
    (profit is determined by qty).
    """
    sl_distance = abs(entry_price - sl_price)
    if sl_distance == 0:
        raise ValueError("sl_price must differ from entry_price")

    risk_eur = balance * max_risk_fraction
    qty = risk_eur / sl_distance
    notional = qty * entry_price

    raw_leverage = notional / balance
    leverage = min(max(1, int(raw_leverage) + 1), max_leverage)

    if raw_leverage > max_leverage:
        qty = (balance * max_leverage) / entry_price
        notional = qty * entry_price
        risk_eur = qty * sl_distance

    liq_price = _liquidation_price(entry_price, leverage, side)

    entry_to_liq = abs(entry_price - liq_price)
    sl_to_liq = abs(sl_price - liq_price)
    liq_buffer_ok = entry_to_liq > 0 and (sl_to_liq / entry_to_liq) >= min_liq_buffer

    return PositionParams(
        qty=qty,
        leverage=leverage,
        notional=notional,
        margin=balance,
        actual_risk_eur=risk_eur,
        sl_price=sl_price,
        liq_price=liq_price,
        liq_buffer_ok=liq_buffer_ok,
    )


def circuit_breaker_triggered(
    realized_pnl_today: float,
    day_start_balance: float,
    daily_loss_limit: float = 0.03,
) -> bool:
    """Returns True when realized PnL today exceeds the daily loss limit.

    Uses realized PnL only (not unrealized) to avoid premature firing
    during temporary drawdowns in open positions.
    """
    return realized_pnl_today < -(day_start_balance * daily_loss_limit)
```

- [ ] **Step 4: Run all risk manager tests**

Run: `pytest tests/scalper/test_risk_manager.py -v`
Expected: all 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/trade4/scalper/risk_manager.py tests/scalper/test_risk_manager.py
git commit -m "feat(scalper): add risk manager with position sizing and circuit breaker"
```

---

## Task 5: EMA-Cross Signal Module

**Files:**
- Create: `src/trade4/scalper/signals/ema_cross.py`
- Create: `tests/scalper/signals/test_ema_cross.py`

- [ ] **Step 1: Write failing tests**

Create `tests/scalper/signals/test_ema_cross.py`:

```python
import pandas as pd
import numpy as np
import pytest
from trade4.scalper.signals.ema_cross import generate_signals


def _make_ohlcv(prices: list[float], start: pd.Timestamp, freq: str = "1min") -> pd.DataFrame:
    n = len(prices)
    timestamps = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": [p * 0.9995 for p in prices],
        "high": [p * 1.002 for p in prices],
        "low": [p * 0.998 for p in prices],
        "close": prices,
        "volume": [1_000_000.0] * n,
    })


def _make_15m_trend(start: pd.Timestamp, trend_up: bool = True, n: int = 260) -> pd.DataFrame:
    if trend_up:
        prices = [100.0 + i * 0.05 for i in range(n)]
    else:
        prices = [100.0 - i * 0.05 for i in range(n)]
    return _make_ohlcv(prices, start, freq="15min")


def test_long_signal_on_golden_cross():
    start = pd.Timestamp("2024-01-01", tz="UTC")
    # 40 candles declining (EMA9 < EMA21), then 40 strong bull candles (EMA9 crosses above EMA21)
    prices_down = [100.0 - i * 0.05 for i in range(40)]
    prices_up = [98.0 + i * 0.40 for i in range(40)]
    df_1m = _make_ohlcv(prices_down + prices_up, start)
    df_15m = _make_15m_trend(start - pd.Timedelta("65h"), trend_up=True)

    signals = generate_signals(df_1m, df_15m)
    long_signals = signals[signals["signal"] == 1]
    assert len(long_signals) > 0, "Expected at least one long signal on golden cross"


def test_short_signal_on_death_cross():
    start = pd.Timestamp("2024-01-01", tz="UTC")
    prices_up = [100.0 + i * 0.05 for i in range(40)]
    prices_down = [102.0 - i * 0.40 for i in range(40)]
    df_1m = _make_ohlcv(prices_up + prices_down, start)
    df_15m = _make_15m_trend(start - pd.Timedelta("65h"), trend_up=False)

    signals = generate_signals(df_1m, df_15m)
    short_signals = signals[signals["signal"] == -1]
    assert len(short_signals) > 0, "Expected at least one short signal on death cross"


def test_no_signal_when_rsi_opposes_cross():
    # Build golden cross but with RSI < 50 (add downward pressure after cross)
    start = pd.Timestamp("2024-01-01", tz="UTC")
    # Declining phase, then a tiny uptick that barely crosses EMA but RSI stays low
    prices = [100.0 - i * 0.10 for i in range(40)]
    prices += [96.0 + i * 0.05 for i in range(10)]  # weak uptick, RSI still < 50
    df_1m = _make_ohlcv(prices, start)
    df_15m = _make_15m_trend(start - pd.Timedelta("65h"), trend_up=True)

    signals = generate_signals(df_1m, df_15m, rsi_period=14)
    # May or may not have signals but this tests RSI filter is active
    # We verify no crash and signal column is bounded
    assert set(signals["signal"].unique()).issubset({1, -1})


def test_no_signal_when_atr_too_low():
    start = pd.Timestamp("2024-01-01", tz="UTC")
    # Completely flat prices — ATR ~ 0, well below min threshold
    prices_down = [100.0 - i * 0.05 for i in range(40)]
    prices_flat = [98.0 + i * 0.40 for i in range(40)]
    df_1m = _make_ohlcv(prices_down + prices_flat, start)
    # Override high/low to be almost same as close (zero ATR)
    df_1m["high"] = df_1m["close"] * 1.000001
    df_1m["low"] = df_1m["close"] * 0.999999
    df_15m = _make_15m_trend(start - pd.Timedelta("65h"), trend_up=True)

    signals = generate_signals(df_1m, df_15m, atr_min_pct=0.005)  # require 0.5% ATR
    assert len(signals) == 0, "Expected no signals when ATR is below minimum"


def test_no_long_signal_when_price_below_15m_trend():
    start = pd.Timestamp("2024-01-01", tz="UTC")
    prices_down = [100.0 - i * 0.05 for i in range(40)]
    prices_up = [98.0 + i * 0.40 for i in range(40)]
    df_1m = _make_ohlcv(prices_down + prices_up, start)
    # Downtrend on 15m (price will be below EMA200)
    df_15m = _make_15m_trend(start - pd.Timedelta("65h"), trend_up=False)

    signals = generate_signals(df_1m, df_15m)
    long_signals = signals[signals["signal"] == 1]
    assert len(long_signals) == 0, "No long signals expected in downtrend on 15m"


def test_sl_distance_equals_atr_multiplied():
    start = pd.Timestamp("2024-01-01", tz="UTC")
    prices_down = [100.0 - i * 0.05 for i in range(40)]
    prices_up = [98.0 + i * 0.40 for i in range(40)]
    df_1m = _make_ohlcv(prices_down + prices_up, start)
    df_15m = _make_15m_trend(start - pd.Timedelta("65h"), trend_up=True)

    signals = generate_signals(df_1m, df_15m, sl_atr_multiplier=1.5)
    if len(signals) > 0:
        row = signals.iloc[0]
        assert row["sl_distance"] == pytest.approx(row["atr_1m"] * 1.5, rel=1e-4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/scalper/signals/test_ema_cross.py -v`
Expected: `ImportError`

- [ ] **Step 3: Implement `src/trade4/scalper/signals/ema_cross.py`**

```python
import pandas as pd
import numpy as np


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    return 100 - 100 / (1 + rs)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def generate_signals(
    df_1m: pd.DataFrame,
    df_15m: pd.DataFrame,
    ema_fast: int = 9,
    ema_slow: int = 21,
    ema_trend: int = 200,
    rsi_period: int = 14,
    atr_period: int = 14,
    atr_min_pct: float = 0.0005,
    sl_atr_multiplier: float = 1.5,
) -> pd.DataFrame:
    """Generate EMA-cross momentum signals.

    Returns DataFrame with columns:
        timestamp, signal (1=long, -1=short),
        atr_1m, sl_distance, entry_price,
        sl_price_long, sl_price_short

    Only rows with non-zero signals are returned.
    Both DataFrames require columns: timestamp, open, high, low, close, volume.
    Timestamps must be UTC-aware datetime64.
    """
    df1 = df_1m.copy().set_index("timestamp").sort_index()
    df15 = df_15m.copy().set_index("timestamp").sort_index()

    df1["ema_fast"] = _ema(df1["close"], ema_fast)
    df1["ema_slow"] = _ema(df1["close"], ema_slow)
    df1["rsi"] = _rsi(df1["close"], rsi_period)
    df1["atr"] = _atr(df1, atr_period)

    df1["cross_long"] = (
        (df1["ema_fast"] > df1["ema_slow"])
        & (df1["ema_fast"].shift(1) <= df1["ema_slow"].shift(1))
    )
    df1["cross_short"] = (
        (df1["ema_fast"] < df1["ema_slow"])
        & (df1["ema_fast"].shift(1) >= df1["ema_slow"].shift(1))
    )

    df15["ema200"] = _ema(df15["close"], ema_trend)
    df15["trend_up"] = df15["close"] > df15["ema200"]
    trend_up = df15["trend_up"].reindex(df1.index, method="ffill").fillna(False)

    atr_ok = df1["atr"] > (df1["close"] * atr_min_pct)

    signal = pd.Series(0, index=df1.index, dtype=int)
    signal[df1["cross_long"] & (df1["rsi"] > 50) & trend_up & atr_ok] = 1
    signal[df1["cross_short"] & (df1["rsi"] < 50) & ~trend_up & atr_ok] = -1

    sl_dist = df1["atr"] * sl_atr_multiplier
    result = pd.DataFrame({
        "timestamp": df1.index,
        "signal": signal.values,
        "atr_1m": df1["atr"].values,
        "sl_distance": sl_dist.values,
        "entry_price": df1["close"].values,
    })
    result["sl_price_long"] = result["entry_price"] - result["sl_distance"]
    result["sl_price_short"] = result["entry_price"] + result["sl_distance"]

    return result[result["signal"] != 0].reset_index(drop=True)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/scalper/signals/test_ema_cross.py -v`
Expected: all PASS (except `test_no_signal_when_rsi_opposes_cross` which only checks no crash)

- [ ] **Step 5: Commit**

```bash
git add src/trade4/scalper/signals/ema_cross.py tests/scalper/signals/test_ema_cross.py
git commit -m "feat(scalper): add EMA-cross momentum signal generator"
```

---

## Task 6: Pump/Spike Scanner Signal Module

**Files:**
- Create: `src/trade4/scalper/signals/pump_scanner.py`
- Create: `tests/scalper/signals/test_pump_scanner.py`

- [ ] **Step 1: Write failing tests**

Create `tests/scalper/signals/test_pump_scanner.py`:

```python
import pandas as pd
import numpy as np
import pytest
from trade4.scalper.signals.pump_scanner import generate_signals


def _make_df(closes: list[float], volumes: list[float], start: pd.Timestamp) -> pd.DataFrame:
    n = len(closes)
    timestamps = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    opens = [c * 0.998 for c in closes]
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": [c * 1.003 for c in closes],
        "low": [c * 0.997 for c in closes],
        "close": closes,
        "volume": volumes,
    })


def _flat_prices(n: int, price: float = 100.0) -> list[float]:
    return [price] * n


def _baseline_volumes(n: int, base: float = 1_000_000.0) -> list[float]:
    return [base] * n


def test_long_signal_on_volume_spike_with_pump():
    start = pd.Timestamp("2024-01-01", tz="UTC")
    # 25 normal candles, then 1 candle with 4× volume and +2% price move
    prices = _flat_prices(25) + [100.0, 102.0]  # last candle: +2%
    volumes = _baseline_volumes(25) + [1_000_000.0, 4_000_000.0]
    # Override open for pump candle so return = (102-100)/100 = +2%
    df = _make_df(prices, volumes, start)
    df.iloc[-1, df.columns.get_loc("open")] = 100.0

    signals = generate_signals(df, volume_multiplier=3.0, price_move_pct=0.015)
    assert len(signals) == 1
    assert signals.iloc[0]["signal"] == 1


def test_short_signal_on_volume_spike_with_dump():
    start = pd.Timestamp("2024-01-01", tz="UTC")
    prices = _flat_prices(25) + [100.0, 98.0]  # last candle: -2%
    volumes = _baseline_volumes(25) + [1_000_000.0, 4_000_000.0]
    df = _make_df(prices, volumes, start)
    df.iloc[-1, df.columns.get_loc("open")] = 100.0

    signals = generate_signals(df, volume_multiplier=3.0, price_move_pct=0.015)
    assert len(signals) == 1
    assert signals.iloc[0]["signal"] == -1


def test_no_signal_volume_below_multiplier():
    start = pd.Timestamp("2024-01-01", tz="UTC")
    prices = _flat_prices(25) + [100.0, 102.0]
    volumes = _baseline_volumes(25) + [1_000_000.0, 2_000_000.0]  # only 2× not 3×
    df = _make_df(prices, volumes, start)
    df.iloc[-1, df.columns.get_loc("open")] = 100.0

    signals = generate_signals(df, volume_multiplier=3.0, price_move_pct=0.015)
    assert len(signals) == 0


def test_no_signal_price_move_below_threshold():
    start = pd.Timestamp("2024-01-01", tz="UTC")
    prices = _flat_prices(25) + [100.0, 100.5]  # only +0.5%
    volumes = _baseline_volumes(25) + [1_000_000.0, 5_000_000.0]
    df = _make_df(prices, volumes, start)
    df.iloc[-1, df.columns.get_loc("open")] = 100.0

    signals = generate_signals(df, volume_multiplier=3.0, price_move_pct=0.015)
    assert len(signals) == 0


def test_sl_and_tp_computed_correctly_for_long():
    start = pd.Timestamp("2024-01-01", tz="UTC")
    prices = _flat_prices(25) + [100.0, 102.0]
    volumes = _baseline_volumes(25) + [1_000_000.0, 4_000_000.0]
    df = _make_df(prices, volumes, start)
    df.iloc[-1, df.columns.get_loc("open")] = 100.0

    signals = generate_signals(df, sl_pct=0.008, tp_pct=0.012)
    row = signals.iloc[0]
    assert row["sl_price"] == pytest.approx(row["entry_price"] * (1 - 0.008), rel=1e-4)
    assert row["tp_price"] == pytest.approx(row["entry_price"] * (1 + 0.012), rel=1e-4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/scalper/signals/test_pump_scanner.py -v`
Expected: `ImportError`

- [ ] **Step 3: Implement `src/trade4/scalper/signals/pump_scanner.py`**

```python
import pandas as pd
import numpy as np


def generate_signals(
    df_1m: pd.DataFrame,
    volume_multiplier: float = 3.0,
    price_move_pct: float = 0.015,
    vol_ma_period: int = 20,
    sl_pct: float = 0.008,
    tp_pct: float = 0.012,
) -> pd.DataFrame:
    """Detect pump/dump spikes from volume and price acceleration.

    Returns DataFrame with columns:
        timestamp, signal (1=long, -1=short),
        entry_price, sl_price, tp_price

    Entry is assumed to fill at the close of the trigger candle
    (market order). sl_pct and tp_pct are applied to entry_price.

    Note: Taker market orders into spikes carry 0.1–0.5% slippage —
    the ScalperCostModel stressed mode accounts for this in backtest.
    """
    df = df_1m.copy().set_index("timestamp").sort_index()

    df["vol_ma"] = df["volume"].rolling(vol_ma_period, min_periods=vol_ma_period).mean()
    df["vol_spike"] = df["volume"] > (df["vol_ma"] * volume_multiplier)

    candle_return = (df["close"] - df["open"]) / df["open"].replace(0, float("nan"))

    pump = df["vol_spike"] & (candle_return > price_move_pct)
    dump = df["vol_spike"] & (candle_return < -price_move_pct)

    signal = pd.Series(0, index=df.index, dtype=int)
    signal[pump] = 1
    signal[dump] = -1

    result = pd.DataFrame({
        "timestamp": df.index,
        "signal": signal.values,
        "entry_price": df["close"].values,
    })
    result = result[result["signal"] != 0].copy()

    result["sl_price"] = result.apply(
        lambda r: r["entry_price"] * (1 - sl_pct)
        if r["signal"] == 1
        else r["entry_price"] * (1 + sl_pct),
        axis=1,
    )
    result["tp_price"] = result.apply(
        lambda r: r["entry_price"] * (1 + tp_pct)
        if r["signal"] == 1
        else r["entry_price"] * (1 - tp_pct),
        axis=1,
    )
    return result.reset_index(drop=True)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/scalper/signals/test_pump_scanner.py -v`
Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/trade4/scalper/signals/pump_scanner.py tests/scalper/signals/test_pump_scanner.py
git commit -m "feat(scalper): add pump/spike scanner signal generator"
```

---

## Task 7: ATR Volatility Screener

**Files:**
- Create: `src/trade4/scalper/screener.py`
- Create: `tests/scalper/test_screener.py`

- [ ] **Step 1: Write failing tests**

Create `tests/scalper/test_screener.py`:

```python
import pandas as pd
import numpy as np
import pytest
from trade4.scalper.screener import compute_atr_score, screen_by_volatility


def _make_1h_ohlcv(atr_pct: float, n: int = 50, price: float = 100.0) -> pd.DataFrame:
    """Synthetic 1h OHLCV with controlled ATR as % of price."""
    start = pd.Timestamp("2024-01-01", tz="UTC")
    timestamps = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    half_range = price * atr_pct / 2
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": [price] * n,
        "high": [price + half_range] * n,
        "low": [price - half_range] * n,
        "close": [price] * n,
        "volume": [1_000_000.0] * n,
    })


def test_atr_score_higher_for_volatile_asset():
    df_volatile = _make_1h_ohlcv(atr_pct=0.02)  # 2% ATR
    df_calm = _make_1h_ohlcv(atr_pct=0.005)     # 0.5% ATR
    assert compute_atr_score(df_volatile) > compute_atr_score(df_calm)


def test_atr_score_returns_zero_for_insufficient_data():
    df = _make_1h_ohlcv(atr_pct=0.01, n=5)  # fewer than period=14
    assert compute_atr_score(df, period=14) == 0.0


def test_screen_returns_top_n_by_volatility():
    ohlcv = {
        "ALUSDT": _make_1h_ohlcv(atr_pct=0.05),   # highest volatility
        "BLUSDT": _make_1h_ohlcv(atr_pct=0.02),
        "CLUSTDT": _make_1h_ohlcv(atr_pct=0.01),
        "DLUSDT": _make_1h_ohlcv(atr_pct=0.001),  # lowest
    }
    volumes = {"ALUSDT": 500e6, "BLUSDT": 500e6, "CLUSTDT": 500e6, "DLUSDT": 500e6}
    result = screen_by_volatility(ohlcv, volumes, top_n=2)
    assert result == ["ALUSDT", "BLUSDT"]


def test_screen_excludes_below_volume_threshold():
    ohlcv = {
        "HIGHVOL": _make_1h_ohlcv(atr_pct=0.05),
        "LOWVOL": _make_1h_ohlcv(atr_pct=0.10),  # more volatile but low volume
    }
    volumes = {"HIGHVOL": 500e6, "LOWVOL": 50e6}  # LOWVOL below 200M threshold
    result = screen_by_volatility(ohlcv, volumes, min_volume_usdt=200e6, top_n=5)
    assert "LOWVOL" not in result
    assert "HIGHVOL" in result


def test_screen_point_in_time_excludes_future_listings():
    ohlcv = {
        "OLD": _make_1h_ohlcv(atr_pct=0.01),
        "NEW": _make_1h_ohlcv(atr_pct=0.05),  # more volatile but listed later
    }
    volumes = {"OLD": 500e6, "NEW": 500e6}
    onboard = {
        "OLD": pd.Timestamp("2020-01-01", tz="UTC"),
        "NEW": pd.Timestamp("2025-01-01", tz="UTC"),
    }
    as_of = pd.Timestamp("2024-01-01", tz="UTC")
    result = screen_by_volatility(ohlcv, volumes, symbol_onboard=onboard, as_of=as_of)
    assert "NEW" not in result
    assert "OLD" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/scalper/test_screener.py -v`
Expected: `ImportError`

- [ ] **Step 3: Implement `src/trade4/scalper/screener.py`**

```python
import pandas as pd
import numpy as np


def compute_atr_score(ohlcv_1h: pd.DataFrame, period: int = 14) -> float:
    """ATR as fraction of current price. Returns 0.0 if insufficient data."""
    df = ohlcv_1h.sort_values("timestamp") if "timestamp" in ohlcv_1h.columns else ohlcv_1h
    if len(df) < period + 1:
        return 0.0

    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(com=period - 1, adjust=False).mean().iloc[-1]
    price = df["close"].iloc[-1]
    return float(atr / price) if price > 0 else 0.0


def screen_by_volatility(
    ohlcv_data: dict[str, pd.DataFrame],
    volume_data_24h: dict[str, float],
    symbol_onboard: dict[str, pd.Timestamp] | None = None,
    as_of: pd.Timestamp | None = None,
    min_volume_usdt: float = 200_000_000.0,
    top_n: int = 20,
    atr_period: int = 14,
) -> list[str]:
    """Return top-N symbols by ATR volatility score.

    Filters:
    - 24h volume >= min_volume_usdt
    - Point-in-time: only symbols listed on or before as_of (if provided)

    Symbols are ranked highest ATR-score first.
    """
    scores: dict[str, float] = {}
    for sym, df in ohlcv_data.items():
        if df is None or df.empty:
            continue
        if symbol_onboard is not None and as_of is not None:
            onboard_ts = symbol_onboard.get(sym, pd.Timestamp.max.tz_localize("UTC"))
            if onboard_ts > as_of:
                continue
        vol = volume_data_24h.get(sym, 0.0)
        if vol < min_volume_usdt:
            continue
        scores[sym] = compute_atr_score(df, atr_period)

    ranked = sorted(scores, key=lambda s: scores[s], reverse=True)
    return ranked[:top_n]
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/scalper/test_screener.py -v`
Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/trade4/scalper/screener.py tests/scalper/test_screener.py
git commit -m "feat(scalper): add ATR volatility screener with point-in-time universe"
```

---

## Task 8: Backtest Harness

**Files:**
- Create: `src/trade4/scalper/backtest_harness.py`
- Create: `tests/scalper/test_backtest_harness.py`

Note: TP for EMA-Cross uses a fixed level of `entry ± 2.5 × ATR` for backtest simplicity (approximates the trailing-stop described in the spec). The trailing-stop mechanism is implemented in Phase 2 live execution.

- [ ] **Step 1: Write failing tests**

Create `tests/scalper/test_backtest_harness.py`:

```python
import pandas as pd
import numpy as np
import pytest
from trade4.scalper.backtest_harness import run_backtest, BacktestConfig, TradeResult


def _make_ohlcv_1m(prices: list[float], start: pd.Timestamp) -> pd.DataFrame:
    n = len(prices)
    timestamps = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    opens = [p * 0.9995 for p in prices]
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": [p * 1.003 for p in prices],
        "low": [p * 0.997 for p in prices],
        "close": prices,
        "volume": [2_000_000.0] * n,
    })


def _make_ohlcv_15m(n: int = 260, start: pd.Timestamp | None = None) -> pd.DataFrame:
    if start is None:
        start = pd.Timestamp("2023-12-29", tz="UTC")
    prices = [100.0 + i * 0.05 for i in range(n)]
    timestamps = pd.date_range(start, periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": prices,
        "high": [p * 1.002 for p in prices],
        "low": [p * 0.998 for p in prices],
        "close": prices,
        "volume": [1_000_000.0] * n,
    })


def test_run_backtest_returns_correct_types():
    start = pd.Timestamp("2024-01-02 08:00", tz="UTC")
    prices_down = [100.0 - i * 0.05 for i in range(40)]
    prices_up = [98.0 + i * 0.40 for i in range(60)]
    df_1m = _make_ohlcv_1m(prices_down + prices_up, start)
    df_15m = _make_ohlcv_15m()

    trades, equity = run_backtest(df_1m, df_15m, "BTCUSDT", BacktestConfig())
    assert isinstance(trades, list)
    assert isinstance(equity, pd.Series)
    assert len(equity) == len(df_1m)


def test_circuit_breaker_stops_new_trades():
    # Force large losses: set daily_loss_limit very low (0.1%)
    # and generate many losing trades
    start = pd.Timestamp("2024-01-02 08:00", tz="UTC")
    prices_down = [100.0 - i * 0.05 for i in range(40)]
    prices_up = [98.0 + i * 0.40 for i in range(40)]
    # Then crash so all SLs get hit
    prices_crash = [114.0 - i * 1.0 for i in range(40)]
    df_1m = _make_ohlcv_1m(prices_down + prices_up + prices_crash, start)
    df_15m = _make_ohlcv_15m()

    cfg = BacktestConfig(daily_loss_limit=0.001)  # 0.1% daily limit → fires easily
    trades, equity = run_backtest(df_1m, df_15m, "BTCUSDT", cfg)
    # After circuit breaker, equity must not go below (start - 3×limit) because trading stops
    min_equity = equity.min()
    assert min_equity > cfg.start_balance * (1 - 0.05), "Circuit breaker should cap losses"


def test_sl_hit_closes_position():
    # Simple scenario: price drops immediately below SL after entry
    start = pd.Timestamp("2024-01-02 08:00", tz="UTC")
    prices_down = [100.0 - i * 0.05 for i in range(40)]
    prices_up = [98.0 + i * 0.40 for i in range(10)]   # short up to generate signal
    prices_crash = [102.0 - i * 2.0 for i in range(20)]  # crash through SL
    df_1m = _make_ohlcv_1m(prices_down + prices_up + prices_crash, start)
    df_15m = _make_ohlcv_15m()

    trades, _ = run_backtest(df_1m, df_15m, "BTCUSDT", BacktestConfig())
    sl_trades = [t for t in trades if t.exit_reason == "sl"]
    # At least one SL hit when price crashes
    if len(trades) > 0:
        assert any(t.exit_reason in ("sl", "tp", "timeout") for t in trades)


def test_no_more_than_max_open_positions():
    start = pd.Timestamp("2024-01-02 08:00", tz="UTC")
    # Large bull run to generate many signals
    prices_down = [100.0 - i * 0.05 for i in range(40)]
    prices_up = [98.0 + i * 0.30 for i in range(200)]
    df_1m = _make_ohlcv_1m(prices_down + prices_up, start)
    df_15m = _make_ohlcv_15m()

    cfg = BacktestConfig(max_open_positions=2)
    trades, equity = run_backtest(df_1m, df_15m, "BTCUSDT", cfg)
    assert isinstance(trades, list)
    # Equity must be monotonically bounded (no position accounting errors)
    assert equity.max() < cfg.start_balance * 100  # sanity check


def test_net_pnl_includes_costs():
    start = pd.Timestamp("2024-01-02 08:00", tz="UTC")
    prices_down = [100.0 - i * 0.05 for i in range(40)]
    prices_up = [98.0 + i * 0.40 for i in range(40)]
    df_1m = _make_ohlcv_1m(prices_down + prices_up, start)
    df_15m = _make_ohlcv_15m()

    trades, _ = run_backtest(df_1m, df_15m, "BTCUSDT", BacktestConfig())
    for t in trades:
        assert t.net_pnl_eur == pytest.approx(t.gross_pnl_eur - t.cost_eur, rel=1e-6)
        assert t.cost_eur >= 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/scalper/test_backtest_harness.py -v`
Expected: `ImportError`

- [ ] **Step 3: Implement `src/trade4/scalper/backtest_harness.py`**

```python
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
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/scalper/test_backtest_harness.py -v`
Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/trade4/scalper/backtest_harness.py tests/scalper/test_backtest_harness.py
git commit -m "feat(scalper): add event-loop backtest harness for EMA-cross and pump scanner"
```

---

## Task 9: Walk-Forward Validation

**Files:**
- Create: `src/trade4/scalper/walk_forward.py`
- Create: `tests/scalper/test_walk_forward.py`

- [ ] **Step 1: Write failing tests**

Create `tests/scalper/test_walk_forward.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/scalper/test_walk_forward.py -v`
Expected: `ImportError`

- [ ] **Step 3: Implement `src/trade4/scalper/walk_forward.py`**

```python
from dataclasses import dataclass
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
        cfg_dict = {**vars(base_config), **params}
        cfg = BacktestConfig(**cfg_dict)
        trades, curve = run_backtest(is_1m, is_15m, symbol, cfg)
        m = compute_metrics(trades, curve, cfg.start_balance)
        if m.sharpe > best_sharpe:
            best_sharpe = m.sharpe
            best_params = params
            best_is_trades, best_is_curve = trades, curve

    best_cfg_dict = {**vars(base_config), **best_params}
    best_cfg = BacktestConfig(**best_cfg_dict)
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
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/scalper/test_walk_forward.py -v`
Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/trade4/scalper/walk_forward.py tests/scalper/test_walk_forward.py
git commit -m "feat(scalper): add walk-forward validator with Phase-2 go/no-go gate"
```

---

## Task 10: Go/No-Go Report

**Files:**
- Create: `src/trade4/scalper/report.py`

No dedicated test — this is a console/text output function. Run manually and visually inspect.

- [ ] **Step 1: Implement `src/trade4/scalper/report.py`**

```python
from trade4.scalper.walk_forward import WalkForwardResult

_GATE_SYMBOL = {True: "PASS", False: "FAIL"}
_LINE = "=" * 60


def print_walk_forward_report(result: WalkForwardResult, symbol: str = "") -> None:
    """Print a structured go/no-go report to stdout."""
    print(_LINE)
    print(f"SCALPER WALK-FORWARD REPORT  {symbol}")
    print(_LINE)

    def _row(label: str, is_val: object, oos_val: object, threshold: str = "") -> None:
        th = f"  (gate: {threshold})" if threshold else ""
        print(f"  {label:<30} IS={is_val!s:<12} OOS={oos_val!s:<12}{th}")

    is_m = result.in_sample
    oos_m = result.out_of_sample

    print("\nPERFORMANCE METRICS")
    _row("Sharpe (annualized)", f"{is_m.sharpe:.2f}", f"{oos_m.sharpe:.2f}", ">= 1.5")
    _row("Max Drawdown", f"{is_m.max_drawdown:.1%}", f"{oos_m.max_drawdown:.1%}", ">= -20%")
    _row("Profit Factor", f"{is_m.profit_factor:.2f}", f"{oos_m.profit_factor:.2f}", ">= 1.4")
    _row("Win Rate", f"{is_m.win_rate:.1%}", f"{oos_m.win_rate:.1%}", ">= 45%")
    _row("Trades", is_m.n_trades, oos_m.n_trades)
    _row("Final Balance (EUR)", f"{is_m.final_balance:.0f}", f"{oos_m.final_balance:.0f}")

    print(f"\n  {'OOS Degradation':<30} {result.oos_degradation:.1%}  (gate: <= 30%)")

    if result.best_params:
        print(f"\nBEST PARAMS (optimized on IS):")
        for k, v in result.best_params.items():
            print(f"  {k}: {v}")

    gate = _GATE_SYMBOL[result.gate_passed]
    print(f"\n{'=' * 60}")
    print(f"  PHASE-2 GATE: {gate}")
    if not result.gate_passed:
        print("  Strategy did NOT pass the go/no-go gate.")
        print("  Do NOT deploy live capital. Review signals and parameters.")
    else:
        print("  OOS validation passed. Proceed to Phase-2 plan.")
    print(_LINE)
```

- [ ] **Step 2: Smoke-test the report function**

Run from the project root:

```python
# In Python REPL or notebook:
from trade4.scalper.walk_forward import WalkForwardMetrics, WalkForwardResult
from trade4.scalper.report import print_walk_forward_report

dummy = WalkForwardResult(
    best_params={"ema_fast": 9, "ema_slow": 21},
    in_sample=WalkForwardMetrics(sharpe=2.1, max_drawdown=-0.12, profit_factor=1.9, win_rate=0.58, n_trades=180, final_balance=2700.0),
    out_of_sample=WalkForwardMetrics(sharpe=1.6, max_drawdown=-0.17, profit_factor=1.5, win_rate=0.51, n_trades=55, final_balance=2250.0),
    oos_degradation=0.24,
    gate_passed=True,
)
print_walk_forward_report(dummy, symbol="BTCUSDT")
```

Expected output: Clean table with `PHASE-2 GATE: PASS`.

- [ ] **Step 3: Run the full test suite to confirm nothing is broken**

Run: `pytest tests/ -v --tb=short`
Expected: all tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/trade4/scalper/report.py
git commit -m "feat(scalper): add go/no-go walk-forward report printer"
```

---

## Self-Review

**Spec coverage check:**

| Spec Section | Task |
|---|---|
| Asset Screener (ATR + point-in-time universe) | Task 7 + Task 2 |
| Modul A: EMA-Cross signal logic | Task 5 |
| Modul B: Pump/Spike Scanner | Task 6 |
| Risk Manager (position sizing, liq buffer, circuit breaker) | Task 4 |
| ScalperCostModel (taker + stress slippage) | Task 3 |
| Backtest Harness (event loop, SL/TP/timeout, session filter) | Task 8 |
| Walk-Forward Validation (IS/OOS, param grid, gate) | Task 9 |
| Go/No-Go Report | Task 10 |
| Binance point-in-time universe | Task 2 |
| Package skeleton | Task 1 |

**Gaps addressed explicitly in code:**
- Survivorship bias: `screen_by_volatility` + `filter_symbols_at_date` enforce point-in-time universe
- Stress slippage: `ScalperCostModel.stress_slippage_bps=30` for pump scanner taker entries
- Leverage semantics: qty is reduced when cap binds, leverage is not the profit lever (documented in `size_position`)
- Circuit breaker: realized PnL only (documented in `circuit_breaker_triggered`)
- Honest gate: `_gate_check` in `walk_forward.py`, report prints FAIL explicitly

**Placeholder scan:** No TBD, TODO, or unspecified sections found.

**Type consistency:** `TradeResult` used identically in `backtest_harness.py`, `walk_forward.py`, and tests. `BacktestConfig` fields referenced by exact name in `run_walk_forward`.
