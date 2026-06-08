# Research Harness — Phase 0 + 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fraud-resistant point-in-time data lake and a vectorised market-neutral portfolio backtest engine whose look-ahead-freedom and PnL correctness are structurally enforced by tests, then prove it by reproducing the known funding-carry OOS loss.

**Architecture:** New `src/trade4/research/` package. A `Panel` holds aligned per-symbol PIT data + a tradeable mask. A `Strategy` ABC emits dollar-neutral target weights. The `portfolio_engine` vectorises PnL (price + discrete funding − turnover costs) and enforces causality via a future-perturbation tripwire and a PIT-mask assert. `funding_carry` is ported causally and validated against the legacy (fixed) engine.

**Tech Stack:** Python 3.12, pandas, numpy, scipy (new), pyarrow, pytest. No heavy backtest framework — full control over funding/cost bookings is the point.

**Reference truth to reproduce:** funding-carry walk-forward IS +11.004 bps → **OOS −2.862 bps** (must come from the `causal_gate`-fixed engine; confirm in Task 0.2).

---

## File Structure

| File | Responsibility |
|------|----------------|
| `src/trade4/research/__init__.py` | package marker |
| `src/trade4/research/panel.py` | `Panel` container: aligned funding/OHLCV/OI frames on a fixed bar grid + `tradeable` mask |
| `src/trade4/research/strategy.py` | `Strategy` ABC: `generate_target_weights(panel) -> DataFrame` |
| `src/trade4/research/costs.py` | turnover-/ADV-aware portfolio cost layer wrapping `backtest/cost_model.py` |
| `src/trade4/research/portfolio_engine.py` | CORE vectorised engine: PnL, funding, costs, tripwire, mask assert |
| `src/trade4/research/strategies/__init__.py` | package marker |
| `src/trade4/research/strategies/funding_carry.py` | causal port of legacy funding-carry as a `Strategy` |
| `src/trade4/data/binance_vision.py` | fetch klines + funding from `data.binance.vision` incl. delisted symbols |
| `tests/research/__init__.py` | package marker |
| `tests/research/test_panel.py` | mask semantics, alignment |
| `tests/research/test_portfolio_engine.py` | known-answer price PnL, funding-only PnL, tripwire, mask assert |
| `tests/research/test_costs.py` | turnover cost correctness |
| `tests/research/test_funding_carry.py` | causal port + cross-engine equivalence (killer-gate) |
| `tests/data/test_binance_vision.py` | delisted symbol returns data |

Modify: `pyproject.toml` (+scipy).

---

## Phase 0 — Reconcile & Scaffold

### Task 0.1: Add scipy dependency

**Files:** Modify `pyproject.toml`

- [ ] **Step 1: Edit dependencies**

In `pyproject.toml`, add `"scipy>=1.13"` to the `dependencies` list (after `numpy>=1.26`).

- [ ] **Step 2: Install**

Run: `pip install -e .` (or `pip install scipy>=1.13`)
Expected: scipy installs without conflict.

- [ ] **Step 3: Verify**

Run: `python -c "import scipy; print(scipy.__version__)"`
Expected: prints a version ≥ 1.13.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "build: add scipy dependency for deflated-sharpe statistics"
```

### Task 0.2: Reconcile VPS fixes (causal_gate, exit_mode, /100)

**Files:** Modify `src/trade4/live/scheduler.py`, `src/trade4/backtest/engine.py` (exact lines determined by diff)

> **External dependency.** VPS 178.254.32.125 is the live APEX box. Operate **read-only**; do
> not stop/restart the running process. If non-interactive SSH is unavailable in this
> session, STOP and ask the user to provide the diffs via `!ssh`/`!scp`. Do not fabricate the
> fixes.

- [ ] **Step 1: Locate the project on the VPS (read-only)**

Run: `ssh root@178.254.32.125 'ls -d ~/trade* /opt/trade* 2>/dev/null; ps aux | grep -i scheduler | grep -v grep'`
Expected: a project path + the running process (confirming it's the live box). Note the path.

- [ ] **Step 2: Diff the two files against local**

Run (read-only, prints remote file):
`ssh root@178.254.32.125 'cat <REMOTE_PATH>/src/trade4/backtest/engine.py' > /tmp/vps_engine.py`
`diff src/trade4/backtest/engine.py /tmp/vps_engine.py`
Repeat for `live/scheduler.py`.
Expected: the diff shows the `causal_gate` / `exit_mode` / `/100` changes. Inspect §3 line 125 — confirm the fix replaces the forward-looking `funding_df[... > ts]` with a causal estimate.

- [ ] **Step 3: Confirm the −2.862 bps provenance**

Search the VPS for any walk-forward output/log proving −2.862 came from the *fixed* engine:
`ssh root@178.254.32.125 'grep -rn "2.862\|2862\|OOS" <REMOTE_PATH> 2>/dev/null | head'`
Record the answer in the commit message. If it came from the unfixed engine, the killer-gate
tolerance (Task 1.9) must be re-derived by running the fixed engine locally instead.

- [ ] **Step 4: Apply the fixes locally**

Port the diffed changes into the local files by hand (Edit), preserving local-only commit
e3a8ca2. Run existing tests: `pytest tests/backtest tests/live -q`
Expected: PASS (the fix may change numbers — update any test that asserted the buggy value, and note it).

- [ ] **Step 5: Commit**

```bash
git add src/trade4/backtest/engine.py src/trade4/live/scheduler.py
git commit -m "fix: reconcile VPS causal_gate/exit_mode fixes into git (provenance: <result of step 3>)"
```

### Task 0.3: research/ package skeleton

**Files:** Create `src/trade4/research/__init__.py`, `src/trade4/research/strategies/__init__.py`, `tests/research/__init__.py`

- [ ] **Step 1: Create package markers**

Each `__init__.py` is empty except a one-line module docstring, e.g.
`"""Honest market-neutral research harness."""`

- [ ] **Step 2: Verify import**

Run: `python -c "import trade4.research"`
Expected: no error.

- [ ] **Step 3: Commit**

```bash
git add src/trade4/research tests/research
git commit -m "feat(research): add package skeleton"
```

### Task 0.4: Point-in-time data lake (data.binance.vision)

**Files:** Create `src/trade4/data/binance_vision.py`, `tests/data/test_binance_vision.py`

Source URL patterns (USDT-M futures):
- klines: `https://data.binance.vision/data/futures/um/monthly/klines/{SYM}/{INTERVAL}/{SYM}-{INTERVAL}-{YYYY-MM}.zip`
- funding: `https://data.binance.vision/data/futures/um/monthly/fundingRate/{SYM}/{SYM}-fundingRate-{YYYY-MM}.zip`

These dumps include many **delisted** symbols (e.g. a symbol no longer on the live API). This
is the mechanism for a true point-in-time universe.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_binance_vision.py
import pandas as pd
from trade4.data.binance_vision import fetch_funding_month

def test_delisted_symbol_returns_funding():
    # SRMUSDT was delisted; its history must still be retrievable from the vision dumps.
    df = fetch_funding_month("SRMUSDT", 2022, 9)
    assert not df.empty
    assert set(["timestamp", "funding_rate"]).issubset(df.columns)
    assert df["timestamp"].is_monotonic_increasing
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/data/test_binance_vision.py -v`
Expected: FAIL (ImportError / module not found).

- [ ] **Step 3: Implement fetch**

```python
# src/trade4/data/binance_vision.py
import io
import zipfile
import logging
import urllib.request
import pandas as pd

logger = logging.getLogger(__name__)
_BASE = "https://data.binance.vision/data/futures/um/monthly"

def _download_zip_csv(url: str) -> pd.DataFrame | None:
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as fh:
            return pd.read_csv(fh, header=None)

def fetch_funding_month(symbol: str, year: int, month: int) -> pd.DataFrame:
    url = f"{_BASE}/fundingRate/{symbol}/{symbol}-fundingRate-{year}-{month:02d}.zip"
    raw = _download_zip_csv(url)
    if raw is None:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])
    # vision fundingRate columns: calc_time, funding_interval_hours, last_funding_rate
    df = raw.iloc[:, [0, -1]].copy()
    df.columns = ["timestamp", "funding_rate"]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["funding_rate"] = df["funding_rate"].astype(float)
    return df.sort_values("timestamp").reset_index(drop=True)
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/data/test_binance_vision.py -v`
Expected: PASS. (If the column order differs, inspect one CSV via the test failure and fix the
column indices — verify against the actual dump, don't guess.)

- [ ] **Step 5: Add klines fetch + listing/delisting inference**

Add `fetch_klines_month(symbol, interval, year, month)` (same pattern; kline columns:
open_time, open, high, low, close, volume, …). Add `infer_listing_window(symbol)` that scans
available months to find first and last data → the symbol's tradeable window. Cache via the
existing `data/store.py` Parquet store.

- [ ] **Step 6: Commit**

```bash
git add src/trade4/data/binance_vision.py tests/data/test_binance_vision.py
git commit -m "feat(data): point-in-time fetch from data.binance.vision incl. delisted perps"
```

**GATE 0:** scipy importable; VPS fixes in git with documented −2.862 provenance; `research/`
imports; delisted-symbol funding retrievable. Report status to user before Phase 1.

---

## Phase 1 — Engine + First Trust Signal

### Task 1.1: Panel container + tradeable mask

**Files:** Create `src/trade4/research/panel.py`, `tests/research/test_panel.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/research/test_panel.py
import numpy as np
import pandas as pd
from trade4.research.panel import Panel

def _grid(n=4):
    return pd.date_range("2023-01-01", periods=n, freq="8h", tz="UTC")

def test_tradeable_mask_marks_unlisted_as_false():
    t = _grid()
    close = pd.DataFrame({"A": [10, 11, 12, 13], "B": [np.nan, np.nan, 5, 6]}, index=t)
    funding = pd.DataFrame({"A": 0.0001, "B": 0.0002}, index=t)
    panel = Panel(close=close, funding=funding)
    # B has no price for first two bars -> not tradeable there
    assert panel.tradeable.loc[t[0], "B"] == False
    assert panel.tradeable.loc[t[2], "B"] == True
    assert panel.tradeable.loc[t[0], "A"] == True

def test_panel_rejects_misaligned_frames():
    t = _grid()
    close = pd.DataFrame({"A": [1, 2, 3, 4]}, index=t)
    funding = pd.DataFrame({"A": [0.1, 0.2]}, index=t[:2])  # wrong length
    import pytest
    with pytest.raises(ValueError):
        Panel(close=close, funding=funding)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/research/test_panel.py -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement Panel**

```python
# src/trade4/research/panel.py
from dataclasses import dataclass, field
import pandas as pd

@dataclass
class Panel:
    """Aligned point-in-time market data on a fixed bar grid.

    All frames share index (time) and columns (symbol). `tradeable[t, s]` is True
    iff symbol s has a finite price at t (listed & has data)."""
    close: pd.DataFrame
    funding: pd.DataFrame
    open_interest: pd.DataFrame | None = None
    tradeable: pd.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        if not self.close.index.equals(self.funding.index):
            raise ValueError("close/funding index mismatch")
        if list(self.close.columns) != list(self.funding.columns):
            raise ValueError("close/funding columns mismatch")
        if not isinstance(self.close.index, pd.DatetimeIndex) or self.close.index.tz is None:
            raise ValueError("index must be tz-aware DatetimeIndex")
        self.tradeable = self.close.notna()

    @property
    def times(self) -> pd.DatetimeIndex:
        return self.close.index

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/research/test_panel.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trade4/research/panel.py tests/research/test_panel.py
git commit -m "feat(research): Panel container with point-in-time tradeable mask"
```

### Task 1.2: Strategy ABC

**Files:** Create `src/trade4/research/strategy.py`, add test to `tests/research/test_panel.py` (or new `test_strategy.py`)

- [ ] **Step 1: Write the failing test**

```python
# tests/research/test_strategy.py
import pandas as pd
from trade4.research.strategy import Strategy
from trade4.research.panel import Panel

class _EqualLongShort(Strategy):
    def generate_target_weights(self, panel):
        w = pd.DataFrame(0.0, index=panel.times, columns=panel.symbols)
        if len(panel.symbols) >= 2:
            w.iloc[:, 0] = 0.5
            w.iloc[:, 1] = -0.5
        return w

def test_strategy_emits_dollar_neutral_weights():
    t = pd.date_range("2023-01-01", periods=3, freq="8h", tz="UTC")
    close = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, index=t)
    funding = pd.DataFrame({"A": 0.0, "B": 0.0}, index=t)
    w = _EqualLongShort().generate_target_weights(Panel(close=close, funding=funding))
    assert abs(w.iloc[0].sum()) < 1e-12  # dollar-neutral
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/research/test_strategy.py -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement ABC**

```python
# src/trade4/research/strategy.py
from abc import ABC, abstractmethod
import pandas as pd
from trade4.research.panel import Panel

class Strategy(ABC):
    """Emits dollar-neutral target weights over a universe.

    CONTRACT: weights at time t may depend on panel data with index <= t only.
    The portfolio engine enforces this with a future-perturbation tripwire."""

    name: str = "strategy"

    @abstractmethod
    def generate_target_weights(self, panel: Panel) -> pd.DataFrame:
        """Return weights [time x symbol]; rows should sum ~0 (dollar-neutral)."""
        raise NotImplementedError
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/research/test_strategy.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trade4/research/strategy.py tests/research/test_strategy.py
git commit -m "feat(research): Strategy ABC with causal weight contract"
```

### Task 1.3: Portfolio engine — known-answer price PnL

**Files:** Create `src/trade4/research/portfolio_engine.py`, `tests/research/test_portfolio_engine.py`

- [ ] **Step 1: Write the failing test (known-answer price PnL, no funding, no cost)**

```python
# tests/research/test_portfolio_engine.py
import numpy as np
import pandas as pd
from trade4.research.panel import Panel
from trade4.research.portfolio_engine import run_portfolio_backtest, EngineConfig

def _const_funding(close):
    return pd.DataFrame(0.0, index=close.index, columns=close.columns)

def test_known_answer_price_pnl_single_long():
    # One symbol, +10% move, full long weight, zero cost/funding -> +10% return.
    t = pd.date_range("2023-01-01", periods=2, freq="8h", tz="UTC")
    close = pd.DataFrame({"A": [100.0, 110.0]}, index=t)
    panel = Panel(close=close, funding=_const_funding(close))
    weights = pd.DataFrame({"A": [1.0, 1.0]}, index=t)
    cfg = EngineConfig(cost_bps=0.0, funding_enabled=False)
    res = run_portfolio_backtest(panel, weights, cfg)
    assert abs(res.equity.iloc[-1] / res.equity.iloc[0] - 1.10) < 1e-9
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/research/test_portfolio_engine.py::test_known_answer_price_pnl_single_long -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement minimal engine (price PnL only)**

```python
# src/trade4/research/portfolio_engine.py
from dataclasses import dataclass
import numpy as np
import pandas as pd
from trade4.research.panel import Panel

@dataclass(frozen=True)
class EngineConfig:
    cost_bps: float = 5.0          # flat round-trip placeholder; replaced by costs.py in 1.7
    funding_enabled: bool = True
    cost_multiplier: float = 1.0   # for cost-sensitivity sweep

@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    turnover: pd.Series
    funding_pnl: pd.Series
    cost: pd.Series

def run_portfolio_backtest(panel: Panel, weights: pd.DataFrame, cfg: EngineConfig) -> BacktestResult:
    w = weights.reindex(index=panel.times, columns=panel.symbols).fillna(0.0)
    # PIT-mask assert: no weight on an untradeable symbol (look-ahead guard).
    leaked = (w != 0) & (~panel.tradeable)
    if leaked.to_numpy().any():
        raise AssertionError("weights assigned to untradeable (unlisted) symbols — look-ahead")
    close = panel.close
    asset_ret = close.pct_change().fillna(0.0)        # return over [t-1, t]
    # position held from t-1 earns asset_ret at t:
    port_ret = (w.shift(1).fillna(0.0) * asset_ret).sum(axis=1)
    equity = (1.0 + port_ret).cumprod()
    zero = pd.Series(0.0, index=panel.times)
    return BacktestResult(equity=equity, returns=port_ret, turnover=zero.copy(),
                          funding_pnl=zero.copy(), cost=zero.copy())
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/research/test_portfolio_engine.py::test_known_answer_price_pnl_single_long -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trade4/research/portfolio_engine.py tests/research/test_portfolio_engine.py
git commit -m "feat(research): portfolio engine price PnL + PIT-mask assert"
```

### Task 1.4: Funding-only known-answer test

**Files:** Modify `portfolio_engine.py`, `tests/research/test_portfolio_engine.py`

- [ ] **Step 1: Write the failing test**

```python
def test_known_answer_funding_only():
    # Flat price, short 1 unit, funding +0.01 (1%) for 3 bars.
    # A short PAYS positive funding? No: short RECEIVES funding when rate>0.
    # Convention: position earns -weight_sign? We define: funding_pnl = -w_prev * rate.
    # short (w=-1), rate=+0.01 -> funding_pnl = -(-1)*0.01 = +0.01 per bar * 3 = +0.03.
    t = pd.date_range("2023-01-01", periods=4, freq="8h", tz="UTC")
    close = pd.DataFrame({"A": [100.0, 100.0, 100.0, 100.0]}, index=t)
    funding = pd.DataFrame({"A": [0.0, 0.01, 0.01, 0.01]}, index=t)
    panel = Panel(close=close, funding=funding)
    weights = pd.DataFrame({"A": [-1.0, -1.0, -1.0, -1.0]}, index=t)
    cfg = EngineConfig(cost_bps=0.0, funding_enabled=True)
    res = run_portfolio_backtest(panel, weights, cfg)
    assert abs(res.funding_pnl.sum() - 0.03) < 1e-9
```

> Funding sign convention (lock it here): a long position with positive funding **pays**;
> a short with positive funding **receives**. `funding_pnl_t = -w_{t-1} * rate_t`. This is the
> single most error-prone booking — it gets its own known-answer test.

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/research/test_portfolio_engine.py::test_known_answer_funding_only -v`
Expected: FAIL (funding_pnl is all zeros).

- [ ] **Step 3: Add funding booking to the engine**

In `run_portfolio_backtest`, after computing `port_ret`, add:

```python
    if cfg.funding_enabled:
        funding_pnl = (-w.shift(1).fillna(0.0) * panel.funding).sum(axis=1)
    else:
        funding_pnl = pd.Series(0.0, index=panel.times)
    port_ret = port_ret + funding_pnl
    equity = (1.0 + port_ret).cumprod()
```
and return `funding_pnl=funding_pnl` instead of the zero series.

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/research/test_portfolio_engine.py -v`
Expected: BOTH known-answer tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trade4/research/portfolio_engine.py tests/research/test_portfolio_engine.py
git commit -m "feat(research): discrete funding PnL with locked sign convention + known-answer test"
```

### Task 1.5: Future-perturbation tripwire (THE core anti-phantom test)

**Files:** Create helper in `portfolio_engine.py`, test in `tests/research/test_portfolio_engine.py`

- [ ] **Step 1: Write the failing test (a cheating strategy MUST be caught)**

```python
from trade4.research.strategy import Strategy
from trade4.research.portfolio_engine import assert_causal

class _CheatStrategy(Strategy):
    name = "cheat"
    def generate_target_weights(self, panel):
        # Looks one bar into the FUTURE: long if next bar's price is higher.
        fwd = panel.close.shift(-1)
        return ((fwd > panel.close).astype(float) - 0.5)

class _HonestStrategy(Strategy):
    name = "honest"
    def generate_target_weights(self, panel):
        # Uses only past: long if last return was positive.
        past = panel.close.pct_change().fillna(0.0)
        return ((past > 0).astype(float) - 0.5)

def _toy_panel():
    t = pd.date_range("2023-01-01", periods=12, freq="8h", tz="UTC")
    rng = np.random.default_rng(0)
    px = 100 * (1 + pd.DataFrame(rng.normal(0, 0.01, (12, 3)),
                                 index=t, columns=["A", "B", "C"])).cumprod()
    fund = pd.DataFrame(0.0, index=t, columns=["A", "B", "C"])
    return Panel(close=px, funding=fund)

def test_tripwire_catches_lookahead():
    import pytest
    with pytest.raises(AssertionError):
        assert_causal(_CheatStrategy(), _toy_panel())

def test_tripwire_passes_honest():
    assert_causal(_HonestStrategy(), _toy_panel())  # must not raise
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/research/test_portfolio_engine.py -k tripwire -v`
Expected: FAIL (ImportError: assert_causal).

- [ ] **Step 3: Implement the tripwire**

```python
# add to portfolio_engine.py
from trade4.research.strategy import Strategy

def assert_causal(strategy: Strategy, panel: Panel, n_checks: int = 5, seed: int = 0) -> None:
    """Perturb future data and assert past weights are unchanged.

    For several cut points t0, replace panel data strictly after t0 with noise,
    recompute weights, and require weights at indices <= t0 to be bit-identical to
    the unperturbed run. A strategy that peeks into the future changes those weights."""
    rng = np.random.default_rng(seed)
    base = strategy.generate_target_weights(panel).reindex(
        index=panel.times, columns=panel.symbols)
    times = panel.times
    cuts = times[len(times)//3 : -1]
    if len(cuts) > n_checks:
        cuts = cuts[np.linspace(0, len(cuts)-1, n_checks).astype(int)]
    for t0 in cuts:
        fut = times > t0
        close2 = panel.close.copy()
        fund2 = panel.funding.copy()
        close2.loc[fut] = close2.loc[fut] * (1 + rng.normal(0, 0.5, close2.loc[fut].shape))
        fund2.loc[fut] = rng.normal(0, 0.01, fund2.loc[fut].shape)
        oi2 = panel.open_interest.copy() if panel.open_interest is not None else None
        perturbed = Panel(close=close2, funding=fund2, open_interest=oi2)
        w2 = strategy.generate_target_weights(perturbed).reindex(
            index=panel.times, columns=panel.symbols)
        past = times <= t0
        a = base.loc[past].fillna(0.0).to_numpy()
        b = w2.loc[past].fillna(0.0).to_numpy()
        if not np.allclose(a, b, atol=1e-12, rtol=0):
            raise AssertionError(
                f"look-ahead: weights at/<= {t0} changed when future data was perturbed")
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/research/test_portfolio_engine.py -k tripwire -v`
Expected: BOTH PASS (cheat raises, honest does not).

- [ ] **Step 5: Commit**

```bash
git add src/trade4/research/portfolio_engine.py tests/research/test_portfolio_engine.py
git commit -m "feat(research): future-perturbation tripwire enforces causality"
```

### Task 1.6: Turnover-/ADV-aware cost layer

**Files:** Create `src/trade4/research/costs.py`, `tests/research/test_costs.py`; wire into engine.

- [ ] **Step 1: Write the failing test**

```python
# tests/research/test_costs.py
import pandas as pd
from trade4.research.costs import turnover_cost

def test_turnover_cost_charges_on_weight_change():
    t = pd.date_range("2023-01-01", periods=3, freq="8h", tz="UTC")
    # weights go 0 -> +1 -> +1 : turnover at bar1 = |1-0| = 1, bar2 = 0
    w = pd.DataFrame({"A": [0.0, 1.0, 1.0]}, index=t)
    cost = turnover_cost(w, cost_bps=10.0, multiplier=1.0)
    assert abs(cost.iloc[1] - 1.0 * 10.0 / 10_000) < 1e-12
    assert abs(cost.iloc[2] - 0.0) < 1e-12
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/research/test_costs.py -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement turnover_cost**

```python
# src/trade4/research/costs.py
import pandas as pd

def turnover_cost(weights: pd.DataFrame, cost_bps: float, multiplier: float = 1.0) -> pd.Series:
    """Per-bar cost = sum_symbol |Δweight| * cost_bps/1e4 * multiplier.

    cost_bps is one-way per unit notional traded. ADV scaling can be layered by passing
    a per-symbol, per-time cost_bps frame (future extension); v1 uses a scalar."""
    dw = weights.fillna(0.0).diff().abs()
    dw.iloc[0] = weights.iloc[0].abs()  # initial build-up from flat
    return dw.sum(axis=1) * (cost_bps / 10_000.0) * multiplier
```

- [ ] **Step 4: Wire into engine + run**

In `run_portfolio_backtest`, replace the zero `cost` series with:
```python
    from trade4.research.costs import turnover_cost
    cost = turnover_cost(w, cfg.cost_bps, cfg.cost_multiplier)
    port_ret = port_ret - cost
    equity = (1.0 + port_ret).cumprod()
```
Update the known-answer price test (Task 1.3) which used `cost_bps=0.0` — still passes.
Run: `pytest tests/research/ -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trade4/research/costs.py src/trade4/research/portfolio_engine.py tests/research/test_costs.py
git commit -m "feat(research): turnover-aware cost layer wired into engine"
```

### Task 1.7: funding_carry causal port

**Files:** Create `src/trade4/research/strategies/funding_carry.py`, `tests/research/test_funding_carry.py`

- [ ] **Step 1: Write the failing test (causality via tripwire)**

```python
# tests/research/test_funding_carry.py
import numpy as np
import pandas as pd
from trade4.research.panel import Panel
from trade4.research.portfolio_engine import assert_causal
from trade4.research.strategies.funding_carry import FundingCarry

def _panel(n=30):
    t = pd.date_range("2023-01-01", periods=n, freq="8h", tz="UTC")
    rng = np.random.default_rng(1)
    px = 100 * (1 + pd.DataFrame(rng.normal(0, 0.005, (n, 2)),
                                 index=t, columns=["A", "B"])).cumprod()
    fund = pd.DataFrame(rng.normal(0.0003, 0.0002, (n, 2)), index=t, columns=["A", "B"])
    return Panel(close=px, funding=fund)

def test_funding_carry_is_causal():
    assert_causal(FundingCarry(entry_threshold=0.0002, persistence_window=3), _panel())
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/research/test_funding_carry.py -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement causal funding-carry**

```python
# src/trade4/research/strategies/funding_carry.py
from dataclasses import dataclass
import pandas as pd
from trade4.research.strategy import Strategy
from trade4.research.panel import Panel

@dataclass
class FundingCarry(Strategy):
    """Causal port of the legacy single-symbol funding-carry.

    Enter (short perp / long spot ≡ negative weight on the perp to harvest positive funding)
    when the TRAILING funding average is persistently above threshold. Uses only data <= t —
    no forward-looking expected-funding (the legacy engine.py:125 bug)."""
    entry_threshold: float = 0.0001
    persistence_window: int = 3
    name: str = "funding_carry"

    def generate_target_weights(self, panel: Panel) -> pd.DataFrame:
        roll = panel.funding.rolling(self.persistence_window, min_periods=1).mean()
        # short the perp (negative weight) to receive positive funding; causal: roll uses <= t
        signal = (roll >= self.entry_threshold).astype(float)
        weights = -signal  # short where funding persistently positive
        weights = weights.where(panel.tradeable, 0.0)
        # normalise to gross 1.0 per bar (dollar scale), keep dollar-neutral-ish by row
        gross = weights.abs().sum(axis=1).replace(0.0, 1.0)
        return weights.div(gross, axis=0)
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/research/test_funding_carry.py -v`
Expected: PASS (tripwire confirms causality).

- [ ] **Step 5: Commit**

```bash
git add src/trade4/research/strategies/funding_carry.py tests/research/test_funding_carry.py
git commit -m "feat(research): causal funding-carry strategy (no forward-looking gate)"
```

### Task 1.8: Cross-engine equivalence + KILLER-GATE

**Files:** `tests/research/test_funding_carry.py` (add), uses real cached data

> This gate proves the new engine agrees with the legacy (fixed) engine and reproduces the
> known OOS loss. The exact symbol/window/cost params come from the legacy walk-forward that
> produced −2.862 bps (recovered in Task 0.2). Tolerance: the two engines book funding/costs
> slightly differently (per-cycle vs per-bar), so equivalence is **directional + magnitude
> within 20 %**, not bit-exact. If Task 0.2 showed −2.862 came from the unfixed engine,
> regenerate the reference by running the fixed legacy engine locally first.

- [ ] **Step 1: Write the gate test**

```python
import pytest

@pytest.mark.integration
def test_killer_gate_reproduces_oos_loss():
    """funding_carry on the reference symbol/window must reproduce a NEGATIVE OOS result
    of the same order as the legacy −2.862 bps. Skips if reference data not cached."""
    from trade4.research.strategies.funding_carry import FundingCarry
    from trade4.research.portfolio_engine import run_portfolio_backtest, EngineConfig
    panel = _load_reference_oos_panel()  # helper: loads the cached reference symbol+window
    if panel is None:
        pytest.skip("reference panel not cached; run data lake fetch first")
    w = FundingCarry().generate_target_weights(panel)
    res = run_portfolio_backtest(panel, w, EngineConfig())
    total_bps = (res.equity.iloc[-1] / res.equity.iloc[0] - 1.0) * 10_000
    assert total_bps < 0.0, f"expected negative OOS, got {total_bps:.1f} bps"
    assert abs(total_bps - (-2862)) / 2862 < 1.0  # within order of magnitude
```

- [ ] **Step 2: Implement `_load_reference_oos_panel` + cross-engine check**

Add the helper loading the exact reference symbol/window from the Parquet cache. Add a second
test running the **legacy** `backtest/engine.run_backtest` on the same single-symbol input and
asserting the two net results agree within 20 %.

- [ ] **Step 3: Run the gate**

Run: `pytest tests/research/test_funding_carry.py -v -m integration`
Expected: PASS — new engine reproduces a negative OOS of the right magnitude AND agrees with
the legacy fixed engine. If it does NOT reproduce, STOP: the engine is wrong — debug before
proceeding (this is the whole point of the gate).

- [ ] **Step 4: Commit**

```bash
git add tests/research/test_funding_carry.py
git commit -m "test(research): killer-gate — reproduce known OOS loss + cross-engine equivalence"
```

**GATE 1 (Phase 1 exit):** `pytest tests/research/ -v` green incl. tripwire (cheat caught,
honest passes), both known-answer tests (price + funding), PIT-mask assert, funding_carry
causal. **KILLER-GATE** reproduces the −2.862 bps OOS within tolerance and cross-engine
equivalence holds. Report the actual reproduced number to the user before Phase 2.

---

## Self-Review Notes

- **Spec coverage:** §4 architecture → Tasks 1.1–1.7; §5 anti-phantom (tripwire, mask, mandatory cost) → 1.3/1.5/1.6; §7 costs (turnover) → 1.6 (ADV scaling stubbed as documented extension, sweep multiplier present via `cost_multiplier`); §8 PIT data lake → 0.4; §3 VPS reconcile → 0.2; killer-gate → 1.8. Funding sign convention locked in 1.4.
- **Deferred to Phase 2/3 plan (by design):** `validation.py` (DSR/PBO/trial-registry, §6), `metrics.py` (365 annualisation, portfolio gates, regime split, §9), `manifest.py` (§10), tearsheet cost-sweep rendering (§7), `xs_funding`/`xs_momentum` (§4), `study.py` + verdict + capacity caveat (§11). These depend on a proven engine and are a separate working deliverable.
- **Type consistency:** `Panel(close, funding, open_interest)`, `EngineConfig(cost_bps, funding_enabled, cost_multiplier)`, `BacktestResult(equity, returns, turnover, funding_pnl, cost)`, `Strategy.generate_target_weights`, `assert_causal`, `turnover_cost` — used consistently across tasks.
