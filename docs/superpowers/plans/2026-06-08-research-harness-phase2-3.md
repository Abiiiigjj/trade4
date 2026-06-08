# Research Harness — Phase 2 + 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the research-grade validation layer (multi-window walk-forward, Deflated Sharpe, PBO, trial registry, run manifest, regime-conditioned tearsheet), build the heterogeneous point-in-time panel builder and the two cross-sectional strategies, then run the first honest study and deliver a verdict.

**Architecture:** `validation.py` orchestrates IS-optimise → OOS-gate over rolling windows, logging every evaluated config to a `TrialRegistry`; `metrics.py` provides crypto-correct (365) Sharpe, PSR/DSR, and a regime classifier; `PBO` runs CSCV over the registry matrix. `panel_builder.py` assembles a ragged real universe from `binance_vision`. `xs_funding`/`xs_momentum` are dollar-neutral perp strategies. `study.py` ranks strategies by DSR and writes tearsheets + verdict.

**Tech Stack:** Python 3.12, pandas, numpy, scipy.stats, pytest. Builds on the Phase 0+1 engine (`Panel`, `Strategy`, `run_portfolio_backtest`, `assert_causal`).

**Depends on:** Phase 0+1 (done, branch `research-harness`). The funding_carry **killer-gate (Phase-1 Task 1.8) and two-leg carry cost are prerequisites only for the carry *baseline* verdict**; the XS strategy verdicts are independent of the VPS reconcile.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `src/trade4/research/metrics.py` | Sharpe(365), max_dd, PSR, DSR, expected-max-Sharpe SR0, regime classifier |
| `src/trade4/research/trial_registry.py` | records every evaluated config; exposes N and the T×N PnL matrix |
| `src/trade4/research/pbo.py` | Probability of Backtest Overfitting via CSCV |
| `src/trade4/research/validation.py` | multi-window walk-forward + portfolio gates, feeds the registry |
| `src/trade4/research/manifest.py` | run manifest: data hash + git commit + seed + params |
| `src/trade4/research/panel_builder.py` | assemble a ragged PIT `Panel` from `binance_vision` fetches |
| `src/trade4/research/tearsheet.py` | render per-strategy tearsheet: IS/OOS, cost sweep, regime split, DSR |
| `src/trade4/research/strategies/xs_funding.py` | long lowest / short highest funding (dollar-neutral) |
| `src/trade4/research/strategies/xs_momentum.py` | long winners / short losers, weekly (dollar-neutral) |
| `src/trade4/research/study.py` | run all strategies, rank by DSR, write verdict |
| `tests/research/test_metrics.py` | PSR/DSR/Sharpe known-answers |
| `tests/research/test_pbo.py` | PBO behavioural known-answers (noise→~0.5, dominant→~0) |
| `tests/research/test_trial_registry.py` | N counting, matrix shape |
| `tests/research/test_validation.py` | window splitting, gate logic, registry integration |
| `tests/research/test_manifest.py` | determinism, data-hash stability |
| `tests/research/test_panel_builder.py` | **misaligned funding schedules**, ragged listing |
| `tests/research/test_xs_funding.py` | causality (tripwire), dollar-neutrality, sign |
| `tests/research/test_xs_momentum.py` | causality (tripwire), dollar-neutrality |

Modify: `src/trade4/research/portfolio_engine.py` (+`n_legs` cost), `src/trade4/research/costs.py` (+`n_legs`).

---

## Phase 2 — Validation, Statistics, Reporting

### Task 2.1: Crypto-correct metrics (Sharpe 365, max_dd)

**Files:** Create `src/trade4/research/metrics.py`, `tests/research/test_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/research/test_metrics.py
import numpy as np
import pandas as pd
from trade4.research.metrics import sharpe_ratio, max_drawdown, PERIODS_PER_YEAR

def test_periods_per_year_is_crypto_8h():
    # 8h bars, 24/7 -> 3 per day * 365 = 1095
    assert PERIODS_PER_YEAR["8h"] == 1095

def test_sharpe_zero_mean_is_zero():
    r = pd.Series([0.01, -0.01, 0.01, -0.01])
    assert abs(sharpe_ratio(r, bar="8h")) < 1e-9

def test_max_drawdown_simple():
    eq = pd.Series([1.0, 1.2, 0.9, 1.1])
    # peak 1.2 -> trough 0.9 => dd = (0.9-1.2)/1.2 = -0.25
    assert abs(max_drawdown(eq) - (-0.25)) < 1e-12
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/research/test_metrics.py -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement metrics**

```python
# src/trade4/research/metrics.py
import numpy as np
import pandas as pd

# bars per year for crypto (24/7, 365 days)
PERIODS_PER_YEAR = {"1h": 24 * 365, "4h": 6 * 365, "8h": 3 * 365, "1d": 365}

def sharpe_ratio(returns: pd.Series, bar: str = "8h") -> float:
    r = returns.dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    ann = PERIODS_PER_YEAR[bar] ** 0.5
    return float(r.mean() / r.std(ddof=1) * ann)

def max_drawdown(equity: pd.Series) -> float:
    eq = equity.dropna()
    if eq.empty:
        return 0.0
    running_max = eq.cummax()
    return float(((eq - running_max) / running_max).min())
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/research/test_metrics.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trade4/research/metrics.py tests/research/test_metrics.py
git commit -m "feat(research): crypto-correct Sharpe (365) and max drawdown"
```

### Task 2.2: Probabilistic & Deflated Sharpe Ratio

**Files:** Modify `src/trade4/research/metrics.py`, `tests/research/test_metrics.py`

Formulas (López de Prado 2014, "The Deflated Sharpe Ratio"):
- PSR(SR*) = Φ[ (ŜR − SR*)·√(T−1) / √(1 − γ3·ŜR + ((γ4−1)/4)·ŜR²) ]
  where ŜR is the **per-period** (non-annualised) Sharpe, T = number of returns, γ3 = skew, γ4 = kurtosis (normal = 3), Φ = standard-normal CDF.
- DSR = PSR(SR0), with the expected maximum Sharpe under the null of N trials:
  SR0 = √V · [ (1−γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) ]
  where V = variance of the per-period Sharpes across the N trials, γ = Euler–Mascheroni ≈ 0.5772156649.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/research/test_metrics.py
import math
from trade4.research.metrics import (
    probabilistic_sharpe_ratio, expected_max_sharpe, deflated_sharpe_ratio,
)

def test_psr_half_when_observed_equals_benchmark():
    # ŜR == SR* -> numerator 0 -> Φ(0) = 0.5
    r = pd.Series(np.random.default_rng(0).normal(0.001, 0.01, 500))
    sr_hat = r.mean() / r.std(ddof=1)  # per-period
    psr = probabilistic_sharpe_ratio(r, sr_benchmark=sr_hat)
    assert abs(psr - 0.5) < 1e-6

def test_psr_increases_with_more_data():
    rng = np.random.default_rng(1)
    short = pd.Series(rng.normal(0.001, 0.01, 100))
    long = pd.Series(rng.normal(0.001, 0.01, 2000))
    assert (probabilistic_sharpe_ratio(long, 0.0)
            > probabilistic_sharpe_ratio(short, 0.0))

def test_expected_max_sharpe_grows_with_trials():
    v = 0.01  # variance of trial Sharpes
    assert expected_max_sharpe(v, n_trials=100) > expected_max_sharpe(v, n_trials=2)

def test_dsr_below_psr_when_many_trials():
    r = pd.Series(np.random.default_rng(2).normal(0.002, 0.01, 1000))
    psr0 = probabilistic_sharpe_ratio(r, 0.0)
    dsr = deflated_sharpe_ratio(r, trial_sharpe_var=0.02, n_trials=50)
    assert dsr < psr0  # deflation reduces the score
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/research/test_metrics.py -k "psr or sharpe or dsr" -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement**

```python
# add to src/trade4/research/metrics.py
from scipy.stats import norm

_EULER = 0.5772156649015329

def probabilistic_sharpe_ratio(returns: pd.Series, sr_benchmark: float = 0.0) -> float:
    """PSR: probability that the true per-period Sharpe exceeds sr_benchmark."""
    r = returns.dropna()
    t = len(r)
    if t < 3 or r.std(ddof=1) == 0:
        return 0.0
    sr = r.mean() / r.std(ddof=1)            # per-period Sharpe
    skew = float(r.skew())
    kurt = float(r.kurtosis()) + 3.0         # pandas gives EXCESS kurtosis -> add 3
    denom = (1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr ** 2) ** 0.5
    if denom == 0:
        return 0.0
    z = (sr - sr_benchmark) * ((t - 1) ** 0.5) / denom
    return float(norm.cdf(z))

def expected_max_sharpe(trial_sharpe_var: float, n_trials: int) -> float:
    """Expected maximum per-period Sharpe under the null across n_trials (SR0)."""
    if n_trials <= 1 or trial_sharpe_var <= 0:
        return 0.0
    n = n_trials
    term = ((1 - _EULER) * norm.ppf(1 - 1.0 / n)
            + _EULER * norm.ppf(1 - 1.0 / (n * math.e)))
    return float((trial_sharpe_var ** 0.5) * term)

def deflated_sharpe_ratio(returns: pd.Series, trial_sharpe_var: float, n_trials: int) -> float:
    """DSR = PSR evaluated against the expected-max-Sharpe benchmark SR0."""
    sr0 = expected_max_sharpe(trial_sharpe_var, n_trials)
    return probabilistic_sharpe_ratio(returns, sr_benchmark=sr0)
```

Add `import math` at the top of the module.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/research/test_metrics.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trade4/research/metrics.py tests/research/test_metrics.py
git commit -m "feat(research): probabilistic + deflated Sharpe ratio (Lopez de Prado)"
```

### Task 2.3: Trial registry

**Files:** Create `src/trade4/research/trial_registry.py`, `tests/research/test_trial_registry.py`

The DSR/PBO deflation is only honest if N counts **every configuration ever evaluated**, not just the final strategies. The registry is the single sink every backtest run reports to.

- [ ] **Step 1: Write the failing test**

```python
# tests/research/test_trial_registry.py
import numpy as np
import pandas as pd
from trade4.research.trial_registry import TrialRegistry

def test_registry_counts_every_trial():
    reg = TrialRegistry()
    idx = pd.date_range("2023-01-01", periods=10, freq="8h", tz="UTC")
    for i in range(7):
        reg.record(label=f"cfg{i}", returns=pd.Series(np.zeros(10), index=idx))
    assert reg.n_trials == 7

def test_registry_builds_aligned_matrix():
    reg = TrialRegistry()
    idx = pd.date_range("2023-01-01", periods=5, freq="8h", tz="UTC")
    reg.record("a", pd.Series([0.1] * 5, index=idx))
    reg.record("b", pd.Series([0.2] * 5, index=idx))
    m = reg.pnl_matrix()
    assert m.shape == (5, 2)
    assert list(m.columns) == ["a", "b"]

def test_registry_sharpe_variance_nonnegative():
    reg = TrialRegistry()
    idx = pd.date_range("2023-01-01", periods=50, freq="8h", tz="UTC")
    rng = np.random.default_rng(0)
    for i in range(5):
        reg.record(f"c{i}", pd.Series(rng.normal(0, 0.01, 50), index=idx))
    assert reg.trial_sharpe_variance() >= 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/research/test_trial_registry.py -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement**

```python
# src/trade4/research/trial_registry.py
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

@dataclass
class _Trial:
    label: str
    returns: pd.Series

@dataclass
class TrialRegistry:
    """Records every evaluated config so DSR/PBO see the true number of trials."""
    trials: list[_Trial] = field(default_factory=list)

    def record(self, label: str, returns: pd.Series) -> None:
        self.trials.append(_Trial(label=label, returns=returns))

    @property
    def n_trials(self) -> int:
        return len(self.trials)

    def pnl_matrix(self) -> pd.DataFrame:
        """T x N matrix of per-bar returns, columns = trial labels (outer-joined index)."""
        cols = {t.label: t.returns for t in self.trials}
        return pd.DataFrame(cols)

    def trial_sharpe_variance(self) -> float:
        """Variance of per-period Sharpes across trials (V for SR0)."""
        srs = []
        for t in self.trials:
            r = t.returns.dropna()
            if len(r) > 1 and r.std(ddof=1) > 0:
                srs.append(r.mean() / r.std(ddof=1))
        return float(np.var(srs, ddof=1)) if len(srs) > 1 else 0.0
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/research/test_trial_registry.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trade4/research/trial_registry.py tests/research/test_trial_registry.py
git commit -m "feat(research): trial registry — N counts every evaluated config for DSR/PBO"
```

### Task 2.4: PBO via CSCV

**Files:** Create `src/trade4/research/pbo.py`, `tests/research/test_pbo.py`

CSCV (Bailey et al. 2017): split T rows into S even partitions; for every C(S, S/2) way to pick S/2 as IS (complement OOS), choose the column best by IS Sharpe, find its OOS rank ω∈(0,1), set logit λ=ln(ω/(1−ω)); **PBO = fraction of splits with λ<0** (best-IS config below median OOS).

- [ ] **Step 1: Write the failing test (behavioural known-answers)**

```python
# tests/research/test_pbo.py
import numpy as np
import pandas as pd
from trade4.research.pbo import probability_of_backtest_overfitting

def _matrix(data):
    idx = pd.date_range("2023-01-01", periods=data.shape[0], freq="8h", tz="UTC")
    return pd.DataFrame(data, index=idx, columns=[f"c{i}" for i in range(data.shape[1])])

def test_pbo_noise_is_near_half():
    rng = np.random.default_rng(0)
    m = _matrix(rng.normal(0, 0.01, (240, 8)))
    pbo = probability_of_backtest_overfitting(m, n_splits=8)
    assert 0.3 < pbo < 0.7  # pure noise -> no real edge -> ~0.5

def test_pbo_dominant_config_is_low():
    rng = np.random.default_rng(1)
    data = rng.normal(0, 0.01, (240, 8))
    data[:, 0] += 0.01  # column 0 is genuinely, persistently best
    m = _matrix(data)
    pbo = probability_of_backtest_overfitting(m, n_splits=8)
    assert pbo < 0.2
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/research/test_pbo.py -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement**

```python
# src/trade4/research/pbo.py
import itertools
import numpy as np
import pandas as pd

def _sharpe_cols(block: np.ndarray) -> np.ndarray:
    mean = block.mean(axis=0)
    std = block.std(axis=0, ddof=1)
    std[std == 0] = np.nan
    return mean / std

def probability_of_backtest_overfitting(pnl_matrix: pd.DataFrame, n_splits: int = 10) -> float:
    """CSCV PBO. pnl_matrix: T x N per-bar returns, columns = configs."""
    m = pnl_matrix.dropna(axis=1, how="any").to_numpy()
    t, n = m.shape
    if n < 2:
        return float("nan")
    s = n_splits - (n_splits % 2)  # even
    rows_per = t // s
    if rows_per == 0:
        raise ValueError("not enough rows for the requested number of splits")
    parts = [m[i * rows_per:(i + 1) * rows_per] for i in range(s)]
    logits = []
    for is_idx in itertools.combinations(range(s), s // 2):
        oos_idx = [i for i in range(s) if i not in is_idx]
        is_block = np.vstack([parts[i] for i in is_idx])
        oos_block = np.vstack([parts[i] for i in oos_idx])
        is_sr = _sharpe_cols(is_block)
        oos_sr = _sharpe_cols(oos_block)
        if np.all(np.isnan(is_sr)):
            continue
        best = int(np.nanargmax(is_sr))
        # OOS rank of the IS-best config (1 = worst ... n = best)
        order = pd.Series(oos_sr).rank(method="average").to_numpy()
        rank = order[best]
        omega = rank / (n + 1)
        omega = min(max(omega, 1e-6), 1 - 1e-6)
        logits.append(np.log(omega / (1 - omega)))
    if not logits:
        return float("nan")
    return float(np.mean(np.array(logits) < 0))
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/research/test_pbo.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trade4/research/pbo.py tests/research/test_pbo.py
git commit -m "feat(research): probability of backtest overfitting via CSCV"
```

### Task 2.5: Multi-window walk-forward validation

**Files:** Create `src/trade4/research/validation.py`, `tests/research/test_validation.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/research/test_validation.py
import numpy as np
import pandas as pd
from trade4.research.panel import Panel
from trade4.research.strategy import Strategy
from trade4.research.portfolio_engine import EngineConfig
from trade4.research.trial_registry import TrialRegistry
from trade4.research.validation import walk_forward, WindowResult

class _FixedStrat(Strategy):
    name = "fixed"
    def __init__(self, w=-1.0): self.w = w
    def generate_target_weights(self, panel):
        return pd.DataFrame(self.w, index=panel.times, columns=panel.symbols)

def _panel(n=300):
    t = pd.date_range("2022-01-01", periods=n, freq="8h", tz="UTC")
    rng = np.random.default_rng(0)
    close = pd.DataFrame(100.0, index=t, columns=["A"])
    funding = pd.DataFrame(rng.normal(0.0005, 0.0002, (n, 1)), index=t, columns=["A"])
    return Panel(close=close, funding=funding)

def test_walk_forward_produces_multiple_windows():
    reg = TrialRegistry()
    res = walk_forward(
        _panel(), _FixedStrat, param_grid=[{"w": -1.0}, {"w": -0.5}],
        cfg=EngineConfig(cost_bps=2.0, price_pnl_enabled=False),
        is_bars=120, oos_bars=60, registry=reg,
    )
    assert len(res) >= 2  # several rolling windows
    assert all(isinstance(r, WindowResult) for r in res)
    # registry saw every param for every window
    assert reg.n_trials == sum(2 for _ in res)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/research/test_validation.py -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement**

```python
# src/trade4/research/validation.py
from dataclasses import dataclass
from typing import Any, Callable
import pandas as pd

from trade4.research.panel import Panel
from trade4.research.strategy import Strategy
from trade4.research.portfolio_engine import run_portfolio_backtest, EngineConfig
from trade4.research.trial_registry import TrialRegistry
from trade4.research.metrics import sharpe_ratio, max_drawdown

@dataclass
class WindowResult:
    is_start: pd.Timestamp
    oos_start: pd.Timestamp
    best_params: dict[str, Any]
    is_sharpe: float
    oos_sharpe: float
    oos_max_dd: float
    oos_degradation: float

def _slice(panel: Panel, lo: int, hi: int) -> Panel:
    return Panel(close=panel.close.iloc[lo:hi], funding=panel.funding.iloc[lo:hi],
                 open_interest=None if panel.open_interest is None
                 else panel.open_interest.iloc[lo:hi])

def walk_forward(
    panel: Panel,
    strategy_factory: Callable[..., Strategy],
    param_grid: list[dict[str, Any]],
    cfg: EngineConfig,
    is_bars: int,
    oos_bars: int,
    registry: TrialRegistry,
    bar: str = "8h",
) -> list[WindowResult]:
    """Rolling IS-optimise (by IS Sharpe) -> OOS-evaluate. Logs every config."""
    n = len(panel.times)
    results: list[WindowResult] = []
    start = 0
    while start + is_bars + oos_bars <= n:
        is_panel = _slice(panel, start, start + is_bars)
        oos_panel = _slice(panel, start + is_bars, start + is_bars + oos_bars)
        best = None
        for params in param_grid:
            strat = strategy_factory(**params)
            w = strat.generate_target_weights(is_panel)
            res = run_portfolio_backtest(is_panel, w, cfg)
            sr = sharpe_ratio(res.returns, bar)
            registry.record(f"{strat.name}|{params}|is{start}", res.returns)
            if best is None or sr > best[1]:
                best = (params, sr)
        bparams, is_sr = best
        strat = strategy_factory(**bparams)
        ow = strat.generate_target_weights(oos_panel)
        ores = run_portfolio_backtest(oos_panel, ow, cfg)
        oos_sr = sharpe_ratio(ores.returns, bar)
        deg = (is_sr - oos_sr) / abs(is_sr) if is_sr != 0 else float("inf")
        results.append(WindowResult(
            is_start=panel.times[start], oos_start=panel.times[start + is_bars],
            best_params=bparams, is_sharpe=is_sr, oos_sharpe=oos_sr,
            oos_max_dd=max_drawdown(ores.equity), oos_degradation=deg,
        ))
        start += oos_bars  # roll forward by the OOS window
    return results

# Portfolio-level go/no-go gate (replaces trade-level scalper gates)
_MIN_OOS_SHARPE = 1.5
_MAX_OOS_DD = -0.25
_MAX_OOS_DEGRADATION = 0.30

def gate_passed(windows: list[WindowResult]) -> bool:
    if not windows:
        return False
    import numpy as np
    mean_oos = np.mean([w.oos_sharpe for w in windows])
    worst_dd = min(w.oos_max_dd for w in windows)
    mean_deg = np.mean([w.oos_degradation for w in windows])
    return bool(mean_oos >= _MIN_OOS_SHARPE and worst_dd >= _MAX_OOS_DD
                and mean_deg <= _MAX_OOS_DEGRADATION)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/research/test_validation.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trade4/research/validation.py tests/research/test_validation.py
git commit -m "feat(research): multi-window walk-forward + portfolio gate, feeds trial registry"
```

### Task 2.6: Run manifest

**Files:** Create `src/trade4/research/manifest.py`, `tests/research/test_manifest.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/research/test_manifest.py
import pandas as pd
from trade4.research.manifest import RunManifest, data_hash

def test_data_hash_is_stable():
    idx = pd.date_range("2023-01-01", periods=3, freq="8h", tz="UTC")
    df = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=idx)
    assert data_hash(df) == data_hash(df.copy())

def test_data_hash_changes_with_data():
    idx = pd.date_range("2023-01-01", periods=3, freq="8h", tz="UTC")
    a = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=idx)
    b = pd.DataFrame({"A": [1.0, 2.0, 3.1]}, index=idx)
    assert data_hash(a) != data_hash(b)

def test_manifest_roundtrip():
    m = RunManifest(data_hash="abc", git_commit="def", seed=7, params={"k": 1})
    d = m.to_dict()
    assert d["data_hash"] == "abc" and d["seed"] == 7
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/research/test_manifest.py -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement**

```python
# src/trade4/research/manifest.py
import hashlib
import subprocess
from dataclasses import dataclass, field, asdict
from typing import Any
import pandas as pd

def data_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[:16]

def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"

@dataclass
class RunManifest:
    data_hash: str
    git_commit: str
    seed: int
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/research/test_manifest.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trade4/research/manifest.py tests/research/test_manifest.py
git commit -m "feat(research): run manifest (data hash + git commit + seed) for reproducibility"
```

### Task 2.7: Regime classifier

**Files:** Modify `src/trade4/research/metrics.py`, `tests/research/test_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/research/test_metrics.py
from trade4.research.metrics import funding_regime

def test_funding_regime_labels_high_and_low():
    idx = pd.date_range("2023-01-01", periods=4, freq="8h", tz="UTC")
    mean_funding = pd.Series([0.0008, 0.0008, 0.0001, 0.0001], index=idx)
    reg = funding_regime(mean_funding, break_even=0.0003)
    assert (reg.iloc[:2] == "high").all()
    assert (reg.iloc[2:] == "low").all()
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/research/test_metrics.py -k regime -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement**

```python
# add to src/trade4/research/metrics.py
def funding_regime(mean_funding: pd.Series, break_even: float = 0.0003) -> pd.Series:
    """Label each bar 'high' / 'low' by whether cross-sectional mean funding clears break-even."""
    return mean_funding.apply(lambda x: "high" if x >= break_even else "low")
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/research/test_metrics.py -k regime -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trade4/research/metrics.py tests/research/test_metrics.py
git commit -m "feat(research): funding-regime classifier for conditional reporting"
```

### Task 2.8: Tearsheet renderer

**Files:** Create `src/trade4/research/tearsheet.py`, `tests/research/test_tearsheet.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/research/test_tearsheet.py
import numpy as np
import pandas as pd
from trade4.research.panel import Panel
from trade4.research.portfolio_engine import EngineConfig
from trade4.research.strategies.funding_carry import FundingCarry
from trade4.research.tearsheet import build_tearsheet

def _panel(n=200):
    t = pd.date_range("2023-01-01", periods=n, freq="8h", tz="UTC")
    rng = np.random.default_rng(0)
    close = pd.DataFrame(100.0, index=t, columns=["A", "B"])
    funding = pd.DataFrame(rng.normal(0.0004, 0.0002, (n, 2)), index=t, columns=["A", "B"])
    return Panel(close=close, funding=funding)

def test_tearsheet_has_cost_sweep_and_regime():
    ts = build_tearsheet(
        FundingCarry(), _panel(),
        EngineConfig(cost_bps=2.0, price_pnl_enabled=False),
        n_trials=10, trial_sharpe_var=0.02,
    )
    assert set(ts["cost_sweep"]) == {1.0, 2.0, 3.0}
    assert "dsr" in ts and "sharpe" in ts and "regime" in ts
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/research/test_tearsheet.py -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement**

```python
# src/trade4/research/tearsheet.py
from typing import Any
import pandas as pd
from dataclasses import replace

from trade4.research.panel import Panel
from trade4.research.strategy import Strategy
from trade4.research.portfolio_engine import run_portfolio_backtest, EngineConfig
from trade4.research.metrics import (
    sharpe_ratio, max_drawdown, deflated_sharpe_ratio, funding_regime,
)

def build_tearsheet(
    strategy: Strategy, panel: Panel, cfg: EngineConfig,
    n_trials: int, trial_sharpe_var: float, bar: str = "8h",
) -> dict[str, Any]:
    w = strategy.generate_target_weights(panel)
    base = run_portfolio_backtest(panel, w, cfg)
    cost_sweep = {}
    for mult in (1.0, 2.0, 3.0):
        res = run_portfolio_backtest(panel, w, replace(cfg, cost_multiplier=mult))
        cost_sweep[mult] = (res.equity.iloc[-1] - 1.0) * 10_000  # net bps
    regime = funding_regime(panel.funding.mean(axis=1))
    by_regime = {}
    for label in ("high", "low"):
        mask = (regime == label).to_numpy()
        r = base.returns[mask]
        by_regime[label] = sharpe_ratio(r, bar) if len(r) > 2 else 0.0
    return {
        "strategy": strategy.name,
        "sharpe": sharpe_ratio(base.returns, bar),
        "max_dd": max_drawdown(base.equity),
        "dsr": deflated_sharpe_ratio(base.returns, trial_sharpe_var, n_trials),
        "net_bps": (base.equity.iloc[-1] - 1.0) * 10_000,
        "cost_sweep": cost_sweep,
        "regime": by_regime,
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/research/test_tearsheet.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trade4/research/tearsheet.py tests/research/test_tearsheet.py
git commit -m "feat(research): tearsheet with cost sweep, regime split, DSR"
```

**GATE 2 (Phase 2 exit):** `.venv/bin/python -m pytest tests/research/ -v` green. PSR=0.5 at benchmark; DSR < PSR under many trials; PBO ~0.5 on noise and <0.2 on a dominant config; walk-forward feeds the registry with the full trial count. Report to user.

---

## Phase 3 — Heterogeneous Universe, XS Strategies, Study

### Task 3.1: Two-leg carry cost (close known-gap #1)

**Files:** Modify `src/trade4/research/costs.py`, `src/trade4/research/portfolio_engine.py`, `tests/research/test_costs.py`

The legacy carry cost charges spot + perp, both sides. `turnover_cost` charged one leg. Add an `n_legs` factor so carry uses 2.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/research/test_costs.py
def test_turnover_cost_two_legs_doubles():
    t = pd.date_range("2023-01-01", periods=2, freq="8h", tz="UTC")
    w = pd.DataFrame({"A": [0.0, 1.0]}, index=t)
    one = turnover_cost(w, cost_bps=10.0, n_legs=1)
    two = turnover_cost(w, cost_bps=10.0, n_legs=2)
    assert abs(two.iloc[1] - 2.0 * one.iloc[1]) < 1e-15
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/research/test_costs.py -k two_legs -v`
Expected: FAIL (unexpected keyword `n_legs`).

- [ ] **Step 3: Implement**

In `costs.py`, add `n_legs: int = 1` to `turnover_cost` and multiply the result by `n_legs`:
```python
def turnover_cost(weights, cost_bps, multiplier=1.0, n_legs=1):
    ...
    return dw.sum(axis=1) * (cost_bps / 10_000.0) * multiplier * n_legs
```
In `portfolio_engine.py`, add `n_legs: int = 1` to `EngineConfig` and pass `cfg.n_legs` into the `turnover_cost(...)` call. funding_carry runs with `EngineConfig(price_pnl_enabled=False, n_legs=2)`.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/research/ -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trade4/research/costs.py src/trade4/research/portfolio_engine.py tests/research/test_costs.py
git commit -m "feat(research): two-leg cost factor (n_legs) for delta-neutral carry"
```

### Task 3.2: Heterogeneous PIT panel builder (close known-gap #2)

**Files:** Create `src/trade4/research/panel_builder.py`, `tests/research/test_panel_builder.py`

The real universe has **misaligned funding schedules** (8h vs 4h/1h on different symbols, ragged listing). The builder must produce a single aligned `Panel` on a chosen grid, with `tradeable` reflecting each symbol's real listing window — never fabricating funding where none existed.

- [ ] **Step 1: Write the failing test (the case the smoke test did NOT cover)**

```python
# tests/research/test_panel_builder.py
import numpy as np
import pandas as pd
from trade4.research.panel_builder import build_panel

def test_builder_aligns_misaligned_funding_schedules():
    # A funds every 8h; B funds every 4h and lists 1 day late.
    a_fund = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=6, freq="8h", tz="UTC"),
        "funding_rate": [0.0001] * 6,
    })
    b_fund = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-02", periods=6, freq="4h", tz="UTC"),
        "funding_rate": [0.0002] * 6,
    })
    a_close = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=48, freq="1h", tz="UTC"),
        "close": np.linspace(100, 110, 48),
    })
    b_close = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-02", periods=24, freq="1h", tz="UTC"),
        "close": np.linspace(50, 55, 24),
    })
    panel = build_panel(
        funding={"A": a_fund, "B": b_fund},
        klines={"A": a_close, "B": b_close},
        bar="8h",
    )
    # B must be untradeable before it listed (no fabricated price)
    assert panel.tradeable.loc[panel.times[0], "B"] == False  # noqa: E712
    # where B has no funding event in a bar, funding must be 0, not forward-filled noise
    assert (panel.funding["B"].loc[panel.times < pd.Timestamp("2023-01-02", tz="UTC")] == 0).all()
    # grid is uniform 8h
    assert (panel.times.to_series().diff().dropna() == pd.Timedelta("8h")).all()
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/research/test_panel_builder.py -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement**

```python
# src/trade4/research/panel_builder.py
import pandas as pd
from trade4.research.panel import Panel

def build_panel(
    funding: dict[str, pd.DataFrame],
    klines: dict[str, pd.DataFrame],
    bar: str = "8h",
) -> Panel:
    """Assemble an aligned PIT Panel from per-symbol funding/kline frames.

    * Grid = uniform `bar` spanning the union of all symbols' data.
    * close = last kline close at/<= each grid bar, but only within the symbol's
      listing window (no fabrication before first or after last data).
    * funding = summed funding events falling in (prev_bar, bar]; 0 where none.
    * tradeable derives from close being finite (set inside the listing window only).
    """
    symbols = sorted(funding.keys())
    starts, ends = [], []
    kl_close, fund_ser = {}, {}
    for s in symbols:
        f = funding[s].set_index("timestamp")["funding_rate"].sort_index()
        k = klines[s].set_index("timestamp")["close"].sort_index()
        fund_ser[s], kl_close[s] = f, k
        starts.append(min(f.index.min(), k.index.min()))
        ends.append(max(f.index.max(), k.index.max()))
    grid = pd.date_range(min(starts).floor(bar), max(ends).ceil(bar), freq=bar)

    close_cols, funding_cols = {}, {}
    for s in symbols:
        k = kl_close[s]
        listed_lo, listed_hi = k.index.min(), k.index.max()
        c = k.reindex(grid, method="ffill")
        c[(grid < listed_lo) | (grid > listed_hi)] = float("nan")  # no fabrication
        close_cols[s] = c
        # sum funding events per grid bar (right-closed); 0 where none occurred
        f = fund_ser[s]
        binned = f.groupby(pd.cut(f.index, bins=grid)).sum()
        fc = pd.Series(0.0, index=grid)
        for interval, val in binned.items():
            if pd.notna(interval):
                fc.loc[interval.right] = val
        funding_cols[s] = fc

    close = pd.DataFrame(close_cols, index=grid)
    fund = pd.DataFrame(funding_cols, index=grid)
    return Panel(close=close, funding=fund)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/research/test_panel_builder.py -v`
Expected: PASS. (If the funding-binning edge alignment is off by one bar, inspect the failing assert and fix the bin boundary — do not loosen the test.)

- [ ] **Step 5: Commit**

```bash
git add src/trade4/research/panel_builder.py tests/research/test_panel_builder.py
git commit -m "feat(research): heterogeneous PIT panel builder, handles misaligned funding"
```

### Task 3.3: xs_funding strategy

**Files:** Create `src/trade4/research/strategies/xs_funding.py`, `tests/research/test_xs_funding.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/research/test_xs_funding.py
import numpy as np
import pandas as pd
from trade4.research.panel import Panel
from trade4.research.portfolio_engine import assert_causal
from trade4.research.strategies.xs_funding import XSFunding

def _panel(n=40, k=6):
    t = pd.date_range("2023-01-01", periods=n, freq="8h", tz="UTC")
    rng = np.random.default_rng(0)
    cols = [f"S{i}" for i in range(k)]
    close = 100 * (1 + pd.DataFrame(rng.normal(0, 0.005, (n, k)), index=t, columns=cols)).cumprod()
    funding = pd.DataFrame(rng.normal(0.0003, 0.0004, (n, k)), index=t, columns=cols)
    return Panel(close=close, funding=funding)

def test_xs_funding_is_causal():
    assert_causal(XSFunding(quantile=0.33, lookback=3), _panel())

def test_xs_funding_is_dollar_neutral():
    w = XSFunding(quantile=0.33, lookback=3).generate_target_weights(_panel())
    last = w.iloc[-1]
    assert abs(last.sum()) < 1e-9  # long lowest, short highest funding, balanced

def test_xs_funding_longs_lowest_shorts_highest():
    # Construct clear funding ordering on the last bar.
    t = pd.date_range("2023-01-01", periods=5, freq="8h", tz="UTC")
    close = pd.DataFrame(100.0, index=t, columns=["A", "B", "C"])
    funding = pd.DataFrame({"A": 0.001, "B": 0.0, "C": -0.001}, index=t)  # A high, C low
    w = XSFunding(quantile=0.34, lookback=1).generate_target_weights(
        Panel(close=close, funding=funding)).iloc[-1]
    assert w["C"] > 0 and w["A"] < 0  # long lowest funding (C), short highest (A)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/research/test_xs_funding.py -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement**

```python
# src/trade4/research/strategies/xs_funding.py
from dataclasses import dataclass
import numpy as np
import pandas as pd
from trade4.research.strategy import Strategy
from trade4.research.panel import Panel

@dataclass
class XSFunding(Strategy):
    quantile: float = 0.25
    lookback: int = 3
    name: str = "xs_funding"

    def generate_target_weights(self, panel: Panel) -> pd.DataFrame:
        signal = panel.funding.rolling(self.lookback, min_periods=1).mean()  # causal
        signal = signal.where(panel.tradeable)
        weights = pd.DataFrame(0.0, index=panel.times, columns=panel.symbols)
        for t in panel.times:
            row = signal.loc[t].dropna()
            if len(row) < 2:
                continue
            n_side = max(1, int(len(row) * self.quantile))
            ordered = row.sort_values()
            longs = ordered.index[:n_side]   # lowest funding
            shorts = ordered.index[-n_side:]  # highest funding
            weights.loc[t, longs] = 0.5 / n_side
            weights.loc[t, shorts] = -0.5 / n_side
        return weights
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/research/test_xs_funding.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trade4/research/strategies/xs_funding.py tests/research/test_xs_funding.py
git commit -m "feat(research): cross-sectional funding-dispersion strategy"
```

### Task 3.4: xs_momentum strategy

**Files:** Create `src/trade4/research/strategies/xs_momentum.py`, `tests/research/test_xs_momentum.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/research/test_xs_momentum.py
import numpy as np
import pandas as pd
from trade4.research.panel import Panel
from trade4.research.portfolio_engine import assert_causal
from trade4.research.strategies.xs_momentum import XSMomentum

def _panel(n=80, k=6):
    t = pd.date_range("2023-01-01", periods=n, freq="8h", tz="UTC")
    rng = np.random.default_rng(0)
    cols = [f"S{i}" for i in range(k)]
    close = 100 * (1 + pd.DataFrame(rng.normal(0.0002, 0.01, (n, k)), index=t, columns=cols)).cumprod()
    funding = pd.DataFrame(0.0, index=t, columns=cols)
    return Panel(close=close, funding=funding)

def test_xs_momentum_is_causal():
    assert_causal(XSMomentum(lookback=21, quantile=0.33), _panel())

def test_xs_momentum_is_dollar_neutral():
    w = XSMomentum(lookback=21, quantile=0.33).generate_target_weights(_panel())
    assert abs(w.iloc[-1].sum()) < 1e-9

def test_xs_momentum_longs_winner():
    t = pd.date_range("2023-01-01", periods=30, freq="8h", tz="UTC")
    close = pd.DataFrame({
        "WIN": np.linspace(100, 150, 30),   # strong up
        "FLAT": [100.0] * 30,
        "LOSE": np.linspace(100, 70, 30),   # strong down
    }, index=t)
    funding = pd.DataFrame(0.0, index=t, columns=close.columns)
    w = XSMomentum(lookback=21, quantile=0.34).generate_target_weights(
        Panel(close=close, funding=funding)).iloc[-1]
    assert w["WIN"] > 0 and w["LOSE"] < 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/research/test_xs_momentum.py -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement**

```python
# src/trade4/research/strategies/xs_momentum.py
from dataclasses import dataclass
import pandas as pd
from trade4.research.strategy import Strategy
from trade4.research.panel import Panel

@dataclass
class XSMomentum(Strategy):
    lookback: int = 21        # bars of trailing return (causal)
    quantile: float = 0.25
    name: str = "xs_momentum"

    def generate_target_weights(self, panel: Panel) -> pd.DataFrame:
        # trailing return over `lookback` bars, ending at t (uses close[t] and close[t-lb])
        mom = panel.close / panel.close.shift(self.lookback) - 1.0
        mom = mom.where(panel.tradeable)
        weights = pd.DataFrame(0.0, index=panel.times, columns=panel.symbols)
        for t in panel.times:
            row = mom.loc[t].dropna()
            if len(row) < 2:
                continue
            n_side = max(1, int(len(row) * self.quantile))
            ordered = row.sort_values()
            losers = ordered.index[:n_side]
            winners = ordered.index[-n_side:]
            weights.loc[t, winners] = 0.5 / n_side
            weights.loc[t, losers] = -0.5 / n_side
        return weights
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/research/test_xs_momentum.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trade4/research/strategies/xs_momentum.py tests/research/test_xs_momentum.py
git commit -m "feat(research): cross-sectional momentum strategy"
```

### Task 3.5: The study — run, rank, verdict

**Files:** Create `src/trade4/research/study.py`, `tests/research/test_study.py`

- [ ] **Step 1: Write the failing test (mechanics on a synthetic universe)**

```python
# tests/research/test_study.py
import numpy as np
import pandas as pd
from trade4.research.panel import Panel
from trade4.research.study import run_study

def _panel(n=400, k=6):
    t = pd.date_range("2022-01-01", periods=n, freq="8h", tz="UTC")
    rng = np.random.default_rng(0)
    cols = [f"S{i}" for i in range(k)]
    close = 100 * (1 + pd.DataFrame(rng.normal(0, 0.008, (n, k)), index=t, columns=cols)).cumprod()
    funding = pd.DataFrame(rng.normal(0.0003, 0.0003, (n, k)), index=t, columns=cols)
    return Panel(close=close, funding=funding)

def test_study_runs_all_strategies_and_ranks():
    result = run_study(_panel(), seed=0)
    assert {"funding_carry", "xs_funding", "xs_momentum"}.issubset(result["tearsheets"].keys())
    # verdict mentions whether any survived OOS after costs
    assert "verdict" in result and isinstance(result["verdict"], str)
    # capacity caveat is always present (honesty requirement)
    assert "2" in result["caveats"]["capacity"] or "capacity" in result["caveats"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/research/test_study.py -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement**

```python
# src/trade4/research/study.py
from typing import Any
import numpy as np

from trade4.research.panel import Panel
from trade4.research.portfolio_engine import EngineConfig
from trade4.research.trial_registry import TrialRegistry
from trade4.research.validation import walk_forward, gate_passed
from trade4.research.tearsheet import build_tearsheet
from trade4.research.strategies.funding_carry import FundingCarry
from trade4.research.strategies.xs_funding import XSFunding
from trade4.research.strategies.xs_momentum import XSMomentum

_CAPACITY_CAVEAT = (
    "Results are 'edge at scale after costs'. A 30-40 name dollar-neutral perp "
    "portfolio is infeasible at EUR 2k (min-notional x both legs); a positive XS "
    "result is NOT an immediately deployable signal for this account."
)

# (strategy_factory, param_grid, EngineConfig) per strategy
_SPECS = {
    "funding_carry": (FundingCarry, [{"entry_threshold": 0.0001}, {"entry_threshold": 0.0003}],
                      EngineConfig(cost_bps=5.0, price_pnl_enabled=False, n_legs=2)),
    "xs_funding": (XSFunding, [{"quantile": 0.25, "lookback": 3}, {"quantile": 0.33, "lookback": 6}],
                   EngineConfig(cost_bps=5.0, price_pnl_enabled=True, n_legs=1)),
    "xs_momentum": (XSMomentum, [{"lookback": 21}, {"lookback": 42}],
                    EngineConfig(cost_bps=5.0, price_pnl_enabled=True, n_legs=1)),
}

def run_study(panel: Panel, seed: int = 0, is_bars: int = 180, oos_bars: int = 90) -> dict[str, Any]:
    registry = TrialRegistry()
    windows = {}
    for name, (factory, grid, cfg) in _SPECS.items():
        windows[name] = walk_forward(panel, factory, grid, cfg, is_bars, oos_bars, registry)

    n_trials = registry.n_trials
    var = registry.trial_sharpe_variance()
    tearsheets, ranking = {}, []
    for name, (factory, grid, cfg) in _SPECS.items():
        strat = factory(**grid[0])
        ts = build_tearsheet(strat, panel, cfg, n_trials=n_trials, trial_sharpe_var=var)
        ts["oos_gate_passed"] = gate_passed(windows[name])
        tearsheets[name] = ts
        ranking.append((name, ts["dsr"]))

    ranking.sort(key=lambda x: x[1], reverse=True)
    survivors = [n for n, (f, g, c) in _SPECS.items() if tearsheets[n]["oos_gate_passed"]]
    if survivors:
        verdict = (f"Survived OOS after costs (DSR-ranked): {ranking}. "
                   f"Gate-passing: {survivors}.")
    else:
        verdict = (f"No strategy passed the OOS gate after costs. DSR ranking: {ranking}. "
                   f"Consistent with the funding-collapse regime thesis.")

    return {
        "tearsheets": tearsheets,
        "ranking": ranking,
        "verdict": verdict,
        "n_trials": n_trials,
        "caveats": {"capacity": _CAPACITY_CAVEAT},
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/research/test_study.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trade4/research/study.py tests/research/test_study.py
git commit -m "feat(research): study runner — DSR ranking, OOS gate, honest verdict + capacity caveat"
```

### Task 3.6: Real-universe study run (integration, honest verdict)

**Files:** Create `scripts/run_study.py` (a thin CLI), no new test (integration via the script)

- [ ] **Step 1: Write the runner script**

```python
# scripts/run_study.py
"""Fetch ~30-40 liquid perps (incl. delisted) from data.binance.vision, build the
PIT panel, run the study, print the honest verdict + tearsheets. Network-heavy."""
import logging
import pandas as pd
from trade4.data.binance_vision import fetch_funding_month, fetch_klines_month
from trade4.research.panel_builder import build_panel
from trade4.research.study import run_study

logging.basicConfig(level=logging.INFO)
UNIVERSE = [...]  # fill with ~30-40 symbols incl. known delisted (SRMUSDT, FTTUSDT, ...)
MONTHS = [(y, m) for y in range(2021, 2027) for m in range(1, 13)]

def _load(sym):
    f = pd.concat([fetch_funding_month(sym, y, m) for y, m in MONTHS], ignore_index=True)
    k = pd.concat([fetch_klines_month(sym, "1h", y, m) for y, m in MONTHS], ignore_index=True)
    return f, k

if __name__ == "__main__":
    funding, klines = {}, {}
    for s in UNIVERSE:
        f, k = _load(s)
        if not f.empty and not k.empty:
            funding[s], klines[s] = f, k
    panel = build_panel(funding, klines, bar="8h")
    result = run_study(panel, seed=0)
    print(result["verdict"])
    print("CAVEAT:", result["caveats"]["capacity"])
    for name, ts in result["tearsheets"].items():
        print(name, {k: ts[k] for k in ("sharpe", "dsr", "max_dd", "oos_gate_passed", "cost_sweep")})
```

- [ ] **Step 2: Run it (manual, network)**

Run: `.venv/bin/python scripts/run_study.py`
Expected: prints a verdict. Plausibility-check funding_carry against the known OOS loss; sanity-check XS numbers. **The verdict is the deliverable — whatever it says, it is reported honestly, including "no edge".**

- [ ] **Step 3: Commit**

```bash
git add scripts/run_study.py
git commit -m "feat(research): real-universe study runner CLI"
```

**GATE 3 (final):** Full suite green. Study runs end-to-end on the real PIT universe and emits a verdict with IS/OOS, DSR, PBO, cost-sweep, regime split per strategy. funding_carry baseline plausibility-checked against the known OOS loss (requires the killer-gate / VPS reconcile for the precise number). Report the verdict.

---

## Self-Review Notes

- **Spec coverage:** §6 validation → 2.3/2.4/2.5; §6 DSR/PSR/SR0 → 2.2; §7 cost sweep → 2.8, two-leg → 3.1; §9 regime → 2.7/2.8; §10 manifest → 2.6; §8 heterogeneous PIT builder → 3.2; §4 xs_funding/xs_momentum → 3.3/3.4; §11 verdict + capacity caveat → 3.5; metrics 365 → 2.1.
- **Known-gaps from Phase 0+1 closed here:** two-leg carry cost → 3.1; misaligned-funding panel builder → 3.2 (explicit misaligned test, not a vague loader).
- **VPS dependency:** only the funding_carry *precise* −2.862 reproduction needs the reconcile (Phase-1 Task 1.8). XS verdicts are independent — the study still delivers an honest XS answer if the VPS stays blocked; the carry baseline is then reported as "magnitude-plausible, exact reproduction pending reconcile".
- **Type consistency:** `EngineConfig(cost_bps, funding_enabled, cost_multiplier, price_pnl_enabled, n_legs)`, `TrialRegistry.record/n_trials/pnl_matrix/trial_sharpe_variance`, `walk_forward(...)→list[WindowResult]`, `deflated_sharpe_ratio(returns, trial_sharpe_var, n_trials)`, `build_tearsheet(strategy, panel, cfg, n_trials, trial_sharpe_var)`, `build_panel(funding, klines, bar)`, strategy `name`/dataclass params — consistent across tasks and with the Phase 0+1 code.
- **Deferred (out of scope, separate project):** live deployment of any survivor; ADV-scaled per-symbol slippage (the cost layer leaves a documented hook); open-interest-based strategies (tripwire already covers OI).
