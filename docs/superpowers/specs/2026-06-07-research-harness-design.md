# Design Spec — Honest Market-Neutral Research Harness (`src/trade4/research/`)

**Date:** 2026-06-07
**Status:** Approved (user granted full authority to proceed)
**Author:** Claude (Opus 4.8) with Ahmet

---

## 1. Purpose & Background

Five prior trading bots (trade/APEX, TRADE1, AEGIS, plus two others) showed **phantom
profits** caused by look-ahead, stale prices, and fake generators. Only trade4
(funding-carry) was broadly honest — yet has **no out-of-sample edge in the current
regime** (walk-forward: IS +11.004 bps → OOS −2.862 bps; cause proven = funding collapse
0.9 → 0.3 bps/interval, below break-even).

The root cause across all bots: **there was never an institutionalised honest validation.**
Each bot was a single strategy that could lie. The one session with an honest backtest +
walk-forward delivered in hours the truth that five bots missed over weeks.

**Goal — not bot #6.** Build the *tool* that enforces this honesty and lets arbitrary
market-neutral strategies be tested quickly and fraud-resistantly — and use it to answer a
real research question: **Is there a market-neutral crypto strategy with out-of-sample edge
in the current regime?**

A code audit during design found that even the "honest" trade4 engine contains a
look-ahead bug (`backtest/engine.py:125`, see §3). This reinforces the thesis: honesty must
be **structurally enforced**, not assumed.

---

## 2. Scope

### In scope (v1)
- New package `src/trade4/research/` — one strategy interface, one portfolio engine, one
  validation layer.
- Three market-neutral strategies: `funding_carry` (baseline port), `xs_funding`,
  `xs_momentum`.
- **True point-in-time universe** including delisted/dead perps (no survivorship bias).
- First research study + honest verdict.

### Out of scope (v1)
- Live deployment of any surviving strategy (separate follow-up project).
- Scalper migration onto the new engine (the scalper stays untouched; its two untracked
  files `scalper/backtest_funding.py`, `scalper/signals/funding_momentum.py` are left as-is).
- Spot-borrow modelling → enforced by **perp-only universe** constraint.

---

## 3. Critical Code Findings (grounded in current code)

1. **🔴 Look-ahead in `backtest/engine.py:125–128`.** The entry decision at time `ts` uses
   `funding_df[funding_df["timestamp"] > ts]` (future funding rates) to compute
   `expected_funding_bps` feeding `gate_passed`. This is the phantom-profit pattern. Almost
   certainly the target of the VPS `causal_gate` fix. **Consequence:** the −2.862 bps
   reference is only valid if produced by the *fixed* engine; `funding_carry` must be ported
   **causally**. Retrieving the VPS fix is a **prerequisite**, not cleanup.
2. **🟠 `cost_model.py` costs are flat per-trade** (`compute_round_trip_cost_bps` sums fixed
   bps). A portfolio engine needs **turnover- and ADV-aware** costs.
3. **🟠 `walk_forward.py:56` annualises with `sqrt(252)`** — crypto trades 24/7 → must be
   **365**. Its gates (`win_rate≥0.45`, `profit_factor≥1.4`) are trade-level scalper gates;
   market-neutral portfolios need Sharpe / DSR / OOS-degradation / MaxDD gates.

---

## 4. Architecture

```
src/trade4/research/
  strategy.py          # ABC: generate_target_weights(panel) -> DataFrame[time × symbol]
  panel.py             # PIT panel container + tradeable-universe mask
  portfolio_engine.py  # CORE: vectorised backtest, funding/borrow/turnover costs, tripwire
  costs.py             # turnover-/ADV-aware portfolio cost layer (wraps backtest/cost_model)
  validation.py        # multi-window walk-forward + Deflated Sharpe + PBO + trial registry
  metrics.py           # crypto-correct (365) Sharpe, DSR, portfolio gates, regime split
  manifest.py          # run manifest: data hash + git commit + seed
  study.py             # runs all strategies through the harness, ranks by DSR, verdict
  strategies/
    funding_carry.py   # causal port of existing logic (baseline)
    xs_funding.py      # long lowest / short highest funding (funding dispersion)
    xs_momentum.py     # long winners / short losers, weekly (robust crypto factor)
```

**Reuse / extend:** `data/store.py` + `data/binance.py`/`okx.py` (PIT universe, delisted,
OI), `backtest/cost_model.py` (wrapped by `costs.py`), `report/report.py` (portfolio
tearsheet). The scalper and live layers are untouched.

### Strategy interface
```python
class Strategy(ABC):
    @abstractmethod
    def generate_target_weights(self, panel: Panel) -> pd.DataFrame:
        """Return dollar-neutral target weights [time × symbol].
        MUST be causal: weights at t may use panel data ≤ t only."""
```
Generalises the existing `generate_signals(df)` pattern to a universe; covers both
time-series and cross-sectional strategies.

### Panel & PIT discipline
- `Panel` holds aligned per-symbol frames (funding, OHLCV, OI) on a fixed bar grid (clock
  defined below) plus a **tradeable mask** `tradeable[time × symbol]` (bool): true iff the
  symbol is listed and liquid at `t`.
- Symbols are ragged (different listing/delisting dates). "Not in universe at t" (mask=False)
  is distinct from "zero weight." The engine asserts **weights are 0 where mask is False** —
  trading a not-yet-listed symbol is look-ahead.

### Clock
- Fixed bar grid (default 8h to align natively with funding intervals; configurable to 1h).
- Funding applied on funding timestamps mapped to bars. Weights held between rebalances.
- All timestamps UTC, tz-aware.

---

## 5. Anti-Phantom Construction (non-negotiable)

1. **Future-perturbation invariance tripwire** (THE core test): the engine injects
   NaN/noise into `panel[t+1:]` and asserts `generate_target_weights` output for `≤t` is
   **bit-identical**. Catches the `engine.py:125` class structurally — far stronger than an
   index assert.
2. **Cost model is mandatory** — engine refuses to run without one.
3. **Reporting refuses a verdict without OOS.**
4. **PIT tradeable-mask assert** (§4) — makes point-in-time enforceable, not just documented.

---

## 6. Validation Layer (`validation.py`)

- **Multi-window walk-forward**: IS-optimise → OOS-gate over several rolling windows.
  Generalises `scalper/walk_forward.py` to portfolio level.
- **Portfolio gates** (replace trade-level gates): OOS Sharpe ≥ 1.5, OOS-degradation ≤ 30 %,
  MaxDD bound, plus DSR threshold. Crypto annualisation = **365**.
- **Deflated Sharpe Ratio (DSR)** — corrects for the number of trials.
- **Probability of Backtest Overfitting (PBO)** — over combinatorially-purged splits.
- **Trial registry** — every configuration *ever evaluated* (each param-grid point × each
  strategy) is logged; `N_trials` flows into DSR/PBO. Without this DSR is under-deflated =
  too optimistic = exactly the lie. This is what makes DSR/PBO meaningful with few final
  strategies.

---

## 7. Costs (`costs.py`)

- **Turnover-based**: cost ∝ |Δweight| per symbol per rebalance, not per-position-count.
- **ADV-aware slippage**: participation-scaled (position notional vs symbol ADV).
- **Funding PnL**: discrete, on funding timestamps; sign/timing per the fixed (causal) logic.
- **Borrow**: for perps, short cost = funding paid (already in funding PnL — must **not** be
  double-counted). Perp-only universe avoids spot-borrow.
- **Cost-sensitivity sweep**: every tearsheet reports results at **1× / 2× / 3×** assumed
  costs. The −2.862 loss *was* cost-fragility; an edge that only survives optimistic costs is
  no edge.

---

## 8. Data Lake — True Point-in-Time Universe

- Perp-only, ~30–40 liquid USDT-M perps **including delisted/dead** symbols.
- **Source strategy (assumption, validate in Phase 0):** `data.binance.vision` public dumps
  for klines + funding history (covers many delisted symbols); listing/delisting dates from
  `onboardDate` + last-available-kline timestamp; cross-check OKX. Document method &
  limitations explicitly.
- Reuse `data/store.py` incremental Parquet cache. Add OI where available.
- This is the **major data effort** of v1 (more than the 30→40 count).

---

## 9. Regime Analysis

- First-class regime classifier as a tearsheet axis: funding-level regime (high/low vs
  break-even) and volatility regime (realized vol). Performance reported **conditional on
  regime**, directly testing the core hypothesis: *does funding dispersion (xs_funding)
  survive when the funding level (carry) is dead?*

---

## 10. Reproducibility

- **Run manifest** per result: data hash, git commit, full config, RNG seed. All randomness
  (PBO splits) seeded. A non-reproducible result is an untrusted result.

---

## 11. Honesty Caveats (in the verdict)

- **Capacity / €2k:** a 30–40-name dollar-neutral portfolio is infeasible at €2k
  (min-notional × both legs). The tool answers "edge **at scale** after costs," **not**
  "tradeable with €2k." State this explicitly so a positive XS result is not mistaken for an
  immediately deployable signal.
- **Survivorship:** controlled via true PIT universe (not merely documented). Any residual
  data limitation (e.g. symbols `data.binance.vision` cannot serve) stated explicitly.

---

## 12. Phased Plan (autonomous within phase, evidence-gate at each boundary)

### Phase 0 — Reconcile & scaffold
- SSH to VPS (178.254.32.125, the live APEX box) **read-only**, do not disturb the running
  process. Diff `live/scheduler.py` & `backtest/engine.py`; retrieve `causal_gate`,
  `exit_mode`, `/100` fixes back into git. **Confirm whether −2.862 bps came from the fixed
  engine** (prerequisite for the reference number).
- Add `scipy` dependency (for DSR statistics).
- Validate the PIT data-source assumption (§8); build the data-lake fetch.
- `research/` skeleton + `tests/research/` scaffold.

### Phase 1 — Engine + first trust signal
- Strategy ABC + `panel.py` + `portfolio_engine.py` (funding/borrow/turnover costs) +
  future-perturbation tripwire + PIT-mask assert.
- **Known-answer tests**: synthetic price PnL **and** funding-only PnL (exact, known rates).
- Port `funding_carry` **causally** (pulled forward from Phase 3 — best trust signal first).
- **GATE 1:** tripwire + both known-answer tests green.
- **KILLER-GATE:** `funding_carry` reproduces −2.862 bps within tolerance **AND**
  cross-engine equivalence (old fixed engine vs new portfolio engine on identical 1-symbol
  input) within tolerance. Trust order: synthetic mechanics → real-data reproduction → then
  build statistics.

### Phase 2 — Validation + reporting
- `validation.py` (multi-window walk-forward + DSR + PBO + trial registry) + `metrics.py`
  (365 annualisation, portfolio gates, regime split) + portfolio tearsheet (cost sweep,
  regime axis, IS vs OOS, DSR).
- **GATE 2:** DSR/PBO reproduce reference values on known inputs.

### Phase 3 — Strategy library + study
- `xs_funding` + `xs_momentum` + first study (`study.py`), max data range (~2021+),
  multi-window walk-forward, ranked by DSR.
- Honest verdict: which (if any) survive OOS after costs, in which regime. Explicitly test
  the core hypothesis.

---

## 13. Verification (end-to-end)

1. `pytest tests/research/` green, incl. future-perturbation tripwire, PIT-mask assert, and
   both known-answer tests.
2. First study runs; `funding_carry` baseline reproduces the known OOS loss; XS numbers
   plausibility-checked vs manual session results.
3. Tearsheet report generated showing IS vs OOS + DSR + cost-sweep + regime split per
   strategy.

---

## 14. Dependencies & Decisions

- Build on trade4 (local dev in `~/Schreibtisch/trade4`).
- Dependency-light: pandas/numpy (present) + **scipy** (new, DSR statistics). **No** heavy
  framework (vectorbt/nautilus) — funding/cost correctness is the whole point and demands
  full control; a framework hides exactly the bookings that were wrong everywhere.
- One engine, one validation layer — de-fragments the two legacy single-symbol engines.
- VPS access is read-only; the live process must not be disturbed.
