# Research Findings — Honest Market-Neutral Strategy Validation

**Date:** 2026-06-08
**Branch:** `research-harness`
**Question:** Is there a market-neutral crypto strategy with out-of-sample edge in the
current regime, after costs?

---

## TL;DR — the honest verdict

**No robustly-established edge — but a real candidate.** After building a fraud-resistant
research harness and putting three market-neutral strategies through it on a 23-symbol
point-in-time perp universe (2022–2025, 3832 8h-bars):

- The best system is **vol-targeted `xs_funding`** (cross-sectional funding dispersion,
  perp-only, weekly rebalance, 10% vol target): **Sharpe ≈ 0.79, max drawdown ≈ −19%,
  tradeable with €2k** (~10 perp positions of ~$130 each).
- But the capstone validation **refuses to bless it**: Deflated Sharpe **0.617** (modest
  after deflating for the 17-config search), PBO **0.51** (coin-flip), and the decisive
  signal — **first-half Sharpe −0.01 vs second-half +1.61.** The entire edge sits in
  2024–2025.

**The edge is real but RECENT.** Backtesting alone cannot tell "a new structural edge"
from "recent overfit." The only honest next step is **forward/paper testing** for genuine
out-of-sample time.

This is the opposite of the five prior bots, which showed phantom profits (look-ahead,
stale prices, fake generators). Here every weakness is measured and named.

---

## What was built — the tool (`src/trade4/research/`)

A fraud-resistant harness that makes lying structurally hard:

| Component | Guarantee |
|-----------|-----------|
| `portfolio_engine.py` | Vectorised PnL (price + discrete funding − turnover cost). **Future-perturbation tripwire** (perturbs *every* panel field; catches look-ahead structurally). **PIT-mask assert** (no weight on untradeable symbols). |
| `panel.py` / `panel_builder.py` | Point-in-time universe incl. delisted perps; tradeable mask; handles misaligned funding schedules. |
| `metrics.py` | Crypto-correct Sharpe (365), PSR/DSR/SR0 (López de Prado). |
| `pbo.py` | Probability of Backtest Overfitting (CSCV). |
| `trial_registry.py` / `validation.py` | Multi-window walk-forward; **N counts every evaluated config** so DSR/PBO can't be gamed. |
| `overlay.py` | Causal vol-targeting risk overlay. |
| `blend.py` | Capital-allocation blend of independently-costed sub-strategies. |
| `data/binance_vision.py` | PIT fetch incl. delisted perps, parquet-cached. |

**167 tests**, including known-answer PnL tests, look-ahead tripwires (a cheating strategy
*must* be caught), and DSR/PBO behavioural checks.

A look-ahead bug was even found in the *legacy* "honest" engine (`backtest/engine.py:125`
used future funding to decide entries) and reconciled with the causal fix.

---

## The research narrative (each step honest)

1. **Naive baseline.** All three strategies (funding_carry, xs_funding, xs_momentum) net
   **negative** with naive 8h rebalancing — but the gross (cost=0) run showed **all three
   positive**. The "no edge" verdict was a *transaction-cost* story, not missing alpha.
2. **Low turnover.** Weekly rebalancing (`PeriodicRebalance`) flipped all three
   net-positive. Turnover, not alpha, was the killer.
3. **Honest carry.** Adding a basis-drift holding cost revealed `funding_carry` is a
   *fair-weather* strategy: Sharpe 4.27 collapses to −0.68 at 0.5 bp/bar basis drift. It is
   doubly fragile (cost + hedge). Structural insight: **`xs_funding` is perp-only — no spot
   hedge, no basis risk — the robust core.**
4. **Blend.** A diversified blend cut drawdown but, once everything was honestly costed and
   vol-targeted, **did not beat standalone `xs_funding`** (0.74 vs 0.79). The carry leg
   didn't earn its complexity. Simpler wins.
5. **€2k capacity.** Not the binding constraint — `xs_funding` already holds ~10 positions.
   Concentration (top_k 1–3) *hurts* (gives up Sharpe, no drawdown benefit).
6. **Capstone.** Search-deflated DSR 0.617, PBO 0.51, first/second-half −0.01/+1.61 → not
   robustly established; edge is recent.

---

## The chosen system (if forward-tested)

```
vol-targeted xs_funding
  signal      : long K lowest- / short K highest-funding perps (quantile 0.25, ~5/side)
  universe    : liquid USDT-M perps, point-in-time
  rebalance   : weekly (every 21 bars on 8h)
  hedging     : none (perp-only, dollar-neutral) -> no basis risk
  risk overlay: vol-target 10% annual, 30-bar trailing vol, max 2x leverage (causal)
  costs       : 5 bps/leg one-way (taker), realistic for retail size on liquid perps
  capacity    : tradeable at €2k (~10 positions, ~$130 each)
```

Reproduce: `python -m scripts.run_concentrated`, `python -m scripts.final_validation`.

---

## Honest caveats

- **Edge is recent (2024–2025);** 2022–2023 flat-to-negative. Regime-dependent.
- **Sharpe ~0.8 is modest;** drawdown −19% is real risk for a small account.
- **Universe lacks verified-dead tickers** (SRMUSDT/FTTUSDT were not actually delisted by
  the period end). The PIT mechanism is correct; the universe needs genuinely-dead names
  for full survivorship control.
- **Gross carry assumed a perfect spot hedge** (basis drift modelled as a conservative
  constant; reality is regime-dependent).
- **DSR/PBO trial set is correlated** (variants of one strategy), which blurs PBO.

---

## Next step (separate project)

**Forward/paper testing.** Wire the chosen system to the existing paper-trading layer
(`live/paper_executor.py`) and collect genuine out-of-sample *time* — the only honest way
to distinguish a real recent structural edge from recent overfit. Backtesting more would
only overfit.
