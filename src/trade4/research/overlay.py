"""Volatility-targeting risk overlay.

Scales a strategy's exposure so realised volatility tracks a fixed target: de-risk when
turbulence rises (where the deep drawdowns happen), lever up modestly when calm (capped).
The scaling at bar t uses only returns strictly before t (trailing window, then shifted),
so it is causal. Operates on the returns series, above the weight engine.
"""
import pandas as pd

from trade4.research.metrics import PERIODS_PER_YEAR


def vol_target(
    returns: pd.Series,
    target_vol_annual: float = 0.10,
    lookback: int = 30,
    max_leverage: float = 2.0,
    bar: str = "8h",
) -> tuple[pd.Series, pd.Series]:
    """Return ``(scaled_returns, exposure)``.

    ``exposure_t = clip(target_vol / realised_vol_{<t}, 0, max_leverage)``; zero until the
    lookback window is filled (conservative). Causal: realised vol is shifted by one bar so
    it never sees the current return.
    """
    ann = PERIODS_PER_YEAR[bar] ** 0.5
    realised = returns.rolling(lookback).std(ddof=1) * ann
    realised = realised.shift(1)  # only past vol -> causal
    exposure = (target_vol_annual / realised).clip(upper=max_leverage)
    exposure = exposure.replace([float("inf"), -float("inf")], max_leverage).fillna(0.0)
    return exposure * returns, exposure


def releveraging_cost(exposure: pd.Series, cost_bps: float) -> pd.Series:
    """Per-bar cost of changing leverage: ``|Δexposure| * cost_bps/1e4``.

    Vol-targeting trades the whole book up/down as exposure changes; this charges for it so
    the overlay is not flattered by a free re-leveraging assumption."""
    de = exposure.fillna(0.0).diff().abs()
    de.iloc[0] = exposure.iloc[0] if pd.notna(exposure.iloc[0]) else 0.0
    return de * (cost_bps / 10_000.0)
