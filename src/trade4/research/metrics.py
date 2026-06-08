"""Performance metrics for the research harness.

Crypto-correct: annualisation uses 365 days (24/7 markets), not the 252 trading
days the legacy scalper walk-forward used. Also provides the Probabilistic and
Deflated Sharpe Ratio (Lopez de Prado 2014) and a funding-regime classifier.
"""
import math

import numpy as np
import pandas as pd
from scipy.stats import norm

# bars per year for crypto (24/7, 365 days)
PERIODS_PER_YEAR = {"1h": 24 * 365, "4h": 6 * 365, "8h": 3 * 365, "1d": 365}

_EULER = 0.5772156649015329


def sharpe_ratio(returns: pd.Series, bar: str = "8h") -> float:
    """Annualised Sharpe ratio (crypto: 365-day year)."""
    r = returns.dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    ann = PERIODS_PER_YEAR[bar] ** 0.5
    return float(r.mean() / r.std(ddof=1) * ann)


def max_drawdown(equity: pd.Series) -> float:
    """Maximum drawdown of an equity curve (negative fraction)."""
    eq = equity.dropna()
    if eq.empty:
        return 0.0
    running_max = eq.cummax()
    return float(((eq - running_max) / running_max).min())


def probabilistic_sharpe_ratio(returns: pd.Series, sr_benchmark: float = 0.0) -> float:
    """PSR: probability that the true per-period Sharpe exceeds ``sr_benchmark``.

    Uses the observed per-period (non-annualised) Sharpe and corrects for sample
    length, skew and kurtosis (Lopez de Prado 2014).
    """
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
    """Expected maximum per-period Sharpe under the null across ``n_trials`` (SR0)."""
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


def funding_regime(mean_funding: pd.Series, break_even: float = 0.0003) -> pd.Series:
    """Label each bar 'high'/'low' by whether cross-sectional mean funding clears break-even."""
    return mean_funding.apply(lambda x: "high" if x >= break_even else "low")
