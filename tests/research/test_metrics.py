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


# ----- PSR / DSR / SR0 -----

from trade4.research.metrics import (  # noqa: E402
    probabilistic_sharpe_ratio, expected_max_sharpe, deflated_sharpe_ratio, funding_regime,
)


def test_psr_half_when_observed_equals_benchmark():
    # SR_hat == SR* -> numerator 0 -> Phi(0) = 0.5
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
    v = 0.01
    assert expected_max_sharpe(v, n_trials=100) > expected_max_sharpe(v, n_trials=2)


def test_dsr_below_psr_when_many_trials():
    r = pd.Series(np.random.default_rng(2).normal(0.002, 0.01, 1000))
    psr0 = probabilistic_sharpe_ratio(r, 0.0)
    dsr = deflated_sharpe_ratio(r, trial_sharpe_var=0.02, n_trials=50)
    assert dsr < psr0


# ----- regime classifier -----

def test_funding_regime_labels_high_and_low():
    idx = pd.date_range("2023-01-01", periods=4, freq="8h", tz="UTC")
    mean_funding = pd.Series([0.0008, 0.0008, 0.0001, 0.0001], index=idx)
    reg = funding_regime(mean_funding, break_even=0.0003)
    assert (reg.iloc[:2] == "high").all()
    assert (reg.iloc[2:] == "low").all()
