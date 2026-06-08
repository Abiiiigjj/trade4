import numpy as np
import pandas as pd

from trade4.research.overlay import vol_target, releveraging_cost


def _returns(n=200, seed=0):
    idx = pd.date_range("2023-01-01", periods=n, freq="8h", tz="UTC")
    return pd.Series(np.random.default_rng(seed).normal(0.0005, 0.01, n), index=idx)


def test_vol_target_is_causal_no_future_leak():
    r = _returns()
    scaled, exp = vol_target(r, lookback=20)
    # perturb the future; past exposure must be unchanged
    r2 = r.copy()
    cut = 120
    r2.iloc[cut + 1:] = np.random.default_rng(9).normal(0, 0.5, len(r) - cut - 1)
    _, exp2 = vol_target(r2, lookback=20)
    pd.testing.assert_series_equal(exp.iloc[:cut + 1], exp2.iloc[:cut + 1])


def test_vol_target_constant_vol_gives_constant_exposure():
    idx = pd.date_range("2023-01-01", periods=60, freq="8h", tz="UTC")
    r = pd.Series([0.01, -0.01] * 30, index=idx)  # constant magnitude -> constant vol
    _, exp = vol_target(r, target_vol_annual=0.10, lookback=10, max_leverage=10.0)
    after_warmup = exp.iloc[15:]
    assert after_warmup.std() < 1e-9  # exposure flat once vol is steady


def test_vol_target_caps_leverage():
    idx = pd.date_range("2023-01-01", periods=60, freq="8h", tz="UTC")
    r = pd.Series([1e-6, -1e-6] * 30, index=idx)  # near-zero vol -> would lever to inf
    _, exp = vol_target(r, lookback=10, max_leverage=2.0)
    assert exp.max() <= 2.0 + 1e-12


def test_releveraging_cost_charges_exposure_changes():
    idx = pd.date_range("2023-01-01", periods=3, freq="8h", tz="UTC")
    exp = pd.Series([1.0, 1.5, 1.5], index=idx)
    c = releveraging_cost(exp, cost_bps=10.0)
    assert abs(c.iloc[1] - 0.5 * 10 / 10_000) < 1e-12
    assert abs(c.iloc[2] - 0.0) < 1e-12
