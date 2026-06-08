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
