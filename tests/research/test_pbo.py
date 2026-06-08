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
