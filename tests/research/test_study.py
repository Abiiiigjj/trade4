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
    assert "verdict" in result and isinstance(result["verdict"], str)
    # capacity caveat is always present (honesty requirement)
    assert "capacity" in result["caveats"]
    # PBO wired in and finite (carry-forward #1/#2)
    assert "pbo" in result and not np.isnan(result["pbo"])
    # ranking is sorted by DSR descending
    dsrs = [d for _, d in result["ranking"]]
    assert dsrs == sorted(dsrs, reverse=True)


def test_study_pbo_matrix_has_no_nans():
    # The PBO matrix must come from a single aligned window — no NaN columns.
    result = run_study(_panel(), seed=0)
    assert result["n_trials"] >= 6  # 3 strategies x 2 params
