import numpy as np
import pandas as pd
import pytest

from trade4.research.panel import Panel


def _grid(n=4):
    return pd.date_range("2023-01-01", periods=n, freq="8h", tz="UTC")


def test_tradeable_mask_marks_unlisted_as_false():
    t = _grid()
    close = pd.DataFrame({"A": [10, 11, 12, 13], "B": [np.nan, np.nan, 5, 6]}, index=t)
    funding = pd.DataFrame({"A": 0.0001, "B": 0.0002}, index=t)
    panel = Panel(close=close, funding=funding)
    assert panel.tradeable.loc[t[0], "B"] == False  # noqa: E712
    assert panel.tradeable.loc[t[2], "B"] == True  # noqa: E712
    assert panel.tradeable.loc[t[0], "A"] == True  # noqa: E712


def test_panel_rejects_misaligned_frames():
    t = _grid()
    close = pd.DataFrame({"A": [1, 2, 3, 4]}, index=t)
    funding = pd.DataFrame({"A": [0.1, 0.2]}, index=t[:2])
    with pytest.raises(ValueError):
        Panel(close=close, funding=funding)


def test_panel_rejects_tz_naive_index():
    t = pd.date_range("2023-01-01", periods=3, freq="8h")  # tz-naive
    close = pd.DataFrame({"A": [1, 2, 3]}, index=t)
    funding = pd.DataFrame({"A": [0.0, 0.0, 0.0]}, index=t)
    with pytest.raises(ValueError):
        Panel(close=close, funding=funding)
