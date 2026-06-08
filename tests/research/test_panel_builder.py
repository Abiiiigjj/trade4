import numpy as np
import pandas as pd

from trade4.research.panel_builder import build_panel


def test_builder_aligns_misaligned_funding_schedules():
    # A funds every 8h from day 1; B funds every 4h and lists 1 day late.
    a_fund = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=6, freq="8h", tz="UTC"),
        "funding_rate": [0.0001] * 6,
    })
    b_fund = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-02", periods=6, freq="4h", tz="UTC"),
        "funding_rate": [0.0002] * 6,
    })
    a_close = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=48, freq="1h", tz="UTC"),
        "close": np.linspace(100, 110, 48),
    })
    b_close = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-02", periods=24, freq="1h", tz="UTC"),
        "close": np.linspace(50, 55, 24),
    })
    panel = build_panel(
        funding={"A": a_fund, "B": b_fund},
        klines={"A": a_close, "B": b_close},
        bar="8h",
    )
    # B must be untradeable before it listed (no fabricated price)
    assert panel.tradeable.loc[panel.times[0], "B"] == False  # noqa: E712
    # where B has no data yet, funding must be 0, not forward-filled noise
    before_b = panel.times < pd.Timestamp("2023-01-02", tz="UTC")
    assert (panel.funding["B"].loc[before_b] == 0).all()
    # grid is uniform 8h
    diffs = panel.times.to_series().diff().dropna()
    assert (diffs == pd.Timedelta("8h")).all()
    # B's 4h funding events get aggregated into 8h bars (two events per bar -> 0.0004)
    after_b = panel.times >= pd.Timestamp("2023-01-02 08:00", tz="UTC")
    assert panel.funding["B"].loc[after_b].max() > 0.0002 - 1e-12
