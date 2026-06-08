import numpy as np
import pandas as pd

from trade4.research.panel_builder import build_panel


def test_builder_handles_ms_resolution_and_ragged_universe():
    """Regression: real fetches carry millisecond-resolution timestamps
    (pd.to_datetime(unit='ms')), while date_range's grid is nanosecond. The old
    np.searchsorted-on-.values path crashed on that resolution mismatch under pandas
    3.0 ('can't compare offset-naive and offset-aware'). Build a ragged 3-symbol
    universe with ms-resolution timestamps and assert it builds cleanly."""
    def _ms(start, periods, freq):
        return pd.date_range(start, periods=periods, freq=freq, tz="UTC").as_unit("ms")

    funding, klines = {}, {}
    specs = [("A", "2022-01-01", 200), ("B", "2022-06-01", 150), ("C", "2023-01-01", 100)]
    for name, start, n in specs:
        funding[name] = pd.DataFrame({"timestamp": _ms(start, n, "8h"),
                                      "funding_rate": [0.0001] * n})
        klines[name] = pd.DataFrame({"timestamp": _ms(start, n * 8, "1h"),
                                     "close": np.linspace(100, 110, n * 8)})

    panel = build_panel(funding, klines, bar="8h")
    assert set(panel.symbols) == {"A", "B", "C"}
    # late lister C is untradeable at the universe start (survivorship control)
    assert panel.tradeable.loc[panel.times[0], "C"] == False  # noqa: E712
    assert panel.tradeable["A"].any()  # A trades somewhere
    assert not panel.funding.isna().any().any()


def test_builder_coerces_object_dtype_timestamps():
    """Regression: concatenating empty (404) monthly frames for ragged/late-listing
    symbols can leave the timestamp column as object dtype -> set_index yields a plain
    Index without .ceil -> crash. build_panel must coerce. This was the second real-run
    bug (after the resolution mismatch)."""
    ts = pd.date_range("2023-01-01", periods=6, freq="8h", tz="UTC")
    # object-dtype timestamp column (what a leading-empty concat produces)
    f = pd.DataFrame({"timestamp": pd.Series(list(ts), dtype="object"),
                      "funding_rate": [0.0001] * 6})
    k = pd.DataFrame({"timestamp": pd.Series(list(pd.date_range(
                        "2023-01-01", periods=48, freq="1h", tz="UTC")), dtype="object"),
                      "close": np.linspace(100, 110, 48)})
    panel = build_panel(funding={"A": f}, klines={"A": k}, bar="8h")
    assert panel.symbols == ["A"]
    assert not panel.funding.isna().any().any()


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
