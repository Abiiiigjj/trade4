import pandas as pd

from trade4.research.costs import turnover_cost


def test_turnover_cost_charges_on_weight_change():
    t = pd.date_range("2023-01-01", periods=3, freq="8h", tz="UTC")
    w = pd.DataFrame({"A": [0.0, 1.0, 1.0]}, index=t)
    cost = turnover_cost(w, cost_bps=10.0, multiplier=1.0)
    assert abs(cost.iloc[1] - 1.0 * 10.0 / 10_000) < 1e-12
    assert abs(cost.iloc[2] - 0.0) < 1e-12


def test_turnover_cost_charges_initial_buildup():
    t = pd.date_range("2023-01-01", periods=2, freq="8h", tz="UTC")
    w = pd.DataFrame({"A": [0.5, 0.5], "B": [-0.5, -0.5]}, index=t)
    cost = turnover_cost(w, cost_bps=10.0)
    # bar0 builds |0.5| + |-0.5| = 1.0 of turnover
    assert abs(cost.iloc[0] - 1.0 * 10.0 / 10_000) < 1e-12


def test_turnover_cost_two_legs_doubles():
    t = pd.date_range("2023-01-01", periods=2, freq="8h", tz="UTC")
    w = pd.DataFrame({"A": [0.0, 1.0]}, index=t)
    one = turnover_cost(w, cost_bps=10.0, n_legs=1)
    two = turnover_cost(w, cost_bps=10.0, n_legs=2)
    assert abs(two.iloc[1] - 2.0 * one.iloc[1]) < 1e-15


def test_cost_multiplier_scales_linearly():
    t = pd.date_range("2023-01-01", periods=2, freq="8h", tz="UTC")
    w = pd.DataFrame({"A": [0.0, 1.0]}, index=t)
    c1 = turnover_cost(w, cost_bps=10.0, multiplier=1.0)
    c3 = turnover_cost(w, cost_bps=10.0, multiplier=3.0)
    assert abs(c3.iloc[1] - 3.0 * c1.iloc[1]) < 1e-15
