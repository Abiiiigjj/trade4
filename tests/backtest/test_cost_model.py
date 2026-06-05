import pytest
from trade4.backtest.cost_model import (
    FeeSchedule,
    CostModel,
    DEFAULT_FEE_SCHEDULE,
    compute_round_trip_cost_bps,
    compute_net_edge_bps,
    gate_passed,
    MIN_NET_EDGE_BPS,
    FUNDING_TO_COST_RATIO,
)


def test_default_fee_schedule_base_tier():
    assert DEFAULT_FEE_SCHEDULE.spot_taker_bps == 10
    assert DEFAULT_FEE_SCHEDULE.perp_taker_bps == 5
    assert DEFAULT_FEE_SCHEDULE.fdusd_spot_taker_bps == 0
    assert DEFAULT_FEE_SCHEDULE.fdusd_spot_maker_bps == 0


def test_round_trip_cost_standard_taker():
    model = CostModel(
        fee_schedule=DEFAULT_FEE_SCHEDULE,
        slippage_entry_bps=5.0,
        slippage_exit_bps=5.0,
        basis_drift_bps=2.0,
        fdusd_depeg_bps=0.0,
        use_fdusd=False,
        use_maker_spot=False,
        use_maker_perp=False,
    )
    cost = compute_round_trip_cost_bps(model)
    # spot taker entry 10 + perp taker entry 5 + spot taker exit 10 + perp taker exit 5
    # + slippage 5+5 + basis 2 + depeg 0 = 42
    assert cost == pytest.approx(42.0)


def test_round_trip_cost_fdusd_saves_spot_fees():
    model_standard = CostModel(
        fee_schedule=DEFAULT_FEE_SCHEDULE,
        slippage_entry_bps=5.0, slippage_exit_bps=5.0,
        basis_drift_bps=2.0, fdusd_depeg_bps=0.5,
        use_fdusd=False, use_maker_spot=False, use_maker_perp=False,
    )
    model_fdusd = CostModel(
        fee_schedule=DEFAULT_FEE_SCHEDULE,
        slippage_entry_bps=5.0, slippage_exit_bps=5.0,
        basis_drift_bps=2.0, fdusd_depeg_bps=0.5,
        use_fdusd=True, use_maker_spot=False, use_maker_perp=False,
    )
    cost_standard = compute_round_trip_cost_bps(model_standard)
    cost_fdusd = compute_round_trip_cost_bps(model_fdusd)
    # FDUSD removes 2x spot_taker (entry + exit) = 20 bps, adds 0.5 depeg
    assert cost_fdusd == pytest.approx(cost_standard - 20 + 0.5)


def test_round_trip_cost_maker_is_cheaper():
    model_maker = CostModel(
        fee_schedule=DEFAULT_FEE_SCHEDULE,
        slippage_entry_bps=5.0, slippage_exit_bps=5.0,
        basis_drift_bps=2.0, fdusd_depeg_bps=0.0,
        use_fdusd=False, use_maker_spot=True, use_maker_perp=True,
    )
    model_taker = CostModel(
        fee_schedule=DEFAULT_FEE_SCHEDULE,
        slippage_entry_bps=5.0, slippage_exit_bps=5.0,
        basis_drift_bps=2.0, fdusd_depeg_bps=0.0,
        use_fdusd=False, use_maker_spot=False, use_maker_perp=False,
    )
    assert compute_round_trip_cost_bps(model_maker) < compute_round_trip_cost_bps(model_taker)


def test_net_edge_calculation():
    model = CostModel(
        fee_schedule=DEFAULT_FEE_SCHEDULE,
        slippage_entry_bps=5.0, slippage_exit_bps=5.0,
        basis_drift_bps=2.0, fdusd_depeg_bps=0.0,
        use_fdusd=False, use_maker_spot=False, use_maker_perp=False,
    )
    cost = compute_round_trip_cost_bps(model)  # 42
    net = compute_net_edge_bps(expected_funding_bps=100.0, cost_model=model)
    assert net == pytest.approx(100.0 - cost)


def test_gate_passes_when_above_threshold():
    # funding=100, cost=42, net_edge=58 >= 15, 100 >= 2x42=84
    model = CostModel(
        fee_schedule=DEFAULT_FEE_SCHEDULE,
        slippage_entry_bps=5.0, slippage_exit_bps=5.0,
        basis_drift_bps=2.0, fdusd_depeg_bps=0.0,
        use_fdusd=False, use_maker_spot=False, use_maker_perp=False,
    )
    assert gate_passed(expected_funding_bps=100.0, cost_model=model) is True


def test_gate_fails_when_funding_too_low():
    # funding=30, cost=42, net_edge=-12 < 15 -> fail
    model = CostModel(
        fee_schedule=DEFAULT_FEE_SCHEDULE,
        slippage_entry_bps=5.0, slippage_exit_bps=5.0,
        basis_drift_bps=2.0, fdusd_depeg_bps=0.0,
        use_fdusd=False, use_maker_spot=False, use_maker_perp=False,
    )
    assert gate_passed(expected_funding_bps=30.0, cost_model=model) is False


def test_gate_fails_when_ratio_not_met():
    # funding=50, cost=42, net_edge=8 < 15 -> fail (also ratio 50 < 2x42=84)
    model = CostModel(
        fee_schedule=DEFAULT_FEE_SCHEDULE,
        slippage_entry_bps=5.0, slippage_exit_bps=5.0,
        basis_drift_bps=2.0, fdusd_depeg_bps=0.0,
        use_fdusd=False, use_maker_spot=False, use_maker_perp=False,
    )
    assert gate_passed(expected_funding_bps=50.0, cost_model=model) is False


def test_stress_test_double_fees_double_slippage():
    """Verify gate with 2x fees and 2x slippage -- charter S11 requirement."""
    base = CostModel(
        fee_schedule=FeeSchedule(
            spot_taker_bps=20, spot_maker_bps=18,
            perp_taker_bps=10, perp_maker_bps=4,
            fdusd_spot_taker_bps=0, fdusd_spot_maker_bps=0,
        ),
        slippage_entry_bps=10.0, slippage_exit_bps=10.0,
        basis_drift_bps=4.0, fdusd_depeg_bps=0.0,
        use_fdusd=False, use_maker_spot=False, use_maker_perp=False,
    )
    # 2x stress: funding=200 should still pass
    assert gate_passed(expected_funding_bps=200.0, cost_model=base) is True
    # funding=100 should fail under stress (cost=20+10+20+10+10+10+4=84, 100-84=16>=15 BUT 100<2x84=168 -> fail ratio)
    assert gate_passed(expected_funding_bps=100.0, cost_model=base) is False
