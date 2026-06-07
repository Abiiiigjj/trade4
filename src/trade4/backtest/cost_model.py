from dataclasses import dataclass

MIN_NET_EDGE_BPS: float = 15.0
FUNDING_TO_COST_RATIO: float = 2.0


@dataclass(frozen=True)
class FeeSchedule:
    spot_taker_bps: float = 10.0
    spot_maker_bps: float = 9.0
    perp_taker_bps: float = 5.0
    perp_maker_bps: float = 2.0
    fdusd_spot_taker_bps: float = 0.0
    fdusd_spot_maker_bps: float = 0.0


DEFAULT_FEE_SCHEDULE = FeeSchedule()


@dataclass(frozen=True)
class CostModel:
    fee_schedule: FeeSchedule
    slippage_entry_bps: float
    slippage_exit_bps: float
    basis_drift_bps: float
    fdusd_depeg_bps: float
    use_fdusd: bool
    use_maker_spot: bool
    use_maker_perp: bool


def compute_round_trip_cost_bps(model: CostModel) -> float:
    fs = model.fee_schedule
    if model.use_fdusd and not model.use_maker_spot:
        spot_fee = fs.fdusd_spot_taker_bps
    elif model.use_fdusd and model.use_maker_spot:
        spot_fee = fs.fdusd_spot_maker_bps
    elif model.use_maker_spot:
        spot_fee = fs.spot_maker_bps
    else:
        spot_fee = fs.spot_taker_bps

    perp_fee = fs.perp_maker_bps if model.use_maker_perp else fs.perp_taker_bps

    return (
        spot_fee + perp_fee          # entry
        + spot_fee + perp_fee        # exit
        + model.slippage_entry_bps
        + model.slippage_exit_bps
        + model.basis_drift_bps
        + (model.fdusd_depeg_bps if model.use_fdusd else 0.0)
    )


def compute_net_edge_bps(expected_funding_bps: float, cost_model: CostModel) -> float:
    return expected_funding_bps - compute_round_trip_cost_bps(cost_model)


def gate_passed(expected_funding_bps: float, cost_model: CostModel) -> bool:
    cost = compute_round_trip_cost_bps(cost_model)
    net_edge = expected_funding_bps - cost
    return (
        net_edge >= MIN_NET_EDGE_BPS
        and expected_funding_bps >= FUNDING_TO_COST_RATIO * cost
    )


@dataclass(frozen=True)
class ScalperCostModel:
    """Cost model for taker-only scalping trades on Binance USDT-M Futures."""
    perp_taker_bps: float = 5.0        # Binance Futures taker fee
    calm_slippage_bps: float = 5.0     # Normal taker entry slippage
    stress_slippage_bps: float = 30.0  # Market order into volume spike (pump scanner)


def scalper_round_trip_bps(model: ScalperCostModel, stressed: bool = False) -> float:
    """Total round-trip cost in bps for one scalping trade.

    stressed=True applies pump-scanner slippage (market buy into spike).
    Includes: perp taker fee × 2 (entry + exit) + slippage × 2.
    """
    slippage = model.stress_slippage_bps if stressed else model.calm_slippage_bps
    return 2 * model.perp_taker_bps + 2 * slippage
