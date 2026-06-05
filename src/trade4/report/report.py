import base64
import io
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from jinja2 import Environment, FileSystemLoader

from trade4.backtest.engine import BacktestResult
from trade4.backtest.cost_model import (
    CostModel,
    DEFAULT_FEE_SCHEDULE,
    FeeSchedule,
    compute_net_edge_bps,
)

logger = logging.getLogger(__name__)
TEMPLATE_DIR = Path(__file__).parent / "templates"


@dataclass
class SensitivityResult:
    normal_1x: float
    normal_2x: float
    flip_1x: float
    flip_2x: float


def generate_report(
    screener_df: pd.DataFrame,
    backtest_results: dict[str, BacktestResult],
    output_path: Path,
) -> None:
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))
    template = env.get_template("report.html.j2")

    processed: dict[str, dict] = {}
    for symbol, result in backtest_results.items():
        cycles = result.cycles
        avg_funding = sum(c.funding_received_bps for c in cycles) / max(len(cycles), 1)
        avg_cost = sum(c.round_trip_cost_bps for c in cycles) / max(len(cycles), 1)

        processed[symbol] = {
            "n_cycles": len(cycles),
            "net_pnl_bps": result.net_pnl_bps,
            "max_drawdown_bps": result.max_drawdown_bps,
            "pct_gate_passed": result.pct_gate_passed,
            "avg_funding_bps": avg_funding,
            "avg_cost_bps": avg_cost,
            "equity_chart_b64": _equity_chart_b64(result),
            "pnl_hist_b64": _pnl_hist_b64(result),
            "sensitivity": _compute_sensitivity(result),
        }

    oos_worse = any(r.net_pnl_bps < 0 for r in backtest_results.values())

    html = template.render(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        screener_rows=screener_df.to_dict("records"),
        backtest_results=processed,
        oos_worse_than_insample=oos_worse,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Report written to %s", output_path)


def _equity_chart_b64(result: BacktestResult) -> str:
    if result.equity_curve.empty:
        return ""
    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0d1117")
    ax.set_facecolor("#161b22")
    ax.plot(result.equity_curve.index, result.equity_curve.values, color="#3fb950", linewidth=1.5)
    ax.axhline(0, color="#6e7681", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Datum", color="#c9d1d9")
    ax.set_ylabel("Kumulativer P&L (bps)", color="#c9d1d9")
    ax.tick_params(colors="#c9d1d9")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    fig.tight_layout()
    return _fig_to_b64(fig)


def _pnl_hist_b64(result: BacktestResult) -> str:
    if not result.cycles:
        return ""
    pnls = [c.net_pnl_bps for c in result.cycles]
    fig, ax = plt.subplots(figsize=(8, 3), facecolor="#0d1117")
    ax.set_facecolor("#161b22")
    ax.hist(pnls, bins=20, color="#58a6ff", edgecolor="#30363d")
    ax.axvline(0, color="#f85149", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Net P&L pro Zyklus (bps)", color="#c9d1d9")
    ax.tick_params(colors="#c9d1d9")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    fig.tight_layout()
    return _fig_to_b64(fig)


def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _compute_sensitivity(result: BacktestResult) -> SensitivityResult:
    if not result.cycles:
        return SensitivityResult(0.0, 0.0, 0.0, 0.0)

    avg_funding = sum(c.funding_received_bps for c in result.cycles) / len(result.cycles)
    flip_funding = avg_funding * 0.3

    def net_edge(funding: float, fee_mult: float, slip_mult: float) -> float:
        m = CostModel(
            fee_schedule=FeeSchedule(
                spot_taker_bps=DEFAULT_FEE_SCHEDULE.spot_taker_bps * fee_mult,
                spot_maker_bps=DEFAULT_FEE_SCHEDULE.spot_maker_bps * fee_mult,
                perp_taker_bps=DEFAULT_FEE_SCHEDULE.perp_taker_bps * fee_mult,
                perp_maker_bps=DEFAULT_FEE_SCHEDULE.perp_maker_bps * fee_mult,
                fdusd_spot_taker_bps=0.0,
                fdusd_spot_maker_bps=0.0,
            ),
            slippage_entry_bps=5.0 * slip_mult,
            slippage_exit_bps=5.0 * slip_mult,
            basis_drift_bps=2.0,
            fdusd_depeg_bps=0.0,
            use_fdusd=False,
            use_maker_spot=False,
            use_maker_perp=False,
        )
        return compute_net_edge_bps(funding, m)

    return SensitivityResult(
        normal_1x=net_edge(avg_funding, 1.0, 1.0),
        normal_2x=net_edge(avg_funding, 2.0, 2.0),
        flip_1x=net_edge(flip_funding, 1.0, 1.0),
        flip_2x=net_edge(flip_funding, 2.0, 2.0),
    )
