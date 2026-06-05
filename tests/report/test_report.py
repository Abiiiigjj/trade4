import pandas as pd
import pytest
from pathlib import Path
from trade4.report.report import generate_report
from trade4.backtest.engine import BacktestResult, CycleResult


def _make_cycle(n: int) -> list[CycleResult]:
    return [
        CycleResult(
            entry_ts=pd.Timestamp(f"2024-0{i+1}-01", tz="UTC"),
            exit_ts=pd.Timestamp(f"2024-0{i+1}-15", tz="UTC"),
            intervals_collected=10,
            funding_received_bps=80.0,
            round_trip_cost_bps=42.0,
            net_pnl_bps=38.0,
            gate_passed=True,
            rebalance_count=0,
            exit_reason="funding_flip",
        )
        for i in range(n)
    ]


def _make_result(n_cycles: int) -> BacktestResult:
    cycles = _make_cycle(n_cycles)
    equity = pd.Series(
        [c.net_pnl_bps * (i + 1) for i, c in enumerate(cycles)],
        index=[c.exit_ts for c in cycles],
    )
    return BacktestResult(
        cycles=cycles,
        equity_curve=equity,
        max_drawdown_bps=-5.0,
        net_pnl_bps=sum(c.net_pnl_bps for c in cycles),
        pct_gate_passed=1.0,
        n_intervals_total=100,
    )


def test_generate_report_creates_html_file(tmp_path):
    screener_df = pd.DataFrame([{
        "symbol": "DOGEUSDT",
        "avg_funding_30d": 0.0002,
        "avg_funding_90d": 0.00015,
        "pct_positive_intervals": 0.85,
        "slippage_est_bps": 3.0,
        "fdusd_zero_fee": True,
        "gate_candidate": True,
    }])
    backtest_results = {"DOGEUSDT": _make_result(3)}
    output_path = tmp_path / "report.html"
    generate_report(screener_df, backtest_results, output_path=output_path)
    assert output_path.exists()
    content = output_path.read_text()
    assert "DOGEUSDT" in content
    assert "PAPER" in content


def test_generate_report_includes_sensitivity(tmp_path):
    screener_df = pd.DataFrame([{
        "symbol": "DOGEUSDT", "avg_funding_30d": 0.0002,
        "avg_funding_90d": 0.00015, "pct_positive_intervals": 0.85,
        "slippage_est_bps": 3.0, "fdusd_zero_fee": True, "gate_candidate": True,
    }])
    backtest_results = {"DOGEUSDT": _make_result(3)}
    output_path = tmp_path / "report.html"
    generate_report(screener_df, backtest_results, output_path=output_path)
    content = output_path.read_text()
    assert "Sensitivity" in content or "sensitivity" in content


def test_report_shows_net_pnl_per_coin(tmp_path):
    screener_df = pd.DataFrame([{
        "symbol": "DOGEUSDT", "avg_funding_30d": 0.0002,
        "avg_funding_90d": 0.00015, "pct_positive_intervals": 0.85,
        "slippage_est_bps": 3.0, "fdusd_zero_fee": True, "gate_candidate": True,
    }])
    result = _make_result(3)
    output_path = tmp_path / "report.html"
    generate_report(screener_df, {"DOGEUSDT": result}, output_path=output_path)
    content = output_path.read_text()
    # 3 cycles × 38.0 bps = 114.0 bps net
    assert "114" in content
