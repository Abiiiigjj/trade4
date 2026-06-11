from trade4.scalper.walk_forward import WalkForwardResult

_GATE_SYMBOL = {True: "PASS", False: "FAIL"}
_LINE = "=" * 60


def print_walk_forward_report(result: WalkForwardResult, symbol: str = "") -> None:
    """Print a structured go/no-go report to stdout."""
    print(_LINE)
    print(f"SCALPER WALK-FORWARD REPORT  {symbol}")
    print(_LINE)

    def _row(label: str, is_val: object, oos_val: object, threshold: str = "") -> None:
        th = f"  (gate: {threshold})" if threshold else ""
        print(f"  {label:<30} IS={is_val!s:<12} OOS={oos_val!s:<12}{th}")

    is_m = result.in_sample
    oos_m = result.out_of_sample

    print("\nPERFORMANCE METRICS")
    _row("Sharpe (annualized)", f"{is_m.sharpe:.2f}", f"{oos_m.sharpe:.2f}", ">= 1.5")
    _row("Max Drawdown", f"{is_m.max_drawdown:.1%}", f"{oos_m.max_drawdown:.1%}", ">= -20%")
    _row("Profit Factor", f"{is_m.profit_factor:.2f}", f"{oos_m.profit_factor:.2f}", ">= 1.4")
    _row("Win Rate", f"{is_m.win_rate:.1%}", f"{oos_m.win_rate:.1%}", ">= 45%")
    _row("Trades", is_m.n_trades, oos_m.n_trades)
    _row("Final Balance (EUR)", f"{is_m.final_balance:.0f}", f"{oos_m.final_balance:.0f}")

    print(f"\n  {'OOS Degradation':<30} {result.oos_degradation:.1%}  (gate: <= 30%)")

    if result.best_params:
        print(f"\nBEST PARAMS (optimized on IS):")
        for k, v in result.best_params.items():
            print(f"  {k}: {v}")

    gate = _GATE_SYMBOL[result.gate_passed]
    print(f"\n{'=' * 60}")
    print(f"  PHASE-2 GATE: {gate}")
    if not result.gate_passed:
        print("  Strategy did NOT pass the go/no-go gate.")
        print("  Do NOT deploy live capital. Review signals and parameters.")
    else:
        print("  OOS validation passed. Proceed to Phase-2 plan.")
    print(_LINE)
