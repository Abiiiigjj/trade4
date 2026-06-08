"""Run the honest market-neutral study on a real point-in-time perp universe.

Fetches funding + klines (incl. delisted symbols) from data.binance.vision, builds
the PIT panel, runs the study, prints the honest verdict + per-strategy tearsheets.
Network-heavy. Configurable universe / date range via CLI.

Usage:
    .venv/bin/python scripts/run_study.py --start 2023-01 --end 2025-06
    .venv/bin/python scripts/run_study.py --symbols BTCUSDT,ETHUSDT,SRMUSDT --start 2024-01 --end 2024-06
"""
import argparse
import json
import logging

import pandas as pd

from trade4.data.binance_vision import fetch_funding_month, fetch_klines_month
from trade4.research.panel_builder import build_panel
from trade4.research.study import run_study

logger = logging.getLogger("run_study")

# ~liquid majors + a few known delisted perps (point-in-time, survivorship-controlled)
DEFAULT_UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT",
    "AVAXUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT", "BCHUSDT", "ATOMUSDT", "NEARUSDT",
    "FILUSDT", "ETCUSDT", "APTUSDT", "ARBUSDT",
    # delisted / dead (survivorship-control): include their real history
    "SRMUSDT", "FTTUSDT", "ANTUSDT", "BTSUSDT", "SCUSDT",
]


def _months(start: str, end: str) -> list[tuple[int, int]]:
    s = pd.Period(start, freq="M")
    e = pd.Period(end, freq="M")
    out = []
    p = s
    while p <= e:
        out.append((p.year, p.month))
        p += 1
    return out


def _load(symbol: str, months: list[tuple[int, int]]):
    f = pd.concat([fetch_funding_month(symbol, y, m) for y, m in months], ignore_index=True)
    k = pd.concat([fetch_klines_month(symbol, "1h", y, m) for y, m in months], ignore_index=True)
    return f, k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(DEFAULT_UNIVERSE))
    ap.add_argument("--start", default="2023-01")
    ap.add_argument("--end", default="2025-06")
    ap.add_argument("--cost-bps", type=float, default=None,
                    help="override one-way cost for all strategies (0 = gross run)")
    ap.add_argument("--rebalance-every", type=int, default=None,
                    help="hold weights between rebalances (bars; 21 = weekly on 8h) -> low turnover")
    ap.add_argument("--json", action="store_true", help="print result as JSON")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    months = _months(args.start, args.end)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    funding, klines = {}, {}
    for s in symbols:
        f, k = _load(s, months)
        if not f.empty and not k.empty:
            funding[s], klines[s] = f, k
            logger.info("loaded %s: %d funding, %d klines", s, len(f), len(k))
        else:
            logger.info("skip %s: no data in range", s)

    if len(funding) < 2:
        raise SystemExit("need >=2 symbols with data")

    panel = build_panel(funding, klines, bar="8h")
    logger.info("panel: %d bars x %d symbols", len(panel.times), len(panel.symbols))

    if args.cost_bps is not None:
        logger.info("cost_bps overridden to %.1f", args.cost_bps)
    if args.rebalance_every is not None:
        logger.info("low-turnover: rebalance every %d bars", args.rebalance_every)
    result = run_study(panel, seed=0, cost_bps_override=args.cost_bps,
                       rebalance_every=args.rebalance_every)

    if args.json:
        print(json.dumps({k: v for k, v in result.items() if k != "tearsheets"}, default=str, indent=2))
    print("\n" + "=" * 70)
    print("VERDICT:", result["verdict"])
    print("CAVEAT:", result["caveats"]["capacity"])
    print(f"trials N={result['n_trials']}  PBO={result['pbo']:.2f}")
    print("=" * 70)
    for name, ts in result["tearsheets"].items():
        print(f"\n{name}:")
        print(f"  Sharpe={ts['sharpe']:.2f}  DSR={ts['dsr']:.3f}  maxDD={ts['max_dd']:.1%}  "
              f"OOS_gate={'PASS' if ts['oos_gate_passed'] else 'FAIL'}")
        print(f"  net_bps={ts['net_bps']:.0f}  cost_sweep(1x/2x/3x)="
              f"{ts['cost_sweep'][1.0]:.0f}/{ts['cost_sweep'][2.0]:.0f}/{ts['cost_sweep'][3.0]:.0f}")
        print(f"  regime Sharpe: high={ts['regime']['high']:.2f}  low={ts['regime']['low']:.2f}")


if __name__ == "__main__":
    main()
