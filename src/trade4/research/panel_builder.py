"""Assemble an aligned point-in-time Panel from a ragged real universe.

The real universe is messy: symbols list and delist at different dates, and funding
intervals differ (8h vs 4h/1h) and even change over a symbol's life. This builder
produces a single aligned Panel on a chosen bar grid WITHOUT fabricating data:

* close = last kline close at/<= each grid bar, but only inside the symbol's listing
  window (NaN outside → untradeable, never forward-filled into existence);
* funding = sum of funding events falling in each grid bar (right-closed); 0 where
  none occurred (not forward-filled);
* tradeable derives from close being finite.
"""
import numpy as np
import pandas as pd

from trade4.research.panel import Panel


def build_panel(
    funding: dict[str, pd.DataFrame],
    klines: dict[str, pd.DataFrame],
    bar: str = "8h",
) -> Panel:
    symbols = sorted(funding.keys())
    fund_ser: dict[str, pd.Series] = {}
    kl_close: dict[str, pd.Series] = {}
    starts, ends = [], []
    for s in symbols:
        # Coerce timestamp to a tz-aware DatetimeIndex: concatenating empty (404)
        # monthly frames can leave the column as object dtype for ragged/late-listing
        # symbols, which would otherwise yield a plain Index without .ceil/.floor.
        fdf = funding[s].copy()
        fdf["timestamp"] = pd.to_datetime(fdf["timestamp"], utc=True)
        kdf = klines[s].copy()
        kdf["timestamp"] = pd.to_datetime(kdf["timestamp"], utc=True)
        f = fdf.set_index("timestamp")["funding_rate"].sort_index()
        k = kdf.set_index("timestamp")["close"].sort_index()
        fund_ser[s], kl_close[s] = f, k
        lo = min([x.index.min() for x in (f, k) if not x.empty])
        hi = max([x.index.max() for x in (f, k) if not x.empty])
        starts.append(lo)
        ends.append(hi)

    grid = pd.date_range(min(starts).floor(bar), max(ends).ceil(bar), freq=bar)

    close_cols, funding_cols = {}, {}
    for s in symbols:
        k = kl_close[s]
        listed_lo, listed_hi = k.index.min(), k.index.max()
        c = k.reindex(grid, method="ffill")
        c[(grid < listed_lo) | (grid > listed_hi)] = np.nan  # no fabrication
        close_cols[s] = c

        # sum funding events into the grid bar that closes them: event at ts belongs
        # to bar b if b - bar_td < ts <= b (right-closed). ceil(bar) maps each event
        # to the smallest grid point >= ts (tz-safe, unlike numpy searchsorted on
        # tz-aware .values which breaks under pandas 3.0).
        f = fund_ser[s]
        if f.empty:
            funding_cols[s] = pd.Series(0.0, index=grid)
        else:
            closing = f.index.ceil(bar)
            summed = f.groupby(closing).sum()
            funding_cols[s] = summed.reindex(grid, fill_value=0.0)

    close = pd.DataFrame(close_cols, index=grid)
    fund = pd.DataFrame(funding_cols, index=grid)
    return Panel(close=close, funding=fund)
