"""Probability of Backtest Overfitting via Combinatorially Symmetric Cross-Validation.

Bailey, Borwein, Lopez de Prado, Zhu (2017). Split T observations into S even
partitions; for each way of choosing S/2 as in-sample (complement out-of-sample),
pick the config best by IS Sharpe, find its OOS rank, and check whether it lands
below the OOS median. PBO is the fraction of splits where the IS-best config
underperforms OOS — high PBO means IS selection does not generalise.
"""
import itertools

import numpy as np
import pandas as pd


def _sharpe_cols(block: np.ndarray) -> np.ndarray:
    mean = block.mean(axis=0)
    std = block.std(axis=0, ddof=1)
    std = np.where(std == 0, np.nan, std)
    return mean / std


def probability_of_backtest_overfitting(pnl_matrix: pd.DataFrame, n_splits: int = 10) -> float:
    """CSCV PBO. ``pnl_matrix``: T x N per-bar returns, columns = configs."""
    m = pnl_matrix.dropna(axis=1, how="any").to_numpy()
    t, n = m.shape
    if n < 2:
        return float("nan")
    s = n_splits - (n_splits % 2)  # force even
    rows_per = t // s
    if rows_per == 0:
        raise ValueError("not enough rows for the requested number of splits")
    parts = [m[i * rows_per:(i + 1) * rows_per] for i in range(s)]

    logits = []
    for is_idx in itertools.combinations(range(s), s // 2):
        oos_idx = [i for i in range(s) if i not in is_idx]
        is_block = np.vstack([parts[i] for i in is_idx])
        oos_block = np.vstack([parts[i] for i in oos_idx])
        is_sr = _sharpe_cols(is_block)
        oos_sr = _sharpe_cols(oos_block)
        if np.all(np.isnan(is_sr)):
            continue
        best = int(np.nanargmax(is_sr))
        # OOS rank of the IS-best config (1 = worst ... n = best)
        order = pd.Series(oos_sr).rank(method="average").to_numpy()
        rank = order[best]
        omega = rank / (n + 1)
        omega = min(max(omega, 1e-6), 1 - 1e-6)
        logits.append(np.log(omega / (1 - omega)))

    if not logits:
        return float("nan")
    return float(np.mean(np.array(logits) < 0))
