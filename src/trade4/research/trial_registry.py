"""Trial registry — counts every evaluated configuration.

DSR/PBO deflation is only honest if N counts *every* config ever evaluated, not
just the final strategies. The registry is the single sink every backtest run
reports to, so the number of trials feeding the statistics is the true one.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class _Trial:
    label: str
    returns: pd.Series


@dataclass
class TrialRegistry:
    trials: list[_Trial] = field(default_factory=list)

    def record(self, label: str, returns: pd.Series) -> None:
        self.trials.append(_Trial(label=label, returns=returns))

    @property
    def n_trials(self) -> int:
        return len(self.trials)

    def pnl_matrix(self) -> pd.DataFrame:
        """T x N matrix of per-bar returns, columns = trial labels (outer-joined index)."""
        return pd.DataFrame({t.label: t.returns for t in self.trials})

    def trial_sharpe_variance(self) -> float:
        """Variance of per-period Sharpes across trials (V for SR0)."""
        srs = []
        for t in self.trials:
            r = t.returns.dropna()
            if len(r) > 1 and r.std(ddof=1) > 0:
                srs.append(r.mean() / r.std(ddof=1))
        return float(np.var(srs, ddof=1)) if len(srs) > 1 else 0.0
