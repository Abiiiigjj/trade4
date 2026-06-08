"""Strategy interface for the research harness."""
from abc import ABC, abstractmethod
import pandas as pd

from trade4.research.panel import Panel


class Strategy(ABC):
    """Emits dollar-neutral target weights over a universe.

    CONTRACT: the weights at time ``t`` may depend on panel data with index ``<= t``
    only. The portfolio engine enforces this with a future-perturbation tripwire
    (:func:`trade4.research.portfolio_engine.assert_causal`). A strategy that peeks
    into the future is caught structurally, not by inspection.

    NEUTRALITY is a *per-strategy* property, not a framework invariant. Cross-sectional
    strategies are dollar-neutral (rows sum ~0) and should enforce/check that themselves.
    A delta-neutral strategy like ``funding_carry`` is instead all-shorts on the perp leg,
    neutral via an external spot hedge — run it with ``EngineConfig.price_pnl_enabled=False``.
    Do not assume rows always sum to zero.
    """

    name: str = "strategy"

    @abstractmethod
    def generate_target_weights(self, panel: Panel) -> pd.DataFrame:
        """Return weights ``[time x symbol]``; rows should sum to ~0 (dollar-neutral)."""
        raise NotImplementedError
