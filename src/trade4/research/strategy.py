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
    """

    name: str = "strategy"

    @abstractmethod
    def generate_target_weights(self, panel: Panel) -> pd.DataFrame:
        """Return weights ``[time x symbol]``; rows should sum to ~0 (dollar-neutral)."""
        raise NotImplementedError
