"""Point-in-time market-data container for the portfolio engine."""
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class Panel:
    """Aligned point-in-time market data on a fixed bar grid.

    All frames share the same index (time) and columns (symbol). ``tradeable[t, s]``
    is True iff symbol ``s`` has a finite price at ``t`` (i.e. it is listed and has
    data). This mask makes the point-in-time universe enforceable: the engine asserts
    no weight is placed on an untradeable symbol, which would be universe-level
    look-ahead (knowing a symbol will list before it does).
    """

    close: pd.DataFrame
    funding: pd.DataFrame
    open_interest: pd.DataFrame | None = None
    tradeable: pd.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.close.index, pd.DatetimeIndex) or self.close.index.tz is None:
            raise ValueError("close index must be a tz-aware DatetimeIndex")
        if not self.close.index.equals(self.funding.index):
            raise ValueError("close/funding index mismatch")
        if list(self.close.columns) != list(self.funding.columns):
            raise ValueError("close/funding columns mismatch")
        if self.open_interest is not None:
            if not self.close.index.equals(self.open_interest.index):
                raise ValueError("close/open_interest index mismatch")
            if list(self.close.columns) != list(self.open_interest.columns):
                raise ValueError("close/open_interest columns mismatch")
        self.tradeable = self.close.notna()

    @property
    def times(self) -> pd.DatetimeIndex:
        return self.close.index

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)
