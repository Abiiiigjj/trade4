import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_BASE_DIR = Path("data")


def _parquet_path(exchange: str, data_type: str, symbol: str, base_dir: Path) -> Path:
    return base_dir / exchange / data_type / f"{symbol}.parquet"


def save_df(
    df: pd.DataFrame,
    exchange: str,
    data_type: str,
    symbol: str,
    base_dir: Path = DEFAULT_BASE_DIR,
) -> None:
    path = _parquet_path(exchange, data_type, symbol, base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.debug("Saved %d rows to %s", len(df), path)


def load_df(
    exchange: str,
    data_type: str,
    symbol: str,
    base_dir: Path = DEFAULT_BASE_DIR,
) -> pd.DataFrame | None:
    path = _parquet_path(exchange, data_type, symbol, base_dir)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def get_last_timestamp(
    exchange: str,
    data_type: str,
    symbol: str,
    base_dir: Path = DEFAULT_BASE_DIR,
) -> pd.Timestamp | None:
    df = load_df(exchange, data_type, symbol, base_dir)
    if df is None or df.empty or "timestamp" not in df.columns:
        return None
    return df["timestamp"].max()
