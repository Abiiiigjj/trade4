"""Run manifest — makes every result reproducible.

A result you cannot reproduce is a result you cannot trust. Each run records the
hash of its input data, the git commit of the code, the RNG seed, and the params.
"""
import hashlib
import subprocess
from dataclasses import dataclass, field, asdict
from typing import Any

import pandas as pd


def data_hash(df: pd.DataFrame) -> str:
    """Stable 16-hex-char hash of a DataFrame's values and index."""
    h = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return h.hexdigest()[:16]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


@dataclass
class RunManifest:
    data_hash: str
    git_commit: str
    seed: int
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
