import pandas as pd

from trade4.research.manifest import RunManifest, data_hash


def test_data_hash_is_stable():
    idx = pd.date_range("2023-01-01", periods=3, freq="8h", tz="UTC")
    df = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=idx)
    assert data_hash(df) == data_hash(df.copy())


def test_data_hash_changes_with_data():
    idx = pd.date_range("2023-01-01", periods=3, freq="8h", tz="UTC")
    a = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=idx)
    b = pd.DataFrame({"A": [1.0, 2.0, 3.1]}, index=idx)
    assert data_hash(a) != data_hash(b)


def test_manifest_roundtrip():
    m = RunManifest(data_hash="abc", git_commit="def", seed=7, params={"k": 1})
    d = m.to_dict()
    assert d["data_hash"] == "abc" and d["seed"] == 7
