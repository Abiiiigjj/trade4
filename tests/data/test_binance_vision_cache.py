import pandas as pd

import trade4.data.binance_vision as bv


def test_cache_roundtrip_avoids_second_download(tmp_path, monkeypatch):
    monkeypatch.setattr(bv, "_CACHE_DIR", tmp_path)
    calls = {"n": 0}

    def fake_download(url):
        calls["n"] += 1
        # minimal funding CSV: calc_time(ms), interval_hours, last_funding_rate
        return pd.DataFrame([[1_700_000_000_000, 8, 0.0001]])

    monkeypatch.setattr(bv, "_download_csv", fake_download)

    first = bv.fetch_funding_month("TESTUSDT", 2023, 9)
    second = bv.fetch_funding_month("TESTUSDT", 2023, 9)

    assert calls["n"] == 1  # second call served from cache, no re-download
    pd.testing.assert_frame_equal(first, second)
    assert (tmp_path / "funding" / "TESTUSDT-2023-09.parquet").exists()


def test_cache_stores_empty_404_result(tmp_path, monkeypatch):
    monkeypatch.setattr(bv, "_CACHE_DIR", tmp_path)
    calls = {"n": 0}

    def fake_download(url):
        calls["n"] += 1
        return None  # 404

    monkeypatch.setattr(bv, "_download_csv", fake_download)

    a = bv.fetch_funding_month("DEADUSDT", 2099, 1)
    b = bv.fetch_funding_month("DEADUSDT", 2099, 1)
    assert a.empty and b.empty
    assert calls["n"] == 1  # empty 404 result is cached too
