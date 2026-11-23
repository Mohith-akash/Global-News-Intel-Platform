"""Worker error paths: bad SQL, refused connection, breaker behaviour."""

import pytest

import src.database as db


@pytest.fixture(autouse=True)
def closed_breaker(monkeypatch):
    monkeypatch.setattr(db, "_breaker_until", 0.0)


def test_bad_sql_is_an_empty_result(tmp_path, monkeypatch):
    import duckdb
    path = tmp_path / "w.duckdb"
    duckdb.connect(str(path)).close()
    monkeypatch.setenv("GDELT_DB", str(path))
    out = db.safe_query_batch({"good": ("SELECT 1 AS x", None), "bad": ("SELECT * FROM nope", None)})
    assert out["good"].iloc[0]["x"] == 1 and out["bad"].empty
    assert db._breaker_until == 0.0


def test_refused_connection(tmp_path, monkeypatch):
    monkeypatch.setenv("GDELT_DB", str(tmp_path / "missing" / "w.duckdb"))
    # a single chat query just comes back empty, the dashboard keeps working
    assert db.safe_query("SELECT 1").empty
    assert db._breaker_until == 0.0
    # a dashboard batch opens the breaker
    with pytest.raises(db.WarehouseUnavailable):
        db.safe_query_batch({"a": ("SELECT 1", None)})
    assert db._breaker_until > 0
    with pytest.raises(db.WarehouseUnavailable):
        db.safe_query("SELECT 1")  # breaker open: no worker is started
