from __future__ import annotations

import os
import sys
import urllib.error

import duckdb
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FIXTURE = "/tmp/tg_fixture.parquet"


@pytest.fixture(scope="session", autouse=True)
def fixture_env():
    con = duckdb.connect()
    con.execute(
        f"""
        COPY (
            SELECT i AS user_id,
                   'user' || i        AS username,
                   'First' || i       AS first_name,
                   'Last' || i        AS last_name,
                   '7964741' || (6000 + i) AS phone,
                   NULL::VARCHAR      AS email,
                   'active'           AS status,
                   NULL::VARCHAR      AS linked_id,
                   NULL::VARCHAR      AS linked_name,
                   NULL::VARCHAR      AS linked_handle
            FROM range(1, 501) t(i)
            ORDER BY user_id
        ) TO '{FIXTURE}' (FORMAT PARQUET, ROW_GROUP_SIZE 100)
        """
    )
    con.close()
    os.environ.update(
        PARQUET_URLS=FIXTURE,
        S3_ENDPOINT="",
        PHONE_SOURCE="",
        USERNAME_SOURCE="",
        REQUIRE_AUTH="true",
        API_KEYS="test-key",
        ALLOW_RAW_SQL="true",
        HF_TOKEN="",
    )
    yield


@pytest.fixture(scope="session")
def client(fixture_env):
    from fastapi.testclient import TestClient

    from app.config import get_settings
    from app.main import app

    get_settings.cache_clear()
    with TestClient(app) as c:
        yield c


HEAD = {"X-API-Key": "test-key"}


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["source_mode"] == "https"


def test_auth_required(client):
    assert client.get("/v1/users/1").status_code == 401
    assert client.get("/v1/users/1", headers={"X-API-Key": "nope"}).status_code == 403


def test_lookup_by_user_id(client):
    body = client.get("/v1/users/42", headers=HEAD).json()
    assert body["count"] == 1
    assert body["results"][0]["username"] == "user42"


def test_lookup_by_username_exact_and_prefix(client):
    exact = client.get("/v1/username/@USER7", headers=HEAD).json()
    assert exact["count"] == 1 and exact["results"][0]["user_id"] == 7
    pref = client.get("/v1/username/user7?prefix=true&limit=100", headers=HEAD).json()
    assert pref["count"] > 1


def test_lookup_by_phone_fuzzy_and_exact(client):
    exact = client.get("/v1/phone/79647416010?fuzzy=false", headers=HEAD).json()
    assert exact["count"] == 1 and exact["results"][0]["user_id"] == 10
    fuzzy = client.get("/v1/phone/+7 (964) 7416010", headers=HEAD).json()
    assert fuzzy["count"] >= 1


def test_search_combined_and_validation(client):
    ok = client.get("/v1/search?username=user1&prefix=true&limit=5", headers=HEAD)
    assert ok.status_code == 200 and ok.json()["count"] == 5
    assert client.get("/v1/search", headers=HEAD).status_code == 400


def test_cache_marks_second_call(client):
    client.post("/v1/cache/clear", headers=HEAD)
    first = client.get("/v1/users/99", headers=HEAD).json()
    second = client.get("/v1/users/99", headers=HEAD).json()
    assert first["cached"] is False and second["cached"] is True


def test_stats(client):
    body = client.get("/v1/stats", headers=HEAD).json()
    assert body["rows"] == 500
    assert body["min_user_id"] == 1 and body["max_user_id"] == 500


def test_raw_sql_guardrails(client):
    good = client.post(
        "/v1/sql", headers=HEAD, json={"sql": "SELECT count(*) c FROM tg", "limit": 1}
    )
    assert good.status_code == 200 and good.json()["results"][0]["c"] == 500
    for bad in ("DROP TABLE tg", "SELECT 1; DROP TABLE tg", "COPY tg TO '/tmp/x.csv'"):
        assert client.post("/v1/sql", headers=HEAD, json={"sql": bad}).status_code == 400


def test_limit_is_clamped(client):
    body = client.get("/v1/search?username=user&prefix=true&limit=9999", headers=HEAD)
    assert body.json()["limit"] <= 200


def test_source_and_schema_endpoints(client):
    src = client.get("/v1/source", headers=HEAD).json()
    assert src["mode"] == "https" and src["file_count"] == 1
    cols = client.get("/v1/schema", headers=HEAD).json()["columns"]
    assert {c["column_name"] for c in cols} >= {"user_id", "username", "phone"}


def test_discovery_flatten_shapes():
    from app import discover

    assert discover._flatten(["a.parquet", "b.parquet"]) == ["a.parquet", "b.parquet"]
    assert discover._flatten([{"url": "c.parquet"}]) == ["c.parquet"]
    assert discover._flatten({"parquet_files": [{"url": "d.parquet"}]}) == ["d.parquet"]
    assert discover._flatten({"unexpected": 1}) == []


def test_discovery_uses_cache(monkeypatch, tmp_path):
    from app import discover

    monkeypatch.setattr(discover, "CACHE_PATH", str(tmp_path / "c.json"))
    calls = {"n": 0}

    def fake_get(url, token="", timeout=20):
        calls["n"] += 1
        return ["https://example.invalid/x.parquet"]

    monkeypatch.setattr(discover, "_get_json", fake_get)
    first = discover.parquet_urls("a/b", "default", "train")
    second = discover.parquet_urls("a/b", "default", "train")
    assert first == second == ["https://example.invalid/x.parquet"]
    assert calls["n"] == 1  # second call served from disk cache


def test_discovery_failure_is_soft(monkeypatch, tmp_path):
    from app import discover

    monkeypatch.setattr(discover, "CACHE_PATH", str(tmp_path / "c2.json"))

    def boom(url, token="", timeout=20):
        raise urllib.error.URLError("no network")

    monkeypatch.setattr(discover, "_get_json", boom)
    assert discover.parquet_urls("a/b") == []
