from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import duckdb

from . import discover
from .config import get_settings

log = logging.getLogger("tgapi.db")

# Columns present in the dataset's parquet files (Kzr0xx/telegram).
COLUMNS = [
    "user_id",
    "username",
    "first_name",
    "last_name",
    "phone",
    "email",
    "status",
    "linked_id",
    "linked_name",
    "linked_handle",
]

_conn: Optional[duckdb.DuckDBPyConnection] = None
_lock = threading.Lock()

# View names resolved at bootstrap; overridden when pre-sorted copies exist.
VIEW_PHONE = "tg"
VIEW_USERNAME = "tg"

# Filled in by _source_expr so /health can report what actually got wired up.
RESOLVED_FILES = 0
RESOLVED_SOURCE = ""


def _lit(value: str) -> str:
    """Single-quoted SQL literal with quotes escaped."""
    return "'" + str(value).replace("'", "''") + "'"


def _scan_of(spec: str) -> str:
    """Build a read_parquet(...) call from a comma-separated URL list or a
    single s3://path / glob."""
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        raise RuntimeError("empty parquet source spec")
    if len(parts) == 1:
        return f"read_parquet({_lit(parts[0])}, union_by_name := true)"
    joined = ", ".join(_lit(p) for p in parts)
    return f"read_parquet([{joined}], union_by_name := true)"


def _source_expr(s) -> str:
    """SQL expression naming the parquet source, as a scan function call.

    Priority: explicit PARQUET_URLS > S3 endpoint > hub-discovered shard URLs
    > hf:// glob over the dataset repo.
    """
    global RESOLVED_FILES, RESOLVED_SOURCE
    if s.urls:
        RESOLVED_FILES, RESOLVED_SOURCE = len(s.urls), "PARQUET_URLS"
        return _scan_of(s.parquet_urls)
    if s.use_s3:
        RESOLVED_FILES, RESOLVED_SOURCE = 0, s.parquet_glob
        return _scan_of(s.parquet_glob)
    if s.dataset_repo:
        if s.discover_parquet:
            found = discover.parquet_urls(
                s.dataset_repo,
                s.dataset_config,
                s.dataset_split,
                s.hf_token,
                s.dataset_revision,
            )
            if found:
                RESOLVED_FILES = len(found)
                RESOLVED_SOURCE = f"hub-api {s.dataset_repo}/{s.dataset_split}"
                return _scan_of(",".join(found))
        RESOLVED_FILES, RESOLVED_SOURCE = 0, s.hf_uri
        return _scan_of(s.hf_uri)
    raise RuntimeError(
        "No data source configured. Set DATASET_REPO, PARQUET_URLS or the S3_* vars."
    )


def _bootstrap(con: duckdb.DuckDBPyConnection) -> None:
    s = get_settings()
    ext_dir = os.environ.get("DUCKDB_EXTENSION_DIRECTORY", "/tmp/duckdb_extensions")
    os.makedirs(ext_dir, exist_ok=True)
    con.execute(f"SET extension_directory='{ext_dir}'")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute(f"SET memory_limit='{s.duckdb_memory_limit}'")
    con.execute(f"SET threads={max(1, s.duckdb_threads)}")
    con.execute("SET temp_directory='/tmp/duckdb'")
    # These three are what make repeat queries over remote parquet fast:
    # parquet footers/metadata stay cached instead of being re-fetched.
    for pragma in (
        "SET enable_object_cache=true",
        "SET enable_http_metadata_cache=true",
        "SET http_keep_alive=true",
        "SET http_retries=3",
        "SET http_timeout=60000",
    ):
        try:
            con.execute(pragma)
        except duckdb.Error as exc:  # option name varies across versions
            log.debug("skipping pragma %s (%s)", pragma, exc)

    # DDL can't take bind parameters, so literals are escaped by hand.
    if s.use_s3:
        host = s.s3_endpoint.replace("https://", "").replace("http://", "").rstrip("/")
        con.execute(
            f"""
            CREATE OR REPLACE SECRET hf_bucket (
                TYPE S3,
                KEY_ID {_lit(s.s3_access_key_id)},
                SECRET {_lit(s.s3_secret_access_key)},
                REGION {_lit(s.s3_region)},
                ENDPOINT {_lit(host)},
                URL_STYLE {_lit(s.s3_url_style)},
                USE_SSL true
            )
            """
        )
    elif s.hf_token:
        # hf:// paths authenticate through a HUGGINGFACE secret; plain https
        # resolve URLs need the header injected instead. Create both — unused
        # ones cost nothing.
        try:
            con.execute(
                "CREATE OR REPLACE SECRET hf_hub (TYPE HUGGINGFACE, TOKEN "
                f"{_lit(s.hf_token)})"
            )
        except duckdb.Error as exc:
            log.warning("HUGGINGFACE secret unavailable (%s)", exc)
        con.execute(
            f"""
            CREATE OR REPLACE SECRET hf_http (
                TYPE HTTP,
                EXTRA_HTTP_HEADERS MAP {{'Authorization': {_lit('Bearer ' + s.hf_token)}}}
            )
            """
        )

    con.execute(f"CREATE OR REPLACE VIEW tg AS SELECT * FROM {_source_expr(s)}")

    # Optional pre-sorted copies. When absent, the aliases fall back to `tg`
    # so query code never needs to branch.
    global VIEW_PHONE, VIEW_USERNAME
    VIEW_PHONE = "tg"
    VIEW_USERNAME = "tg"
    if s.phone_source:
        con.execute(
            f"CREATE OR REPLACE VIEW tg_phone AS SELECT * FROM {_scan_of(s.phone_source)}"
        )
        VIEW_PHONE = "tg_phone"
    if s.username_source:
        con.execute(
            "CREATE OR REPLACE VIEW tg_username AS SELECT * FROM "
            f"{_scan_of(s.username_source)}"
        )
        VIEW_USERNAME = "tg_username"
    log.info("duckdb ready (phone=%s username=%s)", VIEW_PHONE, VIEW_USERNAME)


def connection() -> duckdb.DuckDBPyConnection:
    global _conn
    if _conn is None:
        with _lock:
            if _conn is None:
                con = duckdb.connect(":memory:")
                _bootstrap(con)
                _conn = con
    return _conn


def close() -> None:
    global _conn
    with _lock:
        if _conn is not None:
            _conn.close()
            _conn = None


def query(sql: str, params: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
    """Run a query on a private cursor so concurrent requests don't collide."""
    cur = connection().cursor()
    try:
        cur.execute(sql, params or [])
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        cur.close()


def scalar(sql: str, params: Optional[List[Any]] = None) -> Any:
    rows = query(sql, params)
    if not rows:
        return None
    return next(iter(rows[0].values()))


# --------------------------------------------------------------------------
# tiny TTL cache — remote parquet scans are the expensive part, so never do
# the same one twice inside the TTL window
# --------------------------------------------------------------------------
_cache: Dict[Tuple[Any, ...], Tuple[float, Any]] = {}
_cache_lock = threading.Lock()


def cached(key: Tuple[Any, ...], producer):
    s = get_settings()
    now = time.time()
    with _cache_lock:
        hit = _cache.get(key)
        if hit and hit[0] > now:
            return hit[1], True
    value = producer()
    with _cache_lock:
        if len(_cache) >= s.cache_maxsize:
            for stale in [k for k, v in _cache.items() if v[0] <= now][:256]:
                _cache.pop(stale, None)
            if len(_cache) >= s.cache_maxsize:
                _cache.clear()
        _cache[key] = (now + s.cache_ttl_seconds, value)
    return value, False


def cache_stats() -> Dict[str, int]:
    with _cache_lock:
        return {"entries": len(_cache), "maxsize": get_settings().cache_maxsize}


def cache_clear() -> None:
    with _cache_lock:
        _cache.clear()
