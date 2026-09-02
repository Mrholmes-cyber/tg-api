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

# Columns the API exposes. The `tg` view always has exactly these (plus the raw
# id column), regardless of what the underlying parquet files call them.
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

# Parquet column names that should be exposed as `user_id`. Different dumps of
# this dataset use different names; the first one found wins.
ID_ALIASES = ["user_id", "account_id", "id", "uid", "telegram_id"]

_conn: Optional[duckdb.DuckDBPyConnection] = None
_lock = threading.Lock()

# Warm-up lifecycle -----------------------------------------------------------
# cold    -> nothing has happened yet
# warming -> bootstrap running in a background thread
# ready   -> connection usable
# failed  -> bootstrap raised; LAST_ERROR has the message; retried on demand
STATE = "cold"
LAST_ERROR = ""
WARMUP_SECONDS = 0.0

# View names resolved at bootstrap; overridden when pre-sorted copies exist.
VIEW_PHONE = "tg"
VIEW_USERNAME = "tg"

# How the id column is stored in the parquet files. Filtering on the *raw*
# column (instead of the casted alias) is what lets DuckDB prune row groups.
ID_COL = "user_id"
ID_IS_TEXT = False

# Filled in by _source_expr so /health can report what actually got wired up.
RESOLVED_FILES = 0
RESOLVED_SOURCE = ""
RAW_SCHEMA: List[Dict[str, str]] = []


def _lit(value: str) -> str:
    """Single-quoted SQL literal with quotes escaped."""
    return "'" + str(value).replace("'", "''") + "'"


def _ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _scan_of(spec: str, union_by_name: bool) -> str:
    """Build a read_parquet(...) call from a comma-separated URL list or a
    single s3://path / glob."""
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        raise RuntimeError("empty parquet source spec")
    opts = ", union_by_name := true" if union_by_name else ""
    if len(parts) == 1:
        return f"read_parquet({_lit(parts[0])}{opts})"
    joined = ", ".join(_lit(p) for p in parts)
    return f"read_parquet([{joined}]{opts})"


def _source_expr(s) -> str:
    """SQL expression naming the parquet source, as a scan function call.

    Priority: explicit PARQUET_URLS > S3 endpoint > hub-discovered shard URLs
    > hf:// glob over the dataset repo.
    """
    global RESOLVED_FILES, RESOLVED_SOURCE
    ubn = s.parquet_union_by_name
    if s.urls:
        RESOLVED_FILES, RESOLVED_SOURCE = len(s.urls), "PARQUET_URLS"
        return _scan_of(s.parquet_urls, ubn)
    if s.use_s3:
        RESOLVED_FILES, RESOLVED_SOURCE = 0, s.parquet_glob
        return _scan_of(s.parquet_glob, ubn)
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
                RESOLVED_SOURCE = f"hub {s.dataset_repo}@{s.dataset_revision}"
                return _scan_of(",".join(found), ubn)
            log.warning("discovery found nothing; falling back to %s", s.hf_uri)
        RESOLVED_FILES, RESOLVED_SOURCE = 0, s.hf_uri
        return _scan_of(s.hf_uri, ubn)
    raise RuntimeError(
        "No data source configured. Set DATASET_REPO, PARQUET_URLS or the S3_* vars."
    )


def _projection(raw_cols: Dict[str, str]) -> str:
    """Map whatever the parquet files contain onto COLUMNS.

    - the id column is exposed as `user_id` (BIGINT) *and* kept under its raw
      name so lookups can filter on the stored value
    - missing columns become NULL so query code never has to branch
    """
    global ID_COL, ID_IS_TEXT
    lower = {c.lower(): c for c in raw_cols}

    id_src = next((lower[a] for a in ID_ALIASES if a in lower), None)
    parts: List[str] = []
    if id_src is None:
        ID_COL, ID_IS_TEXT = "user_id", False
        parts.append("NULL::BIGINT AS user_id")
    else:
        ID_COL = id_src
        ID_IS_TEXT = "VARCHAR" in raw_cols[id_src].upper()
        if ID_IS_TEXT:
            parts.append(f"TRY_CAST({_ident(id_src)} AS BIGINT) AS user_id")
        elif id_src.lower() != "user_id":
            parts.append(f"{_ident(id_src)}::BIGINT AS user_id")
        else:
            parts.append(f"{_ident(id_src)}::BIGINT AS user_id")
        if id_src.lower() != "user_id":
            parts.append(_ident(id_src))

    for col in COLUMNS[1:]:
        src = lower.get(col)
        if src is None:
            parts.append(f"NULL::VARCHAR AS {col}")
        elif "VARCHAR" in raw_cols[src].upper():
            parts.append(f"{_ident(src)} AS {col}" if src != col else _ident(col))
        else:
            parts.append(f"CAST({_ident(src)} AS VARCHAR) AS {col}")
    return ", ".join(parts)


def _bootstrap(con: duckdb.DuckDBPyConnection) -> None:
    global RAW_SCHEMA
    s = get_settings()
    ext_dir = os.environ.get("DUCKDB_EXTENSION_DIRECTORY", "/tmp/duckdb_extensions")
    os.makedirs(ext_dir, exist_ok=True)
    con.execute(f"SET extension_directory='{ext_dir}'")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute(f"SET memory_limit='{s.duckdb_memory_limit}'")
    con.execute(f"SET threads={max(1, s.duckdb_threads)}")
    os.makedirs("/tmp/duckdb", exist_ok=True)
    con.execute("SET temp_directory='/tmp/duckdb'")
    # Keep parquet footers/metadata cached between queries and be patient with
    # the HF CDN. NOTE: http_timeout is in *seconds* since duckdb 1.2.
    for pragma in (
        "SET enable_http_metadata_cache=true",
        "SET parquet_metadata_cache=true",
        "SET http_keep_alive=true",
        f"SET http_retries={int(s.http_retries)}",
        f"SET http_timeout={int(s.http_timeout_seconds)}",
        "SET http_retry_wait_ms=500",
        "SET http_retry_backoff=2",
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
                SCOPE 'https://huggingface.co',
                EXTRA_HTTP_HEADERS MAP {{'Authorization': {_lit('Bearer ' + s.hf_token)}}}
            )
            """
        )

    src = _source_expr(s)
    con.execute(f"CREATE OR REPLACE VIEW tg_raw AS SELECT * FROM {src}")
    raw = {
        r[0]: r[1]
        for r in con.execute("DESCRIBE SELECT * FROM tg_raw").fetchall()
    }
    RAW_SCHEMA = [{"name": k, "type": v} for k, v in raw.items()]
    con.execute(f"CREATE OR REPLACE VIEW tg AS SELECT {_projection(raw)} FROM tg_raw")
    log.info(
        "schema: raw=%s id_col=%s text_id=%s", list(raw), ID_COL, ID_IS_TEXT
    )

    # Optional pre-sorted copies. When absent, the aliases fall back to `tg`
    # so query code never needs to branch.
    global VIEW_PHONE, VIEW_USERNAME
    VIEW_PHONE = "tg"
    VIEW_USERNAME = "tg"
    if s.phone_source:
        con.execute(
            "CREATE OR REPLACE VIEW tg_phone AS SELECT * FROM "
            f"{_scan_of(s.phone_source, True)}"
        )
        VIEW_PHONE = "tg_phone"
    if s.username_source:
        con.execute(
            "CREATE OR REPLACE VIEW tg_username AS SELECT * FROM "
            f"{_scan_of(s.username_source, True)}"
        )
        VIEW_USERNAME = "tg_username"
    log.info("duckdb ready (phone=%s username=%s)", VIEW_PHONE, VIEW_USERNAME)


def _connect_locked() -> duckdb.DuckDBPyConnection:
    """Bootstrap synchronously. Caller holds _lock."""
    global _conn, STATE, LAST_ERROR, WARMUP_SECONDS
    STATE = "warming"
    started = time.perf_counter()
    con = duckdb.connect(":memory:")
    try:
        _bootstrap(con)
    except Exception as exc:
        con.close()
        STATE, LAST_ERROR = "failed", f"{type(exc).__name__}: {exc}"
        WARMUP_SECONDS = time.perf_counter() - started
        raise
    _conn = con
    STATE, LAST_ERROR = "ready", ""
    WARMUP_SECONDS = time.perf_counter() - started
    return con


def connection() -> duckdb.DuckDBPyConnection:
    """Return the shared connection, bootstrapping it if needed.

    If a background warm-up is in flight, raise instead of piling a second
    bootstrap behind the lock — callers turn that into a 503.
    """
    if _conn is not None:
        return _conn
    if STATE == "warming" and _lock.locked():
        raise RuntimeError("data engine is still warming up; retry in a moment")
    with _lock:
        if _conn is not None:
            return _conn
        return _connect_locked()


def start_warmup() -> threading.Thread:
    """Bootstrap in a daemon thread so the HTTP port binds immediately."""

    def run() -> None:
        try:
            connection()
            log.info(
                "warm-up ok in %.1fs (%s, %d files)",
                WARMUP_SECONDS,
                RESOLVED_SOURCE,
                RESOLVED_FILES,
            )
        except Exception as exc:  # /health reports LAST_ERROR
            log.error("warm-up failed after %.1fs: %s", WARMUP_SECONDS, exc)

    t = threading.Thread(target=run, name="duckdb-warmup", daemon=True)
    t.start()
    return t


def close() -> None:
    global _conn, STATE
    with _lock:
        if _conn is not None:
            _conn.close()
            _conn = None
        STATE = "cold"


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


def id_predicate() -> Tuple[str, Any]:
    """(sql fragment, param-caster) for an equality filter on the id column
    that DuckDB can push into the parquet scan."""
    if ID_IS_TEXT:
        return f"{_ident(ID_COL)} = ?", str
    return f"{_ident(ID_COL)} = ?", int


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
