from __future__ import annotations

import logging
import re
from contextlib import asynccontextmanager
from typing import Optional

import duckdb
from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse

from . import db, discover, queries
from .config import get_settings
from .models import Health, SearchResponse, SQLRequest, TelegramUser
from .security import require_api_key

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
log = logging.getLogger("tgapi")


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    if s.warmup_in_background:
        # Bind the port right away; Render's health check only needs an HTTP
        # 200 from /health, and reading ~100 parquet footers over the network
        # can take a minute on a small instance.
        db.start_warmup()
    else:
        try:
            db.connection()
            log.info("warm-up ok in %.1fs", db.WARMUP_SECONDS)
        except Exception as exc:  # keep the service up so /health can report why
            log.error("warm-up failed: %s", exc)
    yield
    db.close()


app = FastAPI(
    title="Telegram Data API",
    description="Fast read-only API over the Kzr0xx/telegram parquet dataset on "
    "Hugging Face, served by DuckDB. Authenticate with `X-API-Key`.",
    version="1.0.0",
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)

_settings = get_settings()
app.add_middleware(GZipMiddleware, minimum_size=800)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root():
    return {"service": "telegram-data-api", "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=Health, tags=["meta"])
def health():
    """Liveness + readiness in one. Always returns 200 so the platform keeps
    the instance alive while the engine warms up; `engine` says what's going
    on and `error` carries the exact failure if warm-up broke."""
    s = get_settings()
    engine = db.STATE
    error = db.LAST_ERROR
    if engine == "ready":
        try:
            db.scalar("SELECT 1")
        except Exception as exc:
            engine, error = "failed", str(exc)
    elif engine in ("cold", "failed") and not s.warmup_in_background:
        # synchronous mode: probe (and retry) inline
        try:
            db.scalar("SELECT 1")
            engine, error = "ready", ""
        except Exception as exc:
            engine, error = "failed", str(exc)
    elif engine == "failed":
        # a previous background warm-up failed; kick off another attempt so a
        # transient hub hiccup doesn't leave the service degraded forever
        db.start_warmup()

    status = {"ready": "ok", "warming": "starting", "cold": "starting"}.get(
        engine, "degraded"
    )
    return Health(
        status=status,
        engine=engine,
        error=error or None,
        warmup_seconds=round(db.WARMUP_SECONDS, 1),
        duckdb=duckdb.__version__,
        source_mode=s.source_mode,
        source=db.RESOLVED_SOURCE or s.hf_uri,
        files=db.RESOLVED_FILES,
        id_column=db.ID_COL,
        cache=db.cache_stats(),
    )


@app.get("/v1/source", tags=["meta"])
def source(_: str = Depends(require_api_key)):
    """What the engine is actually reading, and the shard list the hub reported."""
    s = get_settings()
    shards = []
    if s.source_mode == "hf-dataset" and s.discover_parquet:
        shards = discover.parquet_urls(
            s.dataset_repo,
            s.dataset_config,
            s.dataset_split,
            s.hf_token,
            s.dataset_revision,
        )
    return {
        "mode": s.source_mode,
        "resolved_source": db.RESOLVED_SOURCE,
        "file_count": db.RESOLVED_FILES,
        "dataset": {
            "repo": s.dataset_repo,
            "config": s.dataset_config,
            "split": s.dataset_split,
            "revision": s.dataset_revision,
        },
        "views": {"phone": db.VIEW_PHONE, "username": db.VIEW_USERNAME},
        "id_column": {"name": db.ID_COL, "stored_as_text": db.ID_IS_TEXT},
        "raw_schema": db.RAW_SCHEMA,
        "shards": shards,
    }


@app.get("/v1/schema", tags=["meta"])
def schema(_: str = Depends(require_api_key)):
    return _guard(lambda: {"columns": db.query("DESCRIBE SELECT * FROM tg")})


@app.get("/v1/users/{user_id}", response_model=SearchResponse, tags=["lookup"])
def get_user(
    user_id: int = Path(ge=0, description="Telegram numeric user id"),
    _: str = Depends(require_api_key),
):
    return _guard(lambda: queries.by_user_id(user_id))


@app.get("/v1/phone/{phone}", response_model=SearchResponse, tags=["lookup"])
def get_by_phone(
    phone: str = Path(min_length=5, max_length=24),
    fuzzy: bool = Query(True, description="match on suffix, ignoring country code"),
    limit: int = Query(50, ge=1),
    offset: int = Query(0, ge=0),
    _: str = Depends(require_api_key),
):
    return _guard(lambda: queries.by_phone(phone, fuzzy, limit, offset))


@app.get("/v1/username/{username}", response_model=SearchResponse, tags=["lookup"])
def get_by_username(
    username: str = Path(min_length=1, max_length=64),
    prefix: bool = Query(False, description="treat the value as a prefix"),
    limit: int = Query(50, ge=1),
    offset: int = Query(0, ge=0),
    _: str = Depends(require_api_key),
):
    return _guard(lambda: queries.by_username(username, prefix, limit, offset))


@app.get("/v1/search", response_model=SearchResponse, tags=["lookup"])
def search(
    user_id: Optional[int] = Query(None, ge=0),
    username: Optional[str] = Query(None, max_length=64),
    phone: Optional[str] = Query(None, max_length=24),
    name: Optional[str] = Query(None, max_length=64),
    email: Optional[str] = Query(None, max_length=128),
    status: Optional[str] = Query(None, max_length=32),
    linked_id: Optional[str] = Query(None, max_length=64),
    limit: int = Query(50, ge=1),
    offset: int = Query(0, ge=0),
    _: str = Depends(require_api_key),
):
    return _guard(
        lambda: queries.search(
            user_id, username, phone, name, email, status, linked_id, limit, offset
        )
    )


@app.get("/v1/stats", tags=["meta"])
def stats(_: str = Depends(require_api_key)):
    return _guard(queries.stats)


@app.post("/v1/cache/clear", tags=["meta"])
def clear_cache(
    discovery: bool = Query(False, description="also drop the cached shard list"),
    _: str = Depends(require_api_key),
):
    db.cache_clear()
    if discovery:
        discover.clear_cache()
    return {"cleared": True, "discovery": discovery}


_FORBIDDEN = re.compile(
    r"\b(attach|copy|create|delete|drop|export|insert|install|load|pragma|"
    r"update|alter|call|set)\b",
    re.IGNORECASE,
)


@app.post("/v1/sql", tags=["advanced"])
def raw_sql(body: SQLRequest, request: Request, _: str = Depends(require_api_key)):
    """Read-only SELECT passthrough. Disabled unless ALLOW_RAW_SQL=true."""
    if not get_settings().allow_raw_sql:
        raise HTTPException(403, "raw SQL is disabled")
    sql = body.sql.strip().rstrip(";")
    if ";" in sql:
        raise HTTPException(400, "one statement only")
    if not re.match(r"^(select|with)\b", sql, re.IGNORECASE):
        raise HTTPException(400, "only SELECT/WITH is allowed")
    if _FORBIDDEN.search(sql):
        raise HTTPException(400, "statement contains a forbidden keyword")
    rows = db.query(f"SELECT * FROM ({sql}) LIMIT {int(body.limit)}")
    return {"count": len(rows), "results": rows}


def _guard(fn):
    try:
        return fn()
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except duckdb.Error as exc:
        log.exception("query failed")
        raise HTTPException(502, f"data source error: {exc}") from exc
    except RuntimeError as exc:
        # engine still warming up, or bootstrap failed — tell the client to
        # come back rather than surfacing a stack trace
        raise HTTPException(
            503, str(exc), headers={"Retry-After": "10"}
        ) from exc


__all__ = ["app", "TelegramUser"]
