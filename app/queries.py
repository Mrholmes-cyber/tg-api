from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Tuple

from . import db
from .config import get_settings

_DIGITS = re.compile(r"\D+")

SELECT_COLS = ", ".join(db.COLUMNS)


def normalize_phone(raw: str) -> str:
    return _DIGITS.sub("", raw or "")


def normalize_username(raw: str) -> str:
    return (raw or "").strip().lstrip("@").lower()


def _clamp(limit: int, offset: int) -> Tuple[int, int]:
    s = get_settings()
    return max(1, min(limit, s.max_limit)), max(0, offset)


def _run(sql: str, params: List[Any], limit: int, offset: int) -> Dict[str, Any]:
    key = (sql, tuple(params))
    started = time.perf_counter()
    rows, was_cached = db.cached(key, lambda: db.query(sql, params))
    return {
        "count": len(rows),
        "limit": limit,
        "offset": offset,
        "took_ms": int((time.perf_counter() - started) * 1000),
        "cached": was_cached,
        "results": rows,
    }


def _id_filter(user_id: int) -> Tuple[str, Any]:
    """Filter on the *raw* id column so DuckDB can push the predicate into the
    parquet scan and prune row groups by min/max statistics. Filtering on the
    casted `user_id` alias would defeat that."""
    predicate, cast = db.id_predicate()
    return predicate, cast(user_id)


def by_user_id(user_id: int) -> Dict[str, Any]:
    """Fastest path when the id column is sorted in the parquet files."""
    predicate, value = _id_filter(user_id)
    sql = f"SELECT {SELECT_COLS} FROM tg WHERE {predicate} LIMIT 50"
    return _run(sql, [value], 50, 0)


def by_phone(phone: str, fuzzy: bool, limit: int, offset: int) -> Dict[str, Any]:
    limit, offset = _clamp(limit, offset)
    digits = normalize_phone(phone)
    view = db.VIEW_PHONE
    # Pre-sorted copies carry a materialized phone_norm column, which lets
    # DuckDB prune row groups instead of regexp-ing every row.
    col = "phone_norm" if view != "tg" else "regexp_replace(phone, '[^0-9]', '', 'g')"
    predicate = f"{col} LIKE '%' || ?" if fuzzy else f"{col} = ?"
    sql = f"SELECT {SELECT_COLS} FROM {view} WHERE {predicate} LIMIT ? OFFSET ?"
    return _run(sql, [digits, limit, offset], limit, offset)


def by_username(
    username: str, prefix: bool, limit: int, offset: int
) -> Dict[str, Any]:
    limit, offset = _clamp(limit, offset)
    handle = normalize_username(username)
    view = db.VIEW_USERNAME
    col = "username_lc" if view != "tg" else "lower(username)"
    if prefix:
        sql = f"SELECT {SELECT_COLS} FROM {view} WHERE {col} LIKE ? LIMIT ? OFFSET ?"
        params: List[Any] = [handle + "%", limit, offset]
    else:
        sql = f"SELECT {SELECT_COLS} FROM {view} WHERE {col} = ? LIMIT ? OFFSET ?"
        params = [handle, limit, offset]
    return _run(sql, params, limit, offset)


def search(
    user_id: Optional[int],
    username: Optional[str],
    phone: Optional[str],
    name: Optional[str],
    email: Optional[str],
    status: Optional[str],
    linked_id: Optional[str],
    limit: int,
    offset: int,
) -> Dict[str, Any]:
    """Combined filter. Every clause is parameterized; nothing is interpolated."""
    limit, offset = _clamp(limit, offset)
    where: List[str] = []
    params: List[Any] = []

    if user_id is not None:
        predicate, value = _id_filter(user_id)
        where.append(predicate)
        params.append(value)
    if username:
        where.append("lower(username) LIKE ?")
        params.append(normalize_username(username) + "%")
    if phone:
        where.append("regexp_replace(phone, '[^0-9]', '', 'g') LIKE '%' || ?")
        params.append(normalize_phone(phone))
    if name:
        where.append(
            "lower(coalesce(first_name,'') || ' ' || coalesce(last_name,'')) LIKE ?"
        )
        params.append("%" + name.strip().lower() + "%")
    if email:
        where.append("lower(email) = ?")
        params.append(email.strip().lower())
    if status:
        where.append("lower(status) = ?")
        params.append(status.strip().lower())
    if linked_id:
        where.append("linked_id = ?")
        params.append(linked_id.strip())

    if not where:
        raise ValueError("provide at least one filter")

    sql = (
        f"SELECT {SELECT_COLS} FROM tg WHERE "
        + " AND ".join(where)
        + " LIMIT ? OFFSET ?"
    )
    params += [limit, offset]
    return _run(sql, params, limit, offset)


def stats() -> Dict[str, Any]:
    def producer() -> Dict[str, Any]:
        row = db.query(
            """
            SELECT count(*) AS "rows",
                   min(user_id) AS min_user_id,
                   max(user_id) AS max_user_id,
                   count(username) AS with_username,
                   count(phone)    AS with_phone,
                   count(email)    AS with_email
            FROM tg
            """
        )[0]
        return {k: (int(v) if v is not None else None) for k, v in row.items()}

    value, was_cached = db.cached(("stats",), producer)
    return {**value, "cached": was_cached}
