from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class TelegramUser(BaseModel):
    user_id: Optional[int] = None
    username: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    status: Optional[str] = None
    linked_id: Optional[str] = None
    linked_name: Optional[str] = None
    linked_handle: Optional[str] = None


class SearchResponse(BaseModel):
    count: int = Field(description="rows returned in this page")
    limit: int
    offset: int
    took_ms: int
    cached: bool = False
    results: List[TelegramUser] = []


class Health(BaseModel):
    status: str = Field(description="ok | starting | degraded")
    engine: str = Field(description="cold | warming | ready | failed")
    error: Optional[str] = None
    warmup_seconds: float = 0.0
    duckdb: str
    source_mode: str
    source: str = ""
    files: int = 0
    id_column: str = "user_id"
    cache: dict


class SQLRequest(BaseModel):
    sql: str = Field(min_length=8, max_length=4000)
    limit: int = Field(default=100, ge=1, le=5000)
