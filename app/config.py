from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # data source
    # Default: the Hugging Face dataset repo, read directly over hf://.
    dataset_repo: str = "sunsau91/fullteegee"
    dataset_revision: str = "main"
    dataset_glob: str = "**/*.parquet"
    dataset_config: str = "default"
    dataset_split: str = "train"
    # Ask the hub which parquet shards exist instead of globbing blindly:
    # GET /api/datasets/{repo}/parquet/{config}/{split}
    discover_parquet: bool = True

    # Overrides. PARQUET_URLS wins over S3, S3 wins over the dataset repo.
    parquet_urls: str = ""
    parquet_glob: str = ""
    hf_token: str = ""

    # optional lookup-optimized copies produced by scripts/build_indexes.py.
    # Same format as the main source (URL list, or s3:// path when in S3 mode).
    phone_source: str = ""
    username_source: str = ""

    s3_endpoint: str = ""
    s3_access_key_id: str = ""
    s3_secret_access_key: str = ""
    s3_region: str = "us-east-1"
    s3_url_style: str = "path"

    # api
    api_keys: str = "change-me-now"
    require_auth: bool = True
    allow_raw_sql: bool = False
    cors_origins: str = "*"

    # engine
    duckdb_memory_limit: str = "350MB"
    duckdb_threads: int = 2
    max_limit: int = 200
    cache_ttl_seconds: int = 900
    cache_maxsize: int = 1024
    query_timeout_seconds: int = 110

    @property
    def urls(self) -> List[str]:
        return [u.strip() for u in self.parquet_urls.split(",") if u.strip()]

    @property
    def keys(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    @property
    def origins(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def use_s3(self) -> bool:
        return bool(self.s3_endpoint and self.parquet_glob)

    @property
    def use_hf_dataset(self) -> bool:
        return not self.urls and not self.use_s3 and bool(self.dataset_repo)

    @property
    def hf_uri(self) -> str:
        """hf://datasets/<repo>@<revision>/<glob>"""
        rev = f"@{self.dataset_revision}" if self.dataset_revision else ""
        return f"hf://datasets/{self.dataset_repo}{rev}/{self.dataset_glob.lstrip('/')}"

    @property
    def source_mode(self) -> str:
        if self.urls:
            return "https"
        if self.use_s3:
            return "s3"
        return "hf-dataset"


@lru_cache
def get_settings() -> Settings:
    return Settings()
