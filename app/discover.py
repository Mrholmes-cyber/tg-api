from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from urllib.parse import quote
from typing import List, Optional

log = logging.getLogger("tgapi.discover")

HUB_API = "https://huggingface.co/api/datasets/{repo}/parquet/{config}/{split}"
DATASET_SERVER_API = "https://datasets-server.huggingface.co/parquet?dataset={repo}"
TREE_API = "https://huggingface.co/api/datasets/{repo}/tree/{revision}?recursive=true"
VIEWER_SPLITS = "https://datasets-server.huggingface.co/splits?dataset={repo}"
CACHE_PATH = os.environ.get("DISCOVER_CACHE", "/tmp/tg_parquet_urls.json")
CACHE_TTL = 15 * 60


def _get_json(url: str, token: str = "", timeout: int = 20):
    req = urllib.request.Request(url, headers={"User-Agent": "tg-api/1.0"})
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _flatten(payload) -> List[str]:
    """The hub returns a plain list of URLs; older/other shapes nest them."""
    if isinstance(payload, list):
        out = []
        for item in payload:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                url = item.get("url") or item.get("filename") or item.get("path")
                if url:
                    out.append(url)
        return out
    if isinstance(payload, dict):
        for key in ("parquet_files", "urls", "files"):
            if key in payload:
                return _flatten(payload[key])
    return []


def _tree_parquet_urls(repo: str, revision: str, token: str = "") -> List[str]:
    """Build resolve URLs from repository files when the parquet API is stale."""
    try:
        payload = _get_json(
            TREE_API.format(repo=repo, revision=quote(revision, safe="")), token
        )
    except (urllib.error.URLError, ValueError, TimeoutError) as exc:
        log.warning("repository tree discovery failed for %s@%s: %s", repo, revision, exc)
        return []

    paths = [path for path in _flatten(payload) if path.lower().endswith(".parquet")]
    urls = [
        f"https://huggingface.co/datasets/{repo}/resolve/"
        f"{quote(revision, safe='')}/{quote(path, safe='/')}"
        for path in paths
    ]
    return [_resolve_download_url(url, token) for url in urls]


def _dataset_server_urls(repo: str, config: str, split: str, token: str = "") -> List[str]:
    """Get the dataset-server parquet files, which are range-readable on Render."""
    try:
        payload = _get_json(DATASET_SERVER_API.format(repo=quote(repo, safe="")), token)
    except (urllib.error.URLError, ValueError, TimeoutError) as exc:
        log.warning("dataset-server discovery failed for %s/%s: %s", repo, split, exc)
        return []
    if not isinstance(payload, dict):
        return []
    return [
        item["url"]
        for item in payload.get("parquet_files", [])
        if isinstance(item, dict)
        and item.get("config") == config
        and item.get("split") == split
        and item.get("url")
    ]


def _resolve_download_url(url: str, token: str = "") -> str:
    """Follow Hugging Face's redirect so DuckDB reads the signed CDN URL."""
    req = urllib.request.Request(url, headers={"User-Agent": "tg-api/1.0"})
    if token:
        req.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.geturl()
    except (urllib.error.URLError, TimeoutError) as exc:
        log.warning("download URL resolution failed for %s: %s", url, exc)
        return url


def _read_cache(key: str) -> Optional[List[str]]:
    try:
        with open(CACHE_PATH) as fh:
            blob = json.load(fh)
        if blob.get("key") == key and time.time() - blob.get("at", 0) < CACHE_TTL:
            return blob.get("urls") or None
    except (OSError, ValueError):
        pass
    return None


def _write_cache(key: str, urls: List[str]) -> None:
    try:
        with open(CACHE_PATH, "w") as fh:
            json.dump({"key": key, "at": time.time(), "urls": urls}, fh)
    except OSError as exc:
        log.debug("could not cache discovery result: %s", exc)


def splits(repo: str, token: str = "") -> List[dict]:
    """Available (config, split) pairs, via the dataset-viewer API."""
    payload = _get_json(VIEWER_SPLITS.format(repo=repo), token)
    return payload.get("splits", []) if isinstance(payload, dict) else []


def parquet_urls(
    repo: str,
    config: str = "default",
    split: str = "train",
    token: str = "",
    revision: str = "main",
) -> List[str]:
    """Resolve the dataset's parquet shard URLs, cached on disk for 15 minutes."""
    key = f"v2:{repo}/{config}/{split}/{revision}"
    cached = _read_cache(key)
    if cached:
        log.info("discovery cache hit: %d files", len(cached))
        return cached

    url = HUB_API.format(repo=repo, config=config, split=split)
    try:
        urls = _flatten(_get_json(url, token))
    except (urllib.error.URLError, ValueError, TimeoutError) as exc:
        log.warning("parquet discovery failed for %s: %s", key, exc)
        return []

    if not urls:
        log.warning("parquet discovery returned nothing for %s", key)
        return []

    if urls and all("/api/datasets/" in item and "/parquet/" in item for item in urls):
        server_urls = _dataset_server_urls(repo, config, split, token)
        if server_urls:
            urls = server_urls
        else:
            tree_urls = _tree_parquet_urls(repo, revision, token)
            if tree_urls:
                urls = tree_urls

    log.info("discovered %d parquet files for %s", len(urls), key)
    _write_cache(key, urls)
    return urls
