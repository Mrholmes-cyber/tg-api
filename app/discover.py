from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import List, Optional

log = logging.getLogger("tgapi.discover")

HUB_API = "https://huggingface.co/api/datasets/{repo}/parquet/{config}/{split}"
VIEWER_SPLITS = "https://datasets-server.huggingface.co/splits?dataset={repo}"
CACHE_PATH = os.environ.get("DISCOVER_CACHE", "/tmp/tg_parquet_urls.json")
CACHE_TTL = 24 * 3600


def _get_json(url: str, token: str = "", timeout: int = 20):
    req = urllib.request.Request(url, headers={"User-Agent": "tg-api/1.0"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
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
                url = item.get("url") or item.get("filename")
                if url:
                    out.append(url)
        return out
    if isinstance(payload, dict):
        for key in ("parquet_files", "urls", "files"):
            if key in payload:
                return _flatten(payload[key])
    return []


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
    repo: str, config: str = "default", split: str = "train", token: str = ""
) -> List[str]:
    """Resolve the dataset's parquet shard URLs, cached on disk for a day."""
    key = f"{repo}/{config}/{split}"
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

    log.info("discovered %d parquet files for %s", len(urls), key)
    _write_cache(key, urls)
    return urls
