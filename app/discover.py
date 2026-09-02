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
TREE_API = "https://huggingface.co/api/datasets/{repo}/tree/{revision}?recursive=true"
VIEWER_SPLITS = "https://datasets-server.huggingface.co/splits?dataset={repo}"
CACHE_PATH = os.environ.get("DISCOVER_CACHE", "/tmp/tg_parquet_urls.json")
CACHE_TTL = 24 * 3600

# Bumped whenever the discovery strategy changes so stale caches from an older
# deployment are never reused.
_CACHE_VERSION = "v3"


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
                url = item.get("url") or item.get("filename") or item.get("path")
                if url:
                    out.append(url)
        return out
    if isinstance(payload, dict):
        for key in ("parquet_files", "urls", "files"):
            if key in payload:
                return _flatten(payload[key])
    return []


def _sort_key(path: str):
    """Natural sort so data_2 comes before data_10."""
    import re

    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path)]


def _tree_parquet_urls(repo: str, revision: str, token: str = "") -> List[str]:
    """Build /resolve/ URLs for every .parquet file actually in the repo.

    This is the most reliable source: the files are the ones the owner
    uploaded, on the branch we asked for, and the URLs never go stale.
    """
    try:
        payload = _get_json(
            TREE_API.format(repo=repo, revision=quote(revision, safe="")), token
        )
    except (urllib.error.URLError, ValueError, TimeoutError) as exc:
        log.warning("repository tree discovery failed for %s@%s: %s", repo, revision, exc)
        return []

    paths = sorted(
        (p for p in _flatten(payload) if p.lower().endswith(".parquet")),
        key=_sort_key,
    )
    return [
        f"https://huggingface.co/datasets/{repo}/resolve/"
        f"{quote(revision, safe='')}/{quote(path, safe='/')}"
        for path in paths
    ]


def _hub_parquet_urls(repo: str, config: str, split: str, token: str = "") -> List[str]:
    """Fallback: the hub's auto-converted parquet shards.

    These live on the ``refs/convert/parquet`` branch and are exposed through
    ``/api/datasets/<repo>/parquet/<config>/<split>/<n>.parquet`` redirects.
    They lag behind the repo and the conversion job can be missing entirely,
    which is why this is only a fallback.
    """
    url = HUB_API.format(repo=repo, config=config, split=split)
    try:
        return _flatten(_get_json(url, token))
    except (urllib.error.URLError, ValueError, TimeoutError) as exc:
        log.warning("hub parquet API failed for %s/%s/%s: %s", repo, config, split, exc)
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


def clear_cache() -> None:
    try:
        os.remove(CACHE_PATH)
    except OSError:
        pass


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
    """Resolve the dataset's parquet shard URLs, cached on disk for a day.

    Order of preference:
      1. real ``.parquet`` files in the repo tree at ``revision``
      2. the hub's auto-converted shards for ``config``/``split``
    """
    key = f"{_CACHE_VERSION}:{repo}/{config}/{split}/{revision}"
    cached = _read_cache(key)
    if cached:
        log.info("discovery cache hit: %d files", len(cached))
        return cached

    urls = _tree_parquet_urls(repo, revision, token)
    source = "repo tree"
    if not urls:
        urls = _hub_parquet_urls(repo, config, split, token)
        source = "hub parquet API"

    if not urls:
        log.warning("parquet discovery returned nothing for %s", key)
        return []

    log.info("discovered %d parquet files for %s via %s", len(urls), key, source)
    _write_cache(key, urls)
    return urls
