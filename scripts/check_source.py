#!/usr/bin/env python3
"""Verify the dataset is reachable and the schema is what the API expects.

    python scripts/check_source.py

Reads the same .env the API uses. Run this before deploying — it fails loudly
with the exact DuckDB error instead of a vague 502 in production.
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, ".")

from app import db, discover  # noqa: E402
from app.config import get_settings  # noqa: E402


def main() -> int:
    s = get_settings()
    print("mode:", s.source_mode)
    if s.source_mode == "hf-dataset":
        try:
            for sp in discover.splits(s.dataset_repo, s.hf_token):
                print(f"  split: config={sp.get('config')} split={sp.get('split')}")
        except Exception as exc:
            print("  (splits API unavailable:", exc, ")")
        shards = discover.parquet_urls(
            s.dataset_repo, s.dataset_config, s.dataset_split, s.hf_token
        )
        print(f"  shards discovered: {len(shards)}")
        for u in shards[:5]:
            print("   ", u)
        if len(shards) > 5:
            print(f"    ... and {len(shards) - 5} more")
        if not shards:
            print("  falling back to glob:", s.hf_uri)
    elif s.source_mode == "s3":
        print("source:", s.parquet_glob)
    else:
        print("source:\n  " + "\n  ".join(s.urls))

    t0 = time.perf_counter()
    cols = db.query("DESCRIBE SELECT * FROM tg")
    print(f"\nschema ({time.perf_counter() - t0:.1f}s):")
    for c in cols:
        print(f"  {c['column_name']:<16} {c['column_type']}")

    missing = {c for c in db.COLUMNS} - {c["column_name"] for c in cols}
    if missing:
        print("\nWARNING missing expected columns:", sorted(missing))

    t0 = time.perf_counter()
    sample = db.query("SELECT * FROM tg LIMIT 3")
    print(f"\nsample rows ({time.perf_counter() - t0:.1f}s):")
    for row in sample:
        print(" ", row)

    t0 = time.perf_counter()
    total = db.scalar("SELECT count(*) FROM tg")
    print(f"\nrow count: {total:,} ({time.perf_counter() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
