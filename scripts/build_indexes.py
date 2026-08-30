#!/usr/bin/env python3
"""Build lookup-optimized parquet copies of the dataset.

Why: parquet only skips data when the filter column is *sorted*, because row
groups are pruned by their min/max statistics. `user_id` is already sorted in
Kzr0xx/telegram, so /v1/users/{id} is fast. Phone and username lookups have to
read the whole column — that is the slow path. Sorting a copy by those columns
turns each lookup into a couple of row-group reads.

Run this once, on a machine with disk and bandwidth:

    python scripts/build_indexes.py --out ./build

Then upload build/by_phone/ and build/by_username/ to a repo you own and point
PHONE_SOURCE / USERNAME_SOURCE at them.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, ".")

from app import db  # noqa: E402

ROW_GROUP = 122_880  # small groups = finer pruning, still efficient to read


def build(out_dir: str, name: str, sort_key: str, extra: str) -> None:
    target = os.path.join(out_dir, name)
    os.makedirs(target, exist_ok=True)
    started = time.perf_counter()
    sql = f"""
        COPY (
            SELECT *, {extra}
            FROM tg
            ORDER BY {sort_key}
        ) TO '{target}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE {ROW_GROUP},
         PER_THREAD_OUTPUT false, FILE_SIZE_BYTES '400MB', OVERWRITE_OR_IGNORE true)
    """
    print(f"building {name} sorted by {sort_key} ...", flush=True)
    db.connection().execute(sql)
    print(f"  done in {time.perf_counter() - started:.0f}s -> {target}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./build")
    ap.add_argument("--memory", default="8GB", help="DuckDB memory_limit for the sort")
    ap.add_argument("--temp", default="/tmp/duckdb", help="spill directory")
    args = ap.parse_args()

    con = db.connection()
    con.execute(f"SET memory_limit='{args.memory}'")
    con.execute(f"SET temp_directory='{args.temp}'")
    con.execute("SET preserve_insertion_order=false")

    os.makedirs(args.out, exist_ok=True)
    build(
        args.out,
        "by_phone",
        "phone_norm",
        "regexp_replace(coalesce(phone,''), '[^0-9]', '', 'g') AS phone_norm",
    )
    build(args.out, "by_username", "username_lc", "lower(username) AS username_lc")
    print("\nUpload these, then set:")
    print("  PHONE_SOURCE=<URLs (or hf:// path) of the by_phone parquet files>")
    print("  USERNAME_SOURCE=<URLs (or hf:// path) of the by_username parquet files>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
