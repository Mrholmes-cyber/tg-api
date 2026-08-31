# Telegram Data API

Read-only FastAPI service over the [`Kzr0xx/telegram`](https://huggingface.co/datasets/Kzr0xx/telegram)
Hugging Face dataset, queried in place by DuckDB. No database to run, no 4.6GB
download — DuckDB range-reads only the parquet byte ranges a query needs.

Built to deploy on Render as-is.

## How it finds the data

At boot the service asks the hub for the exact shard list:

    GET https://huggingface.co/api/datasets/Kzr0xx/telegram/parquet/default/train

That returns the parquet URLs, which get cached to disk for 24 hours and wired
into a DuckDB view called `tg`. If the hub returns synthetic parquet URLs for
repository files, the service resolves the actual `.parquet` paths from the
repository tree. If discovery is unavailable, it falls back to globbing
`hf://datasets/Kzr0xx/telegram@main/**/*.parquet`. Both paths are overridable —
see `PARQUET_URLS` and the `S3_*` block in `.env.example`.

`GET /v1/source` shows exactly what got wired up, including the shard list.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | engine status, source mode, shard count, cache size |
| GET | `/v1/users/{user_id}` | numeric id lookup — the fast path |
| GET | `/v1/phone/{phone}` | phone lookup, `?fuzzy=true` matches on suffix |
| GET | `/v1/username/{username}` | handle lookup, `?prefix=true` for prefix search |
| GET | `/v1/search` | combined filters: `user_id`, `username`, `phone`, `name`, `email`, `status`, `linked_id` |
| GET | `/v1/stats` | row count, id range, column fill rates |
| GET | `/v1/schema` | column names and types |
| GET | `/v1/source` | resolved source + shard list |
| POST | `/v1/cache/clear` | drop the result cache |
| POST | `/v1/sql` | guarded read-only SELECT (off by default) |

Interactive docs at `/docs`.

Every endpoint except `/health` and `/` needs a key:

```bash
curl -H "X-API-Key: $KEY" https://your-app.onrender.com/v1/users/551348190
curl -H "X-API-Key: $KEY" "https://your-app.onrender.com/v1/phone/79647416479"
curl -H "X-API-Key: $KEY" "https://your-app.onrender.com/v1/search?username=deadlox&prefix=true&limit=25"
```

Responses are uniform:

```json
{
  "count": 1, "limit": 50, "offset": 0, "took_ms": 412, "cached": false,
  "results": [{"user_id": 551348190, "username": null, "phone": "79647416479", "...": null}]
}
```

## Run locally

```bash
pip install -r requirements-dev.txt
cp .env.example .env          # set API_KEYS; HF_TOKEN only if the repo is gated
python scripts/check_source.py    # confirms the dataset resolves + prints schema
uvicorn app.main:app --reload
pytest -q                     # offline: builds a local parquet fixture
```

## Deploy to Render

1. Push this directory to a Git repo.
2. Render → New → Blueprint → pick the repo. `render.yaml` defines everything.
3. Render generates `API_KEYS`; copy it from the dashboard. Set `HF_TOKEN` only
   if the dataset is private or gated.
4. Health check is `/health`, so the first successful boot marks it live.

Docker works too if you'd rather: the `Dockerfile` bakes the `httpfs` extension
in so cold starts don't download it.

## Speed

What makes lookups fast is not the API layer, it's whether DuckDB can skip row
groups. Parquet prunes by min/max statistics per row group, which only helps
when the filter column is *sorted*.

- `user_id` is already sorted in this dataset, so `/v1/users/{id}` reads a
  couple of row groups and returns in well under a second once warm.
- `phone` and `username` are unsorted, so those lookups scan the whole column
  across every shard. Expect seconds, not milliseconds.

To make those fast too, build sorted copies once and point the service at them:

```bash
python scripts/build_indexes.py --out ./build --memory 8GB
# upload build/by_phone/ and build/by_username/ to a repo you own, then set:
#   PHONE_SOURCE=<their URLs>
#   USERNAME_SOURCE=<their URLs>
```

Those copies carry materialized `phone_norm` and `username_lc` columns, so
lookups become sorted-column equality probes instead of regexp scans. The API
picks them up automatically and reports the active views in `/v1/source`.

Three other things are doing work here: DuckDB's object cache keeps parquet
footers in memory between queries, HTTP keep-alive avoids reconnecting per
range request, and an in-process TTL cache (`CACHE_TTL_SECONDS`, default 15min)
means a repeated lookup never touches the network at all.

## Notes

- Render's free tier gives 512MB RAM; `DUCKDB_MEMORY_LIMIT` defaults to 350MB
  and DuckDB spills to `/tmp`. Bump both on a paid plan.
- Free instances sleep after inactivity, so the first request after a nap pays
  cold-start plus parquet metadata fetch. A paid plan or an external pinger
  fixes that.
- `/v1/sql` stays off unless you set `ALLOW_RAW_SQL=true`. Even then it accepts
  a single `SELECT`/`WITH`, rejects DDL keywords, and wraps the query in a
  `LIMIT`.
- Keys are compared with `secrets.compare_digest`. `API_KEYS` takes a
  comma-separated list so you can rotate without downtime.
- This service only reads. Nothing in it writes back to the dataset.
