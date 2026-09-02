# Telegram Data API

Read-only FastAPI service over the [`NhiBatauga/Telegram-Database`](https://huggingface.co/datasets/NhiBatauga/Telegram-Database)
Hugging Face dataset, queried in place by DuckDB. No database to run, no 4.6GB
download — DuckDB range-reads only the parquet byte ranges a query needs.

Built to deploy on Render as-is.

## How it finds the data

At boot the service lists the repo's real `.parquet` files through the hub
tree API:

    GET https://huggingface.co/api/datasets/NhiBatauga/Telegram-Database/tree/main?recursive=true

and turns them into `/resolve/main/<file>` URLs, cached to disk for 24 hours
and wired into a DuckDB view called `tg`. If the tree API is unreachable it
falls back to the hub's auto-converted shards
(`/api/datasets/<repo>/parquet/<config>/<split>`), and if that fails too it
globs `hf://datasets/<repo>@main/**/*.parquet`. All of it is overridable — see
`PARQUET_URLS` and the `S3_*` block in `.env.example`.

The view adapts to whatever columns the files actually have: the id column
(`account_id` in this dump, `user_id` in others) is exposed as `user_id`
(BIGINT), and any missing column comes back as `NULL`.

`GET /v1/source` shows exactly what got wired up, including the shard list
and raw parquet schema.

### Boot sequence on Render

The engine warms up in a background thread, so the port binds instantly and
`/health` returns `200` right away with `"engine": "warming"`. Reading ~100
parquet footers takes 30–60s on a small instance; once done `/health` flips
to `"engine": "ready"`. Data endpoints return `503` + `Retry-After` while
warming. If warm-up fails, `/health` reports `"engine": "failed"` with the
exact error and retries on the next probe.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | engine state (`warming`/`ready`/`failed` + error), source, shard count |
| GET | `/v1/users/{user_id}` | numeric id lookup — the fast path |
| GET | `/v1/phone/{phone}` | phone lookup, `?fuzzy=true` matches on suffix |
| GET | `/v1/username/{username}` | handle lookup, `?prefix=true` for prefix search |
| GET | `/v1/search` | combined filters: `user_id`, `username`, `phone`, `name`, `email`, `status`, `linked_id` |
| GET | `/v1/stats` | row count, id range, column fill rates |
| GET | `/v1/schema` | column names and types |
| GET | `/v1/source` | resolved source + shard list |
| POST | `/v1/cache/clear` | drop the result cache (`?discovery=true` also drops the shard list) |
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
                                  # (add --count for a full row count)
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

- Id lookups filter on the *raw* stored column (`account_id` here) so the
  predicate is pushed into the parquet scan. This dump stores ids as text and
  only partially sorted, so `/v1/users/{id}` is tens of seconds cold and
  instant from cache. A sorted copy (below) makes it sub-second.
- `phone` and `username` are unsorted, so those lookups scan the whole column
  across every shard. Expect seconds to minutes, not milliseconds.

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

## Troubleshooting

**`warm-up failed: HTTP Error ... 403` / `404 (Not Found)`** — this was a
DuckDB version problem. Hugging Face now serves files from the Xet CDN via a
redirect that `duckdb<=1.1`'s httpfs can't follow; `requirements.txt` pins
`duckdb==1.5.5` for that reason. Don't downgrade it. Also make sure Render is
using this `requirements.txt` (check the build log for `duckdb-1.5.5`).

**`/health` stuck on `warming`** — normal for the first ~60s. If it goes on
longer, raise `HTTP_TIMEOUT_SECONDS` or check the Render logs for the
background thread's error.

**`/health` says `failed`** — the `error` field carries the DuckDB message.
Run `python scripts/check_source.py` locally with the same env to reproduce.

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
