# Multi-worker migration

Implements the 4-step plan, in order:

1. **Postgres replacing SQLite** (`app/memory.py`) — same public method
   signatures as before, so `app/main.py`'s `asyncio.to_thread(memory.xxx, ...)`
   calls didn't need to change. Connection pooling via `psycopg_pool`.
2. **Redis-backed Gemini concurrency semaphore** (`app/llm.py`) —
   replaces the old `asyncio.Semaphore` (per-process) with a Redis sorted-set
   + Lua script (cluster-wide). `is_gemini_busy()` is now `async`.
3. **Redis-backed message idempotency** (`app/idempotency.py`) — replaces
   `memory.try_mark_message_processed` (SQLite) with an atomic
   `SET NX EX` in Redis, auto-expiring after 7 days (no cron/prune job needed).
4. **Multi-worker startup** (`start.sh`, `Dockerfile`, `docker-compose.yml`) —
   Gunicorn + `uvicorn.workers.UvicornWorker`, worker count via `WEB_CONCURRENCY`.
   The old `app/store.py` in-memory per-process session cache was **removed** —
   it would silently diverge across workers, and `memory.get_conversation_context()`
   already provides the same context durably.

## Running locally

```bash
cp .env.example .env   # fill in your real API keys
docker compose up --build
```

This starts Postgres, Redis, and the app (4 gunicorn workers by default).
Postgres and Redis schemas/keys are created automatically on first boot.

## Running without Docker (dev loop)

```bash
pip install -r requirements.txt
# point DATABASE_URL / REDIS_URL at local instances in .env
python run.py   # single reloading worker — NOT representative of multi-worker behavior
```

To actually exercise the multi-worker path locally without Docker:

```bash
WEB_CONCURRENCY=4 ./start.sh
```

## What changed for you operationally

- **New required infra**: Postgres and Redis (both provided via
  `docker-compose.yml` for local dev; point `DATABASE_URL` / `REDIS_URL`
  at managed instances for production).
- **`conversation_memory.db` (SQLite file) is no longer used or created.**
  If you need to migrate historical data from the old SQLite file, that's
  a one-off script reading from `sqlite3` and writing via
  `ConversationMemory`'s methods — not included here since none was
  requested, but straightforward given the schemas are nearly identical.
- **`gemini_max_concurrent_requests`** now caps concurrency **across all
  workers combined**, not per-worker — no config change needed, the
  number now means what it always should have meant.
- **`/health` and `/debug` no longer report `active_sessions`** — that
  was reading the removed process-local `store.py`, which was never
  meaningful with >1 worker anyway.
