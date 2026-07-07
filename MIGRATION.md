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

## Kafka queue backend (optional)

Adds Kafka as a new `QUEUE_BACKEND` option, alongside the existing `rq` /
`celery` / `huey`, matching this architecture:

```
[ WhatsApp / Meta ] -> [ Webhook Receivers ] --(instant 200 OK)--> Meta
                              |
                              v (publish, partitioned by phone number)
                        [ Apache Kafka: whatsapp.inbound ]
                              |
                              v (consume)
                    [ Worker Microservices ] <--> [ Redis ] (session state)
                              |
                              v
                  [ Core Engines & APIs ] --> [ Postgres ] (logs & data)
                              |
                              v (publish, partitioned by phone number)
                    [ Kafka: whatsapp.outbound ]
                              |
                              v (consume)
                      [ Outbound Worker ] --> [ WhatsApp Cloud API ]
```

Redis and Postgres are unchanged — they already matched the diagram (Redis
for session/concurrency state, Postgres for logs/data). What's new is
Kafka replacing the request/response-style RQ queue for the inbound hop,
and a brand new outbound hop that didn't exist before (previously,
core engine code called the WhatsApp Cloud API inline).

### New files
- `app/kafka_client.py` — shared producer/consumer helpers (confluent-kafka).
  Messages are keyed by phone number, so one user's messages always land
  on the same partition and are processed in order by one consumer —
  different users fan out across partitions/workers for concurrency.
- `app/outbound_queue.py` — producer-side wrapper. When
  `QUEUE_BACKEND=kafka`, `enqueue_send_text` / `enqueue_send_template` /
  `enqueue_mark_as_read` publish to `whatsapp.outbound` instead of calling
  `app/whatsapp.py` inline. On any other backend they just call the
  existing synchronous functions directly — nothing else changes.
- `outbound_worker.py` — new process that consumes `whatsapp.outbound` and
  performs the real WhatsApp Cloud API calls. Run as its own container so
  a WhatsApp-side slowdown/rate-limit can't back up inbound processing.

### Changed files
- `app/config.py` — new `kafka_*` settings (bootstrap servers, topic
  names, partition count, consumer group ids).
- `app/queueing.py` — `enqueue_incoming` gained a `kafka` branch
  (`_enqueue_kafka`), publishing to `whatsapp.inbound` keyed by phone
  number. Reuses the same `_handle_incoming` pipeline as every other
  backend.
- `worker.py` — gained a `kafka` branch that runs a blocking consumer loop
  over `whatsapp.inbound` instead of an RQ/Celery/Huey worker loop.
- `docker-compose.yml` — added a single-broker Kafka service (KRaft mode,
  no Zookeeper needed) and an `outbound_worker` service. Both are inert
  when `QUEUE_BACKEND` isn't `kafka`.
- `requirements.txt` — added `confluent-kafka`.

### Switching to it

```bash
# .env
QUEUE_BACKEND=kafka
KAFKA_BOOTSTRAP_SERVERS=kafka:9092   # or your managed cluster
```

```bash
docker compose up --build
# scale independently, e.g.:
docker compose up --scale worker=3 --scale outbound_worker=2
```

### Notes / tradeoffs to know about
- **At-least-once, not exactly-once.** Offsets commit only after
  successful processing (see `run_consumer_loop`'s docstring in
  `app/kafka_client.py`), so a worker crash mid-job means the message is
  redelivered. Inbound already goes through `app/idempotency.py`'s
  redis-backed guard; outbound sends do **not** currently dedupe, so a
  retried outbound job after a network blip could in rare cases double-
  send a WhatsApp message. Add a dedupe key in `outbound_worker.py` if
  that turns out to matter for you in practice.
- **Ordering is per-partition (i.e. per phone number), not global.**
  That's the intended tradeoff — it's what "partitioned by phone number"
  buys you.
- **Audio/document sends are not queued through Kafka.** Only
  text/template/read-receipt jobs go through `app/outbound_queue.py`
  today — putting raw audio/document bytes through Kafka messages is
  usually the wrong call (message-size limits, broker disk pressure). If
  you need those queued too, upload the media to object storage first
  and queue a reference instead of the raw bytes.
- **Single-broker Kafka in `docker-compose.yml` is for local/dev only.**
  For production, point `KAFKA_BOOTSTRAP_SERVERS` at a real multi-broker
  cluster (Confluent Cloud, MSK, Redpanda, self-hosted, etc.) instead of
  running the compose Kafka service.