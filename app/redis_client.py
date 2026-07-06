"""
Shared Redis connection used by:
- app/idempotency.py  (step 3: message-idempotency guard)
- app/llm.py           (step 2: cross-worker Gemini concurrency semaphore)

STEP 2 & 3 CONTEXT: both of these used to be process-local (an
asyncio.Semaphore for concurrency, a SQLite table for idempotency).
Process-local state is exactly the thing that breaks once you run
multiple worker processes (step 4) — each worker would get its own
semaphore and its own dedup guard, so N workers would let through
N * gemini_max_concurrent_requests requests at once, and a webhook
redelivery landing on a different worker than the one that handled the
original would sail straight through as "never seen before". Redis gives
both of these a single shared source of truth that every worker talks to
over the network instead of in-process memory.

A single redis.asyncio client is safe to share across coroutines/threads
within a process (it manages its own connection pool internally), so one
module-level singleton is enough — no per-request client needed.
"""
import logging

import redis.asyncio as redis

logger = logging.getLogger(__name__)

_redis_client: "redis.Redis | None" = None


def get_redis() -> "redis.Redis":
    global _redis_client
    if _redis_client is None:
        from app.config import get_settings
        settings = get_settings()
        _redis_client = redis.from_url(
            settings.redis_url,
            decode_responses=True,
            # Fail fast rather than hanging a request indefinitely if
            # Redis itself is down — better to surface an error than to
            # silently stall the whole webhook pipeline behind a dead
            # connection.
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        logger.info(f"Redis client created for {_redis_url_safe(settings.redis_url)}")
    return _redis_client


def _redis_url_safe(url: str) -> str:
    try:
        if "@" in url:
            scheme, rest = url.split("://", 1)
            _, host = rest.split("@", 1)
            return f"{scheme}://***@{host}"
    except Exception:
        pass
    return url
