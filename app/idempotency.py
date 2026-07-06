"""
STEP 3: Redis-backed idempotency guard for incoming webhook message IDs.

This replaces `ConversationMemory.try_mark_message_processed` /
`prune_old_processed_messages` (previously a SQLite table). Two reasons
to move it, in order of how much they matter:

1. It needs to be SHARED ACROSS WORKERS (step 4 requirement). If worker A
   handles the first delivery of a webhook and worker B handles a
   redelivery of the exact same message a few seconds later — which
   WhatsApp does, and which a load balancer will happily route to a
   different worker — a per-process guard (or one backed by a store each
   worker has its own connection/cache for but no shared locking on)
   isn't enough on its own; you need one shared, atomically-updated set
   that every worker checks against. Postgres could do this too (it's
   already shared), but reason 2 below is why Redis specifically:

2. It's a pure "have I seen this key before, yes/no" check-and-set on the
   hot path of every single incoming message — this wants to be as cheap
   as possible, and it wants a TTL so old entries expire on their own
   instead of needing a periodic prune job (the old
   prune_old_processed_messages() cron-style method). Redis's `SET key
   value NX EX <ttl>` does exactly this in one atomic round-trip.
"""
import logging

from app.redis_client import get_redis

logger = logging.getLogger(__name__)

# Matches the old prune_old_processed_messages(keep_days=7) default —
# WhatsApp doesn't redeliver webhooks anywhere near this far out, 7 days
# is generous headroom.
_PROCESSED_MESSAGE_TTL_SECONDS = 7 * 24 * 60 * 60

_KEY_PREFIX = "wa:processed_msg:"


async def try_mark_message_processed(message_id: str) -> bool:
    """Atomically check-and-mark a webhook message ID as processed.

    Returns True if this is the first time we've seen this ID (caller
    should proceed), False if it was already processed before —
    including by a different worker process, since this is backed by
    Redis rather than in-memory or per-process state.
    """
    r = get_redis()
    key = _KEY_PREFIX + message_id
    # SET ... NX (only set if not already present) EX (auto-expire) is a
    # single atomic Redis command — no separate "check then set" race
    # window like a naive GET-then-SET would have.
    was_set = await r.set(key, "1", nx=True, ex=_PROCESSED_MESSAGE_TTL_SECONDS)
    return bool(was_set)
