"""
Per-user daily chat-message limit, stored in Redis (NO database table / schema
change, nothing to migrate in production).

Key:   daily_msgs:<phone>:<YYYY-MM-DD>   (date in the configured local day)
Value: how many chat messages the user has sent today (atomic INCR, so it is
       correct across all workers). Expires by itself after 48h — the date in
       the key is what makes the counter "reset" at midnight.

Limit lookup order (so you can change it live, without restart):
  1. Redis key  config:daily_message_limit   (override, e.g. 5 for testing)
  2. settings.DAILY_MESSAGE_LIMIT            (.env, default 25)
A limit <= 0 means unlimited.

Fails OPEN: if Redis is unreachable the message is allowed — the limit is a
cost guard, it must never take the bot down.
"""
import logging
from datetime import datetime, timedelta

from app.core.config import get_settings

logger = logging.getLogger(__name__)

LIMIT_OVERRIDE_KEY = "config:daily_message_limit"
_COUNTER_TTL_SECONDS = 48 * 3600

ALLOWED = "allowed"
REACHED = "reached"    # allowed, and this was the LAST one -> send reminder
BLOCKED = "blocked"    # over the limit -> ignore silently


async def get_limit() -> int:
    settings = get_settings()
    try:
        from app.core.redis_client import get_redis
        raw = await get_redis().get(LIMIT_OVERRIDE_KEY)
        if raw is not None and str(raw).strip() != "":
            return int(raw)
    except Exception as e:
        logger.warning(f"⚠️ Could not read daily-limit override from Redis: {e}")
    return settings.DAILY_MESSAGE_LIMIT


def _today_key_part() -> str:
    offset = get_settings().DAILY_LIMIT_UTC_OFFSET_MINUTES
    return (datetime.utcnow() + timedelta(minutes=offset)).strftime("%Y-%m-%d")


async def register_message(phone_number: str) -> str:
    """Count one chat message for today and say what to do with it."""
    try:
        limit = await get_limit()
        if limit <= 0:
            return ALLOWED

        from app.core.redis_client import get_redis
        key = f"daily_msgs:{phone_number}:{_today_key_part()}"
        redis = get_redis()
        count = await redis.incr(key)
        await redis.expire(key, _COUNTER_TTL_SECONDS)
    except Exception as e:
        logger.error(f"❌ Daily-limit check failed for {phone_number} (allowing): {e}")
        return ALLOWED

    if count < limit:
        return ALLOWED
    if count == limit:
        return REACHED
    return BLOCKED


def reached_message(limit: int) -> str:
    return get_settings().DAILY_LIMIT_REACHED_MESSAGE.format(limit=limit)
