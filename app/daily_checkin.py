"""
Daily premium check-in job.

Part of the 21-day (configurable via settings.premium_plan_days) Premium
plan: once a day, for every user with an active subscription, generate ONE
personalized health suggestion/to-do based on their saved profile/summary
and conversation so far, send it as a WhatsApp message, and record it so
the conversation can continue naturally from the user's reply (handled by
the normal webhook flow in app/main.py — a reply to a check-in is just a
regular incoming message).

Run this on a schedule (cron, a simple `while True: sleep` loop in a
container, APScheduler, etc.) — see run_daily_checkins() below for the
single entry point one job invocation should call once per day.
"""

import asyncio
import logging
from datetime import datetime, timedelta

from app.config import get_settings
from app.memory import ConversationMemory
from app.whatsapp import send_text_message
from app.llm import generate_daily_checkin_message, detect_reply_language, GeminiUnavailableError

logger = logging.getLogger("daily_checkin")

settings = get_settings()

memory = ConversationMemory(
    database_url=settings.database_url,
    pool_min_size=settings.db_pool_min_size,
    pool_max_size=settings.db_pool_max_size,
)


async def _send_checkin_for_user(phone_number: str, started_at: datetime) -> None:
    """Generate and send today's check-in for a single active-premium user,
    unless one was already sent too recently (throttle) or the plan's day
    count has been exhausted."""
    now = datetime.utcnow()
    if started_at.tzinfo is not None:
        started_at = started_at.replace(tzinfo=None)

    # Throttle: don't send a second check-in within
    # daily_checkin_min_gap_hours of the last one for this user (covers
    # the job being triggered more than once in a day, manual re-runs,
    # restarts, etc.).
    last_sent_at = await asyncio.to_thread(memory.get_last_checkin_sent_at, phone_number)
    if last_sent_at is not None:
        if last_sent_at.tzinfo is not None:
            last_sent_at = last_sent_at.replace(tzinfo=None)
        hours_since = (now - last_sent_at).total_seconds() / 3600
        if hours_since < settings.daily_checkin_min_gap_hours:
            logger.info(
                f"⏭️ Skipping check-in for {phone_number} — last one sent "
                f"{hours_since:.1f}h ago (throttle: {settings.daily_checkin_min_gap_hours}h)."
            )
            return

    # How many check-ins already sent since the plan started = the day
    # number we're about to send (1-indexed).
    sent_so_far = await asyncio.to_thread(memory.get_checkins_sent_count, phone_number, started_at)
    day_number = sent_so_far + 1

    if day_number > settings.premium_plan_days:
        logger.info(
            f"⏭️ Skipping check-in for {phone_number} — plan's "
            f"{settings.premium_plan_days} check-ins already sent."
        )
        return

    try:
        customer_data = await asyncio.to_thread(memory.get_customer, phone_number)
        context_text = await asyncio.to_thread(memory.get_conversation_context, phone_number, limit=5)
        recent_checkins = await asyncio.to_thread(memory.get_recent_checkin_messages, phone_number, limit=5)

        customer_summary = customer_data.get("summary", "")

        # No "current user message" to detect language from for a
        # proactive message — fall back to whatever language the user's
        # last real message was in, detected from the recent context text
        # if available, else default to English inside the generator.
        required_language = None
        if context_text:
            required_language = await detect_reply_language(context_text[-500:])

        message = await generate_daily_checkin_message(
            customer_summary=customer_summary,
            context_text=context_text,
            day_number=day_number,
            total_days=settings.premium_plan_days,
            recent_suggestions=recent_checkins,
            required_language=required_language,
        )
    except GeminiUnavailableError:
        logger.warning(f"⏭️ Skipped check-in for {phone_number} — Gemini unavailable, will retry next run.")
        return
    except Exception as e:
        logger.error(f"❌ Failed to generate check-in for {phone_number}: {e}", exc_info=True)
        return

    if not message:
        logger.warning(f"⏭️ Empty check-in message generated for {phone_number}, skipping send.")
        return

    try:
        await send_text_message(phone_number, message)
        await asyncio.to_thread(memory.save_daily_checkin, phone_number, day_number, message)
        await asyncio.to_thread(
            memory.save_message, phone_number, "assistant", message, message_type="text"
        )
        logger.info(f"✅ Sent day {day_number}/{settings.premium_plan_days} check-in to {phone_number}")
    except Exception as e:
        logger.error(f"❌ Failed to send/save check-in for {phone_number}: {e}", exc_info=True)


async def run_daily_checkins() -> None:
    """
    Single entry point for one day's run: fetch every user with an
    active premium subscription and send each of them their check-in for
    today, one at a time (sequential — this runs once a day, not on the
    hot request path, so there's no latency pressure to parallelize it;
    keeping it sequential also naturally respects the shared Gemini
    concurrency limiter in app/llm.py instead of bursting all requests at
    once).
    """
    if not settings.daily_checkin_enabled:
        logger.info("Daily check-in feature disabled (daily_checkin_enabled=False) — skipping run.")
        return

    users = await asyncio.to_thread(memory.get_active_premium_users)
    logger.info(f"📅 Daily check-in run starting — {len(users)} active premium user(s).")

    for user in users:
        phone_number = user["phone_number"]
        started_at = user["started_at"]
        try:
            await _send_checkin_for_user(phone_number, started_at)
        except Exception as e:
            logger.error(f"❌ Unhandled error sending check-in to {phone_number}: {e}", exc_info=True)

    logger.info("📅 Daily check-in run finished.")


async def _run_forever() -> None:
    """
    Minimal built-in scheduler loop: wakes up once a minute, and once per
    UTC calendar day (at settings.daily_checkin_hour_utc) runs the batch.
    Intended for a small dedicated container/process
    (`python -m app.daily_checkin`) — swap for a real cron/scheduler
    (e.g. a cron-triggered one-off `python -m app.daily_checkin --once`)
    if you'd rather not run a long-lived loop.
    """
    last_run_date = None
    logger.info(
        f"Daily check-in scheduler started — will run once a day at "
        f"{settings.daily_checkin_hour_utc}:00 UTC."
    )
    while True:
        now = datetime.utcnow()
        if now.hour == settings.daily_checkin_hour_utc and now.date() != last_run_date:
            try:
                await run_daily_checkins()
            except Exception as e:
                logger.error(f"❌ Daily check-in run crashed: {e}", exc_info=True)
            last_run_date = now.date()
        await asyncio.sleep(60)


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )

    if "--once" in sys.argv:
        # For use from an external cron scheduler instead of the built-in loop.
        asyncio.run(run_daily_checkins())
    else:
        asyncio.run(_run_forever())