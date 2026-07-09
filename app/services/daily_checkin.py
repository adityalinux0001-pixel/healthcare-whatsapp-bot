"""
Daily premium check-in job — REWRITTEN for the pregeneration architecture.

Old behavior: called Gemini once per user, per day, to generate that day's
message on the fly. New behavior: the entire {premium_plan_days}-day plan
(message + same-day follow-up question, for every day) is generated ONCE,
right after onboarding finishes (see app/onboarding.py ->
app/llm.py::generate_premium_plan -> app/memory.py::save_premium_plan).

This job now does exactly what the architecture diagram says and nothing
more:
    1. Compute which day is next for a user (fetch the lowest-numbered
       unsent row — app/memory.py::get_next_unsent_plan_day).
    2. Fetch that row, send message_text.
    3. Mark it sent and open the same-day follow-up window — the actual
       follow-up QUESTION is sent right after, and the user's reply is
       captured by the normal webhook flow in app/main.py (see
       handle_possible_followup_reply there), not by this job.

No LLM call happens anywhere in this file. That means no risk of a Gemini
outage breaking someone's day-14 message, and a human can review/edit any
day's premium_plans.message_text in the database before it ever gets sent.

Run this on a schedule (cron, a simple `while True: sleep` loop in a
container, APScheduler, etc.) — see run_daily_checkins() below for the
single entry point one job invocation should call once per day.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional

from app.core.config import get_settings
from app.services.memory import ConversationMemory
from app.services.whatsapp import send_text_message

logger = logging.getLogger("daily_checkin")

settings = get_settings()

memory = ConversationMemory(
    database_url=settings.DATABASE_URL,
    pool_min_size=settings.DB_POOL_MIN_SIZE,
    pool_max_size=settings.DB_POOL_MAX_SIZE,
)


async def _send_checkin_for_user(phone_number: str, preferred_hour_utc: "int | None" = None) -> None:
    """Send the next pregenerated, not-yet-sent plan day for one user, if
    any is due and the throttle allows it. Pure fetch-and-send — no LLM
    call.

    Day 1 is now sent immediately right after onboarding finishes (see
    app/onboarding.py::_send_plan_day_now), so in practice this function
    is only ever picking up Day 2 onward — get_next_unsent_plan_day just
    naturally skips Day 1 since it's already marked sent.

    preferred_hour_utc: if given, this user's chosen daily check-in hour
    (0-23 UTC), collected during onboarding question 8. run_daily_checkins
    already only calls this once per calendar day per user via
    _run_forever's per-hour wakeups, so passing the hour through just lets
    run_daily_checkins() decide whether THIS run's hour matches THIS
    user's preferred hour before calling this function at all (see
    below) — kept as a parameter here mainly for logging/clarity.
    """
    now = datetime.utcnow()


    last_sent_at = await asyncio.to_thread(memory.get_last_checkin_sent_at, phone_number)
    if last_sent_at is not None:
        if last_sent_at.tzinfo is not None:
            last_sent_at = last_sent_at.replace(tzinfo=None)
        hours_since = (now - last_sent_at).total_seconds() / 3600
        if hours_since < settings.DAILY_CHECKIN_MIN_GAP_HOURS:
            logger.info(
                f"⏭️ Skipping check-in for {phone_number} — last one sent "
                f"{hours_since:.1f}h ago (throttle: {settings.DAILY_CHECKIN_MIN_GAP_HOURS}h)."
            )
            return

    # Step "Compute current day number" + "Fetch row" from the
    # architecture diagram, combined: whichever pregenerated day hasn't
    # been sent yet IS today's day.
    plan_day = await asyncio.to_thread(memory.get_next_unsent_plan_day, phone_number)
    if plan_day is None:
        logger.info(
            f"⏭️ No pending pregenerated plan day for {phone_number} — "
            f"either the plan is fully sent or was never generated (no onboarding completed)."
        )
        return

    day_number = plan_day["day_number"]
    header = f"*Day {day_number} of {settings.PREMIUM_PLAN_DAYS}* 🗓️\n\n"
    message = header + plan_day["message_text"]
    followup_question = plan_day.get("followup_question")

    try:
        await send_text_message(phone_number, message)
        await asyncio.to_thread(memory.mark_plan_day_sent, phone_number, day_number)
        await asyncio.to_thread(
            memory.save_message, phone_number, "assistant", message, message_type="text"
        )
        logger.info(f"✅ Sent day {day_number}/{settings.PREMIUM_PLAN_DAYS} check-in to {phone_number}")
    except Exception as e:
        logger.error(f"❌ Failed to send/save check-in for {phone_number}: {e}", exc_info=True)
        return


    if followup_question:
        try:
            await send_text_message(phone_number, followup_question)
            await asyncio.to_thread(
                memory.save_message, phone_number, "assistant", followup_question, message_type="text"
            )
        except Exception as e:
            logger.error(
                f"❌ Failed to send follow-up question for {phone_number} day {day_number}: {e}",
                exc_info=True,
            )


async def run_daily_checkins(current_hour_utc: "int | None" = None) -> None:
    """
    Single entry point for one hour's run: fetch every user with an
    active premium subscription, and for each one whose preferred
    check-in hour (set during onboarding question 8 - see
    app/onboarding.py) matches `current_hour_utc`, send their next
    pregenerated check-in. Users who never answered a parseable time
    already have settings.DAILY_CHECKIN_HOUR_UTC stored as their
    effective hour (baked in by memory.set_preferred_checkin_hour at
    onboarding time), so no extra fallback logic is needed here.

    current_hour_utc: which UTC hour this run is for. Defaults to
    "right now" so `--once` / ad-hoc invocations still work sensibly;
    the built-in scheduler (_run_forever, below) always passes it
    explicitly since it now wakes up once per hour instead of once per
    day, to support per-user preferred hours.
    """
    if not settings.DAILY_CHECKIN_ENABLED:
        logger.info("Daily check-in feature disabled (daily_checkin_enabled=False) — skipping run.")
        return

    if current_hour_utc is None:
        current_hour_utc = datetime.utcnow().hour

    users = await asyncio.to_thread(memory.get_active_premium_users)
    due_users = [
        u for u in users
        if (u.get("preferred_checkin_hour_utc")
            if u.get("preferred_checkin_hour_utc") is not None
            else settings.DAILY_CHECKIN_HOUR_UTC) == current_hour_utc
    ]
    logger.info(
        f"📅 Daily check-in run starting for hour {current_hour_utc}:00 UTC — "
        f"{len(due_users)} of {len(users)} active premium user(s) due this hour."
    )

    for user in due_users:
        phone_number = user["phone_number"]
        preferred_hour = user.get("preferred_checkin_hour_utc")
        try:
            await _send_checkin_for_user(phone_number, preferred_hour_utc=preferred_hour)
        except Exception as e:
            logger.error(f"❌ Unhandled error sending check-in to {phone_number}: {e}", exc_info=True)

    logger.info(f"📅 Daily check-in run finished for hour {current_hour_utc}:00 UTC.")


async def _run_forever() -> None:
    """
    Scheduler loop: wakes up once a minute and, once per UTC calendar
    hour, runs the batch for that hour. Needed now that each user can
    have their own preferred check-in hour (see app/onboarding.py
    question 8 and memory.set_preferred_checkin_hour) instead of
    everyone sharing one fixed settings.DAILY_CHECKIN_HOUR_UTC.
    """
    last_run_key = None
    logger.info(
        "Daily check-in scheduler started - checking every hour for users "
        "whose preferred check-in hour matches."
    )
    while True:
        now = datetime.utcnow()
        run_key = (now.date(), now.hour)
        if run_key != last_run_key:
            try:
                await run_daily_checkins(current_hour_utc=now.hour)
            except Exception as e:
                logger.error(f"❌ Daily check-in run crashed: {e}", exc_info=True)
            last_run_key = run_key
        await asyncio.sleep(60)


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )

    if "--once" in sys.argv:
        # For use from an external cron scheduler instead of the built-in loop.
        asyncio.run(run_daily_checkins())
    else:
        asyncio.run(_run_forever())