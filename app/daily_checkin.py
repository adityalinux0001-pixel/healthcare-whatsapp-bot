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

from app.config import get_settings
from app.memory import ConversationMemory
from app.whatsapp import send_text_message

logger = logging.getLogger("daily_checkin")

settings = get_settings()

memory = ConversationMemory(
    database_url=settings.database_url,
    pool_min_size=settings.db_pool_min_size,
    pool_max_size=settings.db_pool_max_size,
)


async def _send_checkin_for_user(phone_number: str) -> None:
    """Send the next pregenerated, not-yet-sent plan day for one user, if
    any is due and the throttle allows it. Pure fetch-and-send — no LLM
    call."""
    now = datetime.utcnow()

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
    message = plan_day["message_text"]
    followup_question = plan_day.get("followup_question")

    try:
        await send_text_message(phone_number, message)
        await asyncio.to_thread(memory.mark_plan_day_sent, phone_number, day_number)
        await asyncio.to_thread(
            memory.save_message, phone_number, "assistant", message, message_type="text"
        )
        logger.info(f"✅ Sent day {day_number}/{settings.premium_plan_days} check-in to {phone_number}")
    except Exception as e:
        logger.error(f"❌ Failed to send/save check-in for {phone_number}: {e}", exc_info=True)
        return

    # Same-day follow-up question, sent right after the day's message.
    # The user's reply is captured by app/main.py's webhook handler
    # (checks memory.get_awaiting_followup_day before normal Q&A routing)
    # — this job's job ends here.
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


async def run_daily_checkins() -> None:
    """
    Single entry point for one day's run: fetch every user with an
    active premium subscription and send each of them their next
    pregenerated check-in, one at a time (sequential — this runs once a
    day, not on the hot request path, so there's no latency pressure to
    parallelize it).
    """
    if not settings.daily_checkin_enabled:
        logger.info("Daily check-in feature disabled (daily_checkin_enabled=False) — skipping run.")
        return

    users = await asyncio.to_thread(memory.get_active_premium_users)
    logger.info(f"📅 Daily check-in run starting — {len(users)} active premium user(s).")

    for user in users:
        phone_number = user["phone_number"]
        try:
            await _send_checkin_for_user(phone_number)
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