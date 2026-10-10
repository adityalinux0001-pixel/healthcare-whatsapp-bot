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
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from app.core.config import get_settings
from app.services.memory import ConversationMemory
from app.services.whatsapp import send_template_message
from app.services.plan_delivery import send_plan_day
from app.services.onboarding import generate_and_send_plan

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

    # Today's delivery slot for this user (their chosen hour, else the
    # default). The scheduler now polls regularly and treats the slot as
    # "from this hour on", so a restart / short outage at the exact hour no
    # longer skips the day (catch-up).
    slot_hour = preferred_hour_utc if preferred_hour_utc is not None else settings.DAILY_CHECKIN_HOUR_UTC
    slot_today = now.replace(hour=slot_hour, minute=0, second=0, microsecond=0)
    if now < slot_today:
        return

    last_sent_at = await asyncio.to_thread(memory.get_last_plan_sent_at, phone_number)
    if last_sent_at is not None:
        if last_sent_at.tzinfo is not None:
            last_sent_at = last_sent_at.astimezone(timezone.utc).replace(tzinfo=None)
        if last_sent_at >= slot_today:
            return  # already got today's message
        # The previous message (e.g. Day 1, sent right after payment at ANY
        # time of day) must be at least MIN_GAP hours older than today's
        # slot; otherwise the next day waits for TOMORROW's morning slot
        # instead of being sent a few hours later the same day.
        gap_to_slot_hours = (slot_today - last_sent_at).total_seconds() / 3600
        if gap_to_slot_hours < settings.DAILY_CHECKIN_MIN_GAP_HOURS:
            logger.debug(
                f"⏭️ {phone_number}: last message was only {gap_to_slot_hours:.1f}h before today's "
                f"slot (min {settings.DAILY_CHECKIN_MIN_GAP_HOURS}h) — next one goes out tomorrow morning."
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

    # Already HELD (window closed / previous day unconfirmed): it is released
    # by the user's next message or button tap (main.py), never by polling.
    if plan_day.get("template_nudge_sent_at") is not None:
        return

    # CONFIRMATION GATE: tomorrow's task is only sent if the user confirmed
    # the previous day's task (Done / Not done button, or a text reply to
    # the follow-up). No confirmation = no reply inside 24h = closed WhatsApp
    # window, so we simply don't send. The user is told this in the note
    # attached to every daily message (settings.TASK_CONFIRM_NOTE). Nothing
    # is sent and nothing is billed. Once they tap an old button the held
    # day is pushed right away (see main.py::_flush_pending_checkin_day).
    if day_number > 1:
        prev_day = await asyncio.to_thread(memory.get_premium_plan_day, phone_number, day_number - 1)
        if prev_day is not None and prev_day.get("sent_at") is not None and prev_day.get("task_status") is None:
            await asyncio.to_thread(memory.mark_plan_day_template_nudge_sent, phone_number, day_number)
            logger.info(
                f"⏭️ Day {day_number} for {phone_number} NOT sent — day {day_number - 1} "
                f"not confirmed yet. Holding until they confirm."
            )
            return

    # WhatsApp only delivers free-form text within ~24h of the user's
    # last INBOUND message. Outside that window, send_text_message still
    # gets a 200 OK from Meta (shows up as "sent" in logs) but the
    # message is never actually delivered to the phone — this is exactly
    # what a stuck-at-"sent" check-in looks like. If the window's closed,
    # send an approved re-engagement TEMPLATE instead and hold the real
    # content: the webhook handler pushes it the moment this user
    # replies to anything (see _flush_pending_checkin_day in main.py).
    last_inbound_at = await asyncio.to_thread(memory.get_last_inbound_message_at, phone_number)
    window_open = False
    if last_inbound_at is not None:
        if last_inbound_at.tzinfo is not None:
            last_inbound_at = last_inbound_at.astimezone(timezone.utc).replace(tzinfo=None)
        window_open = (now - last_inbound_at) < timedelta(hours=settings.WHATSAPP_SESSION_WINDOW_HOURS)

    if not window_open:
        if settings.DAILY_CHECKIN_REENGAGEMENT_TEMPLATE:
            # Optional (paid) fallback — only used when an approved template
            # name is configured. Leave the setting empty to send nothing.
            try:
                await send_template_message(
                    phone_number,
                    settings.DAILY_CHECKIN_REENGAGEMENT_TEMPLATE,
                    params=[str(day_number)],
                )
                logger.info(
                    f"📨 Session window closed for {phone_number} — sent re-engagement "
                    f"template for day {day_number}; holding real content until they reply."
                )
            except Exception as e:
                logger.error(
                    f"❌ Failed to send re-engagement template for {phone_number} day "
                    f"{day_number}: {e}", exc_info=True,
                )
        else:
            logger.info(
                f"⏭️ Session window closed for {phone_number} (day {day_number}) — "
                f"holding it; it will be sent as soon as they message again."
            )
        # Held marker (see memory.get_pending_template_nudge_day): the next
        # inbound message from this user pushes the real content.
        await asyncio.to_thread(memory.mark_plan_day_template_nudge_sent, phone_number, day_number)
        return

    claimed = await asyncio.to_thread(
        memory.claim_plan_day_for_send, phone_number, day_number
    )
    if not claimed:
        logger.info(
            f"⏭️ Day {day_number} for {phone_number} is already being sent or was sent; "
            "skipping duplicate delivery."
        )
        return

    await send_plan_day(memory, phone_number, plan_day)


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

    users = await asyncio.to_thread(memory.get_active_premium_users)
    now_hour = datetime.utcnow().hour

    def _slot_hour(u: dict) -> int:
        h = u.get("preferred_checkin_hour_utc")
        return h if h is not None else settings.DAILY_CHECKIN_HOUR_UTC

    if current_hour_utc is not None:
        # Explicit hour (manual / external cron): users whose slot is exactly this hour.
        due_users = [u for u in users if _slot_hour(u) == current_hour_utc]
    else:
        # Normal mode: everyone whose slot has already started today. Users
        # who already got today's message are skipped inside
        # _send_checkin_for_user, so polling is safe and idempotent.
        due_users = [u for u in users if _slot_hour(u) <= now_hour]
    logger.debug(f"📅 Daily check-in poll: {len(due_users)} of {len(users)} active premium user(s) eligible.")

    for user in due_users:
        phone_number = user["phone_number"]
        preferred_hour = user.get("preferred_checkin_hour_utc")
        try:
            await _send_checkin_for_user(phone_number, preferred_hour_utc=preferred_hour)
        except Exception as e:
            logger.error(f"❌ Unhandled error sending check-in to {phone_number}: {e}", exc_info=True)



_REPAIR_COOLDOWN_SECONDS = 240
_REPAIR_MAX_ATTEMPTS = 8


async def _repair_allowed(phone_number: str) -> bool:
    """At most one repair attempt per user every few minutes, and a hard cap,
    so a persistent Gemini problem can't turn into an endless paid-call loop.
    Fails open if Redis is unavailable."""
    try:
        from app.core.redis_client import get_redis
        redis = get_redis()
        if not await redis.set(f"plan_repair_cooldown:{phone_number}", "1", nx=True, ex=_REPAIR_COOLDOWN_SECONDS):
            return False
        attempts = await redis.incr(f"plan_repair_attempts:{phone_number}")
        await redis.expire(f"plan_repair_attempts:{phone_number}", 6 * 3600)
        if attempts > _REPAIR_MAX_ATTEMPTS:
            logger.error(
                f"🛑 Repair: giving up on {phone_number} after {_REPAIR_MAX_ATTEMPTS} attempts "
                f"(check Gemini errors in the logs, then clear the key plan_repair_attempts:{phone_number})."
            )
            return False
    except Exception as e:
        logger.warning(f"⚠️ Repair throttle unavailable ({e}) — allowing.")
    return True


async def repair_missing_plans() -> None:
    """
    Safety net for the "paid but Day 1 never arrived" case: any active
    subscriber who finished onboarding but has NO plan for their current
    subscription (Gemini was down, a worker restarted mid-generation, ...)
    gets their plan generated + Day 1 sent now. Concurrency-safe: the plan
    lock inside generate_and_send_plan stops this racing the payment webhook.
    """
    try:
        phones = await asyncio.to_thread(memory.get_paid_users_without_plan)
    except Exception as e:
        logger.error(f"❌ Repair: could not list paid users without plan: {e}", exc_info=True)
        return
    for phone_number in phones:
        if not await _repair_allowed(phone_number):
            continue
        logger.warning(f"🛠️ Repair: {phone_number} is paid but has no plan — generating now.")
        try:
            await generate_and_send_plan(memory, phone_number)
        except Exception as e:
            logger.error(f"❌ Repair failed for {phone_number}: {e}", exc_info=True)


_REPAIR_INTERVAL_SECONDS = 120


_CHECKIN_POLL_SECONDS = 600


async def _run_forever() -> None:
    """
    Scheduler loop. Every few minutes it sends today's check-in to every
    active premium user whose daily slot (their chosen hour, or
    settings.DAILY_CHECKIN_HOUR_UTC) has started and who hasn't received
    today's message yet. Because it catches up instead of firing only
    inside the exact hour, a container restart / deploy / brief outage
    around the scheduled time no longer makes a user miss the day.
    Also runs the paid-user plan repair job every couple of minutes.
    """
    last_checkin_at = 0.0
    last_repair_at = 0.0
    logger.info(
        f"Daily check-in scheduler started - default slot {settings.DAILY_CHECKIN_HOUR_UTC}:00 UTC, "
        f"polling every {_CHECKIN_POLL_SECONDS // 60} min (catch-up enabled)."
    )
    while True:
        if time.monotonic() - last_checkin_at >= _CHECKIN_POLL_SECONDS or last_checkin_at == 0.0:
            last_checkin_at = time.monotonic()
            try:
                await run_daily_checkins()
            except Exception as e:
                logger.error(f"❌ Daily check-in run crashed: {e}", exc_info=True)
        if time.monotonic() - last_repair_at >= _REPAIR_INTERVAL_SECONDS:
            last_repair_at = time.monotonic()
            await repair_missing_plans()
        await asyncio.sleep(30)


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