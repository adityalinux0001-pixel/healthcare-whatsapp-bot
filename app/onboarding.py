"""
Onboarding question flow — runs ONCE, immediately after a user subscribes
(right after payment is confirmed), before the single "generate the whole
21-day plan" LLM call.

Flow (matches the updated architecture diagram):

    User subscribes
        -> Category selection (default: weight_loss; more categories added
           later purely as new entries in QUESTIONS_BY_CATEGORY / app.llm's
           PLAN_CATEGORY_PROMPTS — nothing else changes)
        -> Onboarding questions (7 profile questions + 1 "what time should
           I check in daily?" question, one at a time via WhatsApp)
        -> LLM generates full plan (ONE call — app.llm.generate_premium_plan)
        -> Saved to database (app.memory.save_premium_plan), preferred
           check-in hour saved (app.memory.set_preferred_checkin_hour)
        -> Day 1 is sent IMMEDIATELY, right here, instead of waiting for
           the scheduler's next run
        -> Daily scheduler takes over from Day 2 onward (app/daily_checkin.py),
           sending each user's check-in at THEIR preferred hour

This module owns steps 2-5. app/main.py calls into it from two places:
  1. Right after subscription activation (razorpay webhook handler) ->
     start_onboarding(phone_number).
  2. On every incoming text message, BEFORE normal Q&A routing, if the user
     has an onboarding session in progress -> handle_onboarding_reply(...).
     Returns True if the message was consumed as an onboarding answer (so
     main.py should stop processing that message normally), False
     otherwise.
"""

import asyncio
import logging
from typing import Optional

from app.config import get_settings
from app.memory import ConversationMemory
from app.llm import generate_premium_plan, detect_reply_language, GeminiUnavailableError
from app.whatsapp import send_text_message

logger = logging.getLogger(__name__)

settings = get_settings()


# Each question is (key, prompt_text). `key` is the field name stored in
# onboarding_sessions.answers and later passed straight into the plan
# generation prompt. Order matters — this is the order asked.
#
# 7 questions chosen to give the LLM everything needed to personalize a
# full multi-week plan in one shot: sizing (1), ambition (2), constraints
# that touch every single day's content (3, 5), how hard it can push (4),
# scheduling realism (6), and what to avoid repeating (7, optional-feel but
# still asked so the plan doesn't recommend something they already know
# doesn't work for them).
QUESTIONS_BY_CATEGORY: dict[str, list[tuple[str, str]]] = {
    "weight_loss": [
        (
            "weight_height",
            "Let's build your personalized plan! 📋\n\n"
            "1️⃣ What's your current weight and height? (e.g. \"72kg, 165cm\")",
        ),
        (
            "goal",
            "2️⃣ What's your goal? (e.g. \"lose 5kg\", \"fit into old jeans\", \"just feel healthier\")",
        ),
        (
            "diet",
            "3️⃣ Any dietary preference or restrictions? (e.g. vegetarian, vegan, no dairy, diabetic, or \"no restrictions\")",
        ),
        (
            "activity_level",
            "4️⃣ How active is your day-to-day? (e.g. \"mostly sitting/desk job\", \"light exercise\", \"already quite active\")",
        ),
        (
            "medical_conditions",
            "5️⃣ Any existing medical conditions or injuries I should know about? "
            "(e.g. knee pain, PCOS, thyroid, pregnancy — or \"none\")\n\n"
            "This helps me keep every suggestion safe for you 🙏",
        ),
        (
            "routine_time",
            "6️⃣ What's your typical routine, and how much time can you realistically give this each day? "
            "(e.g. \"early mornings, 20 minutes\", \"busy till evening, 1 hour at night\")",
        ),
        (
            "past_attempts",
            "7️⃣ Almost done! Have you tried anything before that didn't work for you, or that you'd rather avoid? "
            "(e.g. \"tried keto before\", \"hate running\", or \"nothing to avoid\")",
        ),
        (
            "preferred_checkin_time",
            "8️⃣ Last one! What time of day should I send your daily check-in? "
            "(e.g. \"8am\", \"9:30 pm\", \"7 in the morning\" — reply in your local time, "
            "IST/India time by default)",
        ),
    ],
    # Future categories go here, e.g. "yoga": [...], "bulking": [...].
    # Falls back to the weight_loss question set if a category has no
    # dedicated list yet (see _questions_for below).
}


def _questions_for(category: str) -> list[tuple[str, str]]:
    return QUESTIONS_BY_CATEGORY.get(category, QUESTIONS_BY_CATEGORY["weight_loss"])


# IST is UTC+5:30. Most of this bot's users reply in local (India) time
# when asked "what time works for you", so we assume IST unless the user
# explicitly says UTC/GMT. This is a best-effort parse for a free-text
# WhatsApp reply, not a full NLP time parser — falls back to
# settings.daily_checkin_hour_utc (see app/config.py) if we can't
# confidently parse an hour.
_IST_OFFSET_HOURS = 5.5


def _parse_preferred_hour_to_utc(user_text: str) -> Optional[int]:
    """
    Parse a free-text reply like "8am", "9:30 pm", "7 in the morning",
    "14:00 UTC" into an integer UTC hour (0-23). Returns None if nothing
    resembling a time could be confidently extracted (caller should fall
    back to the default hour rather than guess).
    """
    import re

    text = user_text.strip().lower()
    is_utc = bool(re.search(r"\b(utc|gmt)\b", text))

    # 24-hour "HH:MM" or "HH" with no am/pm marker, e.g. "14:00", "21"
    m = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", text)
    hour = None
    if m:
        hour = int(m.group(1))
    else:
        # "8am", "9:30 pm", "7 pm", "7 in the morning/evening/night"
        m = re.search(r"\b(1[0-2]|0?[1-9])(?::([0-5]\d))?\s*(am|pm|a\.m\.|p\.m\.)\b", text)
        if m:
            hour = int(m.group(1))
            meridiem = m.group(3).replace(".", "")
            if meridiem == "pm" and hour != 12:
                hour += 12
            elif meridiem == "am" and hour == 12:
                hour = 0
        else:
            m = re.search(r"\b(1[0-2]|0?[1-9])\b.*\b(morning|noon|afternoon|evening|night)\b", text)
            if m:
                hour = int(m.group(1))
                part = m.group(2)
                if part in ("afternoon", "evening", "night") and hour != 12:
                    hour += 12
                elif part == "noon":
                    hour = 12

    if hour is None or not (0 <= hour <= 23):
        return None

    if is_utc:
        return hour % 24
    # Convert from assumed IST to UTC.
    return int((hour - _IST_OFFSET_HOURS) % 24)


async def start_onboarding(
    memory: ConversationMemory,
    phone_number: str,
    category: Optional[str] = None,
) -> None:
    """
    Kick off onboarding for a freshly-subscribed user: creates/resets the
    onboarding session at question 0 and sends the first question. Call
    this right after memory.activate_subscription(...) succeeds (see the
    /razorpay/webhook handler in app/main.py).
    """
    category = category or settings.default_plan_category
    questions = _questions_for(category)

    await asyncio.to_thread(memory.start_onboarding_session, phone_number, category)

    intro = (
        f"🎉 You're all set on the {category.replace('_', ' ')} plan!\n\n"
        f"Just {len(questions)} quick questions so I can personalize your "
        f"{settings.premium_plan_days}-day plan, then I'll get it ready for you."
    )
    await send_text_message(phone_number, intro)
    await asyncio.to_thread(
        memory.save_message, phone_number, "assistant", intro, message_type="text"
    )

    first_key, first_prompt = questions[0]
    await send_text_message(phone_number, first_prompt)
    await asyncio.to_thread(
        memory.save_message, phone_number, "assistant", first_prompt, message_type="text"
    )


async def _finish_onboarding_and_generate_plan(
    memory: ConversationMemory, phone_number: str
) -> None:
    """
    Last answer just came in. Mark the session complete, make the SINGLE
    LLM call that generates all {premium_plan_days} days + follow-up
    questions, save the whole plan to the database, and tell the user
    it's ready. If plan generation fails, the user is told to hang tight
    and nothing partial is saved — safe to retry by re-triggering
    onboarding completion (e.g. an admin re-running this function).
    """
    session = await asyncio.to_thread(memory.mark_onboarding_complete, phone_number)
    answers = session.get("answers", {}) or {}
    category = session.get("category") or settings.default_plan_category

    raw_time_answer = answers.get("preferred_checkin_time", "")
    preferred_hour_utc = _parse_preferred_hour_to_utc(raw_time_answer) if raw_time_answer else None
    if preferred_hour_utc is None:
        preferred_hour_utc = settings.daily_checkin_hour_utc
        logger.info(
            f"Couldn't parse preferred check-in time '{raw_time_answer}' for "
            f"{phone_number} - falling back to default hour {preferred_hour_utc}:00 UTC."
        )
    await asyncio.to_thread(memory.set_preferred_checkin_hour, phone_number, preferred_hour_utc)

    await send_text_message(
        phone_number,
        "Perfect, thank you! 🙏 Give me a moment while I put together your "
        f"full {settings.premium_plan_days}-day plan...",
    )

    context_text = await asyncio.to_thread(memory.get_conversation_context, phone_number, limit=5)
    required_language = None
    if context_text:
        required_language = await detect_reply_language(context_text[-500:])

    try:
        days = await generate_premium_plan(
            onboarding_answers=answers,
            category=category,
            total_days=settings.premium_plan_days,
            required_language=required_language,
        )
    except GeminiUnavailableError:
        logger.warning(f"⏭️ Plan generation unavailable for {phone_number} — will need a retry.")
        await send_text_message(
            phone_number,
            "I'm having trouble generating your plan right now due to high demand. "
            "Please message me again in a few minutes and I'll pick up right where we left off 🙏",
        )
        return
    except ValueError as e:
        logger.error(f"❌ Plan generation malformed for {phone_number}: {e}", exc_info=True)
        await send_text_message(
            phone_number,
            "Something went wrong while building your plan. Our team has been notified — "
            "please message me again shortly and I'll retry.",
        )
        return

    await asyncio.to_thread(memory.save_premium_plan, phone_number, category, days)

    confirm = (
        f"✅ Your personalized {settings.premium_plan_days}-day plan is ready!\n\n"
        f"Day 1 is coming up right now 👇 and then one message per day, every day "
        f"around your chosen time. I'll also check in with a quick question after "
        f"each one — just reply whenever you can 💪"
    )
    await send_text_message(phone_number, confirm)
    await asyncio.to_thread(
        memory.save_message, phone_number, "assistant", confirm, message_type="text"
    )

    # Send Day 1 IMMEDIATELY instead of waiting for the scheduler's next
    # run - this is what was missing before: onboarding used to end on a
    # generic "plan is ready" message with no actual Day 1 content until
    # the scheduled job next fired (up to ~24h later).
    await _send_plan_day_now(memory, phone_number, day_number=1)


async def _send_plan_day_now(
    memory: ConversationMemory, phone_number: str, day_number: int
) -> None:
    """
    Fetch a specific pregenerated plan day and send it right now, with a
    clear "Day X of N" header prepended so it's unambiguous in the chat
    which message is the daily plan content (as opposed to onboarding
    confirmations, follow-up questions, etc.). Marks the row as sent and
    opens the same-day follow-up window, exactly like the scheduled job
    in app/daily_checkin.py does - this is just that same send-path,
    triggered immediately instead of on the next scheduler tick.
    """
    plan_day = await asyncio.to_thread(memory.get_premium_plan_day, phone_number, day_number)
    if not plan_day:
        logger.error(
            f"❌ Tried to immediately send day {day_number} for {phone_number} but "
            "no such pregenerated row exists — skipping."
        )
        return

    header = f"*Day {day_number} of {settings.premium_plan_days}* 🗓️\n\n"
    message = header + plan_day["message_text"]
    followup_question = plan_day.get("followup_question")

    await send_text_message(phone_number, message)
    await asyncio.to_thread(memory.mark_plan_day_sent, phone_number, day_number)
    await asyncio.to_thread(
        memory.save_message, phone_number, "assistant", message, message_type="text"
    )
    logger.info(f"✅ Sent day {day_number}/{settings.premium_plan_days} immediately to {phone_number}")

    if followup_question:
        await send_text_message(phone_number, followup_question)
        await asyncio.to_thread(
            memory.save_message, phone_number, "assistant", followup_question, message_type="text"
        )


async def handle_onboarding_reply(
    memory: ConversationMemory, phone_number: str, user_text: str
) -> bool:
    """
    Call this from the webhook text handler BEFORE normal Q&A routing.

    Returns True if `user_text` was consumed as an answer to the current
    onboarding question (caller should stop processing this message any
    further), False if this user has no onboarding session in progress
    (caller should proceed with normal handling).
    """
    session = await asyncio.to_thread(memory.get_onboarding_session, phone_number)
    if not session or session["is_complete"]:
        return False

    category = session["category"]
    questions = _questions_for(category)
    question_index = session["question_index"]

    if question_index >= len(questions):
        # Defensive: shouldn't happen (mark_onboarding_complete runs on the
        # last answer), but don't let a stuck session swallow messages
        # forever.
        await _finish_onboarding_and_generate_plan(memory, phone_number)
        return True

    current_key, _ = questions[question_index]
    new_index = await asyncio.to_thread(
        memory.save_onboarding_answer, phone_number, current_key, user_text
    )

    await asyncio.to_thread(
        memory.save_message, phone_number, "user", user_text, message_type="text"
    )

    if new_index >= len(questions):
        await _finish_onboarding_and_generate_plan(memory, phone_number)
        return True

    next_key, next_prompt = questions[new_index]
    await send_text_message(phone_number, next_prompt)
    await asyncio.to_thread(
        memory.save_message, phone_number, "assistant", next_prompt, message_type="text"
    )
    return True