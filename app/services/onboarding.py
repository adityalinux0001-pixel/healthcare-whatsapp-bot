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

from app.core.config import get_settings
from app.services.memory import ConversationMemory
from app.services.llm import (
    generate_premium_plan,
    detect_reply_language,
    classify_onboarding_answer,
    GeminiUnavailableError,
)
from app.services.whatsapp import send_text_message

logger = logging.getLogger(__name__)

settings = get_settings()



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


# ---------------------------------------------------------------------------
# Strict validators
#
# Only questions whose answers feed real downstream calculations (BMI/plan
# math for weight_height, scheduler hour for preferred_checkin_time) get a
# hard regex/format check here. Everything else (goal, diet, activity_level,
# medical_conditions, routine_time, past_attempts) stays free-text, gated
# only by classify_onboarding_answer's LLM judgment as before.
#
# Each validator returns (parsed_value, error_message):
#   - success:  (parsed_value, None)
#   - failure:  (None, "<short, friendly, WhatsApp-ready error message>")
#
# On failure, handle_onboarding_reply re-sends the error + the SAME
# question, without advancing and without calling the LLM classifier.
# ---------------------------------------------------------------------------

import re as _re

_WEIGHT_KG_RE = _re.compile(r"(\d{2,3}(?:\.\d+)?)\s*(?:kgs?|kilograms?)\b", _re.IGNORECASE)
_WEIGHT_LB_RE = _re.compile(r"(\d{2,3}(?:\.\d+)?)\s*(?:lbs?|pounds?)\b", _re.IGNORECASE)
_HEIGHT_CM_RE = _re.compile(r"(\d{2,3}(?:\.\d+)?)\s*(?:cms?|centimeters?|centimetres?)\b", _re.IGNORECASE)
_HEIGHT_M_RE = _re.compile(r"(\d(?:\.\d+))\s*m\b", _re.IGNORECASE)
_HEIGHT_FT_IN_RE = _re.compile(
    r"(\d)\s*(?:'|ft|feet)\s*(\d{1,2})?\s*(?:\"|in|inches)?", _re.IGNORECASE
)


def _parse_weight_height(user_text: str) -> tuple[Optional[dict], Optional[str]]:
    """
    Strictly parse a free-text reply like "72kg, 165cm", "72 kg 5'5\"",
    "158 lbs, 5 ft 6 in" into {"weight_kg": float, "height_cm": float}.

    Requires BOTH a weight and a height to be unambiguously present with
    units; sanity-range checked (weight 25-300kg, height 100-250cm) to
    catch typos. Returns (None, error_message) if either is missing,
    unparseable, or out of range.
    """
    text = user_text.strip().lower()

    weight_kg: Optional[float] = None
    m = _WEIGHT_KG_RE.search(text)
    if m:
        weight_kg = float(m.group(1))
    else:
        m = _WEIGHT_LB_RE.search(text)
        if m:
            weight_kg = float(m.group(1)) * 0.45359237

    height_cm: Optional[float] = None
    m = _HEIGHT_CM_RE.search(text)
    if m:
        height_cm = float(m.group(1))
    else:
        m = _HEIGHT_M_RE.search(text)
        if m:
            height_cm = float(m.group(1)) * 100
        else:
            m = _HEIGHT_FT_IN_RE.search(text)
            if m:
                feet = int(m.group(1))
                inches = int(m.group(2)) if m.group(2) else 0
                height_cm = (feet * 12 + inches) * 2.54

    if weight_kg is None or height_cm is None:
        missing = []
        if weight_kg is None:
            missing.append("weight (with kg or lbs)")
        if height_cm is None:
            missing.append("height (with cm, m, or ft/in)")
        return None, (
            "Hmm, I couldn't quite catch your " + " and ".join(missing) + " 🤔\n\n"
            "Please include units, e.g. \"72kg, 165cm\" or \"158 lbs, 5 ft 6 in\"."
        )

    if not (25 <= weight_kg <= 300):
        return None, (
            "That weight looks off to me — please double check and resend, "
            "e.g. \"72kg, 165cm\"."
        )
    if not (100 <= height_cm <= 250):
        return None, (
            "That height looks off to me — please double check and resend, "
            "e.g. \"72kg, 165cm\"."
        )

    return {"weight_kg": round(weight_kg, 1), "height_cm": round(height_cm, 1)}, None


def _parse_checkin_time_strict(user_text: str) -> tuple[Optional[dict], Optional[str]]:
    """
    Strict wrapper around _parse_preferred_hour_to_utc: unlike the
    fail-open fallback used at plan-generation time, onboarding itself
    now REQUIRES a parseable time and re-asks if it can't find one,
    rather than silently defaulting.
    """
    hour_utc = _parse_preferred_hour_to_utc(user_text)
    if hour_utc is None:
        return None, (
            "I couldn't quite figure out a time from that 🤔\n\n"
            "Please reply with something like \"8am\", \"9:30 pm\", or "
            "\"7 in the morning\" (your local time, IST by default)."
        )
    return {"checkin_hour_utc": hour_utc}, None


# Maps onboarding answer key -> strict validator function. Only keys
# present here get strict validation; all other keys are untouched.
_STRICT_VALIDATORS = {
    "weight_height": _parse_weight_height,
    "preferred_checkin_time": _parse_checkin_time_strict,
}


# ---------------------------------------------------------------------------
# Minimal-effort filter (applies to the free-text questions that have no
# strict validator above, i.e. everything except weight_height and
# preferred_checkin_time: goal, diet, activity_level, medical_conditions,
# routine_time, past_attempts).
#
# This is a cheap, deterministic pre-check that runs BEFORE the LLM
# classifier. It only screens out replies that are too short / low-content
# to plausibly be a real answer (e.g. "idk", "x", "??", a single emoji) —
# it does NOT judge topical relevance, that's still classify_onboarding_
# answer's job. Genuinely short-but-valid answers ("vegan", "none", "20
# min") are explicitly allowed via a short whitelist + length floor, so
# this stays a low-effort filter, not a strictness filter.
# ---------------------------------------------------------------------------

_LOW_EFFORT_PHRASES = {
    "idk", "dunno", "dont know", "don't know", "whatever", "anything",
    "who knows", "not sure", "no idea", "meh", "idc", "shrug",
}

_MIN_CONTENT_CHARS = 2  # after stripping punctuation/whitespace/emoji


def _looks_low_effort(user_text: str) -> bool:
    """
    True if the reply is too thin to be a genuine answer attempt: empty,
    a single low-info word/phrase, or almost no alphanumeric content
    (e.g. just punctuation or an emoji). Short legitimate answers like
    "vegan", "none", or "20 min" pass through untouched.
    """
    stripped = user_text.strip().lower()
    if not stripped:
        return True

    # Strip common trailing punctuation for the phrase check.
    normalized = stripped.strip(" .!?,;:'\"")
    if normalized in _LOW_EFFORT_PHRASES:
        return True

    # Count actual letters/digits — filters out "??", "...", lone emoji,
    # or other near-empty replies while letting "20 min" or "5'5" through.
    alnum_chars = sum(ch.isalnum() for ch in stripped)
    if alnum_chars < _MIN_CONTENT_CHARS:
        return True

    return False


_LOW_EFFORT_RETRY_MESSAGE = (
    "I need a bit more to go on for this one 🙏\n\n"
)


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
    category = category or settings.DEFAULT_PLAN_CATEGORY
    questions = _questions_for(category)

    await asyncio.to_thread(memory.start_onboarding_session, phone_number, category)

    intro = (
        f"🎉 You're all set on the {category.replace('_', ' ')} plan!\n\n"
        f"Just {len(questions)} quick questions so I can personalize your "
        f"{settings.PREMIUM_PLAN_DAYS}-day plan, then I'll get it ready for you."
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
    category = session.get("category") or settings.DEFAULT_PLAN_CATEGORY

    raw_time_answer = answers.get("preferred_checkin_time", "")
    preferred_hour_utc = _parse_preferred_hour_to_utc(raw_time_answer) if raw_time_answer else None
    if preferred_hour_utc is None:
        preferred_hour_utc = settings.DAILY_CHECKIN_HOUR_UTC
        logger.info(
            f"Couldn't parse preferred check-in time '{raw_time_answer}' for "
            f"{phone_number} - falling back to default hour {preferred_hour_utc}:00 UTC."
        )
    await asyncio.to_thread(memory.set_preferred_checkin_hour, phone_number, preferred_hour_utc)

    await send_text_message(
        phone_number,
        "Perfect, thank you! 🙏 Give me a moment while I put together your "
        f"full {settings.PREMIUM_PLAN_DAYS}-day plan...",
    )

    context_text = await asyncio.to_thread(memory.get_conversation_context, phone_number, limit=5)
    required_language = None
    if context_text:
        required_language = await detect_reply_language(context_text[-500:])

    try:
        days = await generate_premium_plan(
            onboarding_answers=answers,
            category=category,
            total_days=settings.PREMIUM_PLAN_DAYS,
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
        f"✅ Your personalized {settings.PREMIUM_PLAN_DAYS}-day plan is ready!\n\n"
        f"Day 1 is coming up right now 👇 and then one message per day, every day "
        f"around your chosen time. I'll also check in with a quick question after "
        f"each one — just reply whenever you can 💪"
    )
    await send_text_message(phone_number, confirm)
    await asyncio.to_thread(
        memory.save_message, phone_number, "assistant", confirm, message_type="text"
    )


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

    header = f"*Day {day_number} of {settings.PREMIUM_PLAN_DAYS}* 🗓️\n\n"
    message = header + plan_day["message_text"]
    followup_question = plan_day.get("followup_question")

    await send_text_message(phone_number, message)
    await asyncio.to_thread(memory.mark_plan_day_sent, phone_number, day_number)
    await asyncio.to_thread(
        memory.save_message, phone_number, "assistant", message, message_type="text"
    )
    logger.info(f"✅ Sent day {day_number}/{settings.PREMIUM_PLAN_DAYS} immediately to {phone_number}")

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

    current_key, current_prompt = questions[question_index]

    await asyncio.to_thread(
        memory.save_message, phone_number, "user", user_text, message_type="text"
    )

    # Strict gate FIRST: for keys with a registered validator (currently
    # weight_height and preferred_checkin_time — see _STRICT_VALIDATORS),
    # the reply must parse into a well-formed value before we even
    # consider it an "answer". Failing this is a hard re-ask with a
    # specific format hint, unlike the LLM gate below, and skips the LLM
    # classifier call entirely (no ambiguity to resolve — it's just
    # unparseable).
    validator = _STRICT_VALIDATORS.get(current_key)
    if validator is not None:
        parsed_value, error_message = validator(user_text)
        if error_message is not None:
            reply_text = f"{error_message}\n\n{current_prompt}"
            await send_text_message(phone_number, reply_text)
            await asyncio.to_thread(
                memory.save_message, phone_number, "assistant", reply_text, message_type="text"
            )
            return True
        # Store the original text (kept human-readable / language-agnostic
        # for the plan-generation LLM prompt downstream) — the parsed_value
        # dict is what we've just confirmed IS extractable from it, used
        # here only to gate acceptance. preferred_checkin_time's raw text
        # is still what _parse_preferred_hour_to_utc re-parses later in
        # _finish_onboarding_and_generate_plan.
        new_index = await asyncio.to_thread(
            memory.save_onboarding_answer, phone_number, current_key, user_text
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

    # Minimal-effort gate (free-text questions only — Q1/Q8 already
    # returned above via the strict validator branch). Cheap, deterministic
    # rejection of near-empty replies like "idk" or "??" BEFORE spending an
    # LLM call on them. Does not judge topical relevance — that's still the
    # classifier's job right below.
    if _looks_low_effort(user_text):
        reply_text = f"{_LOW_EFFORT_RETRY_MESSAGE}{current_prompt}"
        await send_text_message(phone_number, reply_text)
        await asyncio.to_thread(
            memory.save_message, phone_number, "assistant", reply_text, message_type="text"
        )
        return True

    # Gate: only save + advance if this reply genuinely answers the
    # CURRENT question. Off-topic replies (greetings, small talk, random
    # questions, etc.) are acknowledged gracefully and the SAME question
    # is re-sent, instead of being silently stored as the answer and
    # skipped past — see classify_onboarding_answer's docstring for the
    # fail-open rationale on classifier errors.
    classification = await classify_onboarding_answer(current_prompt, user_text)
    if not classification.get("is_answer", True):
        acknowledgment = classification.get("acknowledgment") or "Got it 🙂"
        reply_text = f"{acknowledgment}\n\n{current_prompt}"
        await send_text_message(phone_number, reply_text)
        await asyncio.to_thread(
            memory.save_message, phone_number, "assistant", reply_text, message_type="text"
        )
        return True

    new_index = await asyncio.to_thread(
        memory.save_onboarding_answer, phone_number, current_key, user_text
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