"""
Onboarding question flow — runs ONCE, now starting immediately on a
user's first message ("hi"/"hello"/etc.), BEFORE payment, so a
non-premium user's free Q&A (see settings.FREE_QUESTION_LIMIT in
app/main.py) can be personalized right away. Generating the actual paid
21-day plan is a SEPARATE step gated on payment — see generate_and_send_plan().

Flow:

    User's first message ("hi")
        -> Category selection (default: weight_loss; more categories added
           later purely as new entries in QUESTIONS_BY_CATEGORY / app.llm's
           PLAN_CATEGORY_PROMPTS — nothing else changes)
        -> Onboarding questions (8 profile questions: age & gender, height &
           weight, city, target weight, sugar intake, exercise, family
           history, dairy — one at a time via WhatsApp) — FREE, no payment
           required. The daily check-in hour is no longer asked; it uses
           settings.DAILY_CHECKIN_HOUR_UTC.
        -> Onboarding complete (_finish_onboarding): answers + preferred
           check-in hour are saved. If the user isn't premium yet (the
           normal case), they're invited to ask questions now — up to
           settings.FREE_QUESTION_LIMIT free answers before the payment
           pitch takes over (see app/main.py's free-question gate).
        -> Once/if the user pays, app/main.py's /razorpay/webhook handler
           calls generate_and_send_plan(): ONE LLM call
           (app.llm.generate_premium_plan) generates the full plan from the
           already-saved onboarding answers, it's saved to the database
           (app.memory.save_premium_plan), and Day 1 is sent IMMEDIATELY,
           instead of waiting for the scheduler's next run.
        -> Daily scheduler takes over from Day 2 onward (app/daily_checkin.py),
           sending each user's check-in at THEIR preferred hour

This module owns onboarding + plan generation. app/main.py calls into it
from three places:
  1. On a user's first-ever message (_maybe_send_greeting) ->
     start_onboarding(phone_number).
  2. On every incoming text message, BEFORE normal Q&A routing, if the user
     has an onboarding session in progress -> handle_onboarding_reply(...).
     Returns True if the message was consumed as an onboarding answer (so
     main.py should stop processing that message normally), False
     otherwise.
  3. Right after subscription activation (razorpay webhook handler), if
     onboarding was already complete at that point -> generate_and_send_plan
     (phone_number). (If onboarding wasn't complete yet, _finish_onboarding
     detects the now-active subscription itself and generates the plan the
     moment onboarding finishes instead.)
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from app.core.config import get_settings
from app.services.memory import ConversationMemory
from app.services.llm import (
    generate_premium_plan,
    detect_reply_language,
    classify_onboarding_answer,
    GeminiUnavailableError,
)
from app.services.whatsapp import send_text_message, send_template_message
from app.services.plan_delivery import send_plan_day

logger = logging.getLogger(__name__)

settings = get_settings()



QUESTIONS_BY_CATEGORY: dict[str, list[tuple[str, str]]] = {
    "weight_loss": [
        (
            "age_gender",
            "Let's build your personalized plan! 📋\n\n"
            "1️⃣ How old are you, and what is your gender?\n\n"
            "Age: ___\n"
            "Gender: Male / Female / Prefer not to say\n\n"
            "(e.g. \"28, Female\")",
        ),
        (
            "height_weight",
            "2️⃣ What are your current height and weight?\n\n"
            "Height: ___ cm / ft & inches\n"
            "Weight: ___ kg / lbs\n\n"
            "(e.g. \"165 cm, 72 kg\" or \"5 ft 6 in, 158 lbs\")",
        ),
        (
            "location",
            "3️⃣ Where do you currently live?\n\n"
            "City: ___ (e.g. \"Indore\")",
        ),
        (
            "target_weight",
            "4️⃣ What is your target weight?\n\n"
            "Enter target weight: ___ kg / lbs\n"
            "(e.g. \"65 kg\")\n\n"
            "Not sure? Just reply \"Not sure\" and I'll suggest a healthy target for you.",
        ),
        (
            "sugar_intake",
            "5️⃣ How much sugar or sugary food do you normally consume?\n\n"
            "1. None or very little\n"
            "2. Low\n"
            "3. Moderate\n"
            "4. High\n"
            "5. Very high\n\n"
            "Examples: sweets, desserts, sugary tea/coffee, soft drinks, packaged juices, etc.\n\n"
            "Reply with the number (1-5) 👇",
        ),
        (
            "exercise_level",
            "6️⃣ How much exercise or physical activity do you usually get?\n\n"
            "1. I don't exercise\n"
            "2. Less than 10 minutes a day\n"
            "3. 10–30 minutes a day\n"
            "4. 30–60 minutes a day\n"
            "5. More than 60 minutes a day\n\n"
            "Reply with the number (1-5) 👇",
        ),
        (
            "family_obesity",
            "7️⃣ Does obesity or significant weight gain run in your family?\n\n"
            "1. Yes\n"
            "2. No\n"
            "3. Not sure\n\n"
            "Reply with the number (1-3) 👇",
        ),
        (
            "dairy_intake",
            "8️⃣ Last one! How much dairy do you normally consume?\n\n"
            "1. None\n"
            "2. Low — occasionally\n"
            "3. Moderate — once a day\n"
            "4. High — 2–3 times a day\n"
            "5. Very high — more than 3 times a day\n\n"
            "Examples: milk, curd/yogurt, paneer, cheese, butter, cream, etc.\n\n"
            "Reply with the number (1-5) 👇",
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


def _parse_height_weight(user_text: str) -> tuple[Optional[dict], Optional[str]]:
    """Q2: reuse the strict weight+height parser, and store a normalized,
    unit-consistent string for the plan generator."""
    parsed, err = _parse_weight_height(user_text)
    if err:
        return None, err
    parsed["normalized"] = f"Height {parsed['height_cm']:g} cm, Weight {parsed['weight_kg']:g} kg"
    return parsed, None


_FEMALE_RE = _re.compile(r"\b(female|woman|girl|lady|f|ladki|mahila)\b", _re.IGNORECASE)
_MALE_RE = _re.compile(r"\b(male|man|boy|gentleman|m|ladka|purush)\b", _re.IGNORECASE)
_PREFER_NOT_RE = _re.compile(r"prefer|rather not|not to say|skip|don'?t want|dont want", _re.IGNORECASE)


def _parse_age_gender(user_text: str) -> tuple[Optional[dict], Optional[str]]:
    """Q1: needs BOTH an age (12-100) and a gender (male / female /
    prefer not to say)."""
    text = user_text.strip()
    nums = _re.findall(r"\b(\d{1,3})\b", text)
    age = int(nums[0]) if nums else None

    female, male = bool(_FEMALE_RE.search(text)), bool(_MALE_RE.search(text))
    if _PREFER_NOT_RE.search(text) and not (female or male):
        gender = "Prefer not to say"
    elif female and not male:
        gender = "Female"
    elif male and not female:
        gender = "Male"
    else:
        gender = None

    if age is None or gender is None:
        missing = []
        if age is None:
            missing.append("age")
        if gender is None:
            missing.append("gender (Male / Female / Prefer not to say)")
        return None, (
            "Hmm, I couldn't quite catch your " + " and ".join(missing) + " 🤔\n\n"
            "Please reply like this: \"28, Female\"."
        )
    if not (12 <= age <= 100):
        return None, "That age looks off to me — please double check and resend, e.g. \"28, Female\"."
    return {"age": age, "gender": gender, "normalized": f"Age {age}, {gender}"}, None


_NOT_SURE_RE = _re.compile(
    r"not sure|unsure|don'?t know|dont know|no idea|idk|you (?:tell|suggest|decide)|suggest|recommend|pata nahi",
    _re.IGNORECASE,
)


def _parse_target_weight(user_text: str) -> tuple[Optional[dict], Optional[str]]:
    """Q4: a target weight with units, OR "not sure"."""
    text = user_text.strip().lower()
    kg: Optional[float] = None
    m = _WEIGHT_KG_RE.search(text)
    if m:
        kg = float(m.group(1))
    else:
        m = _WEIGHT_LB_RE.search(text)
        if m:
            kg = float(m.group(1)) * 0.45359237

    if kg is None:
        if _NOT_SURE_RE.search(text):
            return {"target_kg": None, "normalized": "Not sure — please recommend a healthy target weight"}, None
        return None, (
            "Please include your target weight with units, e.g. \"65 kg\" or \"143 lbs\" — "
            "or reply \"Not sure\" and I'll suggest one 🙂"
        )
    if not (25 <= kg <= 300):
        return None, "That target weight looks off to me — please double check and resend, e.g. \"65 kg\"."
    kg = round(kg, 1)
    return {"target_kg": kg, "normalized": f"{kg:g} kg"}, None


_NOT_A_CITY = {
    "hi", "hello", "hey", "ok", "okay", "yes", "no", "thanks", "thank you",
    "hii", "hlo", "test", "nothing", "none", "na", "n/a",
}


def _parse_city(user_text: str) -> tuple[Optional[dict], Optional[str]]:
    """Q3: a plausible city/place name (no LLM call needed)."""
    text = " ".join(user_text.strip().split())
    letters = sum(ch.isalpha() for ch in text)
    ok = (
        letters >= 2
        and len(text) <= 60
        and len(text.split()) <= 6
        and "?" not in text
        and not any(ch.isdigit() for ch in text)
        and text.lower().strip(" .!,") not in _NOT_A_CITY
    )
    if not ok:
        return None, "Please tell me the city you live in, e.g. \"Indore\" or \"Pune, Maharashtra\" 🏙️"
    return {"city": text, "normalized": text}, None


def _choice_validator(options: list[tuple[str, str]], order: list[int] | None = None):
    """
    Build a validator for a numbered multiple-choice question.

    options: [(label_to_store, regex_of_free_text_synonyms), ...] in the
             same order as the numbered list shown to the user.
    order:   1-based option indexes in the order the free-text regexes
             should be tried (most specific first, e.g. "very high"
             before "high"). Defaults to 1..N.

    Accepts "3", "3.", "option 3", "3 - moderate", or a recognizable phrase
    like "very high". Anything else is rejected with a re-ask. The stored
    answer is the canonical option label.
    """
    n = len(options)
    compiled = [_re.compile(pat, _re.IGNORECASE) for _, pat in options]
    order = order or list(range(1, n + 1))

    def _validate(user_text: str) -> tuple[Optional[dict], Optional[str]]:
        text = user_text.strip().lower()
        digit_groups = _re.findall(r"\d+", text)
        if len(digit_groups) == 1:
            m = _re.match(r"^\W*(?:option\s*|opt\s*|no\.?\s*)?(\d+)(?!\d)", text)
            if m and 1 <= int(m.group(1)) <= n:
                label = options[int(m.group(1)) - 1][0]
                return {"choice": label, "normalized": label}, None
        for idx in order:
            if compiled[idx - 1].search(text):
                label = options[idx - 1][0]
                return {"choice": label, "normalized": label}, None
        return None, f"Please reply with the number of your choice (1-{n}) 🙏"

    return _validate


_SUGAR_OPTIONS = [
    ("None or very little", r"\bnone\b|very\s*little|no\s*sugar|\bzero\b|\bno\b|nahi"),
    ("Low", r"\blow\b|\blittle\b|rarely|occasional"),
    ("Moderate", r"moderate|medium|average|normal|sometimes"),
    ("High", r"\bhigh\b|\bheavy\b|a\s*lot|zyada"),
    ("Very high", r"very\s*high|extreme|too\s*much|bahut\s*zyada"),
]
_EXERCISE_OPTIONS = [
    ("I don't exercise", r"(?:don'?t|do not|dont)\s*exercise|no\s*exercise|\bnone\b|\bnever\b|\bnothing\b|\bno\b|nahi"),
    ("Less than 10 minutes a day", r"less than 10|under 10|<\s*10|few minutes"),
    ("10–30 minutes a day", r"\b10\s*(?:-|–|—|to)\s*30"),
    ("30–60 minutes a day", r"\b30\s*(?:-|–|—|to)\s*60"),
    ("More than 60 minutes a day", r"more than 60|over 60|above 60|60\s*\+|\d+\s*hours?"),
]
_FAMILY_OPTIONS = [
    ("Yes", r"\byes\b|\by\b|\byeah\b|\bhaan?\b|\bha\b"),
    ("No", r"\bno\b|\bn\b|\bnope\b|nahi"),
    ("Not sure", r"not sure|unsure|don'?t know|dont know|no idea|maybe|idk|pata nahi"),
]
_DAIRY_OPTIONS = [
    ("None", r"\bnone\b|\bno\b|\bnever\b|don'?t|do not|\bvegan\b|\bzero\b|nahi"),
    ("Low — occasionally", r"\blow\b|occasional|rarely|sometimes|weekly"),
    ("Moderate — once a day", r"moderate|medium|\bonce\b|one time|1 time|daily|every\s*day"),
    ("High — 2–3 times a day", r"\bhigh\b|\b2\s*(?:-|–|—|to)\s*3\b|\btwo\b|\bthrice\b|\b3 times\b|three times"),
    ("Very high — more than 3 times a day", r"very\s*high|more than 3|\b3\s*\+|>\s*3|\b4\s*\+|many times"),
]


# Maps onboarding answer key -> strict validator function. Every
# question in the weight_loss flow is now validated deterministically
# (no LLM classifier call is needed, which also saves cost/latency).
# A validator returns ({"normalized": "<text to store>", ...}, None) or
# (None, "<friendly error>"); the "normalized" string — not the raw reply
# — is what gets saved and later fed to the plan generator.
_STRICT_VALIDATORS = {
    "age_gender": _parse_age_gender,
    "height_weight": _parse_height_weight,
    "location": _parse_city,
    "target_weight": _parse_target_weight,
    "sugar_intake": _choice_validator(_SUGAR_OPTIONS, order=[5, 1, 2, 3, 4]),
    "exercise_level": _choice_validator(_EXERCISE_OPTIONS, order=[5, 3, 4, 2, 1]),
    "family_obesity": _choice_validator(_FAMILY_OPTIONS, order=[3, 1, 2]),
    "dairy_intake": _choice_validator(_DAIRY_OPTIONS, order=[5, 4, 3, 2, 1]),
    "preferred_checkin_time": _parse_checkin_time_strict,  # not asked by default any more
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
    intro_text: Optional[str] = None,
) -> None:
    """
    Create/reset the onboarding session at question 0 and send the first
    question.

    Sends ONE WhatsApp message (intro + question 1) instead of two/three,
    because Meta bills every outbound service message. `intro_text` lets
    the caller supply its own opener (e.g. the first-"hi" welcome in
    app/api/main.py::_maybe_send_greeting) so that opener is merged into
    this same message instead of being sent separately. If omitted, the
    default "you're all set" intro (used when onboarding is kicked off by
    a payment) is used.
    """
    category = category or settings.DEFAULT_PLAN_CATEGORY
    questions = _questions_for(category)

    await asyncio.to_thread(memory.start_onboarding_session, phone_number, category)

    if intro_text is None:
        intro_text = (
            f"🎉 You're all set on the {category.replace('_', ' ')} plan!\n\n"
            f"Just {len(questions)} quick questions so I can personalize your "
            f"{settings.PREMIUM_PLAN_DAYS}-day plan, then I'll get it ready for you."
        )

    first_key, first_prompt = questions[0]
    first_message = f"{intro_text}\n\n{first_prompt}"
    await send_text_message(phone_number, first_message)
    await asyncio.to_thread(
        memory.save_message, phone_number, "assistant", first_message, message_type="text"
    )


async def _finish_onboarding(
    memory: ConversationMemory, phone_number: str
) -> None:
    """
    Last onboarding answer just came in. Mark the session complete and
    save the preferred check-in hour — both independent of payment
    status, since the collected profile is also used to personalize the
    free Q&A a non-premium user gets (see settings.FREE_QUESTION_LIMIT
    in app/main.py).

    Onboarding is now FREE and starts on the user's first "hi", before
    any payment (see app/main.py's _maybe_send_greeting). The full PAID
    21-day plan is only ever generated once the user is ALSO a paying
    subscriber:
      - If they're somehow ALREADY premium right this moment (paid
        before/during onboarding — an edge case) -> generate and send
        the plan immediately, same as the old combined behavior.
      - Otherwise (the normal case) -> tell them they're set and invite
        them to ask questions now; the actual plan gets generated later,
        from the /razorpay/webhook handler in app/main.py, the moment
        their payment lands.
    """
    session = await asyncio.to_thread(memory.mark_onboarding_complete, phone_number)

    raw_time_answer = (session.get("answers") or {}).get("preferred_checkin_time", "")
    preferred_hour_utc = _parse_preferred_hour_to_utc(raw_time_answer) if raw_time_answer else None
    if preferred_hour_utc is None:
        preferred_hour_utc = settings.DAILY_CHECKIN_HOUR_UTC
        logger.info(
            f"Couldn't parse preferred check-in time '{raw_time_answer}' for "
            f"{phone_number} - falling back to default hour {preferred_hour_utc}:00 UTC."
        )
    await asyncio.to_thread(memory.set_preferred_checkin_hour, phone_number, preferred_hour_utc)

    if await asyncio.to_thread(memory.is_premium_active, phone_number):
        # Rare ordering (paid before finishing onboarding) — behave
        # exactly like the old combined flow and generate the plan now.
        await generate_and_send_plan(memory, phone_number, session=session)
        return

    if settings.FREE_QUESTION_LIMIT <= 0:
        # No free questions: the payment offer goes out right now, as the
        # ONE message that closes onboarding (instead of the "ask me
        # anything" outro). Imported lazily to avoid a circular import.
        from app.api.main import _send_free_limit_message
        await _send_free_limit_message(phone_number, is_first_hit=True)
        return

    outro = (
        "Perfect, thank you! 🙏 I've got everything I need to personalize "
        "my answers for you.\n\n"
        "Go ahead and ask me anything about your health, diet, or fitness "
        "goals 💬"
    )
    await send_text_message(phone_number, outro)
    await asyncio.to_thread(
        memory.save_message, phone_number, "assistant", outro, message_type="text"
    )


def build_derived_profile(answers: dict) -> dict:
    """BMI / healthy range / weight-to-lose computed from the saved
    (normalized) onboarding answers. Best-effort: missing or unparseable
    inputs are simply left out."""
    out: dict = {}
    try:
        hw, _ = _parse_weight_height(answers.get("height_weight", "") or "")
        if hw:
            kg, cm = hw["weight_kg"], hw["height_cm"]
            m = cm / 100.0
            bmi = kg / (m * m)
            out["current_bmi"] = round(bmi, 1)
            out["bmi_category_who"] = (
                "underweight" if bmi < 18.5 else
                "normal" if bmi < 25 else
                "overweight" if bmi < 30 else "obese"
            )
            lo, hi = round(18.5 * m * m, 1), round(24.9 * m * m, 1)
            out["healthy_weight_range_kg"] = f"{lo}-{hi}"
            tw = answers.get("target_weight", "") or ""
            tgt, _ = _parse_target_weight(tw)
            if tgt and tgt.get("target_kg"):
                out["target_weight_kg"] = tgt["target_kg"]
                out["kg_to_lose"] = round(kg - tgt["target_kg"], 1)
            else:
                out["target_weight_note"] = (
                    "User is NOT sure of a target — pick a realistic, healthy target "
                    "inside healthy_weight_range_kg (a first milestone of 5-10% of "
                    "current weight is fine) and frame the plan around it."
                )
    except Exception as e:
        logger.warning(f"⚠️ Could not build derived profile: {e}")
    return out


_PLAN_LOCK_TTL_SECONDS = 600
_PLAN_GEN_ATTEMPTS = 3
_PLAN_GEN_RETRY_DELAYS = (5, 20)  # seconds between attempts


async def _acquire_plan_lock(phone_number: str) -> bool:
    """Cross-process lock so the payment webhook and the repair job never
    generate the same user's plan twice. Fails OPEN if Redis is down."""
    try:
        from app.core.redis_client import get_redis
        return bool(await get_redis().set(
            f"plan_gen_lock:{phone_number}", "1", nx=True, ex=_PLAN_LOCK_TTL_SECONDS
        ))
    except Exception as e:
        logger.warning(f"⚠️ Plan lock unavailable ({e}) — continuing without it.")
        return True


async def _release_plan_lock(phone_number: str) -> None:
    try:
        from app.core.redis_client import get_redis
        await get_redis().delete(f"plan_gen_lock:{phone_number}")
    except Exception:
        pass


async def generate_and_send_plan(
    memory: ConversationMemory, phone_number: str, session: Optional[dict] = None
) -> bool:
    """
    Generate the full {PREMIUM_PLAN_DAYS}-day plan from this user's
    already-saved onboarding answers (one LLM call), save it, and send
    Day 1 immediately. Requires onboarding to already be complete.

    Call from:
      1. _finish_onboarding() — user is ALREADY premium when onboarding ends.
      2. The payment webhook (app/api/main.py) — runs as a BACKGROUND task
         there, so the slow Gemini call never blocks the webhook response.
      3. run_daily_checkins' repair step (app/services/daily_checkin.py) —
         safety net for paid users whose plan never got generated.

    Robustness (paid users must never be left without Day 1):
      - cross-process lock: no double generation / double Day 1
      - skips generation if a plan for the current subscription exists
      - retries Gemini failures (3 attempts); if all fail, the repair job
        retries every few minutes until it works
    Returns True if the plan exists afterwards (Day 1 sent or held).
    """
    if not await _acquire_plan_lock(phone_number):
        logger.info(f"⏭️ Plan generation for {phone_number} already in progress elsewhere — skipping.")
        return False

    try:
        if await asyncio.to_thread(memory.has_current_plan, phone_number):
            logger.info(f"⏭️ {phone_number} already has a plan for the current subscription — skipping.")
            return True

        if session is None:
            session = await asyncio.to_thread(memory.get_onboarding_session, phone_number)
        answers = (session or {}).get("answers", {}) or {}
        category = (session or {}).get("category") or settings.DEFAULT_PLAN_CATEGORY

        context_text = await asyncio.to_thread(memory.get_conversation_context, phone_number, limit=5)
        required_language = None
        if context_text:
            try:
                required_language = await detect_reply_language(context_text[-500:])
            except Exception as e:
                logger.warning(f"⚠️ Language detection failed for {phone_number}: {e}")

        # Computed facts (BMI, kg to lose, healthy range...) so the plan is
        # driven by the user's real numbers, not just their raw text.
        answers = {**answers, "derived_profile": build_derived_profile(answers)}

        # WhatsApp profile name (saved from the webhook); None if unknown.
        user_name = None
        try:
            user_name = (await asyncio.to_thread(memory.get_customer, phone_number)).get("name")
        except Exception as e:
            logger.warning(f"⚠️ Could not load user name for {phone_number}: {e}")

        days = None
        for attempt in range(1, _PLAN_GEN_ATTEMPTS + 1):
            try:
                days = await generate_premium_plan(
                    onboarding_answers=answers,
                    category=category,
                    total_days=settings.PREMIUM_PLAN_DAYS,
                    required_language=required_language,
                    user_name=user_name,
                )
                break
            except (GeminiUnavailableError, ValueError) as e:
                logger.warning(
                    f"⚠️ Plan generation attempt {attempt}/{_PLAN_GEN_ATTEMPTS} failed "
                    f"for {phone_number}: {e}"
                )
                if attempt < _PLAN_GEN_ATTEMPTS:
                    await asyncio.sleep(_PLAN_GEN_RETRY_DELAYS[attempt - 1])

        if days is None:
            logger.error(f"❌ Plan generation failed for {phone_number} — repair job will retry.")
            return False

        await asyncio.to_thread(memory.save_premium_plan, phone_number, category, days)

        confirm = (
            f"✅ Your personalized {settings.PREMIUM_PLAN_DAYS}-day plan is ready!\n"
            f"Here's Day 1 👇 — then one task per day around your chosen time. "
            f"Tap a button after each task so I know how it went 💪\n\n"
        )
        # Plan-ready note + Day 1 + confirm buttons go out together (1-2
        # messages total instead of 4).
        await _send_plan_day_now(memory, phone_number, day_number=1, prefix=confirm)
        return True
    finally:
        await _release_plan_lock(phone_number)


async def _send_plan_day_now(
    memory: ConversationMemory, phone_number: str, day_number: int, prefix: str = ""
) -> None:
    """
    Fetch a specific pregenerated plan day and send it right now (with the
    "Day X of N" header and the Done / Not-done confirmation buttons — see
    app/services/plan_delivery.py), exactly like the scheduled job in
    app/services/daily_checkin.py does; this is just that same send-path
    triggered immediately instead of on the next scheduler tick.
    """
    plan_day = await asyncio.to_thread(memory.get_premium_plan_day, phone_number, day_number)
    if not plan_day:
        logger.error(
            f"❌ Tried to immediately send day {day_number} for {phone_number} but "
            "no such pregenerated row exists — skipping."
        )
        return

    # WhatsApp only delivers free-form messages within 24h of the user's
    # last inbound message. If they paid later than that, free-form Day 1
    # would show "sent" but never arrive — so HOLD it instead: it is
    # pushed the moment the user sends any message (main.py ->
    # _flush_pending_checkin_day), or at their check-in hour if open by then.
    last_inbound_at = await asyncio.to_thread(memory.get_last_inbound_message_at, phone_number)
    window_open = False
    if last_inbound_at is not None:
        if last_inbound_at.tzinfo is not None:
            last_inbound_at = last_inbound_at.astimezone(timezone.utc).replace(tzinfo=None)
        window_open = (datetime.utcnow() - last_inbound_at) < timedelta(
            hours=settings.WHATSAPP_SESSION_WINDOW_HOURS
        )
    if not window_open:
        await asyncio.to_thread(memory.mark_plan_day_template_nudge_sent, phone_number, day_number)
        if settings.DAILY_CHECKIN_REENGAGEMENT_TEMPLATE:
            try:
                await send_template_message(
                    phone_number, settings.DAILY_CHECKIN_REENGAGEMENT_TEMPLATE,
                    params=[str(day_number)],
                )
            except Exception as e:
                logger.error(f"❌ Re-engagement template failed for {phone_number}: {e}", exc_info=True)
        logger.info(
            f"⏸️ Day {day_number} for {phone_number} held — 24h window closed; "
            "will send as soon as they message."
        )
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

    await send_plan_day(memory, phone_number, plan_day, prefix=prefix)


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
        await _finish_onboarding(memory, phone_number)
        return True

    # Production safety: a session that was started under the OLD question
    # set (answers saved under keys that no longer exist) would otherwise
    # mix old and new answers. Restart it cleanly with the new questions.
    saved_answers = session.get("answers") or {}
    if isinstance(saved_answers, str):
        import json as _json_mod
        try:
            saved_answers = _json_mod.loads(saved_answers)
        except Exception:
            saved_answers = {}
    valid_keys = {key for key, _ in questions}
    if set(saved_answers) - valid_keys:
        logger.info(f"🔄 Restarting legacy onboarding session for {phone_number} (question set changed).")
        await start_onboarding(
            memory, phone_number, category=category,
            intro_text="We've upgraded our questions to personalize your plan even better 🙂 "
                       "Sorry for the repeat — it only takes a minute!",
        )
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
        # _finish_onboarding.
        answer_to_store = (parsed_value or {}).get("normalized") or user_text
        new_index = await asyncio.to_thread(
            memory.save_onboarding_answer, phone_number, current_key, answer_to_store
        )
        if new_index >= len(questions):
            await _finish_onboarding(memory, phone_number)
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
        await _finish_onboarding(memory, phone_number)
        return True

    next_key, next_prompt = questions[new_index]
    await send_text_message(phone_number, next_prompt)
    await asyncio.to_thread(
        memory.save_message, phone_number, "assistant", next_prompt, message_type="text"
    )
    return True