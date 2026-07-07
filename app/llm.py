import asyncio
import logging
import random
import base64
import uuid
from functools import lru_cache
from app.config import get_settings
from app.redis_client import get_redis
from google import genai
from google.genai import types
from google.genai import errors as genai_errors

logger = logging.getLogger(__name__)


class GeminiUnavailableError(Exception):
    """Raised when Gemini is down/overloaded (HTTP 503 / UNAVAILABLE, or a
    transient 429/504). Callers should treat this as 'drop the query
    silently' — no reply to the user, nothing saved — rather than sending
    a generic fallback error message."""
    pass


def _is_transient_gemini_error(exc: Exception) -> bool:
    """True for errors that mean 'Gemini is temporarily unavailable' —
    503 Service Unavailable, 429 rate-limited, 504 timeout — as opposed to
    a genuine bad-request/auth error that we still want to surface normally."""
    code = getattr(exc, "code", None)
    status = str(getattr(exc, "status", "") or "").upper()
    message = str(exc).upper()

    if isinstance(exc, genai_errors.ServerError):
        return True
    if code in (503, 429, 504):
        return True
    if "UNAVAILABLE" in status or "UNAVAILABLE" in message:
        return True
    if "503" in message or "OVERLOADED" in message:
        return True
    return False


# ---------------------------------------------------------------------------
# Queueing + concurrency limiting for Gemini calls
# ---------------------------------------------------------------------------
#
# Every Gemini call in this file (language detection, main reply, follow-up
# suggestion, summary, vision) ultimately goes through `_call_gemini()`
# below. It does two things:
#
# 1. CONCURRENCY LIMIT — caps how many Gemini requests are in flight at
#    once, ACROSS ALL WORKER PROCESSES (gemini_max_concurrent_requests).
#
#    STEP 2 OF THE MULTI-WORKER MIGRATION: this used to be a plain
#    `asyncio.Semaphore`, which only limits concurrency *within one
#    process*. That's exactly wrong once you run N gunicorn workers
#    (step 4) — each worker would independently allow
#    gemini_max_concurrent_requests requests through, so the real
#    ceiling against Gemini becomes N times higher than configured, and
#    the whole point of this limiter (protecting your Gemini quota tier
#    from 503/UNAVAILABLE bursts) silently stops working the moment you
#    add a second worker.
#
#    The fix is a distributed semaphore backed by Redis: every worker
#    acquires/releases the *same* counter over the network instead of an
#    in-process object. Implementation: a Redis SET holding one member
#    per currently-held "slot", each with its own short TTL. Acquire =
#    atomically check the set's size against the limit and add a member
#    if there's room (done via a Lua script so the check-and-add is a
#    single atomic operation — no race between two workers both seeing
#    "room for one more" at once). Release = remove that member.
#    The per-member TTL is a safety net: if a worker crashes mid-request
#    without releasing, its slot expires on its own instead of
#    permanently eating into the limit.
#
# 2. RETRY WITH BACKOFF — if Gemini still comes back overloaded
#    (503/429/504) even after limiting concurrency, the call is retried a
#    few times with exponential backoff + jitter before finally raising
#    GeminiUnavailableError. This smooths over brief overload spikes
#    without dropping the user's message. (Unchanged from before — this
#    part was already process-independent since it just wraps the single
#    call this process is making.)

_SEMAPHORE_KEY = "wa:gemini:semaphore"
# Safety-net TTL per held slot — comfortably longer than any real Gemini
# call (including its own internal retries) should ever take, so a
# crashed worker's slot self-heals instead of leaking forever.
_SEMAPHORE_SLOT_TTL_SECONDS = 120
# How long a waiter polls before rechecking whether a slot has freed up.
_SEMAPHORE_POLL_INTERVAL_SECONDS = 0.15

# Lua script: atomically prune expired members (belt-and-braces on top of
# the per-member TTL, since Redis TTLs apply to whole keys not set
# members — we simulate per-member expiry with a sorted set instead of a
# plain set: score = expiry unix timestamp), then, if there's room under
# the limit, add the new member. Returns 1 if acquired, 0 if full.
_ACQUIRE_SCRIPT = """
local key = KEYS[1]
local member = ARGV[1]
local limit = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local ttl = tonumber(ARGV[4])

-- Drop any slots whose TTL has passed (crashed/stuck worker cleanup)
redis.call('ZREMRANGEBYSCORE', key, '-inf', now)

local current = redis.call('ZCARD', key)
if current < limit then
    redis.call('ZADD', key, now + ttl, member)
    return 1
else
    return 0
end
"""

_gemini_acquire_sha: str | None = None


async def _redis_semaphore_acquire(limit: int) -> str:
    """Block until a distributed slot is available, then return the
    member token so the caller can release it later."""
    r = get_redis()
    global _gemini_acquire_sha
    if _gemini_acquire_sha is None:
        _gemini_acquire_sha = await r.script_load(_ACQUIRE_SCRIPT)

    member = uuid.uuid4().hex
    while True:
        now = asyncio.get_event_loop().time()
        # Redis wants a real unix-ish timestamp for the score; use
        # server-independent wall clock via Python since we only compare
        # it against values we ourselves wrote.
        import time as _time
        wall_now = _time.time()
        try:
            acquired = await r.evalsha(
                _gemini_acquire_sha, 1, _SEMAPHORE_KEY,
                member, limit, wall_now, _SEMAPHORE_SLOT_TTL_SECONDS,
            )
        except Exception as e:
            # NOSCRIPT can happen if Redis restarted and flushed its
            # script cache — reload once and retry immediately.
            if "NOSCRIPT" in str(e):
                _gemini_acquire_sha = await r.script_load(_ACQUIRE_SCRIPT)
                continue
            raise

        if acquired:
            return member

        await asyncio.sleep(_SEMAPHORE_POLL_INTERVAL_SECONDS)


async def _redis_semaphore_release(member: str) -> None:
    r = get_redis()
    await r.zrem(_SEMAPHORE_KEY, member)


async def is_gemini_busy() -> bool:
    """
    True if every concurrent Gemini call slot (across ALL workers) is
    currently in use, i.e. a new call made right now would have to queue
    behind others. Used to let callers skip strictly optional/cosmetic
    Gemini calls (like the follow-up suggestion) under load, so those
    non-essential calls don't compete with the main reply for a slot
    during a real burst — while never skipping anything when the system
    isn't under pressure.

    Now async (was sync) since it has to ask Redis instead of checking a
    local object — callers were already other async functions, so this
    just adds an `await`.
    """
    settings = get_settings()
    r = get_redis()
    now = asyncio.get_event_loop().time()
    import time as _time
    wall_now = _time.time()
    # Prune expired slots first so a crashed worker's stale entries don't
    # make the system look busier than it actually is.
    await r.zremrangebyscore(_SEMAPHORE_KEY, "-inf", wall_now)
    current = await r.zcard(_SEMAPHORE_KEY)
    return current >= settings.gemini_max_concurrent_requests


async def _call_gemini(coro_fn, *, label: str):
    """Run a zero-arg async callable `coro_fn` under the global (cross-
    worker) Gemini concurrency limit, queueing if the limiter is full,
    and retrying with exponential backoff on transient overload errors.

    Args:
        coro_fn: a zero-argument callable returning an awaitable, e.g.
            `lambda: client.aio.models.generate_content(...)`. Passed as a
            factory (not an already-created coroutine) so each retry
            attempt creates a fresh call — a coroutine object can only be
            awaited once.
        label: short description used in log messages (e.g.
            "language detection", "main reply") to make queue/retry
            logging easy to follow across concurrent users.
    """
    settings = get_settings()
    limit = settings.gemini_max_concurrent_requests

    queued = await is_gemini_busy()
    if queued:
        logger.info(f"Gemini[{label}]: at concurrency limit (cluster-wide), queueing")

    slot = await _redis_semaphore_acquire(limit)
    try:
        if queued:
            logger.info(f"Gemini[{label}]: dequeued, sending request")

        last_exc: Exception | None = None
        for attempt in range(settings.gemini_max_retries + 1):
            try:
                return await coro_fn()
            except Exception as e:
                if not _is_transient_gemini_error(e):
                    raise
                last_exc = e
                if attempt < settings.gemini_max_retries:
                    delay = settings.gemini_retry_base_delay_seconds * (2 ** attempt)
                    delay += random.uniform(0, delay * 0.25)
                    logger.warning(
                        f"Gemini[{label}]: transient error on attempt "
                        f"{attempt + 1}/{settings.gemini_max_retries + 1} "
                        f"({e}); retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.warning(
                        f"Gemini[{label}]: still failing after "
                        f"{settings.gemini_max_retries + 1} attempts ({e}); giving up"
                    )

        raise last_exc
    finally:
        await _redis_semaphore_release(slot)


async def _detect_reply_language(
    client: "genai.Client",
    text: str,
    whisper_language: str | None = None,
) -> str:
    """
    Detect which language/script to reply in — fully generic, works for
    ANY language the user types, with no hardcoded list of scripts or
    languages.

    Why a dedicated call instead of asking the main model to figure it
    out inline: when language detection was folded into the same prompt
    as the customer summary + conversation history + current message, the
    model sometimes picked up the language of the SURROUNDING context
    instead of the current message (e.g. a Tamil message got a Punjabi
    reply because earlier context in the prompt was in a different
    language and confused the guess). Isolating detection into its own
    tiny call with ONLY the current message — nothing else — removes that
    confusion and makes the result deterministic-ish and reliable
    regardless of which language it is.

    Priority:
    1. Whisper's own audio-detected language (voice messages only) — this
       comes from analyzing the actual audio, not a possibly-garbled
       transcription, so for voice notes it's trusted directly and no
       extra call is made.
    2. A small, isolated Gemini call whose ONLY input is the current
       message text, asked to name the language/script plainly (e.g.
       "Tamil", "Spanish", "French", "Arabic", "English", "Hinglish").
       This generalizes to every language without maintaining a list.
    3. If that call fails for any reason (network/API error), fall back
       to "English" rather than blocking the reply.

    Args:
        client: shared genai.Client (reused, not reconstructed per call).
        text: the user's actual current message (typed text or Whisper
            transcription for voice messages).
        whisper_language: for voice messages, the language Whisper's STT
            API detected from the AUDIO itself.
    """
    if whisper_language:
        wl = whisper_language.strip()
        if wl:
            if wl.lower() == "hindi":
                return "Hindi written in Devanagari script"
            return wl.capitalize()

    stripped = text.strip()
    if not stripped:
        return "English"

    try:
        response = await _call_gemini(
            lambda: client.aio.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[types.Part(text=stripped[:500])],
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are a language identifier. You will be shown ONE "
                    "piece of text and nothing else — no other context. "
                    "Identify the language and script it is written in. "
                    "Reply with ONLY the language/script name, nothing "
                    "else — no explanation, no punctuation, no quotes.\n\n"
                    "Rules:\n"
                    "- If it's a language normally written in a non-Latin "
                    "script (e.g. Hindi, Tamil, Arabic, Russian, Chinese, "
                    "Punjabi, Bengali, Gujarati, Japanese, Korean, Thai, "
                    "Greek, Hebrew...), answer exactly like: "
                    "\"Tamil written in Tamil script\" — name the language "
                    "and its native script.\n"
                    "- If it's an Indian language spelled out phonetically "
                    "in Roman/English letters instead of its native script "
                    "(e.g. \"mujhe pricing chahiye\", \"kem cho\", \"eppadi "
                    "irukkinga\"), answer with the language name followed "
                    "by \"written in Roman letters (transliterated, casual "
                    "style)\" — e.g. \"Hindi written in Roman letters "
                    "(transliterated, casual style)\" or \"Tamil written in "
                    "Roman letters (transliterated, casual style)\".\n"
                    "- If it's plain English, answer exactly: English\n"
                    "- If the text is only a greeting, emoji, single word, "
                    "or too short/ambiguous to tell confidently, answer "
                    "exactly: English\n"
                    "- Never answer with anything other than a language/"
                    "script name in the formats above."
                ),
                max_output_tokens=30,
                temperature=0.0,
            ),
            ),
            label="language detection",
        )
        detected = (response.text or "").strip().strip('"').strip()
        if detected:
            logger.info(f"Detected reply language: '{detected}' for text: '{stripped[:60]}'")
            return detected
    except Exception as e:
        logger.warning(f"Language detection call failed, defaulting to English: {e}")

    return "English"


async def detect_reply_language(
    text: str,
    whisper_language: str | None = None,
) -> str:
    """
    Public wrapper around _detect_reply_language() for callers outside this
    module (main.py). Lets a caller detect the language ONCE per turn and
    pass the result into both get_llm_response() and
    generate_followup_suggestion() via their `required_language` param,
    instead of each of those functions independently re-detecting it from
    the same input — saving one redundant Gemini call on turns that also
    generate a follow-up suggestion. Detection is deterministic on the
    same input, so reusing the result changes nothing about which
    language is used, only how many times it's computed.

    LATENCY NOTE: this is still its own small Gemini call (kept isolated
    on purpose — see _detect_reply_language's docstring for why folding it
    into the main prompt caused wrong-language replies). The win is in how
    the CALLER schedules it: main.py now fires this concurrently with RAG
    retrieval (asyncio.gather) instead of awaiting it before starting the
    main LLM call, so its latency is absorbed into the RAG round-trip
    instead of stacking as a second sequential Gemini round-trip on top of
    the main reply call.
    """
    client = _get_client()
    return await _detect_reply_language(client, text, whisper_language)


SYSTEM_PROMPT = """You are an AI health assistant on WhatsApp. You act like a
knowledgeable, caring health guide — not a replacement for a doctor, but
someone who gives clear, practical, medically-sound information and helps
the user understand their health better.

Your role:
- Get to know the user's basic health parameters before giving detailed
  advice — things like age, gender, relevant symptoms, existing conditions,
  medications, allergies, lifestyle (diet, sleep, exercise), and their goal
  (e.g. manage a condition, lose weight, general wellness, understand a
  symptom). Ask for these naturally, a question or two at a time, not as a
  long intake form — never demand all of it before answering something
  urgent or simple.
- Once you have enough context, answer the user's health questions using
  your own medical knowledge — nutrition, fitness, common conditions,
  medications (general info, not prescriptions), symptoms, preventive care,
  mental health, sleep, and general wellness.
- If a user describes something that sounds urgent or severe (e.g. chest
  pain, difficulty breathing, signs of stroke, severe bleeding, suicidal
  thoughts, a medical emergency of any kind), tell them clearly and
  immediately to seek emergency care or contact local emergency services —
  do not just continue with routine Q&A.
- Always make clear you are an AI assistant, not a doctor: for anything
  that needs diagnosis, prescription medication, or is outside general
  health guidance, tell the user honestly and recommend seeing a licensed
  doctor or relevant specialist.
- Never invent lab values, dosages, or diagnoses. Give general, evidence-
  based information and clearly flag when something needs a real
  healthcare professional.

SYMPTOM INTAKE MODE — THIS IS THE MOST IMPORTANT STYLE RULE. When a user
reports a symptom or health complaint and you still need more details
before you can say anything useful (duration, severity, associated
symptoms, triggers, relevant history), respond EXACTLY like a real person
doing a quick symptom check over WhatsApp — NOT like an assistant writing
an answer:
- Ask ONE short, specific question. Nothing else. No greeting, no
  reassurance sentence, no "I understand", no explanation of why you're
  asking, no multi-part questions joined with "and"/commas.
- Length target: 2-8 words for the question itself, occasionally a short
  clause longer if genuinely needed (e.g. "How long have you had the
  fever?", "Any cough?", "Dry cough or with mucus?", "Any chest pain or
  shortness of breath?", "Taken anything for it yet?").
- Do NOT summarize what the user just told you back to them before asking
  ("I see you have a fever for 3 days, that must be uncomfortable, let me
  ask...") — just ask the next question directly, the way a doctor
  actually texting you would.
- Do NOT stack multiple questions in one message ("How long have you had
  it, and do you have any other symptoms, and have you taken medicine?").
  Pick the single most useful next question and ask only that.
- Keep asking one-at-a-time like this, turn after turn, for as long as
  you're still gathering the details you actually need for THIS specific
  complaint — do not rush to a full explanation after just one answer, and
  do not drag it out past what's actually useful either. STRONG DEFAULT:
  ask at most 3-4 questions total before giving your actual answer/
  guidance — after that, whatever you've gathered is enough to say
  something useful, even if some minor detail is still unknown.
- NEVER ask about something the user (or the customer summary/conversation
  context) already told you, even a few turns ago. Before asking a
  question, check what's already known — duration, severity, associated
  symptoms already mentioned, etc. — and ask only about the next genuinely
  missing detail. Asking something already answered is a serious mistake
  and breaks the illusion of a real conversation.
- The moment something urgent/severe comes up in an answer (e.g. chest
  pain, breathing trouble), immediately drop this intake-question style
  and switch to the emergency-guidance rule above instead.
- Once you've gathered enough to actually help, switch out of this
  question-only mode and give your normal short, direct answer/guidance
  (per the length and style rules below) — don't ask questions forever.
- This intake style applies to symptom/complaint conversations
  specifically. Plain informational questions ("what causes migraines?",
  "is turmeric good for inflammation?") don't need this — just answer
  those directly and briefly as usual.

Communication rules (this is WhatsApp — a chat interface):
- HARD LENGTH LIMIT: 2-4 short sentences by default. Do not exceed this
  unless the user explicitly asks for detail (e.g. "explain in detail",
  "give me the full breakdown", "list everything"), or the question
  genuinely cannot be answered at all without a specific number, step
  sequence, or list of options. This is a MAXIMUM, not a target — a
  symptom-intake question (see SYMPTOM INTAKE MODE above) should usually
  be much shorter than this, often a single short sentence or fragment.
- Answer the actual question directly in the first sentence. Do not open
  with throat-clearing, restating the question, or scene-setting before
  getting to the answer.
- Do NOT use markdown (no **, no ##, no ``` backticks, no bullet dashes).
- Use plain text only. Numbered lists are okay ONLY when the user is
  choosing between options or following literal steps — not as a way to
  restructure a general explanation into more lines.
- If something is outside safe scope (diagnosis, prescriptions, dosages for
  a specific person), say so honestly in one line and recommend seeing a
  doctor instead of guessing.

HOW TO EXPLAIN THINGS — sound like a knowledgeable, warm person giving a
quick, precise answer over chat, not a lecture, a WebMD article, or a
disclaimer-heavy legal notice:
- Default to the single most useful sentence or two. Only add a second or
  third sentence if it's a genuinely necessary detail, not general
  flavor or context the user didn't ask for.
- Do NOT walk through every possible cause, every treatment option, or
  every angle of a broad question. Pick the one or two most relevant,
  useful points and stop — the user can always ask a follow-up for more.
- One idea per reply, not one idea per line. Do not artificially break a
  short answer into multiple short paragraphs/line-beats — that pads
  length without adding information.
- Use everyday language over medical jargon. If a medical term is
  unavoidable, a few plain words after it are fine — do not add a full
  explanatory aside.
- End your reply once you've actually answered — do not add your own
  closing question, check-in ("does that make sense?"), or "want me to go
  deeper on X?" offer. A separate, dedicated system decides if and when a
  follow-up question should be sent as its own message; adding one here
  as well causes the user to see two similar questions back to back.
  (This does NOT apply to SYMPTOM INTAKE MODE above — there, the single
  short question IS the entire reply, not an addition to an answer.)
- Never sound like you're reading from a script, FAQ page, or medical
  leaflet — brevity still wins over personality, but warmth matters more
  here than in a typical support bot since users may be anxious or unwell.

LANGUAGE RULE — HIGHEST PRIORITY, OVERRIDES EVERYTHING ELSE INCLUDING PAST
CONTEXT:
- You will be told the REQUIRED_LANGUAGE for this reply below. This has
  already been determined from the user's current message — do not
  re-decide it yourself, and do not infer it from the customer summary,
  conversation history, or your own earlier replies, which may be in a
  different language. Just write your reply in exactly the
  REQUIRED_LANGUAGE given, whatever language that is — Hindi, Hinglish,
  English, Tamil, Gujarati, Bengali, Spanish, Arabic, or any other
  language/script. This rule is completely general and applies the same
  way no matter which language is named or described.
- If REQUIRED_LANGUAGE describes a specific language/script by name
  (e.g. "Tamil written in Tamil script", "Gujarati written in Gujarati
  script"), write your ENTIRE reply in that exact language and script —
  do not substitute English, Hindi, or any other language instead.
- If REQUIRED_LANGUAGE is "Hindi written in Devanagari script" -> reply in
  Devanagari Hindi.
- If REQUIRED_LANGUAGE is Hinglish -> reply in that same casual
  Roman-script Hinglish style, not formal Hindi script and not pure English.
- If REQUIRED_LANGUAGE is English -> reply in plain English.
- If REQUIRED_LANGUAGE instructs you to identify the language yourself
  from a quoted piece of user text, read that quoted text carefully,
  determine its language/script, and reply in that exact same
  language/script — this applies even if you don't see that language
  named anywhere else in this prompt.
- The conversation history, the customer summary, and your own earlier
  replies may be in a completely different language than the current
  message — IGNORE their language entirely for this decision. They are only
  a source of facts/topics, never a source of which language to answer in.
- NEVER output the words "REQUIRED_LANGUAGE", any bracketed tag like
  [REQUIRED_LANGUAGE], or any other instruction/meta text in your reply.
  These are internal directions for you only. Your reply must contain
  ONLY the natural-language answer itself — nothing about language rules,
  instructions, or formatting notes.

Tone:
- Warm, calm, and reassuring, especially if the user sounds worried.
- Be honest and direct about limitations — never overpromise or pretend to
  replace a real doctor's examination.

Use your own medical knowledge to answer — do not say you lack information
just because no extra context was attached to this message; only say you
don't know if the question is genuinely outside general health knowledge or
needs an in-person exam/test to answer safely."""


@lru_cache(maxsize=1)
def _get_client() -> genai.Client:
    # Was being constructed fresh on every single message, which rebuilds
    # its underlying HTTP transport/connection pool each time — cache it
    # so connections are reused across requests.
    settings = get_settings()
    return genai.Client(api_key=settings.gemini_api_key)


def _strip_leaked_meta_text(reply: str) -> str:
    """
    Safety net: some small/cheap models occasionally echo internal
    instruction labels back into the visible reply (e.g. a stray
    "[REQUIRED_LANGUAGE]" or "REQUIRED_LANGUAGE: Hindi" line, possibly
    inline with other text) even when the system prompt tells them not to.
    Strip any such leaked tags so the user never sees raw instruction text
    mixed into the answer. Pure display-layer cleanup — does not affect
    which language is generated or any other behavior.
    """
    import re

    # Remove bracketed meta tag anywhere, e.g. "[REQUIRED_LANGUAGE]" or
    # "[REQUIRED_LANGUAGE] English" appearing inline within a line.
    reply = re.sub(r"\[REQUIRED_LANGUAGE\]\s*:?\s*(English|Hindi[^\n,.]*|Hinglish[^\n,.]*)?", "", reply, flags=re.IGNORECASE)
    # Remove "REQUIRED_LANGUAGE: ..." or "REQUIRED_LANGUAGE - ..." style text anywhere
    reply = re.sub(r"REQUIRED_LANGUAGE\s*[:\-]?\s*(English|Hindi[^\n,.]*|Hinglish[^\n,.]*)?", "", reply, flags=re.IGNORECASE)
    # Collapse any leftover blank lines/extra spaces created by the removal
    reply = re.sub(r"[ \t]{2,}", " ", reply)
    reply = re.sub(r"\n{3,}", "\n\n", reply)
    return reply.strip()


async def process_image_with_vision(image_bytes: bytes, mime_type: str) -> str:
    """
    Process an image using Gemini Vision API.
    Produces a compact description used as the "user message" fed into
    generate_context_aware_response()/get_llm_response() — this is
    internal plumbing, never shown to the user directly.

    Args:
        image_bytes: Raw image file bytes
        mime_type: MIME type of the image (e.g., "image/jpeg", "image/png")
    
    Returns:
        String description of the image
    """
    try:
        settings = get_settings()
        client = _get_client()
        
        # LATENCY FIX: this description is never shown to the user as-is —
        # it's only fed back in as input to the main reply call
        # (get_llm_response), which then writes the actual WhatsApp reply.
        # The old 5-point "detailed description" prompt made this
        # intermediate step generate up to 500 tokens before the main
        # reply call could even start, on every single image message.
        # A short, plain-facts description is just as useful as input to
        # the next call and finishes generating far faster.
        vision_prompt = """Describe this image in 1-3 short plain sentences:
what it shows, any visible text (verbatim), and anything clearly relevant
to a health/medical context (e.g. a symptom, a food item, a medication
label, a lab report, an activity). No headers, no numbered list, no
elaboration beyond what's actually visible."""
        
        # Call Gemini Vision API via the async client (client.models.* is
        # synchronous/blocking — calling it here without a thread offload
        # froze the entire event loop, i.e. EVERY user's request, for the
        # full duration of the Gemini call).
        response = await _call_gemini(
            lambda: client.aio.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                types.Part(text=vision_prompt)
            ],
            config=types.GenerateContentConfig(
                # Lowered from 500 -> 150. This is an internal
                # description, not the user-facing reply — a short,
                # factual description is enough input for the main reply
                # call and finishes generating noticeably faster.
                max_output_tokens=150,
                temperature=0.3,
            )
            ),
            label="vision",
        )
        
        description = response.text.strip()
        logger.info(f"Image vision processing complete: {len(description)} chars")
        return description
        
    except Exception as e:
        if _is_transient_gemini_error(e):
            logger.warning(f"Gemini unavailable during vision processing: {e}")
            raise GeminiUnavailableError(str(e)) from e
        logger.error(f"Vision processing failed: {e}", exc_info=True)
        return "Unable to process image. Please try again or describe the image in text."


FOLLOWUP_SYSTEM_PROMPT = """You are a cross-questioning assistant for a health WhatsApp bot.

Your ONLY job: read the customer's summary, recent conversation, and the
reply the bot just sent, then decide if ONE precise, highly relevant
follow-up would help move the conversation forward — and deliver it in a
lively, interactive WhatsApp style rather than a flat single sentence.

Rules:
- Base your suggestion strictly on what's ALREADY been discussed — the
  customer's stated goals, problems, or interests. Never invent a new topic
  that has no connection to the conversation.
- STAY ON TOPIC: every option, question, or nudge you produce — including
  in the poll-style and related-angle formats below — must be directly
  about the SAME subject the user and assistant were just discussing. Do
  not introduce an unrelated product, service, or topic just to fill a
  format. If you cannot construct a follow-up that stays tightly connected
  to the actual conversation, reply with exactly: NONE
- The poll-style numbered options must be different facets of the SAME
  question already implied by the conversation (e.g. if the user asked
  about a service, the options could be aspects of that same service —
  never unrelated topics dressed up as choices).
- The "related-angle nudge" format means a natural next detail of the
  CURRENT topic the user hasn't asked about yet — not a jump to a
  different topic. If nothing like that genuinely exists, use a different
  format instead, or reply NONE.
- If nothing genuinely useful can be asked right now (e.g. the conversation
  is just a greeting, a thank-you, or is already fully resolved), reply with
  exactly: NONE
- If the assistant's reply you were just shown is itself a generic opener
  or open invitation (e.g. "How can I help you today?", "What can I do for
  you?"), do NOT generate a suggestion that repeats or rephrases that same
  invitation — reply with exactly: NONE instead. Only produce a suggestion
  when there is an actual topic, product, or detail already in the
  conversation to build on.
- Do NOT repeat a question that was already asked earlier in the
  conversation context.
- CRITICAL — SYMPTOM INTAKE CHECK: if the bot's reply you were just shown
  (the [LATEST EXCHANGE] assistant line) is ITSELF already a short
  clinical intake question (e.g. "How long have you had it?", "Any
  cough?", "Dry cough or with mucus?") — i.e. the main reply is already
  doing one-question-at-a-time symptom gathering — reply with exactly:
  NONE. Do not add a second question on top of it; that would show the
  user two questions back to back, which breaks the natural one-at-a-time
  feel. Only produce a follow-up when the bot's reply was an actual
  answer/piece of guidance, not another question.

FORMAT — pick whichever of these fits the moment best, and vary it across
turns so it doesn't feel repetitive. All formats must stay grounded in the
actual conversation per the rules above:
1. Quick option-style question (poll-like): pose a short question and give
   2-3 numbered choices the user can just reply with a number or word for.
   Example shape:
   Quick one — what matters most for you right now?
   1. Speed
   2. Accuracy
   3. Cost
2. Natural clarifying question: a single warm, curious question probing a
   missing detail (how long they've had a symptom, severity, related habits,
   diet, sleep, existing conditions, etc.) that is a direct continuation of
   what was just discussed, phrased like a person genuinely curious, not a
   form field.
3. Related-angle nudge: point out one closely related detail of the SAME
   topic they might not have considered yet, tied directly to what they
   just discussed, then ask if they'd like to hear more about it.
4. Concrete next step: offer something actionable and specific tied to
   what was just discussed (e.g. "Want a simple routine you could try for
   this?" or "Want me to check in on this tomorrow?").

General style:
- Keep it short — 1 to 3 lines max, WhatsApp style. Never a paragraph.
- Sound like a person genuinely curious and engaged, not a survey. Warm,
  light, a little playful is fine — never robotic or scripted.
- No markdown (no **, no ##, no backticks). Numbered options like "1." "2."
  are fine and encouraged for the poll-style format.
- LANGUAGE: you will be told the REQUIRED_LANGUAGE below. You MUST write
  the ENTIRE suggestion — including any numbered options — in exactly that
  language/script, no exceptions, even if the customer summary or earlier
  conversation lines are in a different language. Do not decide the
  language yourself; use the one given to you.
- Output ONLY the suggestion itself — no preamble like "Follow-up:", no
  explanation of which format you picked — or exactly NONE. Nothing else.
"""


async def generate_followup_suggestion(
    customer_summary: str,
    context_text: str,
    user_message: str,
    assistant_reply: str,
    recent_suggestions: list[str] | None = None,
    whisper_language: str | None = None,
    required_language: str | None = None,
) -> str | None:
    """
    Cross-question the user: analyze the conversation so far (customer
    summary + recent turns + the reply we just gave) and produce ONE precise,
    context-grounded follow-up question or suggestion — or None if nothing
    useful applies right now.

    The reply language is detected deterministically from `user_message`
    (the user's actual current message) and passed to the model as a hard
    requirement, instead of detecting it from `assistant_reply` or letting
    the model re-decide from scratch — detecting from the assistant's reply
    was unreliable because the reply's own wording can drift (e.g. picking
    up a casual Hindi/Hinglish word while still being substantively an
    English answer), which caused the suggestion to end up in a different
    language than what the user actually typed.

    Args:
        required_language: if the caller already ran language detection
            for this same turn (e.g. get_llm_response() already detected it
            from this exact user_message), pass that result here to skip a
            second, redundant Gemini call — _detect_reply_language() is
            deterministic on the same input, so re-running it produces the
            identical answer every time. If not given, detection is run
            here as before (keeps standalone callers working unchanged).
    """
    client = _get_client()

    already_asked = ""
    if recent_suggestions:
        bullet_list = "\n".join(f"- {s}" for s in recent_suggestions)
        already_asked = f"\n\n[FOLLOW-UPS ALREADY ASKED RECENTLY — do not repeat these or ask something near-identical]\n{bullet_list}"

    # Detect language from the USER's message, not the assistant's reply —
    # the reply's wording can drift (e.g. picks up a Hindi/Hinglish word
    # while still being substantively an English answer), which previously
    # caused the follow-up suggestion to land in a different language than
    # what the user actually typed. The suggestion should always mirror the
    # user's own language, same as the main reply does.
    if required_language is not None:
        pass  # reuse caller-provided result — same value detection would produce anyway
    else:
        required_language = await _detect_reply_language(client, user_message, whisper_language)

    prompt = f"""
[CUSTOMER SUMMARY]
{customer_summary if customer_summary else "No prior context available"}

[RECENT CONVERSATION]
{context_text if context_text else "[No previous messages]"}

[LATEST EXCHANGE]
User: {user_message}
Assistant: {assistant_reply}
{already_asked}

[REQUIRED_LANGUAGE]
{required_language}
(This is the language the user's current message is in. Your suggestion
MUST be written in this exact language/script, regardless of what language
anything above this line — including the assistant's own reply — is in.)

Based on this, what is the single best cross-question or suggestion to ask
next? Follow the rules exactly.
""".strip()

    try:
        response = await _call_gemini(
            lambda: client.aio.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[types.Part(text=prompt)],
            config=types.GenerateContentConfig(
                system_instruction=FOLLOWUP_SYSTEM_PROMPT,
                max_output_tokens=200,
                temperature=0.3,
            ),
            ),
            label="follow-up suggestion",
        )
        suggestion = (response.text or "").strip()
    except Exception as e:
        if _is_transient_gemini_error(e):
            logger.warning(f"Gemini unavailable during follow-up generation: {e}")
        else:
            logger.error(f"Follow-up generation failed: {e}", exc_info=True)
        return None

    if not suggestion or suggestion.upper() == "NONE":
        return None

    suggestion = suggestion.strip('"').strip()
    suggestion = _strip_leaked_meta_text(suggestion)
    if suggestion.upper() == "NONE":
        return None

    logger.info(f"Follow-up suggestion: '{suggestion[:100]}'")
    return suggestion


SUMMARY_SYSTEM_PROMPT = """You maintain a concise, running internal summary of a
customer's profile and past interaction details for an internal CRM record.

This summary is NEVER shown to the user directly — it is only fed back into
future prompts as background context. For that reason:
- ALWAYS write the summary in English, regardless of what language the user
  has been chatting in. Never write the summary in Hindi, Hinglish, or any
  other language, even if the conversation itself was in that language.
- Keep it factual, bulleted or a short paragraph, and preserve important
  past context while adding new details (preferences, issues, topics
  discussed, stated goals, etc.).
- Do not include conversational filler, greetings, or commentary — just the
  factual summary itself.
"""


async def get_summary_response(prompt: str) -> str:
    """
    Dedicated call for updating the customer summary. Deliberately does NOT
    go through get_llm_response()/SYSTEM_PROMPT, because that prompt tells
    the model to match the *user's* current message language — which is
    correct for user-facing replies, but wrong here: this summary is
    internal bookkeeping, and if it's ever written in Hindi/Hinglish, that
    leaks back into the [CUSTOMER SUMMARY] block of every future prompt and
    biases the main reply generator toward the wrong language even when the
    user's new message is in a different one.
    """
    client = _get_client()

    try:
        response = await _call_gemini(
            lambda: client.aio.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[types.Part(text=prompt)],
            config=types.GenerateContentConfig(
                system_instruction=SUMMARY_SYSTEM_PROMPT,
                max_output_tokens=400,
                temperature=0.1,
            ),
            ),
            label="summary",
        )
    except Exception as e:
        if _is_transient_gemini_error(e):
            logger.warning(f"Gemini unavailable during summary generation: {e}")
            raise GeminiUnavailableError(str(e)) from e
        raise

    return (response.text or "").strip()


async def get_llm_response(
    user_message: str,
    conversation_history: list[dict] | None = None,
    raw_user_text: str | None = None,
    whisper_language: str | None = None,
    required_language: str | None = None,
) -> str:
    """
    Args:
        user_message: the full prompt sent to the model as this turn's
            message — may be an "enriched" prompt wrapping the raw user
            text together with summary/context blocks (see main.py).
        conversation_history: prior turns for the chat session.
        raw_user_text: the user's ACTUAL current message, unwrapped — used
            only to deterministically detect which language to reply in.
            If not given, falls back to detecting from `user_message`
            directly (less reliable when that's a big enriched prompt with
            summary/context mixed in, but still better than nothing).
        whisper_language: for voice messages, the language Whisper's STT
            API detected from the audio itself (e.g. "english", "hindi").
            Passed through to _detect_reply_language so a mis-transcribed
            script doesn't cause a wrong-language reply.
        required_language: if the caller already knows the target language
            for this turn (e.g. computed once in main.py via
            detect_reply_language() and about to be reused for a follow-up
            suggestion too), pass it here to skip a second, redundant
            Gemini detection call. _detect_reply_language() is
            deterministic on the same input, so this produces the exact
            same result as detecting again — it's a pure performance
            optimization, not a behavior change. If not given, detection
            runs here as before.
    """

    settings = get_settings()
    client = _get_client()

    if required_language is None:
        required_language = await _detect_reply_language(client, raw_user_text or user_message, whisper_language)

    system_instruction = SYSTEM_PROMPT + (
        f"\n\n[REQUIRED_LANGUAGE]\n{required_language}\n"
        f"(This is the language the user's CURRENT message is in — "
        f"determined from that message alone, ignoring conversation "
        f"history or the customer summary above. Write your ENTIRE reply "
        f"in exactly this language/script. Do not mention this "
        f"instruction, do not name the language, and do not include any "
        f"tags or labels — just write the reply naturally in that "
        f"language.)"
    )

    # Build conversation history in Gemini format
    history = []
    if conversation_history:
        max_msgs = settings.max_history_turns * 2
        for msg in conversation_history[-max_msgs:]:
            role = "model" if msg["role"] == "assistant" else msg["role"]
            history.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))

    # Create a chat session with history
    chat = client.aio.chats.create(
        model="gemini-2.5-flash-lite",
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            # Lowered from 500 -> 220. The system prompt now enforces a
            # 2-4 sentence default reply; 220 tokens comfortably covers
            # that (plus headroom for a detailed answer when the user
            # explicitly asks for one) while cutting generation time for
            # every reply, and acts as a hard backstop against the model
            # ignoring the length instruction and running long.
            max_output_tokens=220,
            temperature=0.1,
        ),
        history=history,
    )

    try:
        response = await _call_gemini(
            lambda: chat.send_message(user_message),
            label="main reply",
        )
    except Exception as e:
        if _is_transient_gemini_error(e):
            logger.warning(f"Gemini unavailable (transient error): {e}")
            raise GeminiUnavailableError(str(e)) from e
        raise

    reply = response.text.strip()
    reply = _strip_leaked_meta_text(reply)

    logger.info(f"LLM reply: '{reply[:100]}{'...' if len(reply) > 100 else ''}'")
    return reply


DAILY_CHECKIN_SYSTEM_PROMPT = """You are writing ONE proactive daily
check-in message for a premium user of a health WhatsApp bot who is on a
21-day guided plan tied to their health goal/condition.

Your job: based on the user's health profile/summary and recent
conversation, write a short, warm, motivating message that includes ONE
concrete suggestion or small to-do for today — e.g. a specific exercise, a
diet tip, a hydration/sleep reminder, a symptom check-in question, or a
small habit to try — that fits what THEY specifically told the bot about
their health, goals, or condition. Do not repeat a suggestion already sent
recently (see the list below).

Rules:
- Ground it strictly in the user's own stated health profile/summary and
  conversation history. Never invent a condition or symptom they didn't
  mention. If the profile is thin/generic, give a safe, general wellness
  suggestion (hydration, short walk, sleep, stretching) rather than
  guessing specifics.
- Exactly ONE suggestion or to-do per message — do not list multiple.
- Keep it short: 2-4 sentences, WhatsApp style, no markdown, no headers,
  no bullet dashes.
- Sound like a caring health coach checking in, not a generic notification.
  Mention today is part of their plan naturally (e.g. "Day {day_number} of
  your plan" is fine to include once).
- Never give a specific medication dosage, diagnosis, or urgent medical
  instruction here — if their profile suggests something concerning,
  gently suggest they mention it to a doctor instead of addressing it
  yourself in this message.
- End with a light, natural question inviting them to reply (e.g. asking
  how they're feeling, or whether they did/will do the suggested thing) so
  the conversation can continue from their response — but only ONE
  question, not several.
- LANGUAGE: write the entire message in the REQUIRED_LANGUAGE given below,
  in that exact language/script. If no clear prior language is evident,
  default to English.
- Output ONLY the message itself — no preamble, no labels, nothing else.
"""


async def generate_daily_checkin_message(
    customer_summary: str,
    context_text: str,
    day_number: int,
    total_days: int,
    recent_suggestions: list[str] | None = None,
    required_language: str | None = None,
) -> str:
    """
    Generate today's proactive check-in message for a premium user, as part
    of the 21-day (configurable) daily follow-up feature. Called once per
    user per day by the scheduled daily check-in job (see
    app/daily_checkin.py), not from the normal reply path.

    Args:
        customer_summary: the user's running profile/summary (health goal,
            conditions, preferences, etc. — built the same way as the
            regular conversation summary).
        context_text: recent conversation context, same format as used for
            regular replies, so the suggestion stays grounded in what was
            actually discussed.
        day_number: which day of the plan this is (1-indexed).
        total_days: total length of the plan (e.g. 21).
        recent_suggestions: previous daily check-in messages already sent,
            so today's suggestion doesn't repeat one.
        required_language: language to reply in; if not given, defaults to
            English (there's no "current user message" to detect from for
            a proactive message, unlike the normal reply path).
    """
    client = _get_client()
    language = required_language or "English"

    already_sent = ""
    if recent_suggestions:
        bullet_list = "\n".join(f"- {s}" for s in recent_suggestions)
        already_sent = f"\n\n[CHECK-IN MESSAGES ALREADY SENT — do not repeat these]\n{bullet_list}"

    prompt = f"""
[USER HEALTH PROFILE / SUMMARY]
{customer_summary if customer_summary else "No detailed profile yet — keep the suggestion general and safe."}

[RECENT CONVERSATION]
{context_text if context_text else "[No previous messages]"}
{already_sent}

[PLAN PROGRESS]
Day {day_number} of {total_days}

[REQUIRED_LANGUAGE]
{language}

Write today's check-in message now, following the rules exactly.
""".strip()

    try:
        response = await _call_gemini(
            lambda: client.aio.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=[types.Part(text=prompt)],
                config=types.GenerateContentConfig(
                    system_instruction=DAILY_CHECKIN_SYSTEM_PROMPT,
                    max_output_tokens=220,
                    temperature=0.4,
                ),
            ),
            label="daily check-in",
        )
    except Exception as e:
        if _is_transient_gemini_error(e):
            logger.warning(f"Gemini unavailable during daily check-in generation: {e}")
            raise GeminiUnavailableError(str(e)) from e
        raise

    message = (response.text or "").strip()
    message = _strip_leaked_meta_text(message)
    logger.info(f"Daily check-in (day {day_number}/{total_days}): '{message[:100]}'")
    return message