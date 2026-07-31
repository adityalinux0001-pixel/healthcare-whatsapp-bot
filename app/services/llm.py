import asyncio
import logging
import random
import uuid
import json as _json
from functools import lru_cache
from app.core.config import get_settings
from app.core.redis_client import get_redis
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




_SEMAPHORE_KEY = "wa:gemini:semaphore"
# Safety-net TTL per held slot — comfortably longer than any real Gemini
# call (including its own internal retries) should ever take, so a
# crashed worker's slot self-heals instead of leaking forever.
_SEMAPHORE_SLOT_TTL_SECONDS = 120
# How long a waiter polls before rechecking whether a slot has freed up.
_SEMAPHORE_POLL_INTERVAL_SECONDS = 0.15


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
    import time as _time
    wall_now = _time.time()
    # Prune expired slots first so a crashed worker's stale entries don't
    # make the system look busier than it actually is.
    await r.zremrangebyscore(_SEMAPHORE_KEY, "-inf", wall_now)
    current = await r.zcard(_SEMAPHORE_KEY)
    return current >= settings.GEMINI_MAX_CONCURRENT_REQUESTS


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
    limit = settings.GEMINI_MAX_CONCURRENT_REQUESTS

    queued = await is_gemini_busy()
    if queued:
        logger.info(f"Gemini[{label}]: at concurrency limit (cluster-wide), queueing")

    slot = await _redis_semaphore_acquire(limit)
    try:
        if queued:
            logger.info(f"Gemini[{label}]: dequeued, sending request")

        last_exc: Exception | None = None
        for attempt in range(settings.GEMINI_MAX_RETRIES + 1):
            try:
                return await coro_fn()
            except Exception as e:
                if not _is_transient_gemini_error(e):
                    raise
                last_exc = e
                if attempt < settings.GEMINI_MAX_RETRIES:
                    delay = settings.GEMINI_RETRY_BASE_DELAY_SECONDS * (2 ** attempt)
                    delay = min(delay, settings.GEMINI_RETRY_MAX_DELAY_SECONDS)
                    delay += random.uniform(0, delay * 0.25)
                    logger.warning(
                        f"Gemini[{label}]: transient error on attempt "
                        f"{attempt + 1}/{settings.GEMINI_MAX_RETRIES + 1} "
                        f"({e}); retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.warning(
                        f"Gemini[{label}]: still failing after "
                        f"{settings.GEMINI_MAX_RETRIES + 1} attempts ({e}); giving up"
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
                    "You are a language identifier. Review the provided "
                    "isolated text and output ONLY the language and script "
                    "name. Do not include explanations, punctuation, or "
                    "quotes.\n\n"
                    "Rules:\n"
                    "- Non-Latin script languages (e.g., Hindi, Tamil, "
                    "Arabic, Chinese): Respond exactly as \"Language "
                    "written in Native script\" (e.g., \"Tamil written in "
                    "Tamil script\").\n"
                    "- Transliterated Indian languages (Roman/English "
                    "letters; e.g., \"mujhe pricing chahiye\"): Respond "
                    "exactly as \"Language written in Roman letters "
                    "(transliterated, casual style)\" (e.g., \"Hindi "
                    "written in Roman letters (transliterated, casual "
                    "style)\").\n"
                    "- Plain English, greetings, emojis, or ambiguous/"
                    "short text: Respond exactly as: English\n"
                    "- Never output any format other than the specified "
                    "strings."
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
    the CALLER schedules it: main.py now fires this concurrently with context
    retrieval (asyncio.gather) instead of awaiting it sequentially.
    """
    client = _get_client()
    return await _detect_reply_language(client, text, whisper_language)


async def _translate_fixed_text(
    client: "genai.Client",
    text: str,
    required_language: str,
) -> str:
    """
    Translate a fixed, pre-written English template (premium offer /
    expiry notice) into the user's detected reply language.

    Kept as its own tiny, isolated Gemini call — same reasoning as
    _detect_reply_language: mixing this into a larger prompt risks the
    model drifting in tone or partially translating. Here the input is
    ONLY the template text plus the target language name, so the model
    has nothing else to get confused by.

    Numbers, the ₹ amount, emojis, the payment link URL, and the
    1./2./3. list structure must be preserved exactly — only the
    surrounding words get translated. If the target language is plain
    English, the original text is returned unchanged (no wasted call).
    If the call fails for any reason, the original English text is
    returned rather than blocking the send.
    """
    if not required_language or required_language.strip().lower() == "english":
        return text

    try:
        response = await _call_gemini(
            lambda: client.aio.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[types.Part(text=text)],
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are a precise translator. Translate the given "
                    f"WhatsApp message into {required_language}. "
                    "Rules:\n"
                    "- Preserve the exact structure: line breaks, the "
                    "1./2./3. numbered list, and all emojis exactly where "
                    "they are.\n"
                    "- Do NOT translate or alter URLs (e.g. links "
                    "starting with http/https), numbers, or the ₹ "
                    "currency amount — keep those exactly as written.\n"
                    "- Translate only the surrounding natural-language "
                    "words/sentences.\n"
                    "- Output ONLY the translated message, nothing else "
                    "— no preamble, no explanation, no quotes."
                ),
                max_output_tokens=600,
                temperature=0.0,
            ),
            ),
            label="premium offer translation",
        )
        translated = (response.text or "").strip()
        if translated:
            return translated
    except Exception as e:
        logger.warning(
            f"Premium offer translation failed, sending English original: {e}"
        )

    return text


async def translate_premium_offer_text(text: str, required_language: str) -> str:
    """
    Public wrapper around _translate_fixed_text() for main.py to translate
    the fixed premium-offer / expiry-notice templates into the user's
    detected reply language, mirroring the detect_reply_language()
    public-wrapper pattern above.
    """
    client = _get_client()
    return await _translate_fixed_text(client, text, required_language)


async def _classify_premium_intent(
    client: "genai.Client",
    user_text: str,
    recent_context: str = "",
) -> dict:
    """
    Small, isolated Gemini call that decides whether the user's CURRENT
    message is actually about the premium plan/subscription/payment (and,
    if so, whether they're directly asking for it right now).

    Replaces the old approach of `keyword in text.lower()` substring
    matching against a big static list (_PREMIUM_INTEREST_KEYWORDS /
    _PREMIUM_EXPLICIT_REQUEST_KEYWORDS). That approach broke in two
    concrete, observed ways:

    1. False positives from unrelated words that happen to contain a
       trigger substring or co-occur with one — e.g. "vegetarian diet
       for tuberculosis" matched "diet" and got treated as premium
       interest, and "no, tuberculosis diet, not weight loss" still
       matched "weight loss" even though the user was explicitly
       correcting/declining that topic.
    2. No concept of negation or topic-correction at all — a keyword
       list can't tell "I want the plan" apart from "I don't want the
       plan" or "that's not what I meant".

    This mirrors _detect_reply_language's pattern exactly: an isolated,
    cheap, low-latency classifier call with a tightly constrained output
    format, given only the current message plus a little recent context
    (not the full conversation, to keep it fast and avoid the model
    getting confused by older unrelated turns — same rationale as
    language detection).

    Returns a dict:
      {
        "premium_related": bool,   # current message is about the plan/
                                    # payment/subscription (topic gate —
                                    # equivalent to old _shows_premium_interest)
        "explicit_request": bool,  # user is directly asking for the plan
                                    # or its payment link RIGHT NOW
                                    # (equivalent to old
                                    # _requests_premium_plan_explicitly)
      }

    On any failure (network/API error, unparsable response), falls back
    to {"premium_related": False, "explicit_request": False} — i.e. fail
    CLOSED. A missed upsell opportunity is a minor, recoverable business
    cost; a wrongly-triggered upsell that hijacks a health question (the
    original bug) is a worse user-facing failure, so the safe default on
    error is "don't trigger", not "trigger".
    """
    stripped = user_text.strip()
    if not stripped:
        return {"premium_related": False, "explicit_request": False}

    prompt_parts = []
    if recent_context:
        prompt_parts.append(f"Recent conversation (oldest to newest):\n{recent_context}\n")
    prompt_parts.append(f"CURRENT user message: {stripped[:500]}")
    contents_text = "\n".join(prompt_parts)

    try:
        response = await _call_gemini(
            lambda: client.aio.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=[types.Part(text=contents_text)],
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "You are an intent classifier for a WhatsApp health "
                        "bot offering an optional paid 21-day premium plan. "
                        "Evaluate the user's CURRENT message against recent "
                        "context to determine two boolean properties:\n\n"
                        "1. premium_related: Is the current message "
                        "explicitly about the paid premium plan, pricing, "
                        "or subscription?\n"
                        "   - FALSE if discussing diet/exercise for a "
                        "specific medical condition (e.g., 'meal plan for "
                        "diabetes').\n"
                        "   - FALSE if correcting/clarifying a previous "
                        "misunderstanding (e.g., 'no, I meant TB, not "
                        "weight loss').\n"
                        "   - FALSE if saying no, stop, cancel, or "
                        "expressing zero interest.\n"
                        "   - TRUE if asking about price, content, payment "
                        "links, subscription steps, or expressing a "
                        "personal goal to lose weight/get fit (not as a "
                        "topic correction).\n\n"
                        "2. explicit_request: Evaluated only if "
                        "premium_related is true. Is the user explicitly "
                        "and unambiguously demanding the plan or payment "
                        "link RIGHT NOW (e.g., 'send link', 'I want to "
                        "buy')? General questions or passing mentions are "
                        "FALSE.\n\n"
                        "Respond ONLY with a raw JSON object (no markdown, "
                        "no explanations):\n"
                        '{"premium_related": true|false, '
                        '"explicit_request": true|false}'
                    ),
                    max_output_tokens=60,
                    temperature=0.0,
                ),
            ),
            label="premium intent classification",
        )
        raw = (response.text or "").strip()
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
        parsed = _json.loads(raw)
        premium_related = bool(parsed.get("premium_related", False))
        explicit_request = bool(parsed.get("explicit_request", False)) and premium_related
        return {"premium_related": premium_related, "explicit_request": explicit_request}
    except Exception as e:
        logger.warning(
            f"Premium intent classification failed, defaulting to False/False: {e}"
        )
        return {"premium_related": False, "explicit_request": False}


async def classify_premium_intent(
    user_text: str,
    recent_context: str = "",
) -> dict:
    """
    Public wrapper around _classify_premium_intent() for callers outside
    this module (main.py). See that function's docstring for the full
    rationale and return shape.
    """
    client = _get_client()
    return await _classify_premium_intent(client, user_text, recent_context)


SYSTEM_PROMPT = """You are an AI WhatsApp health assistant—a knowledgeable, caring guide (not a doctor) providing clear, practical, evidence-based information.

ROLE
- Context Gathering: Naturally ask 1-2 questions at a time about age, gender, symptoms, conditions, medications, allergies, lifestyle, and goals. Never delay urgent or simple answers for a full intake.
- Scope: Answer using medical knowledge (nutrition, fitness, conditions, medications, prevention, mental health, sleep). Never invent lab values, dosages, or diagnoses.
- Medical Disclaimer: State clearly you are an AI. For diagnoses, prescriptions, or advanced guidance, recommend a licensed doctor.
- Emergency Rule: If urgent/severe symptoms (chest pain, dyspnea, stroke signs, severe bleeding, suicidal thoughts) occur, immediately halt routine Q&A and direct them to emergency services.

SYMPTOM INTAKE MODE
Triggered when key details (duration, severity, associated symptoms, triggers, history) are missing from a reported symptom.
- Respond like a real person texting a quick check—not a formal assistant.
- Ask exactly ONE short, specific question (2-8 words; e.g., "Any cough?", "How long?"). No greetings, reassurance, explanations, or stacked questions.
- Ask maximum 3-4 questions total before providing guidance based on gathered facts.
- Never re-ask details already present in the history or summary.
- Exit mode immediately if emergency signs appear, or once sufficient details are known. Does not apply to general questions (e.g., "what causes migraines?").

COMMUNICATION STYLE
- Length: Max 2-4 short sentences, unless details require a list/sequence. Intake questions are single fragments.
- Directness: Answer immediately in the first sentence. No preamble or meta-text.
- Formatting: Plain text only—no markdown. Use numbered lists only for steps or options.
- Out of Scope: State in one line that it is outside your safe scope and recommend a doctor.

EXPLANATION STYLE
- Keep it to 1-2 useful sentences (max 3 if essential); address the 1-2 most relevant points without padding.
- One idea per reply. Use plain language over medical jargon; provide brief inline definitions if jargon is unavoidable.
- End immediately after answering—do not ask closing or follow-up questions (managed by an external system).
- Sound natural, human, and unscripted.

LANGUAGE RULE: Always reply in the exact script and language specified in REQUIRED_LANGUAGE (e.g., Tamil in Tamil script, Hindi in Devanagari, Hinglish in casual Roman script, English in plain English). Never infer language from history. Do not output the label "REQUIRED_LANGUAGE" or any bracketed tags.

TONE: Warm, calm, reassuring, and completely honest about AI limitations."""


@lru_cache(maxsize=1)
def _get_client() -> genai.Client:
    # Was being constructed fresh on every single message, which rebuilds
    # its underlying HTTP transport/connection pool each time — cache it
    # so connections are reused across requests.
    settings = get_settings()
    return genai.Client(api_key=settings.GEMINI_API_KEY)


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
        client = _get_client()
        

        vision_prompt = """Describe this image in 1-3 short, plain sentences. Specify what it shows, extract any visible text verbatim, and highlight details relevant to health or medicine (e.g., symptoms, food, medication labels, lab reports, activities). Do not include headers, numbered lists, or extrapolation."""
        

        response = await _call_gemini(
            lambda: client.aio.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                types.Part(text=vision_prompt)
            ],
            config=types.GenerateContentConfig(

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


FOLLOWUP_SYSTEM_PROMPT = """You are a cross-questioning assistant for a WhatsApp health bot. Analyze the customer summary, recent conversation, and the bot's latest reply to determine if a precise, highly relevant follow-up can engage the user.

Rules:
- Grounding: Base suggestions strictly on existing goals, problems, or interests. Never introduce completely new topics.
- Stay on Topic: Every option or nudge must align directly with the active subject. If you cannot create a highly relevant follow-up, reply exactly: NONE
- Avoid Redundancy: Do not repeat questions asked earlier. If the bot's latest reply is an open invitation (e.g., "How can I help you?"), reply: NONE
- Symptom Intake Check: If the bot's latest reply is ALREADY a short intake question (e.g., "Any cough?", "How long?"), you must reply exactly: NONE (avoids double questioning).
- Output: Output ONLY the text of the suggestion or exactly NONE. No preambles ("Follow-up:"), formatting explanations, or markdown labels.

FORMATS (Select the most appropriate and vary across turns):
1. Poll Option-Style: A short question followed by 2-3 numbered choices (e.g., "Quick one—what matters most right now? 1. Speed 2. Cost"). Choices must target the same active topic.
2. Natural Clarifying Question: A warm, conversational question probing a missing lifestyle or symptom detail relevant to the current topic.
3. Related-Angle Nudge: Highlight an unaddressed aspect of the active topic and ask if they want to explore it.
4. Concrete Next Step: Offer an actionable next step (e.g., "Want a simple routine to try for this?").

Style Guidelines:
- Max 1-3 lines, WhatsApp style. No paragraphs.
- Tone should be warm, engaged, and human. No markdown symbols (no **, ##, or backticks).
- LANGUAGE: Write the entire output (including numbered options) strictly in the language/script designated under [REQUIRED_LANGUAGE].
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

Identify the single best follow-up question or nudge following the rules exactly.
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


SUMMARY_SYSTEM_PROMPT = """You maintain a concise, factual, running internal summary of a customer's profile, goals, preferences, and interaction history for a CRM record.

Rules:
- Always write the summary in English, regardless of the conversation language.
- Format as a short paragraph or bullet points. Preserve key past context while incorporating new facts.
- Exclude conversational filler, metadata, greetings, or commentary.
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
            Passed through to _detect_reply_language so a miss-transcribed
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
        max_msgs = settings.MAX_HISTORY_TURNS * 2
        for msg in conversation_history[-max_msgs:]:
            role = "model" if msg["role"] == "assistant" else msg["role"]
            history.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))

    # Create a chat session with history
    chat = client.aio.chats.create(
        model="gemini-2.5-flash-lite",
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,

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


DAILY_CHECKIN_SYSTEM_PROMPT = """You are a caring health coach writing ONE proactive daily check-in message for a premium WhatsApp user on a 21-day guided health plan.

Rules:
- Personalization: Base the message strictly on the user's profile and history. Never invent symptoms or conditions. If data is sparse, provide a safe, generic wellness tip.
- Content: Deliver exactly ONE concrete, actionable suggestion or daily to-do (e.g., a food swap, specific stretch, hydration goal). Never provide medication dosages or diagnostic claims.
- Length & Style: 2-4 sentences max, WhatsApp style. Plain text only (no markdown, headers, or bullet dashes). Include 2-3 relevant emojis (e.g., 💪, 🥗, 💧) to add warmth without replacing words.
- Structure: Seamlessly reference their progress (e.g., "Day {day_number} of your plan"). End with exactly ONE natural, engaging question inviting a response.
- Language: Write completely in the specified REQUIRED_LANGUAGE. Default to English if ambiguous. Output ONLY the final message text.
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

Write today's check-in message following the rules exactly.
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


PLAN_CATEGORY_PROMPTS: dict[str, str] = {
    "weight_loss": """
You are a practical health coach building a progressive {total_days}-day WhatsApp weight-loss coaching plan based on the onboarding data below. This is a structured, evidence-informed behavior-change program — grounded in real clinical weight-management guidance (CDC lifestyle-change program principles and the NHLBI/NIH obesity treatment guidelines) — not medical prescriptions, not a crash diet, and not a generic tip-of-the-day list.

Grounding facts to build every day's guidance on (do not state these as raw facts to the user — apply them, don't lecture):
- Healthy, sustainable weight loss is about 1-2 lb (0.5-1 kg) per week, coming from a moderate daily energy deficit built through food choices and movement — never frame anything as rapid or extreme loss.
- Small, specific, trackable actions compound. Self-monitoring (food/water/sleep/mood logging) in the early days measurably improves outcomes because it builds awareness before it builds change.
- Movement adherence is higher when broken into short bouts (e.g., three 10-minute walks) rather than demanding one long session, especially early on or when time is limited.
- Early, achievable wins build momentum; guidance should never front-load the hardest asks.
- Long-term success depends on habit and identity change (what a person consistently does), not willpower spikes — so plateaus, slip-ups, and real-life disruptions (travel, parties, stress, cravings) must be planned for, not treated as failure.

Program arc — the plan MUST progress through these three phases across the {total_days} days (split roughly into thirds; adjust boundaries slightly if {total_days} isn't divisible by 3):

PHASE 1 — AWARENESS & BASELINE (first third):
- Focus: self-monitoring, noticing current patterns, identifying personal obstacles (schedule, cravings, social pressure, sleep), and ONE small, low-friction swap at a time.
- Tone: gentle, curious, zero pressure. No big asks. This phase builds the habit of paying attention, not the habit of restriction.

PHASE 2 — BUILDING CORE HABITS (middle third):
- Focus: structured movement matched to their stated time limit and physical constraints (broken into short bouts where useful), protein- and fiber-forward food swaps, consistent meal timing, hydration, and sleep hygiene.
- Tone: encouraging and a little more structured — this is where real behavior change is being built, day over day, referencing what earlier days already established.

PHASE 3 — CONSOLIDATION & MINDSET (final third):
- Focus: handling setbacks and plateaus without spiraling, navigating real-life situations (eating out, travel, stress-eating, low-motivation days), and locking in the habits that should continue after Day {total_days}.
- Tone: resilience-focused and forward-looking — helping them own the change as their own, not something that ends when the plan ends.

Message quality requirements (this is what makes each day feel like real coaching, not a random tip):
- Every message should briefly connect the ONE action to why it matters in plain, non-clinical language (e.g., grounding it in momentum, energy, consistency, or how the body responds) — one short reason, not a lecture.
- Give exactly ONE specific, concrete, doable action per day. Never vague ("eat healthier", "be more active") — always a specific swap, a specific movement, a specific check-in, or a specific reflection prompt.
- Each day must clearly build on or connect to what came before within its phase, so the plan reads as one coherent 21-day arc, not {total_days} unrelated tips.
- Reference the plan's day number and phase-appropriate framing (e.g., early days = "let's just notice", later days = "you've already built X, now let's add Y") so continuity is felt.

Safety & Customization Rules:
- Adhere strictly to any listed injuries, medical conditions, or physical restrictions (e.g., no high-impact moves if knee pain is noted). Keep guidance safe and generic for chronic conditions (PCOS, diabetes, thyroid) — never contradict a doctor's plan, always defer to one for anything medical.
- Fully respect all dietary preferences and restrictions.
- Match their daily time constraints (e.g., do not exceed 15 mins if that is their stated limit).
- NEVER prescribe specific calorie counts, macro targets, or numeric weight-loss targets/timelines to the user — the 1-2 lb/week pace above is for your own internal calibration of tone and ambition only, not something to state as a number/promise to the user.
- Do not repeat identical suggestions across the {total_days} days. Progressively vary the focus across movement, food swaps, mindset, sleep, and reflection within each phase. Avoid strategies they explicitly noted disliking.
""",
}


def _plan_category_system_prompt(category: str, total_days: int) -> str:
    template = PLAN_CATEGORY_PROMPTS.get(category, PLAN_CATEGORY_PROMPTS["weight_loss"])
    return template.format(total_days=total_days).strip()


_PLAN_OUTPUT_FORMAT_INSTRUCTIONS = """
[OUTPUT FORMAT — FOLLOW EXACTLY]
Respond ONLY with a single JSON array containing exactly {total_days} objects corresponding to each ordered day. No markdown fences, preambles, or trailing text. Each object must contain exactly these two keys:

  "message": A plain-text WhatsApp message (4-7 sentences, in {language}) — long enough to feel like genuine, descriptive coaching rather than a one-line tip. Structure it as: (1) a brief natural reference to today/their progress, (2) the ONE specific action for today stated clearly, (3) one short "why this matters" line connecting it to momentum, energy, or consistency, and (4) a brief note on how to actually do it today (timing, a simple substitution, or a way to make it easier). Friendly, conversational, genuinely encouraging tone decorated with 2-3 well-placed emojis (e.g., 💪, 🥗, 🎯). Vary emoji combinations day-to-day to avoid repetition. Never replace text clarity with symbols, and never state calorie counts, macro numbers, or specific weight-loss amounts/timelines.
  "followup_question": A short, single-sentence engagement question (in {language}) decorated with exactly ONE relevant emoji, designed to check in on their progress later that day. Do not reference this question inside the "message".

Example structure:
[
  {{"message": "Day 1 text...", "followup_question": "Day 1 question?"}},
  {{"message": "Day 2 text...", "followup_question": "Day 2 question?"}}
]
""".strip()


async def generate_premium_plan(
    onboarding_answers: dict,
    category: str,
    total_days: int,
    required_language: str | None = None,
) -> list[dict]:
    """
    ONE Gemini call that generates the entire {total_days}-day premium plan
    up front: for every day, both the message to send AND that day's
    same-day follow-up question. Called exactly once, right after
    onboarding finishes (see app/onboarding.py), and the result is written
    to the premium_plans table (see app/memory.py) as one row per day.

    No LLM calls happen again for this user's plan after this point — the
    daily scheduler (app/daily_checkin.py) only ever fetches pre-written
    rows and sends them.

    Args:
        onboarding_answers: dict of the answers collected during onboarding
            (weight/height, goal, diet, activity level, medical conditions,
            routine/time available, past attempts — see app/onboarding.py
            for the exact question set).
        category: plan category, e.g. "weight_loss" (see
            settings.DEFAULT_PLAN_CATEGORY and PLAN_CATEGORY_PROMPTS above).
        total_days: length of the plan (settings.PREMIUM_PLAN_DAYS).
        required_language: language to write in; defaults to English.

    Returns:
        A list of exactly `total_days` dicts, each shaped
        {"message": str, "followup_question": str}, in day order
        (index 0 = day 1).

    Raises:
        GeminiUnavailableError: Gemini is down/overloaded — caller should
            NOT activate/save a half-formed plan and should let the
            subscription-activation flow retry or alert an operator,
            since silently failing here means the user paid for a premium
            plan that never materializes.
        ValueError: Gemini responded but not with valid, complete JSON for
            all {total_days} days — treated the same as unavailable by
            callers (don't save a partial/malformed plan).
    """
    client = _get_client()
    language = required_language or "English"

    system_prompt = _plan_category_system_prompt(category, total_days)
    output_format = _PLAN_OUTPUT_FORMAT_INSTRUCTIONS.format(
        total_days=total_days, language=language
    )
    full_system_prompt = f"{system_prompt}\n\n{output_format}"

    answers_json = _json.dumps(onboarding_answers, indent=2, ensure_ascii=False)
    prompt = f"""
[ONBOARDING ANSWERS FOR THIS USER]
{answers_json}

[PLAN LENGTH]
{total_days} days

[REQUIRED_LANGUAGE]
{language}

Generate the full {total_days}-day JSON plan now, following the rules and format exactly.
""".strip()

    try:
        response = await _call_gemini(
            lambda: client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=[types.Part(text=prompt)],
                config=types.GenerateContentConfig(
                    system_instruction=full_system_prompt,
                    max_output_tokens=get_settings().PLAN_GENERATION_MAX_OUTPUT_TOKENS,
                    temperature=0.6,
                    response_mime_type="application/json",
                ),
            ),
            label="premium plan pregeneration",
        )
    except Exception as e:
        if _is_transient_gemini_error(e):
            logger.warning(f"Gemini unavailable during premium plan pregeneration: {e}")
            raise GeminiUnavailableError(str(e)) from e
        raise

    raw_text = (response.text or "").strip()

    try:
        parsed = _json.loads(raw_text)
    except _json.JSONDecodeError as e:
        logger.error(f"❌ Plan pregeneration returned invalid JSON: {e} | raw[:500]={raw_text[:500]}")
        raise ValueError(f"Invalid JSON from plan generation: {e}") from e

    if not isinstance(parsed, list) or len(parsed) != total_days:
        logger.error(
            f"❌ Plan pregeneration returned wrong shape — expected a {total_days}-item "
            f"list, got {type(parsed).__name__} of len "
            f"{len(parsed) if isinstance(parsed, list) else 'n/a'}."
        )
        raise ValueError(
            f"Expected {total_days} plan days, got "
            f"{len(parsed) if isinstance(parsed, list) else type(parsed).__name__}"
        )

    days: list[dict] = []
    for i, item in enumerate(parsed, start=1):
        if not isinstance(item, dict) or "message" not in item or "followup_question" not in item:
            logger.error(f"❌ Plan pregeneration day {i} malformed: {item!r}")
            raise ValueError(f"Day {i} missing required keys 'message'/'followup_question'")
        message = _strip_leaked_meta_text(str(item["message"]).strip())
        followup_question = str(item["followup_question"]).strip()
        if not message or not followup_question:
            raise ValueError(f"Day {i} has an empty message or followup_question")
        days.append({"message": message, "followup_question": followup_question})

    logger.info(
        f"✅ Pregenerated {len(days)}-day '{category}' plan in one Gemini call "
        f"(language={language})."
    )
    return days


async def _classify_onboarding_answer(
    client: "genai.Client",
    question_text: str,
    user_text: str,
) -> dict:
    """
    Small, isolated Gemini call used ONLY during onboarding
    (app/services/onboarding.py). Decides whether the user's reply is a
    genuine attempt to answer the CURRENT onboarding question, or
    off-topic text (greetings, small talk, random questions, complaints,
    anything else) that should NOT be saved as the answer.

    Same pattern as _classify_premium_intent/_detect_reply_language: a
    single tightly-scoped call given only the current question + current
    reply (not the full onboarding history), kept cheap and fast.

    When the reply is off-topic, this ALSO generates the short, warm
    acknowledgment message to send back to the user in the SAME call, so
    an off-topic reply costs exactly one Gemini call, not two.

    Returns a dict:
      {
        "is_answer": bool,        # True if this genuinely answers the
                                    # current question and should be saved
        "acknowledgment": str,     # only meaningful when is_answer is
                                    # False - a brief, friendly reply to
                                    # whatever the user actually said
      }

    On any failure (network/API error, unparsable response), falls back
    to {"is_answer": True, "acknowledgment": ""} - i.e. fail OPEN. If we
    can't classify, treating the reply as a real answer and moving on
    keeps onboarding from ever getting stuck in a loop; the worst case is
    an occasional imperfect answer reaching the plan generator, which is
    far less disruptive than a user being unable to progress at all.
    """
    stripped = user_text.strip()
    if not stripped:
        return {"is_answer": True, "acknowledgment": ""}

    contents_text = (
        f"QUESTION asked to the user:\n{question_text.strip()[:500]}\n\n"
        f"USER's reply:\n{stripped[:500]}"
    )

    try:
        response = await _call_gemini(
            lambda: client.aio.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=[types.Part(text=contents_text)],
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "You are checking replies during a WhatsApp health "
                        "coach's onboarding flow, where the user is asked a "
                        "fixed sequence of profile questions one at a time.\n\n"
                        "Decide: does the user's reply genuinely attempt to "
                        "answer the QUESTION asked (even if brief, informal, "
                        "or incomplete)? Judge substance, not politeness or "
                        "grammar.\n"
                        "- TRUE if it's a real attempt at the requested info, "
                        "even if partial, vague, or in another language.\n"
                        "- FALSE if it's a greeting ('hi', 'hello'), small "
                        "talk ('how are you'), an unrelated question, a "
                        "complaint, or otherwise doesn't address the "
                        "question at all.\n\n"
                        "If FALSE, also write a short acknowledgment "
                        "(max 2 short sentences, warm and natural, 0-1 "
                        "emoji) that directly responds to what the user "
                        "said, then gently reminds them the question is "
                        "still waiting. Do NOT restate the full original "
                        "question text in the acknowledgment — just the "
                        "reminder (e.g. 'still need that from you above' "
                        "or 'whenever you're ready to answer that'). If "
                        "TRUE, leave acknowledgment as an empty string.\n\n"
                        "Respond ONLY with a raw JSON object (no markdown, "
                        "no explanations):\n"
                        '{"is_answer": true|false, "acknowledgment": "..."}'
                    ),
                    max_output_tokens=200,
                    temperature=0.4,
                ),
            ),
            label="onboarding answer classification",
        )
        raw = (response.text or "").strip()
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
        parsed = _json.loads(raw)
        is_answer = bool(parsed.get("is_answer", True))
        acknowledgment = str(parsed.get("acknowledgment", "") or "").strip()
        return {"is_answer": is_answer, "acknowledgment": acknowledgment}
    except Exception as e:
        logger.warning(
            f"Onboarding answer classification failed, defaulting to "
            f"is_answer=True (fail open): {e}"
        )
        return {"is_answer": True, "acknowledgment": ""}


async def classify_onboarding_answer(
    question_text: str,
    user_text: str,
) -> dict:
    """
    Public wrapper around _classify_onboarding_answer() for callers
    outside this module (app/services/onboarding.py). See that
    function's docstring for the full rationale and return shape.
    """
    client = _get_client()
    return await _classify_onboarding_answer(client, question_text, user_text)