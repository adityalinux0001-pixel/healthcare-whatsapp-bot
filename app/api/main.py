"""
Enhanced Main Application
Features:
- Database storage of all messages with timestamps
- Audio file storage and retrieval
- Context-based responses using last 5 messages
- Support for both audio and text models
- Intelligent conversation routing
"""

import logging
import sys
import json
import httpx
import asyncio
import time
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException, Query, BackgroundTasks
from fastapi.responses import PlainTextResponse, HTMLResponse
from app.core.config import get_settings
from app.models.schemas import (
    WebhookPayload,
    IncomingMessage,
    TestMessageRequest,
    TestTemplateRequest,
)
from app.services.whatsapp import (
    send_text_message,
    send_template_message,
    mark_as_read,
    verify_token_valid,
)
from app.services.llm import (
    get_llm_response,
    get_summary_response,
    process_image_with_vision,
    generate_followup_suggestion,
    detect_reply_language,
    translate_premium_offer_text,
    classify_premium_intent,
    classify_payment_status_intent,
    is_gemini_busy,
    GeminiUnavailableError,
)
from app.services.memory import ConversationMemory
from app.core.idempotency import try_mark_message_processed
from app.services.audio_handler import (
    transcribe_audio, 
    get_available_models,
    get_model_info,
    get_audio_duration_seconds,
)
from app.core.queueing import enqueue_incoming
from app.services.razorpay_client import create_payment_link, verify_webhook_signature
from app.services.onboarding import start_onboarding, handle_onboarding_reply, generate_and_send_plan
from app.services.plan_delivery import send_plan_day, TASK_DONE_PREFIX, TASK_NOT_DONE_PREFIX
from app.services import daily_limit
from contextvars import ContextVar

from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.web.router import router as web_router, templates as web_templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
# Voice notes longer than this are rejected outright — client requirement.
MAX_AUDIO_DURATION_SECONDS = 30

# Logging
settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("whatsapp_bot_enhanced")

# Initialize Postgres-backed conversation memory (step 1)
memory = ConversationMemory(
    database_url=settings.DATABASE_URL,
    pool_min_size=settings.DB_POOL_MIN_SIZE,
    pool_max_size=settings.DB_POOL_MAX_SIZE,
)


_MAX_MESSAGE_AGE_SECONDS = 120

# App
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("AI Health Assistant — WhatsApp Bot")
    logger.info(f"Phone Number ID : {settings.PHONE_NUMBER_ID}")
    logger.info(f"Database        : {memory._safe_url()}")
    logger.info(f"Audio Storage   : {memory.audio_dir}")
    logger.info("Swagger UI      : http://localhost:8000/docs")
    logger.info("=" * 60)

    try:
        from app.core.redis_client import get_redis
        await get_redis().ping()
        logger.info("✅ Redis connected")
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")

    yield


    try:
        memory.pool.close()
    except Exception:
        logger.warning("Error closing Postgres pool on shutdown", exc_info=True)

app = FastAPI(
    title="AI Health Assistant — WhatsApp Bot",
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(web_router)


@app.exception_handler(404)
async def not_found(request: Request, exc):
    return web_templates.TemplateResponse(request, "404.html", {}, status_code=404)

# Webhook GET — Meta verification
@app.get("/webhook")
async def verify_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
    hub_challenge: str = Query(None, alias="hub.challenge"),
):
    logger.info(f"Webhook verify — mode={hub_mode} token={hub_verify_token}")
    if hub_mode == "subscribe" and hub_verify_token == settings.VERIFY_TOKEN:
        logger.info("Webhook verified")
        return PlainTextResponse(content=hub_challenge)
    logger.warning("Webhook verification failed")
    raise HTTPException(status_code=403, detail="Verification token mismatch")


# Reuse a single pooled/keep-alive HTTP client for media downloads instead
# of opening two brand new TCP+TLS connections (one per request) every
# time a user sends a voice note or image.
_media_http_client: httpx.AsyncClient | None = None


def _media_client() -> httpx.AsyncClient:
    global _media_http_client
    if _media_http_client is None:
        _media_http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
        )
    return _media_http_client


async def download_media(media_id: str) -> tuple[bytes, str]:
    """Download media from WhatsApp using media ID."""
    try:
        url = f"https://graph.facebook.com/v25.0/{media_id}"
        headers = {"Authorization": f"Bearer {settings.WHATSAPP_TOKEN}"}
        client = _media_client()

        resp = await client.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        media_data = resp.json()
        media_url = media_data.get("url")
        mime_type = media_data.get("mime_type", "audio/ogg")

        resp = await client.get(media_url, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.content, mime_type
    except Exception as e:
        logger.error(f"Failed to download media {media_id}: {e}")
        raise


def _looks_like_a_question(reply: str) -> bool:
    text = reply.strip()
    if not text:
        return False

    first_line = text.splitlines()[0].strip()

    if not first_line.endswith("?"):
        return False


    return len(first_line) <= 120


async def generate_context_aware_response(
    phone_number: str,
    user_message: str,
    is_audio: bool = False,
    whisper_language: str | None = None,
) -> tuple[str, str]:
    """
    Generate response using conversation context from last 5 messages.
    
    Args:
        phone_number: User's phone number
        user_message: User's current message
        is_audio: Whether the message was audio
        whisper_language: for voice messages, the language Whisper's STT
            API detected from the audio itself — passed through so the
            reply language decision isn't fooled by a mis-transcribed
            script on short/accented clips.
        
    Returns:
        (reply_text, required_language) — required_language is the
        detected reply language for this turn's user_message, returned so
        the caller can pass it straight into maybe_send_followup() /
        generate_followup_suggestion() instead of having that function
        re-detect the same deterministic result via a second Gemini call.
    """

    async def _get_context():
        return await asyncio.to_thread(memory.get_conversation_context, phone_number, limit=5)

    async def _get_customer():
        return await asyncio.to_thread(memory.get_customer, phone_number)

    async def _get_onboarding_profile():
        return await asyncio.to_thread(memory.get_onboarding_session, phone_number)

    async def _get_language():
        return await detect_reply_language(user_message, whisper_language)

    context_text, customer_data, onboarding_session, required_language = await asyncio.gather(
        _get_context(), _get_customer(), _get_onboarding_profile(), _get_language()
    )
    customer_summary = customer_data.get("summary", "")
    onboarding_answers = (onboarding_session or {}).get("answers") or {}
    if isinstance(onboarding_answers, str):
        try:
            onboarding_answers = json.loads(onboarding_answers)
        except json.JSONDecodeError:
            onboarding_answers = {"saved_profile": onboarding_answers}
    onboarding_profile = json.dumps(onboarding_answers, ensure_ascii=False, indent=2)


    session_age = await asyncio.to_thread(memory.get_symptom_session_age_seconds, phone_number)
    if session_age is not None and session_age > settings.SYMPTOM_INTAKE_SESSION_TIMEOUT_SECONDS:
        # Stale session (user went quiet for a long time) — treat this as
        # a fresh conversation instead of continuing to count against an
        # abandoned symptom discussion.
        await asyncio.to_thread(memory.reset_symptom_session, phone_number)
        questions_asked_so_far = 0
    else:
        questions_asked_so_far = await asyncio.to_thread(memory.get_symptom_question_count, phone_number)

    intake_limit_reached = questions_asked_so_far >= settings.SYMPTOM_INTAKE_MAX_QUESTIONS


    topic_switch_instruction = ""
    if not intake_limit_reached:
        premium_intent = await classify_premium_intent(user_message, recent_context=context_text)
        if premium_intent["premium_related"]:
            topic_switch_instruction = f"""

HARD OVERRIDE — READ FIRST: The user's CURRENT message is about the premium plan/subscription/payment, not symptoms. Do NOT ask a symptom-intake question. Answer only their plan-related question (pricing, content, payment link, etc.). Do not pivot back to previous symptoms. Intake can resume only if they explicitly return to symptoms in a later message. IMPORTANT: still write your ENTIRE reply in [REQUIRED_LANGUAGE] ({required_language}) exactly as instructed above — do not switch to English just because the plan details or prior messages were in English."""

    # Build enriched prompt with context
    message_type = "[AUDIO MESSAGE]" if is_audio else ""
    force_answer_instruction = ""
    if intake_limit_reached:

        force_answer_instruction = f"""

HARD OVERRIDE — READ FIRST: You have already asked {questions_asked_so_far} intake questions (accurate internal count). Do NOT ask another question under any circumstances. Provide your best possible guidance now using existing data, adhering to normal length/style rules. Recommend an in-person doctor exam if required. This overrides SYMPTOM INTAKE MODE."""

    enriched_prompt = f"""
{message_type}

[CUSTOMER SUMMARY]
{customer_summary if customer_summary else "No prior context available"}

[SAVED ONBOARDING PROFILE]
{onboarding_profile if onboarding_answers else "No onboarding profile saved yet"}

{context_text if context_text else "[No previous messages]"}

[CURRENT USER MESSAGE]
{user_message}

Answer only the current user message directly and briefly. Use the conversation context solely for consistency—do not re-summarize or restate past messages.

The SAVED ONBOARDING PROFILE contains the user's answers collected earlier. Treat those answers as authoritative facts for this user. Do not ask the user again for any detail that is present there, even if it is not repeated in recent conversation history. If the profile contains a field such as age, gender, weight, height, goal, diet, activity, medical conditions, routine, or past attempts, use it when relevant and never ask for it again unless the user clearly says it has changed.

CRITICAL INTAKE CHECK: Review [SAVED ONBOARDING PROFILE], [CUSTOMER SUMMARY], and context carefully. If in SYMPTOM INTAKE MODE, NEVER ask for details (duration, severity, etc.) already provided. Ask only for the next missing detail. If all necessary information is present, stop questioning and provide your final guidance.
{force_answer_instruction}
{topic_switch_instruction}
    """.strip()



    # Get LLM response
    try:
        response = await get_llm_response(
            user_message=enriched_prompt,
            conversation_history=[],
            raw_user_text=user_message,
            whisper_language=whisper_language,
            required_language=required_language,
        )
    except GeminiUnavailableError:
        # Gemini is down/overloaded (503 etc.) — bubble this up so the
        # caller can drop the query silently instead of sending any
        # fallback reply or saving a "sorry" message to history.
        raise
    except Exception as e:
        logger.error(f"LLM error: {e}", exc_info=True)
        return "Sorry, I ran into an issue. Please try again in a moment.", required_language


    try:
        if _looks_like_a_question(response):
            await asyncio.to_thread(memory.increment_symptom_question_count, phone_number)
        else:
            await asyncio.to_thread(memory.reset_symptom_session, phone_number)
    except Exception as e:
        logger.error(f"⚠️ Failed to update symptom session counter for {phone_number}: {e}", exc_info=True)

    return response, required_language


SUMMARY_REFRESH_EVERY_N_MESSAGES = 3


async def generate_new_summary(phone_number: str, current_summary: str, user_msg: str, ai_msg: str):
    total_messages = await asyncio.to_thread(memory.get_message_count, phone_number)
    if total_messages % SUMMARY_REFRESH_EVERY_N_MESSAGES != 0:
        logger.info(
            f"⏭️ Skipping summary refresh for {phone_number} "
            f"(message #{total_messages}, refreshes every {SUMMARY_REFRESH_EVERY_N_MESSAGES})"
        )
        return

    prompt = f"""
    Current Summary: "{current_summary}"
    New Interaction:
    User: {user_msg}
    AI: {ai_msg}

    Update the summary incorporating any new crucial details (e.g., preferences, issues, context). Keep it factual, bulleted, or a short paragraph. Do not lose vital past context.
    Updated Summary:
    """
    try:
        new_summary = await get_summary_response(prompt)

        await asyncio.to_thread(memory.update_summary, phone_number, new_summary.strip())
        logger.info(f"✅ Updated summary for {phone_number}")
    except GeminiUnavailableError:
        logger.warning(f"⏭️ Skipped summary update for {phone_number} — Gemini unavailable.")
    except Exception as e:
        logger.error(f"❌ Error updating summary for {phone_number}: {e}", exc_info=True)



_PREMIUM_OFFER_COPY = {
    "weight_loss": {
        "hook": "Want a real shot at losing weight, not just tips you'll forget by tomorrow?",
        "plan_name": "Fat Loss Challenge",
        "daily_item": "one specific food, workout, or habit tweak each day, built around your weight-loss goal",
        "adapt_line": "A plan that adapts as you lose weight — it evolves with your progress instead of staying static",
    },
    "bulking": {
        "hook": "Want to actually pack on muscle instead of guessing at your macros?",
        "plan_name": "Lean Bulk Program",
        "daily_item": "one specific meal, lift, or recovery tip each day, built around your bulking goal",
        "adapt_line": "A plan that adapts as you gain — it evolves with your strength and progress instead of staying static",
    },
    "yoga": {
        "hook": "Want a real yoga practice that builds week over week, not random poses?",
        "plan_name": "Yoga Progress Plan",
        "daily_item": "one guided sequence or focus area each day, built around your practice goals",
        "adapt_line": "A plan that adapts as your flexibility and strength improve — it evolves with your progress instead of staying static",
    },
}

_DEFAULT_PREMIUM_OFFER_COPY = {
    "hook": "Want real progress on your health goals, not just generic tips?",
    "plan_name": "Premium Health Plan",
    "daily_item": "one small suggestion or to-do each day based on your health goals",
    "adapt_line": "A running conversation that adapts as you make progress",
}


async def _maybe_send_premium_offer(
    phone_number: str, force_resend: bool = False, required_language: str | None = None
) -> bool:
    """
    Runs on EVERY incoming message and decides whether to send a Razorpay
    payment link for the 21-day premium plan. Three cases:

    1. ACTIVE SUBSCRIPTION (is_premium_active() True):
       Send nothing at all. A paying user is never pitched again while
       their current plan is still running.

    2. PLAN JUST EXPIRED (had a subscription row, expires_at is in the
       past, and we haven't sent the one-time expiry notice for THIS
       subscription yet — subscriptions.expiry_notified is False):
       Send a distinct "your plan has expired" message together with a
       fresh payment link, then mark expiry_notified = TRUE so this exact
       message is sent only once per expiry (not repeated on the user's
       next 50 messages). It gets reset back to False automatically the
       next time they buy again (see activate_subscription), so the
       *next* expiry gets its own one-time notice too.

    3. NEVER SUBSCRIBED, OR ALREADY NOTIFIED OF THIS EXPIRY (falls through
       past case 2): send the normal recurring reminder+payment-link, but
       throttled to at most once per premium_reoffer_min_gap_seconds
       (default 24h) so a non-paying user chatting all day doesn't get a
       fresh link on every single message — they get it roughly once a
       day until they either pay or stop chatting.

       force_resend=True bypasses this throttle: used when the CURRENT
       message is an explicit ask for the plan/link (see
       _requests_premium_plan_explicitly) — a user who directly asks
       "send me the link" should always get one back, even if the
       last unpaid link was sent minutes ago. Re-sends the EXISTING
       unpaid link when there is one (no new Razorpay link/row created)
       rather than generating a fresh link every time they ask, since
       the old one is still valid and paying against it works the same.

    Runs on every message (no session-gap gate) so case 2 fires on the
    user's very next message after expiry, whenever that happens to be —
    it doesn't wait for a "new session".

    Failure here (Razorpay API down, etc.) is logged and swallowed — it
    must never block or break the user's actual conversation.

    Returns True only if a message (expiry notice or offer/resend) was
    actually sent this call — False for case 1, a throttled case-3 skip,
    or any exception. Callers use this to avoid ALSO firing
    _maybe_send_unpaid_plan_reminder in the same turn, which would put
    two separate payment-link messages back to back for one reply.
    """
    try:
        subscription = await asyncio.to_thread(memory.get_subscription, phone_number)

        if subscription:
            expires_at = subscription["expires_at"]
            if expires_at.tzinfo is not None:
                expires_at = expires_at.replace(tzinfo=None)

            if datetime.utcnow() < expires_at:
                # Case 1: still active — say nothing.
                logger.info(f"💎 {phone_number} already has active premium — skipping upsell.")
                return False

            if not subscription.get("expiry_notified"):
                # Case 2: just expired, one-time notice not sent yet.
                logger.info(f"⌛ Premium expired for {phone_number} — sending expiry notice + new link.")
                link = await create_payment_link(
                    phone_number=phone_number,
                    amount_rupees=settings.PREMIUM_PLAN_AMOUNT_RUPEES,
                    description=f"AI Health Assistant — {settings.PREMIUM_PLAN_DAYS}-day Premium",
                )
                await asyncio.to_thread(
                    memory.save_payment_link,
                    link["id"],
                    phone_number,
                    settings.PREMIUM_PLAN_AMOUNT_RUPEES * 100,
                    link.get("short_url"),
                )
                await asyncio.to_thread(memory.mark_subscription_expiry_notified, phone_number)

                expiry_text = (
                    f"Your {settings.PREMIUM_PLAN_DAYS}-day Premium plan has expired. ⌛\n\n"
                    f"Please buy again to keep getting your daily check-ins and "
                    f"priority answers — here's your payment link:\n"
                    f"{link['short_url']}"
                )
                expiry_text = await translate_premium_offer_text(
                    expiry_text, required_language or "English"
                )
                await send_text_message(phone_number, expiry_text)
                await asyncio.to_thread(
                    memory.save_message, phone_number, "assistant", expiry_text, message_type="text"
                )
                logger.info(f"⌛ Sent expiry notice to {phone_number} | link_id={link['id']}")
                return True

        # Case 3: never subscribed, or already notified of this expiry —
        # fall through to the normal throttled recurring reminder below.
        latest_link = await asyncio.to_thread(memory.get_latest_payment_link_for_user, phone_number)
        if latest_link and latest_link.get("status") != "paid" and not force_resend:
            created_at = latest_link.get("created_at")
            if created_at is not None:
                if created_at.tzinfo is not None:
                    created_at = created_at.replace(tzinfo=None)
                seconds_since = (datetime.utcnow() - created_at).total_seconds()
                if seconds_since < settings.PREMIUM_REOFFER_MIN_GAP_SECONDS:
                    logger.info(
                        f"⏭️ Skipping premium offer for {phone_number} — an unpaid link was "
                        f"already sent {seconds_since:.0f}s ago (throttle: "
                        f"{settings.PREMIUM_REOFFER_MIN_GAP_SECONDS}s)."
                    )
                    return False

        resent_existing_link = False
        if latest_link and latest_link.get("status") != "paid" and force_resend:

            logger.info(f"🔁 Explicit request — resending existing unpaid link for {phone_number}.")

            link = {"id": latest_link["payment_link_id"], "short_url": latest_link.get("short_url")}
            resent_existing_link = True
            if not link["short_url"]:
                # Older rows/back-compat safety: if short_url wasn't
                # stored for some reason, fall back to creating a fresh
                # link rather than sending a message with no URL in it.
                link = await create_payment_link(
                    phone_number=phone_number,
                    amount_rupees=settings.PREMIUM_PLAN_AMOUNT_RUPEES,
                    description=f"AI Health Assistant — {settings.PREMIUM_PLAN_DAYS}-day Premium",
                )
                await asyncio.to_thread(
                    memory.save_payment_link,
                    link["id"],
                    phone_number,
                    settings.PREMIUM_PLAN_AMOUNT_RUPEES * 100,
                    link.get("short_url"),
                )
        else:
            logger.info(f"🆕 No active premium for {phone_number} — sending premium offer.")
            link = await create_payment_link(
                phone_number=phone_number,
                amount_rupees=settings.PREMIUM_PLAN_AMOUNT_RUPEES,
                description=f"AI Health Assistant — {settings.PREMIUM_PLAN_DAYS}-day Premium",
            )
            await asyncio.to_thread(
                memory.save_payment_link,
                link["id"],
                phone_number,
                settings.PREMIUM_PLAN_AMOUNT_RUPEES * 100,
                link.get("short_url"),
            )
    except Exception as e:
        logger.error(f"❌ Failed to create/send premium offer for {phone_number}: {e}", exc_info=True)
        return False

    try:
        if not resent_existing_link:

            await asyncio.to_thread(
                memory.save_payment_link,
                link["id"],
                phone_number,
                settings.PREMIUM_PLAN_AMOUNT_RUPEES * 100,
                link.get("short_url"),
            )

        category = await asyncio.to_thread(memory.get_user_category, phone_number)
        copy = _PREMIUM_OFFER_COPY.get(category, _DEFAULT_PREMIUM_OFFER_COPY)

        offer_text = (
            f"Before we dive in — quick heads up 👋\n\n"
            f"{copy['hook']} "
            f"Our {settings.PREMIUM_PLAN_DAYS}-Day {copy['plan_name']} is ₹{settings.PREMIUM_PLAN_AMOUNT_RUPEES} and gives you:\n"
            f"1. A daily action plan for {settings.PREMIUM_PLAN_DAYS} days — {copy['daily_item']}\n"
            f"2. Priority, more detailed answers whenever you're stuck or plateauing\n"
            f"3. {copy['adapt_line']}\n\n"
            f"Totally optional — you can keep chatting normally either way. "
            f"If you'd like to grab it, here's your secure payment link:\n"
            f"{link['short_url']}"
        )
        offer_text = await translate_premium_offer_text(
            offer_text, required_language or "English"
        )
        await send_text_message(phone_number, offer_text)
        await asyncio.to_thread(
            memory.save_message, phone_number, "assistant", offer_text, message_type="text"
        )
        logger.info(f"💳 Sent premium offer to {phone_number} | link_id={link['id']}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to send/save premium offer message for {phone_number}: {e}", exc_info=True)
        return False






_GREETING_WORDS = ("hi", "hii", "hiii", "hello", "hey", "heya", "yo", "namaste")


async def _maybe_send_greeting(phone_number: str) -> bool:
    """
    For a message that does NOT show premium interest (see
    _shows_premium_interest / now classify_premium_intent) — typically a
    first "hii"/"hello" — send a short welcome and immediately KICK OFF
    ONBOARDING (see app.services.onboarding.start_onboarding) instead of
    a static intro. Onboarding no longer waits for payment: the user
    answers the onboarding questions first, those answers are saved
    (handle_onboarding_reply, called earlier in _handle_incoming, is
    what actually captures each answer turn-by-turn), and afterwards the
    bot answers using that profile — up to settings.FREE_QUESTION_LIMIT
    free questions (see the gate in _handle_incoming) before the
    payment-link pitch takes over.

    Skipped entirely for users with an active premium subscription (they
    already have a profile and don't need to be re-onboarded), and only
    fires on this user's FIRST-EVER message (message_count == 0 before
    this turn's save) so a chatty user doesn't get re-onboarded every
    time they happen to say "hii"/"hello" again later.

    Returns True if the welcome + onboarding kickoff were actually sent
    (caller should treat this as the full reply for the turn and stop —
    see call site in _handle_incoming), False if skipped for any reason
    (caller should fall through to normal Q&A handling instead).
    """
    try:
        if await asyncio.to_thread(memory.is_premium_active, phone_number):
            return False

        message_count = await asyncio.to_thread(memory.get_message_count, phone_number)
        if message_count > 0:
            # Not this user's first message — onboarding has already
            # started (or finished) for them; don't restart it.
            return False

        welcome_text = (
            "Hi! 👋 I'm your AI health assistant — happy to help with diet, "
            "workouts, or any health questions you've got.\n\n"
            "Let's quickly get to know you so I can personalize my answers 🙂"
        )

        # ONE message: welcome + first onboarding question (previously
        # three separate outbound messages). Every subsequent user reply is
        # captured by handle_onboarding_reply() (checked first thing in
        # _handle_incoming, before any intent classification) until
        # onboarding is complete.
        await start_onboarding(
            memory, phone_number,
            category=settings.DEFAULT_PLAN_CATEGORY,
            intro_text=welcome_text,
        )

        logger.info(f"👋 Sent welcome + started onboarding for {phone_number}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to send welcome/start onboarding for {phone_number}: {e}", exc_info=True)
        return False


async def _maybe_send_unpaid_plan_reminder(
    phone_number: str, required_language: str | None = None
) -> bool:
    """Send a soft 21-day plan reminder WITH a payment link to users who
    are not premium. Reuses the existing unpaid link if one already
    exists (no duplicate Razorpay link/row created); otherwise creates a
    fresh one.
    """
    try:
        if await asyncio.to_thread(memory.is_premium_active, phone_number):
            return False

        # Reuse existing unpaid link instead of creating a new one every time.
        latest_link = await asyncio.to_thread(
            memory.get_latest_payment_link_for_user, phone_number
        )
        if latest_link and latest_link.get("status") != "paid" and latest_link.get("short_url"):
            short_url = latest_link["short_url"]
        else:
            link = await create_payment_link(
                phone_number=phone_number,
                amount_rupees=settings.PREMIUM_PLAN_AMOUNT_RUPEES,
                description=f"AI Health Assistant — {settings.PREMIUM_PLAN_DAYS}-day Premium",
            )
            await asyncio.to_thread(
                memory.save_payment_link,
                link["id"],
                phone_number,
                settings.PREMIUM_PLAN_AMOUNT_RUPEES * 100,
                link.get("short_url"),
            )
            short_url = link["short_url"]

        reminder_text = (
            "If you want more structure, I also offer a "
            f"{settings.PREMIUM_PLAN_DAYS}-day guided health plan with "
            "daily check-ins and personalized follow-up 👇\n\n"
            f"{short_url}"
        )
        reminder_text = await translate_premium_offer_text(
            reminder_text, required_language or "English"
        )
        await send_text_message(phone_number, reminder_text)
        await asyncio.to_thread(
            memory.save_message, phone_number, "assistant", reminder_text, message_type="text"
        )
        logger.info(f"📣 Sent unpaid plan reminder + link to {phone_number}")
        return True
    except Exception as e:
        logger.error(
            f"❌ Failed to send unpaid plan reminder for {phone_number}: {e}",
            exc_info=True,
        )
        return False


async def _send_payment_status_message(
    phone_number: str, required_language: str | None = None
) -> None:
    """
    Sent when classify_payment_status_intent() detects the user is
    asking about THEIR OWN payment/subscription status (e.g. "is my
    payment done"), instead of the general premium-offer pitch. Looks
    up the real subscription row and replies with a plain done/not-done
    status — no payment link on the "done" branch (they don't need one),
    a payment link on the "not done" branch so they can act right away.
    """
    try:
        subscription = await asyncio.to_thread(memory.get_subscription, phone_number)
        is_active = await asyncio.to_thread(memory.is_premium_active, phone_number)

        if is_active and subscription:
            expires_at = subscription["expires_at"]
            if expires_at.tzinfo is not None:
                expires_at = expires_at.replace(tzinfo=None)
            status_text = (
                f"✅ Your payment is done — your {settings.PREMIUM_PLAN_DAYS}-day "
                f"Premium plan is active until {expires_at.strftime('%d %b %Y')}."
            )
        else:
            # Either no subscription row at all, or one that's expired —
            # both read as "not done" from the user's point of view.
            latest_link = await asyncio.to_thread(
                memory.get_latest_payment_link_for_user, phone_number
            )
            if latest_link and latest_link.get("status") != "paid" and latest_link.get("short_url"):
                short_url = latest_link["short_url"]
            else:
                link = await create_payment_link(
                    phone_number=phone_number,
                    amount_rupees=settings.PREMIUM_PLAN_AMOUNT_RUPEES,
                    description=f"AI Health Assistant — {settings.PREMIUM_PLAN_DAYS}-day Premium",
                )
                await asyncio.to_thread(
                    memory.save_payment_link,
                    link["id"], phone_number,
                    settings.PREMIUM_PLAN_AMOUNT_RUPEES * 100,
                    link.get("short_url"),
                )
                short_url = link["short_url"]

            status_text = (
                f"❌ Your payment is not done yet — you don't have an active "
                f"Premium plan right now.\n\nYou can complete it here:\n{short_url}"
            )

        status_text = await translate_premium_offer_text(
            status_text, required_language or "English"
        )
        await send_text_message(phone_number, status_text)
        await asyncio.to_thread(
            memory.save_message, phone_number, "assistant", status_text, message_type="text"
        )
        logger.info(f"💳 Sent payment status ({'done' if is_active else 'not done'}) to {phone_number}")
    except Exception as e:
        logger.error(f"❌ Failed to send payment status for {phone_number}: {e}", exc_info=True)


async def _send_free_limit_message(
    phone_number: str, is_first_hit: bool, required_language: str | None = None
) -> None:
    """
    Sent INSTEAD OF an actual answer once a non-premium user has used up
    settings.FREE_QUESTION_LIMIT free questions (see the gate in
    _handle_incoming, right before generate_context_aware_response is
    called). No LLM call is spent answering the question itself.

    is_first_hit=True is sent right AFTER the last free answer is
    delivered (question #FREE_QUESTION_LIMIT; see
    _handle_free_quota_after_answer) — gets the "you've used up your
    free questions" framing. Every question after that (is_first_hit=
    False) gets a differently-worded repeat so it doesn't read like a
    stuck bot repeating itself, while still always including the app
    features + payment link.

    Reuses an existing unpaid payment link when one is already on file,
    same pattern as _maybe_send_unpaid_plan_reminder, instead of minting
    a fresh Razorpay link on every single blocked question.

    HARDENED: this used to be one big try/except around the whole body
    that only logged on failure — a Razorpay error, a category-lookup
    error, or a translation error meant the user got NOTHING back after
    crossing the free-question limit (looked like the bot had gone
    silent). Each risky step now degrades gracefully instead of
    aborting the whole message.
    """
    short_url = None
    try:
        latest_link = await asyncio.to_thread(
            memory.get_latest_payment_link_for_user, phone_number
        )
        if latest_link and latest_link.get("status") != "paid" and latest_link.get("short_url"):
            short_url = latest_link["short_url"]
        else:
            link = await create_payment_link(
                phone_number=phone_number,
                amount_rupees=settings.PREMIUM_PLAN_AMOUNT_RUPEES,
                description=f"AI Health Assistant — {settings.PREMIUM_PLAN_DAYS}-day Premium",
            )
            await asyncio.to_thread(
                memory.save_payment_link,
                link["id"], phone_number,
                settings.PREMIUM_PLAN_AMOUNT_RUPEES * 100,
                link.get("short_url"),
            )
            short_url = link["short_url"]
    except Exception as e:
        logger.error(
            f"⚠️ Could not get/create payment link for {phone_number} — "
            f"sending free-limit message without a link instead of nothing: {e}",
            exc_info=True,
        )
        short_url = None

    try:
        category = await asyncio.to_thread(memory.get_user_category, phone_number)
    except Exception as e:
        logger.warning(f"⚠️ Failed to look up category for {phone_number}, using default copy: {e}")
        category = None
    copy = _PREMIUM_OFFER_COPY.get(category, _DEFAULT_PREMIUM_OFFER_COPY)

    link_line = (
        f"Pay here to keep going:\n{short_url}"
        if short_url
        else "Reply here and we'll get a payment link sent to you to keep going."
    )

    if is_first_hit and settings.FREE_QUESTION_LIMIT <= 0:
        # No free questions at all: this is the message right after
        # onboarding finishes (see onboarding._finish_onboarding).
        text = (
            f"Perfect, thank you! 🙏 I've got everything I need to personalize your plan.\n\n"
            f"To get started, grab our {settings.PREMIUM_PLAN_DAYS}-Day "
            f"{copy['plan_name']} for ₹{settings.PREMIUM_PLAN_AMOUNT_RUPEES} — you get:\n"
            f"1. A daily action plan for {settings.PREMIUM_PLAN_DAYS} days — {copy['daily_item']}\n"
            f"2. Priority, more detailed answers whenever you're stuck or plateauing\n"
            f"3. {copy['adapt_line']}\n\n"
            f"{link_line}"
        )
    elif is_first_hit:
        text = (
            f"You've used up your {settings.FREE_QUESTION_LIMIT} free questions "
            f"for now. 🙌\n\n"
            f"To keep getting answers, grab our {settings.PREMIUM_PLAN_DAYS}-Day "
            f"{copy['plan_name']} for ₹{settings.PREMIUM_PLAN_AMOUNT_RUPEES} — you get:\n"
            f"1. A daily action plan for {settings.PREMIUM_PLAN_DAYS} days — {copy['daily_item']}\n"
            f"2. Priority, more detailed answers whenever you're stuck or plateauing\n"
            f"3. {copy['adapt_line']}\n\n"
            f"{link_line}"
        )
    else:
        text = (
            f"Still here whenever you're ready 🙂\n\n"
            f"Premium unlocks full access, a personalized "
            f"{settings.PREMIUM_PLAN_DAYS}-day plan with daily check-ins, and "
            f"priority answers — all for ₹{settings.PREMIUM_PLAN_AMOUNT_RUPEES}.\n\n"
            f"{link_line}"
        )

    try:
        text = await translate_premium_offer_text(text, required_language or "English")
    except Exception as e:
        logger.warning(
            f"⚠️ Failed to translate free-limit message for {phone_number}, "
            f"sending untranslated instead of nothing: {e}"
        )

    try:
        await send_text_message(phone_number, text)
        logger.info(
            f"🚫 Free question limit reached for {phone_number} "
            f"(first_hit={is_first_hit}) — sent app features + link instead of an answer."
        )
    except Exception as e:
        logger.error(f"❌ Failed to send free-limit message for {phone_number}: {e}", exc_info=True)
        return

    try:
        await asyncio.to_thread(
            memory.save_message, phone_number, "assistant", text, message_type="text"
        )
    except Exception as e:
        logger.error(
            f"⚠️ Free-limit message was sent to {phone_number} but failed to save to history: {e}",
            exc_info=True,
        )


async def _handle_free_quota_after_answer(
    phone_number: str,
    required_language: str | None = None,
    link_already_sent: bool = False,
) -> bool:
    """
    Runs right AFTER a real answer was sent. For non-premium users it
    counts the question against settings.FREE_QUESTION_LIMIT and, ONLY when
    this answer was the last free one (e.g. question 5 of 5), sends the
    payment link message.

    Questions 1..LIMIT-1 get NO extra message at all — just the answer
    (the old per-answer "plan reminder + payment link" and the cross-
    question follow-up are gone, since Meta bills every outbound message).

    Returns True if the user is in the free phase (non-premium); callers
    use that to skip the optional follow-up message. Returns False for
    premium users (nothing counted, nothing sent).
    """
    if await asyncio.to_thread(memory.is_premium_active, phone_number):
        return False

    new_count = await asyncio.to_thread(memory.increment_free_question_count, phone_number)
    if new_count == settings.FREE_QUESTION_LIMIT and not link_already_sent:
        try:
            await _send_free_limit_message(
                phone_number, is_first_hit=True, required_language=required_language
            )
        except Exception as e:
            logger.error(f"❌ Failed to send payment message after last free question for {phone_number}: {e}", exc_info=True)
    return True


async def _flush_pending_checkin_day(phone_number: str, plan_day: dict, prefix: str = "") -> bool:
    """
    Push a premium plan day's REAL content as free-form text, right after
    this user's WhatsApp session window just reopened (they sent
    something — any message, any type). Counterpart to the re-engagement
    template sent by daily_checkin.py when the window was closed at the
    scheduled check-in time: that template only nudges the user; this is
    what actually delivers the day's message + follow-up question,
    exactly like _send_checkin_for_user does when the window was already
    open. Called from _handle_incoming before that message is otherwise
    processed, so the plan never falls further behind schedule just
    because a user didn't message for a day or two.
    """
    day_number = plan_day["day_number"]
    claimed = await asyncio.to_thread(
        memory.claim_plan_day_for_send, phone_number, day_number
    )
    if not claimed:
        logger.info(
            f"⏭️ Held day {day_number} for {phone_number} is already being sent or was sent; "
            "skipping duplicate delivery."
        )
        return False

    return await send_plan_day(memory, phone_number, plan_day, prefix=prefix)


async def _send_ack_or_release_next_day(phone_number: str, ack: str) -> None:
    """
    After the user confirmed a task (button tap or text reply): if the
    daily job already HELD the next day because this confirmation was
    missing, release it NOW with the acknowledgement glued in front of it
    (one message instead of two); otherwise just send the acknowledgement.
    """
    try:
        if await asyncio.to_thread(memory.is_premium_active, phone_number):
            pending_day = await asyncio.to_thread(memory.get_pending_template_nudge_day, phone_number)
            if pending_day:
                sent = await _flush_pending_checkin_day(
                    phone_number, pending_day, prefix=f"{ack}\n\n"
                )
                if sent:
                    return
    except Exception as e:
        logger.error(f"❌ Failed to release held day for {phone_number}: {e}", exc_info=True)

    await send_text_message(phone_number, ack)
    await asyncio.to_thread(memory.save_message, phone_number, "assistant", ack, message_type="text")


async def _handle_interactive_reply(sender: str, interactive: dict) -> None:
    """
    A button tap from the daily "Done / Not done" message
    (see app/services/plan_delivery.py). Records the status for that plan
    day — which is what unlocks tomorrow's message — and acknowledges it.
    The tap is also saved as an inbound message so the 24h service window
    is correctly tracked from it.
    """
    button = (interactive or {}).get("button_reply") or {}
    button_id = button.get("id") or ""
    title = button.get("title") or button_id

    if button_id.startswith(TASK_DONE_PREFIX):
        status, prefix, ack = "done", TASK_DONE_PREFIX, settings.TASK_CONFIRM_DONE_ACK
    elif button_id.startswith(TASK_NOT_DONE_PREFIX):
        status, prefix, ack = "not_done", TASK_NOT_DONE_PREFIX, settings.TASK_CONFIRM_NOT_DONE_ACK
    else:
        logger.info(f"ℹ️ Ignoring unknown interactive reply from {sender}: {button_id!r}")
        return

    try:
        day_number = int(button_id[len(prefix):])
    except ValueError:
        logger.warning(f"⚠️ Bad task-confirmation button id from {sender}: {button_id!r}")
        return

    await asyncio.to_thread(
        memory.save_message, sender, "user", f"[Button]: {title}", message_type="text"
    )

    plan_day = await asyncio.to_thread(memory.get_premium_plan_day, sender, day_number)
    if plan_day and plan_day.get("task_status") is not None:
        # Already confirmed (double tap / old button) — stay silent, every
        # outbound message is billed.
        logger.info(f"⏭️ Day {day_number} already confirmed for {sender} — ignoring repeat tap.")
        return

    if not await asyncio.to_thread(memory.save_task_confirmation, sender, day_number, status):
        logger.warning(f"⚠️ Confirmation for unknown/unsent day {day_number} from {sender}.")
        return

    logger.info(f"✅ {sender} confirmed day {day_number}: {status}")
    await _send_ack_or_release_next_day(sender, ack)


async def maybe_send_followup(
    phone_number: str,
    customer_summary: str,
    context_text: str,
    user_message: str,
    assistant_reply: str,
    whisper_language: str | None = None,
    required_language: str | None = None,
) -> None:
    """
    Background task: cross-question the user based on the conversation so
    far. If the LLM decides a precise, relevant follow-up question or
    suggestion applies, send it as a short separate WhatsApp message right
    after the main reply, and record it so we don't repeat it later.

    Runs after the main reply has already been sent, so a slow/failed call
    here never delays the user's actual answer.

    whisper_language: for voice-triggered replies, Whisper's detected
        source language — passed through so the follow-up matches the
        same language decision as the main reply.
    required_language: the language already detected for this turn (by
        generate_context_aware_response()/detect_reply_language()) — reused
        here instead of generate_followup_suggestion() re-detecting the
        same deterministic result via a second Gemini call. If not given,
        detection still runs as before (keeps this function safe to call
        standalone).

    LOAD SHEDDING: this follow-up is a cosmetic nice-to-have, not part of
    the core reply the user is waiting for. If Gemini is currently at its
    concurrency limit (is_gemini_busy()), skip it entirely for this turn
    rather than adding more contenders for the same limited slots — this
    protects main replies (and other users' requests) from queueing behind
    a feature nobody is blocked on. Under normal load this never triggers
    and the follow-up behaves exactly as before.
    """

    if _looks_like_a_question(assistant_reply):
        logger.info(
            f"⏭️ Skipping follow-up for {phone_number} — main reply is "
            f"already a short question, avoiding a duplicate/second question."
        )
        return

    if await is_gemini_busy():
        logger.info(f"⏭️ Skipping follow-up for {phone_number} — Gemini busy, protecting main replies.")
        return

    try:
        recent = await asyncio.to_thread(memory.get_recent_followups, phone_number, limit=5)
        suggestion = await generate_followup_suggestion(
            customer_summary=customer_summary,
            context_text=context_text,
            user_message=user_message,
            assistant_reply=assistant_reply,
            recent_suggestions=recent,
            whisper_language=whisper_language,
            required_language=required_language,
        )
        if not suggestion:
            return

        await send_text_message(phone_number, suggestion)
        await asyncio.to_thread(memory.save_followup_suggestion, phone_number, suggestion)
        await asyncio.to_thread(
            memory.save_message, phone_number, "assistant", suggestion, message_type="text"
        )
        logger.info(f"❓ Follow-up → [{phone_number}]: {suggestion[:100]}")
    except GeminiUnavailableError:
        logger.warning(f"⏭️ Skipped follow-up for {phone_number} — Gemini unavailable.")
    except Exception as e:
        logger.error(f"❌ Error generating/sending follow-up for {phone_number}: {e}", exc_info=True)


def _safe_enqueue(raw_msg: dict) -> None:
    """BackgroundTasks-safe wrapper around enqueue_incoming().

    enqueue_incoming() already retries and dead-letters on failure (see
    app/queueing.py), so this should basically never raise. But
    BackgroundTasks gives zero visibility if something still does — no
    crash, no logged traceback by default — so this makes sure any
    residual exception is at least logged loudly instead of vanishing the
    way the original message did in the incident this fix addresses.
    """
    try:
        enqueue_incoming(raw_msg)
    except Exception:
        logger.critical(
            "🔥 _safe_enqueue: enqueue_incoming raised even after its own "
            "retry + dead-letter fallback — message may be lost. "
            f"raw_msg={raw_msg}",
            exc_info=True,
        )


# Webhook POST — incoming WhatsApp events
@app.post("/webhook")
async def receive_message(request: Request, background_tasks: BackgroundTasks):
    raw = await request.body()
    if not raw:
        return {"status": "ok"}

    try:
        body = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning(f"Invalid JSON in webhook: {e}")
        return {"status": "ok"}

    logger.debug(f"📨 Webhook received:\n{json.dumps(body, indent=2)}")

    try:
        payload = WebhookPayload(**body)
    except Exception as e:
        logger.warning(f"Payload parse error: {e}")
        return {"status": "ok"}

    for entry in payload.entry:
        for change in entry.changes:
            value = change.value

            if value.statuses:
                for s in value.statuses:
                    if s.errors:
                        error_detail = "; ".join(
                            f"code={e.code} title={e.title!r} message={e.message!r}"
                            for e in s.errors
                        )
                        logger.info(
                            f"📊 Status: {s.status} | to={s.recipient_id} | errors: {error_detail}"
                        )
                        if any(e.code == 131047 for e in s.errors):
                            logger.warning(
                                f"⏰ Message to {s.recipient_id} failed with code 131047 — "
                                f"more than 24h since their last reply. This is the WhatsApp "
                                f"session-window issue (see daily_checkin.py's re-engagement "
                                f"template handling)."
                            )
                    else:
                        logger.info(f"📊 Status: {s.status} | to={s.recipient_id}")
                continue

            if not value.messages:
                continue


            if value.contacts:
                for contact in value.contacts:
                    wa_id = contact.wa_id
                    profile = contact.profile or {}
                    name = profile.get("name") if isinstance(profile, dict) else None
                    if wa_id and name:
                        background_tasks.add_task(memory.set_user_name, wa_id, name)

            for raw_msg in value.messages:

                background_tasks.add_task(_safe_enqueue, raw_msg)

    return {"status": "ok"}


# PhonePe webhook — fires on checkout.order.completed / checkout.order.failed.
# Configure this URL in the PhonePe Business Dashboard -> Webhooks, and set the
# same username/password you enter there as PHONEPE_WEBHOOK_USERNAME and
# PHONEPE_WEBHOOK_PASSWORD in .env.
_bg_tasks: set = set()


def _spawn_background(coro) -> None:
    task = asyncio.create_task(coro)
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)


async def _post_payment_flow(phone_number: str) -> None:
    """Onboarding / plan generation after a successful payment (background)."""
    try:
        # Onboarding now normally already ran (for FREE) when this user
        # first said "hi" (see _maybe_send_greeting), well before they
        # ever paid. Payment landing is what should trigger the actual
        # paid plan generation + Day 1 send now — see
        # app.services.onboarding.generate_and_send_plan().
        session = await asyncio.to_thread(memory.get_onboarding_session, phone_number)
        if session and session.get("is_complete"):
            # Normal case: onboarding already finished, so generate and
            # send the paid plan right now using those saved answers.
            await generate_and_send_plan(memory, phone_number, session=session)
        elif session and not session.get("is_complete"):
            # User is mid-onboarding when payment landed. Nothing to do
            # here — _finish_onboarding() checks is_premium_active() the
            # moment they answer the last question and will generate the
            # plan itself right then, since the subscription is already
            # active by that point.
            logger.info(
                f"ℹ️ {phone_number} paid mid-onboarding — plan will "
                f"generate automatically once they finish answering."
            )
        else:
            # Edge case: payment landed without the user ever messaging
            # the bot first, so onboarding never had a chance to start.
            # Kick it off now — _finish_onboarding will see this user is
            # already premium and generate the plan itself once
            # onboarding finishes.
            await start_onboarding(memory, phone_number, category=settings.DEFAULT_PLAN_CATEGORY)
    except Exception as e:
        logger.error(f"❌ Failed to trigger onboarding/plan generation for {phone_number}: {e}", exc_info=True)


@app.post("/razorpay/webhook")
async def razorpay_webhook(request: Request):
    raw = await request.body()
    signature = request.headers.get("X-Razorpay-Signature", "")

    if not verify_webhook_signature(raw, signature):
        logger.warning("⛔ Razorpay webhook: invalid signature — rejecting.")
        raise HTTPException(status_code=400, detail="Invalid signature")

    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Razorpay webhook: invalid JSON body.")
        return {"status": "ok"}

    event = body.get("event", "")
    logger.info(f"💰 Razorpay webhook event: {event}")


    if event != "payment_link.paid":
        return {"status": "ok"}

    try:
        payload = body["payload"]
        payment_link_entity = payload["payment_link"]["entity"]
        payment_entity = payload["payment"]["entity"]

        payment_link_id = payment_link_entity.get("id") or payment_entity.get("payment_link_id")
        razorpay_payment_id = payment_entity.get("id")
    except (KeyError, TypeError) as e:
        logger.error(f"❌ Razorpay webhook: unexpected payload shape: {e} | body={body}")
        return {"status": "ok"}

    if not payment_link_id or not razorpay_payment_id:
        logger.error(
            "❌ Razorpay webhook: missing payment_link_id or payment id in payload"
            f" | payload={payload}"
        )
        return {"status": "ok"}

    phone_number = await asyncio.to_thread(
        memory.mark_payment_link_paid, payment_link_id, razorpay_payment_id
    )
    if not phone_number:
        existing_payment = await asyncio.to_thread(
            memory.get_payment_link, payment_link_id
        )
        if existing_payment and existing_payment.get("status") == "paid":
            logger.info(
                f"⏭️ Duplicate payment webhook for {payment_link_id}, skipping."
            )
            return {"status": "ok"}

        logger.warning(
            f"⚠️ Razorpay webhook: no local record for payment_link_id={payment_link_id}"
        )
        raw_phone_number = None
        customer = payment_link_entity.get("customer") or {}
        if isinstance(customer, dict):
            raw_phone_number = customer.get("contact")
        if not raw_phone_number:
            notes = payment_link_entity.get("notes") or {}
            raw_phone_number = notes.get("phone_number")
        if not raw_phone_number:
            logger.warning(
                "⚠️ Razorpay webhook: could not recover phone number from payload"
            )
            return {"status": "ok"}

        phone_number = str(raw_phone_number).strip()
        if phone_number.startswith("+"):
            phone_number = phone_number[1:]

        logger.info(
            f"ℹ️ Recovered phone number from webhook payload: {phone_number}"
        )
        if payment_link_entity:
            amount_paise = payment_link_entity.get("amount") or 0
            short_url = payment_link_entity.get("short_url")
            try:
                await asyncio.to_thread(
                    memory.save_payment_link,
                    payment_link_id,
                    phone_number,
                    amount_paise,
                    short_url,
                )
                phone_number = await asyncio.to_thread(
                    memory.mark_payment_link_paid,
                    payment_link_id,
                    razorpay_payment_id,
                )
            except Exception as e:
                logger.error(
                    f"❌ Failed to recover payment link record for {payment_link_id}: {e}",
                    exc_info=True,
                )
                return {"status": "ok"}

        if not phone_number:
            return {"status": "ok"}

    expires_at = await asyncio.to_thread(
        memory.activate_subscription,
        phone_number,
        settings.PREMIUM_PLAN_DAYS,
        payment_link_id,
    )
    logger.info(f"✅ Activated {settings.PREMIUM_PLAN_DAYS}-day premium for {phone_number}, expires {expires_at}")

    confirm_text = (
        f"Payment received, thank you! 🎉\n\n"
        f"Your {settings.PREMIUM_PLAN_DAYS}-day Premium plan is now active. ⏳ "
        f"I'm preparing your personalized plan right now — Day 1 will arrive in this "
        f"chat in about a minute."
    )
    try:
        await send_text_message(phone_number, confirm_text)
        await asyncio.to_thread(
            memory.save_message, phone_number, "assistant", confirm_text, message_type="text"
        )
    except Exception as e:
        # Subscription is already activated in DB even if this confirmation
        # message fails to send — don't let a WhatsApp API hiccup undo the
        # payment activation or make the webhook look like it failed to Razorpay.
        logger.error(f"❌ Failed to send payment confirmation to {phone_number}: {e}", exc_info=True)


    # Plan generation (one big Gemini call, often 30-120s) must NOT run
    # inside the webhook request: Razorpay times out and retries, and a
    # worker restart mid-request would silently lose the plan. Run it in the
    # background and answer Razorpay immediately; the repair job in
    # daily_checkin.py retries if this ever fails.
    _spawn_background(_post_payment_flow(phone_number))

    return {"status": "ok"}


# Where PhonePe sends the user's browser once they finish (or abandon) checkout.
# PhonePe requires a redirectUrl on every order, but this page is purely
# informational: activation happens in /phonepe/webhook, which is the only
# thing we trust. Set PHONEPE_REDIRECT_URL to this endpoint's public URL.
@app.api_route("/phonepe/redirect", methods=["GET", "POST"], response_class=HTMLResponse)
async def phonepe_redirect():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Payment status</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
           display: flex; align-items: center; justify-content: center; min-height: 100vh;
           margin: 0; background: #f5f6f8; color: #1c1e21; text-align: center; }
    .card { background: #fff; padding: 32px 28px; border-radius: 12px; max-width: 380px;
            box-shadow: 0 2px 12px rgba(0,0,0,.08); }
    h1 { font-size: 20px; margin: 0 0 12px; }
    p { font-size: 15px; line-height: 1.5; margin: 0; color: #4b4f56; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Thanks! We're confirming your payment.</h1>
    <p>You can close this page and head back to WhatsApp &mdash;
       we'll message you there as soon as your plan is active.</p>
  </div>
</body>
</html>"""


# Per-message scratchpad shared between _handle_incoming (wrapper) and
# _handle_incoming_core: lets the core flag "this was the user's last
# allowed message today" so the wrapper sends the limit reminder AFTER the
# answer has been delivered.
_limit_state: ContextVar[dict | None] = ContextVar("daily_limit_state", default=None)


def _limit_reached_now() -> bool:
    state = _limit_state.get()
    return bool(state and state.get("remind"))


async def _daily_limit_blocks(sender: str, exempt_followup: bool = False) -> bool:
    """
    True -> the user is over today's limit: the caller must drop the message
    silently (no reply, no cost). Applies to PAID (premium) users only.
    Only real chat messages are counted; button
    taps, onboarding answers and replies to the daily follow-up question are
    never routed through here (they unlock tomorrow's plan, so they must
    always work).
    """
    try:
        # The daily limit only starts AFTER payment. Unpaid users are
        # already capped by settings.FREE_QUESTION_LIMIT (5 free questions
        # total), so they are neither counted nor blocked here.
        if not await asyncio.to_thread(memory.is_premium_active, sender):
            return False
        if exempt_followup and await asyncio.to_thread(memory.get_awaiting_followup_day, sender):
            return False
        status = await daily_limit.register_message(sender)
    except Exception as e:
        logger.error(f"❌ Daily-limit check crashed for {sender} (allowing): {e}", exc_info=True)
        return False

    if status == daily_limit.BLOCKED:
        logger.info(f"🚫 {sender} is over the daily message limit — message ignored.")
        return True
    if status == daily_limit.REACHED:
        state = _limit_state.get()
        if state is not None:
            state["remind"] = True
            state["sender"] = sender
    return False


async def _handle_incoming(raw_msg: dict) -> None:
    """Entry point used by the queue workers: runs the normal message
    handling, then — if that message was the user's last allowed one today —
    sends the single "daily limit reached, resets tomorrow" reminder."""
    state = {"remind": False, "sender": None}
    token = _limit_state.set(state)
    try:
        await _handle_incoming_core(raw_msg)
    finally:
        _limit_state.reset(token)

    if state["remind"] and state["sender"]:
        try:
            limit = await daily_limit.get_limit()
            text = daily_limit.reached_message(limit)
            await send_text_message(state["sender"], text)
            await asyncio.to_thread(
                memory.save_message, state["sender"], "assistant", text, message_type="text"
            )
            logger.info(f"⚠️ Daily limit ({limit}) reached for {state['sender']} — reminder sent.")
        except Exception as e:
            logger.error(f"❌ Failed to send daily-limit reminder: {e}", exc_info=True)


async def _handle_incoming_core(raw_msg: dict) -> None:
    """Enhanced message handler with context awareness and audio support.

    Runs as a detached background task (scheduled from the webhook handler),
    so it must not rely on a request-bound BackgroundTasks instance for its
    own fire-and-forget work — we use asyncio.create_task for that instead.
    """
    try:
        msg = IncomingMessage.from_raw(raw_msg)
    except Exception as e:
        logger.error(f"❌ Cannot parse message: {e} | raw={raw_msg}")
        return

    sender = msg.from_


    if not await try_mark_message_processed(msg.id):
        logger.info(f"⏭️ Duplicate webhook delivery for message id={msg.id}, skipping.")
        return


    try:
        age_seconds = time.time() - int(msg.timestamp)
    except (TypeError, ValueError):
        age_seconds = 0
    if age_seconds > _MAX_MESSAGE_AGE_SECONDS:
        logger.info(
            f"⏭️ Dropping stale message id={msg.id} from {sender} "
            f"(age={age_seconds:.0f}s > {_MAX_MESSAGE_AGE_SECONDS}s)."
        )
        return

    logger.info(f"📱 From={sender} | type={msg.type} | id={msg.id}")

    # This inbound message (any type) just reopened this user's 24h
    # WhatsApp session window. If a premium daily check-in was held back
    # earlier because the window was closed (see daily_checkin.py — it
    # sends a re-engagement template instead of silently-undelivered free
    # text in that case), push the real content now, before anything else.
    try:
        if await asyncio.to_thread(memory.is_premium_active, sender):
            pending_day = await asyncio.to_thread(memory.get_pending_template_nudge_day, sender)
            if pending_day:
                await _flush_pending_checkin_day(sender, pending_day)
    except Exception as e:
        logger.error(f"❌ Failed to check/flush pending check-in for {sender}: {e}", exc_info=True)

    background_tasks: list[asyncio.Task[None]] = []


    background_tasks.append(asyncio.create_task(mark_as_read(msg.id, show_typing=True)))

    # ============ TEXT MESSAGE ============
    if msg.type == "text" and msg.text:
        user_text = msg.text.body.strip()
        logger.info(f"👤 [{sender}]: {user_text}")


        # Onboarding answers MUST be captured before any intent
        # classification runs. Otherwise a reply like "8am" to onboarding
        # question 8 ("What time should I send your daily check-in?") gets
        # fed into classify_premium_intent() along with recent context that
        # literally contains the words "premium"/"check-in"/"plan" (the
        # question prompt itself), which can flag it as premium_related /
        # explicit_request and `return` before handle_onboarding_reply ever
        # runs — silently stranding the user on question 8 forever, with no
        # plan ever generated and therefore no Day 2+ check-ins possible.
        # An in-progress onboarding session should always win over any other
        # routing.
        try:
            consumed = await handle_onboarding_reply(memory, sender, user_text)
            if consumed:
                return
        except Exception as e:
            logger.error(f"❌ Onboarding reply handling failed for {sender}: {e}", exc_info=True)

        # Daily chat limit (onboarding answers above are exempt; so are
        # replies to the daily follow-up question).
        if await _daily_limit_blocks(sender, exempt_followup=True):
            return

        # Tracks whether a payment-link message already went out earlier
        # in THIS turn (e.g. the premium-offer branch below), so we can
        # skip _maybe_send_unpaid_plan_reminder later and never send two
        # separate payment-link messages for one reply.
        premium_link_already_sent_this_turn = False

        recent_context_for_intent = await asyncio.to_thread(
            memory.get_conversation_context, sender, limit=5
        )
        premium_intent = await classify_premium_intent(
            user_text, recent_context=recent_context_for_intent
        )
        if premium_intent["premium_related"]:
            # Narrow further: is this specifically "is my payment done?"
            # rather than general interest/pricing/an explicit buy
            # request? Those two need completely different replies (a
            # real status lookup vs. the sales pitch), so this only runs
            # once we already know the message is premium-related at all.
            payment_status_intent = await classify_payment_status_intent(
                user_text, recent_context=recent_context_for_intent
            )
            if payment_status_intent["payment_status_query"]:
                await asyncio.to_thread(
                    memory.save_message, sender, "user", user_text, message_type="text"
                )
                status_required_language = await detect_reply_language(user_text)
                await _send_payment_status_message(
                    sender, required_language=status_required_language
                )
                return

            explicit_request = premium_intent["explicit_request"]
            offer_required_language = await detect_reply_language(user_text)
            premium_link_already_sent_this_turn = await _maybe_send_premium_offer(
                sender, force_resend=explicit_request, required_language=offer_required_language
            )
            if explicit_request:

                await asyncio.to_thread(
                    memory.save_message, sender, "user", user_text, message_type="text"
                )
                return
        else:
            greeted = await _maybe_send_greeting(sender)
            if greeted:

                await asyncio.to_thread(
                    memory.save_message, sender, "user", user_text, message_type="text"
                )
                return


        try:
            awaiting = await asyncio.to_thread(memory.get_awaiting_followup_day, sender)
            if awaiting:
                await asyncio.to_thread(
                    memory.save_followup_answer, sender, awaiting["day_number"], user_text
                )
                await asyncio.to_thread(
                    memory.save_message, sender, "user", user_text, message_type="text"
                )
                logger.info(
                    f"📝 Logged day {awaiting['day_number']} follow-up answer for {sender}: "
                    f"'{user_text[:80]}'"
                )
                ack = "Thanks for the update! 👍 Keep it up, and I'll check in again tomorrow."
                await _send_ack_or_release_next_day(sender, ack)
                return
        except Exception as e:
            logger.error(f"❌ Follow-up capture failed for {sender}: {e}", exc_info=True)

        # Handle special commands
        if user_text.lower() in ("/reset", "/clear", "reset", "clear"):
            await asyncio.to_thread(memory.update_summary, sender, "")
            stats = await asyncio.to_thread(memory.get_user_stats, sender)
            logger.info(f"🗑️ Cleared conversation for {sender}. Stats: {stats}")
            await send_text_message(sender, "Conversation cleared! How can I help you with your health today?")
            return
        
        if user_text.lower() == "/help":
            summary = """
🤖 AI Health Assistant — Help

Just tell me what's going on with your health, and I'll ask a few quick
questions if I need more context, then give you clear, practical guidance.

Commands:
  • /reset — clear our conversation and start fresh
  • /stats — see your conversation statistics
  • /models — see the AI models powering this bot

Note: I'm an AI assistant, not a doctor. For diagnosis, prescriptions, or
anything urgent, please see a licensed healthcare professional.
            """
            await send_text_message(sender, summary.strip())
            return

        if user_text.lower() == "/models":
            # Send summary instead of full list (WhatsApp message limit)
            summary = """
📊 Available Models:

🎙️ Audio Models:
  • Whisper Large V3 (transcription)
  • TTS-1 & TTS-1-HD (text-to-speech)
  • Google Cloud TTS & AWS Polly

💬 Text Models:
  • GPT-4 & GPT-4 Mini
  • Claude 3 (Opus/Sonnet/Haiku)
  • Gemini Pro
  • Mistral Large
  • LLaMA 2

Type /stats to see your conversation statistics.
            """
            await send_text_message(sender, summary.strip())
            return
        
        if user_text.lower() == "/stats":
            stats = await asyncio.to_thread(memory.get_user_stats, sender)
            customer_data = await asyncio.to_thread(memory.get_customer, sender)
            stats_msg = f"""
📈 Your Statistics:
  • Total messages: {stats['total_messages']}
  • Your messages: {stats['user_messages']}
  • Bot responses: {stats['assistant_messages']}
  • Audio messages: {stats['audio_messages']}
  • Last message: {customer_data.get('last_message_at', 'Never')[:19]}
            """
            await send_text_message(sender, stats_msg.strip())
            return

        # Free-question gate: a non-premium user gets settings.
        # FREE_QUESTION_LIMIT questions actually answered. Every question
        # after that gets the payment link + app features instead of an
        # answer — no Gemini call spent on it. Checked BEFORE calling the
        # LLM (not after) so a blocked question never touches Gemini.
        if not await asyncio.to_thread(memory.is_premium_active, sender):
            free_count = await asyncio.to_thread(memory.get_free_question_count, sender)
            if free_count >= settings.FREE_QUESTION_LIMIT:
                await asyncio.to_thread(
                    memory.save_message, sender, "user", user_text, message_type="text"
                )
                try:
                    gate_required_language = await detect_reply_language(user_text)
                except Exception as e:
                    logger.warning(f"⚠️ Language detection failed for {sender}, defaulting to English: {e}")
                    gate_required_language = None
                await _send_free_limit_message(
                    sender,
                    is_first_hit=False,
                    required_language=gate_required_language,
                )
                return

        # Generate context-aware response
        try:
            reply, required_language = await generate_context_aware_response(sender, user_text, is_audio=False)
        except GeminiUnavailableError:
            # Gemini is down/overloaded — don't save anything to DB (so no
            # junk "sorry" turns pollute history/context), but do let the
            # user know their message didn't get lost, instead of silence.
            logger.warning(f"⛔ Gemini unavailable — dropping query from {sender}: '{user_text[:80]}'")
            await send_text_message(
                sender,
                "Sorry, I'm a bit overloaded right now. Please try sending that again in a moment 🙏",
            )
            return


        try:
            await asyncio.to_thread(memory.save_message, sender, "user", user_text, message_type="text")
            await asyncio.to_thread(memory.save_message, sender, "assistant", reply, message_type="text")
        except Exception as e:
            logger.error(f"❌ Failed to persist chat history for {sender}: {e}", exc_info=True)


        try:
            await send_text_message(sender, reply)
            logger.info(f"🤖 → [{sender}]: {reply[:100]}...")
        except Exception as e:
            logger.error(f"❌ Failed to send reply: {e}", exc_info=True)

        # Free-question accounting: nothing extra is sent for questions
        # 1..LIMIT-1; the payment link goes out right after the LIMIT-th
        # answer. premium_link_already_sent_this_turn avoids a duplicate if
        # the user explicitly asked for the plan in this same message.
        in_free_phase = await _handle_free_quota_after_answer(
            sender,
            required_language=required_language,
            link_already_sent=premium_link_already_sent_this_turn,
        )

        # Update summary in background
        customer_data = await asyncio.to_thread(memory.get_customer, sender)
        background_tasks.append(asyncio.create_task(
            generate_new_summary(
                sender,
                customer_data.get("summary", ""),
                user_text,
                reply
            )
        ))

        # Cross-question the user with a precise, context-grounded
        # follow-up (or suggestion of what to ask next) — fire-and-forget
        # so it never delays the primary reply.
        # (skipped during the free phase: it's one more billed message)
        if not in_free_phase and not _limit_reached_now():
            context_for_followup = await asyncio.to_thread(
                memory.get_conversation_context, sender, limit=5
            )
            background_tasks.append(asyncio.create_task(
                maybe_send_followup(
                    sender,
                    customer_data.get("summary", ""),
                    context_for_followup,
                    user_text,
                    reply,
                    required_language=required_language,
                )
            ))

    # ============ AUDIO MESSAGE ============
    elif msg.type == "audio" and msg.audio:
        logger.info(f"🎤 [{sender}] Audio received | ID: {msg.audio.id}")

        if await _daily_limit_blocks(sender):
            return
        
        try:
            media_bytes, mime_type = await download_media(msg.audio.id)
            logger.info(f"✅ Downloaded audio: {len(media_bytes)} bytes, type: {mime_type}")

            # Enforce 30s max — reject longer voice notes before we spend
            # time/money saving + transcribing them.
            duration = await get_audio_duration_seconds(media_bytes, audio_format="ogg")
            if duration is not None and duration > MAX_AUDIO_DURATION_SECONDS:
                logger.info(f"⛔ [{sender}] Audio rejected: {duration:.1f}s > {MAX_AUDIO_DURATION_SECONDS}s limit")
                await send_text_message(
                    sender,
                    f"That voice note is a bit long — please keep it under {MAX_AUDIO_DURATION_SECONDS} seconds so I can process it."
                )
                return

            # Save audio file (disk write — offload to a thread)
            audio_path = await asyncio.to_thread(memory.save_audio_file, sender, media_bytes)
            logger.info(f"💾 Audio saved to {audio_path}")
            

            transcription_result = await transcribe_audio(media_bytes, audio_format="ogg")
            
            if not transcription_result or not transcription_result.get("text"):
                await send_text_message(sender, "❌ Could not transcribe audio. Please try again.")
                return

            transcription = transcription_result["text"]
            whisper_language = transcription_result.get("language") or None
            
            logger.info(f"📝 Transcription: {transcription[:100]}... | detected_language={whisper_language}")

            # Free-question gate — same rule as the text handler: a
            # non-premium user gets settings.FREE_QUESTION_LIMIT answered
            # questions, voice included, before every further question
            # gets the payment link + app features instead of an answer.
            if not await asyncio.to_thread(memory.is_premium_active, sender):
                free_count = await asyncio.to_thread(memory.get_free_question_count, sender)
                if free_count >= settings.FREE_QUESTION_LIMIT:
                    await asyncio.to_thread(
                        memory.save_message,
                        sender, "user", f"[Voice Message]: {transcription}",
                        message_type="audio", audio_file_path=audio_path,
                        audio_transcription=transcription,
                    )
                    if whisper_language:
                        gate_required_language = whisper_language
                    else:
                        try:
                            gate_required_language = await detect_reply_language(transcription)
                        except Exception as e:
                            logger.warning(f"⚠️ Language detection failed for {sender}, defaulting to English: {e}")
                            gate_required_language = None
                    await _send_free_limit_message(
                        sender,
                        is_first_hit=False,
                        required_language=gate_required_language,
                    )
                    return

            try:
                reply, required_language = await generate_context_aware_response(
                    sender, transcription, is_audio=True, whisper_language=whisper_language,
                )
            except GeminiUnavailableError:
                logger.warning(f"⛔ Gemini unavailable — dropping voice query from {sender}: '{transcription[:80]}'")
                await send_text_message(
                    sender,
                    "Sorry, I'm a bit overloaded right now. Please try sending that again in a moment 🙏",
                )
                return

            # Save audio message with transcription
            await asyncio.to_thread(
                memory.save_message,
                sender, 
                "user", 
                f"[Voice Message]: {transcription}",
                message_type="audio",
                audio_file_path=audio_path,
                audio_transcription=transcription
            )

            # Save response
            await asyncio.to_thread(memory.save_message, sender, "assistant", reply, message_type="text")
            
            # Send response
            await send_text_message(sender, reply)
            logger.info(f"🤖 → [{sender}]: {reply[:100]}...")

            in_free_phase = await _handle_free_quota_after_answer(
                sender, required_language=required_language
            )
            
            # Update summary in background
            customer_data = await asyncio.to_thread(memory.get_customer, sender)
            background_tasks.append(asyncio.create_task(
                generate_new_summary(
                    sender,
                    customer_data.get("summary", ""),
                    f"[Voice Message]: {transcription}",
                    reply
                )
            ))

            # Cross-question the user based on this exchange (skipped in
            # the free phase — one more billed message).
            if not in_free_phase and not _limit_reached_now():
                context_for_followup = await asyncio.to_thread(
                    memory.get_conversation_context, sender, limit=5
                )
                background_tasks.append(asyncio.create_task(
                    maybe_send_followup(
                        sender,
                        customer_data.get("summary", ""),
                        context_for_followup,
                        f"[Voice Message]: {transcription}",
                        reply,
                        whisper_language=whisper_language,
                        required_language=required_language,
                    )
                ))

            # Clean up old audio files
            background_tasks.append(asyncio.create_task(asyncio.to_thread(memory.delete_old_audio_files, sender, keep_count=10)))
            
        except Exception as e:
            logger.error(f"❌ Audio processing error: {e}", exc_info=True)
            await send_text_message(sender, "Sorry, I couldn't process the audio. Please try again or send text instead.")

    # ============ IMAGE MESSAGE ============
    elif msg.type == "image" and msg.image:
        logger.info(f"📸 [{sender}] Image received | ID: {msg.image.id}")

        if await _daily_limit_blocks(sender):
            return
        
        try:
            media_bytes, mime_type = await download_media(msg.image.id)
            logger.info(f"Downloaded image: {len(media_bytes)} bytes, type: {mime_type}")

            try:
                image_description = await process_image_with_vision(media_bytes, mime_type)
                logger.info(f"Image description: {image_description[:100]}...")

                # Free-question gate — same rule as text/voice: a
                # non-premium user gets settings.FREE_QUESTION_LIMIT
                # answered questions before every further one gets the
                # payment link + app features instead of an answer.
                if not await asyncio.to_thread(memory.is_premium_active, sender):
                    free_count = await asyncio.to_thread(memory.get_free_question_count, sender)
                    if free_count >= settings.FREE_QUESTION_LIMIT:
                        await asyncio.to_thread(
                            memory.save_message, sender, "user",
                            f"[Sent an Image]: {image_description}", message_type="text",
                        )
                        try:
                            gate_required_language = await detect_reply_language(image_description)
                        except Exception as e:
                            logger.warning(f"⚠️ Language detection failed for {sender}, defaulting to English: {e}")
                            gate_required_language = None
                        await _send_free_limit_message(
                            sender,
                            is_first_hit=False,
                            required_language=gate_required_language,
                        )
                        return

                # Generate context-aware response FIRST — only persist
                # anything once we know Gemini actually answered.
                reply, required_language = await generate_context_aware_response(sender, f"[Image]: {image_description}", is_audio=False)
            except GeminiUnavailableError:
                logger.warning(f"⛔ Gemini unavailable — dropping image query from {sender}.")
                await send_text_message(
                    sender,
                    "Sorry, I'm a bit overloaded right now. Please try sending that again in a moment 🙏",
                )
                return

            # Save image message
            await asyncio.to_thread(
                memory.save_message, sender, "user", f"[Sent an Image]: {image_description}", message_type="text"
            )

            # Save and send response
            await asyncio.to_thread(memory.save_message, sender, "assistant", reply, message_type="text")
            
            await send_text_message(sender, reply)
            logger.info(f"🤖 → [{sender}]: {reply[:100]}...")

            in_free_phase = await _handle_free_quota_after_answer(
                sender, required_language=required_language
            )
            
            # Update summary in background
            customer_data = await asyncio.to_thread(memory.get_customer, sender)
            background_tasks.append(asyncio.create_task(
                generate_new_summary(
                    sender,
                    customer_data.get("summary", ""),
                    f"[Sent an Image]: {image_description}",
                    reply
                )
            ))

            # Cross-question the user based on this exchange (skipped in
            # the free phase — one more billed message).
            if not in_free_phase and not _limit_reached_now():
                context_for_followup = await asyncio.to_thread(
                    memory.get_conversation_context, sender, limit=5
                )
                background_tasks.append(asyncio.create_task(
                    maybe_send_followup(
                        sender,
                        customer_data.get("summary", ""),
                        context_for_followup,
                        f"[Sent an Image]: {image_description}",
                        reply,
                        required_language=required_language,
                    )
                ))

        except Exception as e:
            logger.error(f"❌ Image processing error: {e}", exc_info=True)
            await send_text_message(sender, "Sorry, I couldn't process the image. Please try again or send text instead.")

    # ============ BUTTON TAP (daily task confirmation) ============
    elif msg.type == "interactive" and msg.interactive:
        try:
            await _handle_interactive_reply(sender, msg.interactive)
        except Exception as e:
            logger.error(f"❌ Interactive reply handling failed for {sender}: {e}", exc_info=True)

    else:
        await send_text_message(sender, "I can handle text, audio, and images. Please send one of those formats.")

    if background_tasks:
        await asyncio.gather(*background_tasks, return_exceptions=True)


# ============ EXISTING ENDPOINTS (Testing, Health) ============

@app.post("/test/send-template", tags=["Testing"])
async def test_send_template(req: TestTemplateRequest):
    result = await send_template_message(req.to, req.template_name)
    return {"status": "sent", "meta_response": result}


@app.post("/test/full-flow", tags=["Testing"])
async def test_full_flow(req: TestMessageRequest, background_tasks: BackgroundTasks):
    sender = "test_user"
    try:
        reply, _required_language = await generate_context_aware_response(sender, req.message)
    except GeminiUnavailableError:
        raise HTTPException(status_code=503, detail="Gemini is currently unavailable. Query was dropped, nothing saved.")

    await asyncio.to_thread(memory.save_message, sender, "user", req.message)
    await asyncio.to_thread(memory.save_message, sender, "assistant", reply)
    wa_result = await send_text_message(req.to, reply)
    
    customer_data = await asyncio.to_thread(memory.get_customer, sender)
    background_tasks.add_task(
        generate_new_summary,
        sender,
        customer_data.get("summary", ""),
        req.message,
        reply
    )
    
    return {
        "input": req.message,
        "llm_reply": reply,
        "sent_to": req.to,
        "meta_response": wa_result,
    }


@app.post("/admin/run-daily-checkins", tags=["Health"])
async def run_daily_checkins_endpoint():
    """
    Manually trigger one run of the daily premium check-in job (normally
    run on a schedule by the dedicated daily_checkin service/process — see
    app/daily_checkin.py). Useful for testing without waiting for the
    scheduled hour.
    """
    from app.services.daily_checkin import run_daily_checkins
    await run_daily_checkins()
    return {"status": "ok", "message": "Daily check-in run completed."}


@app.get("/health", tags=["Health"])
async def health():

    pending_dead_letters = await asyncio.to_thread(memory.count_pending_dead_letters)
    return {
        "status": "ok",
        "bot": "AI Health Assistant",
        "phone_number_id": settings.PHONE_NUMBER_ID,
        "supported_media": ["text", "audio", "image"],
        "features": [
            "Context-aware health Q&A",
            "Health profile capture",
            "Audio transcription",
            "Message persistence",
            "Conversation history",
            "21-day premium daily check-ins",
        ],
        "pending_dead_letters": pending_dead_letters,
    }


@app.get("/debug/dead-letters", tags=["Health"])
async def list_dead_letters(limit: int = Query(50, ge=1, le=500)):
    """Inspect pending dead-lettered inbound messages — i.e. messages
    that failed to enqueue after retries and are waiting for
    dead_letter_worker.py's next cycle (or manual replay below)."""
    rows = await asyncio.to_thread(memory.get_pending_dead_letters, limit)
    return {"count": len(rows), "pending_dead_letters": rows}


@app.post("/debug/dead-letters/{dead_letter_id}/replay", tags=["Health"])
async def replay_dead_letter(dead_letter_id: int):
    """Manually force-replay a single dead-lettered message right now,
    instead of waiting for dead_letter_worker.py's next poll cycle —
    useful right after fixing whatever caused the original outage."""
    rows = await asyncio.to_thread(memory.get_pending_dead_letters, 500)
    row = next((r for r in rows if r["id"] == dead_letter_id), None)
    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"No pending dead-letter with id={dead_letter_id}",
        )

    job_ref = await asyncio.to_thread(enqueue_incoming, row["payload"])
    if job_ref.startswith("dead_letter:"):
        return {
            "status": "still_failing",
            "dead_letter_id": dead_letter_id,
            "detail": "Replay attempted but the queue backend is still failing; left pending.",
        }

    await asyncio.to_thread(memory.mark_dead_letter_resolved, dead_letter_id)
    return {"status": "resolved", "dead_letter_id": dead_letter_id, "job_ref": job_ref}


@app.get("/debug", tags=["Health"])
async def debug():
    token_check = await verify_token_valid()

    return {
        "config": {
            "phone_number_id": settings.PHONE_NUMBER_ID,
            "openai_key_set": bool(settings.OPENAI_API_KEY),
            "verify_token_set": bool(settings.VERIFY_TOKEN),
        },
        "token_check": token_check,
        "database": {
            "type": "PostgreSQL",
            "location": memory._safe_url(),
            "features": ["Message storage", "Audio tracking", "Timestamps", "Context retrieval"]
        },
        "media_support": "text, audio, image",
    }


@app.delete("/debug/clear-session/{phone}", tags=["Health"])
async def clear_session(phone: str):
    await asyncio.to_thread(memory.update_summary, phone, "")
    return {"cleared": phone, "message": f"Session cleared for {phone}"}


@app.get("/models", tags=["Models"])
async def get_models(model_type: str = Query("all", description="audio, text, or all")):
    """Get available AI models."""
    return get_available_models(model_type)


@app.get("/models/info", tags=["Models"])
async def get_model_details(
    model_type: str = Query(..., description="audio or text"),
    model_key: str = Query(..., description="Model identifier")
):
    """Get detailed information about a specific model."""
    info = get_model_info(model_type, model_key)
    if not info:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"model": model_key, "type": model_type, **info}