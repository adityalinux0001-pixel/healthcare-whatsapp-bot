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
from fastapi import FastAPI, Request, HTTPException, Query, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import PlainTextResponse
from app.config import get_settings
from app.models import (
    WebhookPayload,
    IncomingMessage,
    TestMessageRequest,
    TestTemplateRequest,
    TestRAGRequest,
    IngestTextRequest,
    DeleteSourceRequest,
)
from app.whatsapp import (
    send_text_message,
    send_template_message,
    mark_as_read,
    verify_token_valid,
)
from app.llm import (
    get_llm_response,
    get_summary_response,
    process_image_with_vision,
    generate_followup_suggestion,
    detect_reply_language,
    is_gemini_busy,
    GeminiUnavailableError,
)
from app.vector_utils import retrieve_context, get_pinecone_index, get_index_stats
from app.ingest import ingest_text, delete_source, read_file
from app.memory import ConversationMemory
from app.idempotency import try_mark_message_processed
from app.audio_handler import (
    transcribe_audio, 
    get_available_models,
    get_model_info,
    get_audio_duration_seconds,
)
from app.queueing import enqueue_incoming
from app.razorpay_client import create_payment_link, verify_webhook_signature

# Voice notes longer than this are rejected outright — client requirement.
MAX_AUDIO_DURATION_SECONDS = 30

# Logging
settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("whatsapp_bot_enhanced")

# Initialize Postgres-backed conversation memory (step 1)
memory = ConversationMemory(
    database_url=settings.database_url,
    pool_min_size=settings.db_pool_min_size,
    pool_max_size=settings.db_pool_max_size,
)

# Ignore any incoming message older than this many seconds by the time we
# actually get to process it. WhatsApp can redeliver a webhook for a
# message hours after it was first sent (e.g. after our server was slow,
# down, or restarted) — if we didn't check this, that redelivery would
# run through the whole pipeline as if it were a live, brand-new query
# and could fire a "Gemini unavailable" reply long after the user's
# conversation had already ended.
_MAX_MESSAGE_AGE_SECONDS = 120

# App
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("Steve's AI Lab — WhatsApp RAG Bot (ENHANCED)")
    logger.info(f"Phone Number ID : {settings.phone_number_id}")
    logger.info(f"Pinecone Index  : {settings.pinecone_index_name}")
    logger.info(f"RAG top-k       : {settings.rag_top_k}")
    logger.info(f"Database        : {memory._safe_url()}")
    logger.info(f"Audio Storage   : {memory.audio_dir}")
    logger.info("Swagger UI      : http://localhost:8000/docs")
    logger.info("=" * 60)
    try:
        get_pinecone_index()
        logger.info("✅ Pinecone connected")
    except Exception as e:
        logger.error(f"❌ Pinecone connection failed: {e}")

    try:
        from app.redis_client import get_redis
        await get_redis().ping()
        logger.info("✅ Redis connected")
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")

    yield

    # Shutdown: release the Postgres connection pool cleanly rather than
    # letting connections dangle when a worker process exits (matters
    # more now than under SQLite, since a lingering connection here holds
    # a slot against Postgres's max_connections until the OS notices the
    # socket is dead).
    try:
        memory.pool.close()
    except Exception:
        logger.warning("Error closing Postgres pool on shutdown", exc_info=True)


app = FastAPI(title="Steve's AI Lab — WhatsApp RAG Bot (ENHANCED)")


# Webhook GET — Meta verification
@app.get("/webhook")
async def verify_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
    hub_challenge: str = Query(None, alias="hub.challenge"),
):
    logger.info(f"Webhook verify — mode={hub_mode} token={hub_verify_token}")
    if hub_mode == "subscribe" and hub_verify_token == settings.verify_token:
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
        headers = {"Authorization": f"Bearer {settings.whatsapp_token}"}
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
    # Run local DB lookups, the (slower) RAG retrieval, AND language
    # detection all concurrently — they're all independent of each other,
    # so no need to do any of them sequentially.
    #
    # LATENCY FIX: detect_reply_language() used to be awaited on its own,
    # AFTER this gather() had already finished — meaning every turn paid
    # for two full sequential Gemini round-trips (detect language, THEN
    # generate the main reply) even though detection doesn't depend on
    # anything this gather produces. Folding it into the same gather()
    # means its latency overlaps with the RAG/DB round-trips instead of
    # stacking on top of them, and the main Gemini reply call is now the
    # only Gemini call still purely sequential on the critical path.
    #
    # NOTE: memory.* methods use blocking sqlite3 calls. Without
    # asyncio.to_thread here, calling them directly inside these
    # coroutines would run synchronously on the event loop, defeating the
    # whole point of asyncio.gather — they'd effectively execute one after
    # another (and block every other in-flight request) instead of really
    # running concurrently with the RAG retrieval.
    async def _get_context():
        return await asyncio.to_thread(memory.get_conversation_context, phone_number, limit=5)

    async def _get_customer():
        return await asyncio.to_thread(memory.get_customer, phone_number)

    async def _get_chunks():
        try:
            return await retrieve_context(user_message, top_k=settings.rag_top_k)
        except Exception as e:
            logger.error(f"RAG retrieval error: {e}")
            return []

    async def _get_language():
        return await detect_reply_language(user_message, whisper_language)

    context_text, customer_data, chunks, required_language = await asyncio.gather(
        _get_context(), _get_customer(), _get_chunks(), _get_language()
    )
    customer_summary = customer_data.get("summary", "")
    
    # Build enriched prompt with context
    message_type = "[AUDIO MESSAGE]" if is_audio else ""
    enriched_prompt = f"""
{message_type}

[CUSTOMER SUMMARY]
{customer_summary if customer_summary else "No prior context available"}

{context_text if context_text else "[No previous messages]"}

[CURRENT USER MESSAGE]
{user_message}

Answer only the current user message above, directly and briefly (per the
system prompt's length rule), using the conversation context only to stay
consistent — do not re-summarize the context or restate prior messages.
    """.strip()

    # required_language was already detected above, concurrently with the
    # context/RAG gather() — reused here and (via the returned tuple) for
    # the follow-up suggestion in main.py, instead of get_llm_response()
    # and generate_followup_suggestion() each running their own
    # independent (but identical, deterministic) detection call.

    # Get LLM response
    try:
        response = await get_llm_response(
            user_message=enriched_prompt,
            conversation_history=[],
            context_chunks=chunks if chunks else None,
            raw_user_text=user_message,
            whisper_language=whisper_language,
            required_language=required_language,
        )
        return response, required_language
    except GeminiUnavailableError:
        # Gemini is down/overloaded (503 etc.) — bubble this up so the
        # caller can drop the query silently instead of sending any
        # fallback reply or saving a "sorry" message to history.
        raise
    except Exception as e:
        logger.error(f"LLM error: {e}", exc_info=True)
        return "Sorry, I ran into an issue. Please try again in a moment.", required_language


SUMMARY_REFRESH_EVERY_N_MESSAGES = 3


async def generate_new_summary(phone_number: str, current_summary: str, user_msg: str, ai_msg: str):
    """Background task to async update the conversation summary using LLM.

    IMPORTANT: this summary is internal bookkeeping only — it is never sent
    to the user. It must always be written in English regardless of what
    language the user is chatting in, otherwise its language leaks back
    into future prompts (via [CUSTOMER SUMMARY]) and biases the main reply
    generator toward whatever language the summary happens to be in, even
    when the user's current message is in a different language.

    THROTTLING: this is a background-only Gemini call — never shown to the
    user directly — so it's the safest one to run less often under load.
    The raw last-5-messages window ([CUSTOMER SUMMARY]'s companion context
    in generate_context_aware_response) is rebuilt fresh from the DB on
    every single turn regardless, so recent detail is never lost even when
    the summary itself is a message or two stale. Only regenerate the
    summary every SUMMARY_REFRESH_EVERY_N_MESSAGES messages for this user;
    skip it otherwise (leaving the existing summary as-is) to cut Gemini
    calls without any visible change in behavior.
    """
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
        # Was a bare (blocking) call — synchronous sqlite3 writes block the
        # entire event loop, so under concurrent users every other user's
        # request stalls for the duration of this disk write. Offload to a
        # thread like every other memory.* call in this file.
        await asyncio.to_thread(memory.update_summary, phone_number, new_summary.strip())
        logger.info(f"✅ Updated summary for {phone_number}")
    except GeminiUnavailableError:
        logger.warning(f"⏭️ Skipped summary update for {phone_number} — Gemini unavailable.")
    except Exception as e:
        logger.error(f"❌ Error updating summary for {phone_number}: {e}", exc_info=True)


async def _maybe_send_premium_offer(phone_number: str) -> None:
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

    Runs on every message (no session-gap gate) so case 2 fires on the
    user's very next message after expiry, whenever that happens to be —
    it doesn't wait for a "new session".

    Failure here (Razorpay API down, etc.) is logged and swallowed — it
    must never block or break the user's actual conversation.
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
                return

            if not subscription.get("expiry_notified"):
                # Case 2: just expired, one-time notice not sent yet.
                logger.info(f"⌛ Premium expired for {phone_number} — sending expiry notice + new link.")
                link = await create_payment_link(
                    phone_number=phone_number,
                    amount_rupees=settings.premium_plan_amount_rupees,
                    description=f"Steve's AI Lab — {settings.premium_plan_days}-day Premium",
                )
                await asyncio.to_thread(
                    memory.save_payment_link,
                    link["id"],
                    phone_number,
                    settings.premium_plan_amount_rupees * 100,
                )
                await asyncio.to_thread(memory.mark_subscription_expiry_notified, phone_number)

                expiry_text = (
                    f"Your {settings.premium_plan_days}-day Premium plan has expired. ⌛\n\n"
                    f"Please buy again to keep enjoying priority access, regular project "
                    f"updates, and the extra features — here's your payment link:\n"
                    f"{link['short_url']}"
                )
                await send_text_message(phone_number, expiry_text)
                await asyncio.to_thread(
                    memory.save_message, phone_number, "assistant", expiry_text, message_type="text"
                )
                logger.info(f"⌛ Sent expiry notice to {phone_number} | link_id={link['id']}")
                return

        # Case 3: never subscribed, or already notified of this expiry —
        # fall through to the normal throttled recurring reminder below.
        latest_link = await asyncio.to_thread(memory.get_latest_payment_link_for_user, phone_number)
        if latest_link and latest_link.get("status") != "paid":
            created_at = latest_link.get("created_at")
            if created_at is not None:
                if created_at.tzinfo is not None:
                    created_at = created_at.replace(tzinfo=None)
                seconds_since = (datetime.utcnow() - created_at).total_seconds()
                if seconds_since < settings.premium_reoffer_min_gap_seconds:
                    logger.info(
                        f"⏭️ Skipping premium offer for {phone_number} — an unpaid link was "
                        f"already sent {seconds_since:.0f}s ago (throttle: "
                        f"{settings.premium_reoffer_min_gap_seconds}s)."
                    )
                    return

        logger.info(f"🆕 No active premium for {phone_number} — sending premium offer.")

        link = await create_payment_link(
            phone_number=phone_number,
            amount_rupees=settings.premium_plan_amount_rupees,
            description=f"Steve's AI Lab — {settings.premium_plan_days}-day Premium",
        )
    except Exception as e:
        logger.error(f"❌ Failed to create/send premium offer for {phone_number}: {e}", exc_info=True)
        return

    try:
        await asyncio.to_thread(
            memory.save_payment_link,
            link["id"],
            phone_number,
            settings.premium_plan_amount_rupees * 100,
        )

        offer_text = (
            f"Before we dive in — quick heads up 👋\n\n"
            f"We offer a {settings.premium_plan_days}-day Premium plan for ₹{settings.premium_plan_amount_rupees}. "
            f"With it you get:\n"
            f"1. Daily-ish follow-up updates on how we're progressing on your project\n"
            f"2. Priority access to our team\n"
            f"3. Extra features unlocked in this chat\n\n"
            f"Totally optional — you can keep chatting normally either way. "
            f"If you'd like to grab it, here's your secure payment link:\n"
            f"{link['short_url']}"
        )
        await send_text_message(phone_number, offer_text)
        await asyncio.to_thread(
            memory.save_message, phone_number, "assistant", offer_text, message_type="text"
        )
        logger.info(f"💳 Sent premium offer to {phone_number} | link_id={link['id']}")
    except Exception as e:
        logger.error(f"❌ Failed to send/save premium offer message for {phone_number}: {e}", exc_info=True)


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
                    logger.info(f"📊 Status: {s.status} | to={s.recipient_id}")
                continue

            if not value.messages:
                continue

            # Save/update contact name (from WhatsApp profile) for each sender.
            # This is a blocking sqlite write — don't do it inline before the
            # webhook ack; the whole point of the ack is to return fast so
            # Meta doesn't time out and redeliver the event.
            if value.contacts:
                for contact in value.contacts:
                    wa_id = contact.wa_id
                    profile = contact.profile or {}
                    name = profile.get("name") if isinstance(profile, dict) else None
                    if wa_id and name:
                        background_tasks.add_task(memory.set_user_name, wa_id, name)

            for raw_msg in value.messages:
                # Don't await this inline — webhook must ack fast (200 OK)
                # so WhatsApp/Meta never times out and redelivers the same
                # event, which is what was causing duplicate delayed replies
                # when Gemini was slow/erroring.
                background_tasks.add_task(enqueue_incoming, raw_msg)

    return {"status": "ok"}


# Razorpay webhook — fires on payment.captured / payment_link.paid events.
# Configure this URL in Razorpay Dashboard -> Settings -> Webhooks, and set
# the same secret you enter there as RAZORPAY_WEBHOOK_SECRET in .env.
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

    # We only care about a payment link actually being paid. Razorpay also
    # sends payment.captured, order.paid, etc. — payment_link.paid is the
    # one tied to what we created, and its payload carries the
    # payment_link_id we stored when we created the link.
    if event != "payment_link.paid":
        return {"status": "ok"}

    try:
        payload = body["payload"]
        payment_link_entity = payload["payment_link"]["entity"]
        payment_entity = payload["payment"]["entity"]

        payment_link_id = payment_link_entity["id"]
        razorpay_payment_id = payment_entity["id"]
    except (KeyError, TypeError) as e:
        logger.error(f"❌ Razorpay webhook: unexpected payload shape: {e} | body={body}")
        return {"status": "ok"}

    phone_number = await asyncio.to_thread(
        memory.mark_payment_link_paid, payment_link_id, razorpay_payment_id
    )
    if not phone_number:
        logger.warning(f"⚠️ Razorpay webhook: no local record for payment_link_id={payment_link_id}")
        return {"status": "ok"}

    expires_at = await asyncio.to_thread(
        memory.activate_subscription,
        phone_number,
        settings.premium_plan_days,
        payment_link_id,
    )
    logger.info(f"✅ Activated {settings.premium_plan_days}-day premium for {phone_number}, expires {expires_at}")

    confirm_text = (
        f"Payment received, thank you! 🎉\n\n"
        f"Your {settings.premium_plan_days}-day Premium plan is now active. "
        f"You'll start getting regular updates on your project and priority access right here in this chat."
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

    return {"status": "ok"}


async def _handle_incoming(raw_msg: dict) -> None:
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

    # Skip if WhatsApp already redelivered this exact message (e.g. retry
    # after our previous response was slow/failed, or after a server
    # restart). Backed by Redis (see app/idempotency.py) so it survives
    # restarts AND is shared across worker processes — an in-memory set,
    # or state private to one worker, would not catch a redelivery that
    # lands on a *different* worker than the one that handled it first.
    if not await try_mark_message_processed(msg.id):
        logger.info(f"⏭️ Duplicate webhook delivery for message id={msg.id}, skipping.")
        return

    # Skip if this message is too old to be a live query anymore (stale
    # webhook redelivery). Without this, a message from hours ago could
    # still trigger a fresh Gemini call and an out-of-nowhere "sorry"
    # reply long after the conversation ended.
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

    background_tasks: list[asyncio.Task[None]] = []

    # Fire-and-forget: don't block reply generation on this network call.
    # show_typing=True also shows WhatsApp's native "typing…" indicator to
    # the user while we're generating the reply (via Gemini/RAG/etc.), so
    # it doesn't look like the message went nowhere during that wait.
    background_tasks.append(asyncio.create_task(mark_as_read(msg.id, show_typing=True)))

    # ============ PREMIUM UPSELL: fire on the first message of a new session ============
    # Awaited (not fire-and-forget) and placed BEFORE any save_message()/
    # append_turn() call for this turn — those overwrite last_message_at
    # with "now", so reading it after saving (or racing a background task
    # against the save) would always look like 0 seconds have passed.
    await _maybe_send_premium_offer(sender)

    # ============ TEXT MESSAGE ============
    if msg.type == "text" and msg.text:
        user_text = msg.text.body.strip()
        logger.info(f"👤 [{sender}]: {user_text}")

        # Handle special commands
        if user_text.lower() in ("/reset", "/clear", "reset", "clear"):
            await asyncio.to_thread(memory.update_summary, sender, "")
            stats = await asyncio.to_thread(memory.get_user_stats, sender)
            logger.info(f"🗑️ Cleared conversation for {sender}. Stats: {stats}")
            await send_text_message(sender, "Conversation cleared! How can I help you with Steve's AI Lab?")
            return
        
        if user_text.lower() in ("/models", "/help"):
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

        # Save user message + assistant reply (only once we know we have a
        # real reply). Wrapped in try/except so a DB failure (locked file,
        # disk full, bad path, etc.) is logged loudly instead of silently
        # disappearing inside this background task — and so we still send
        # the reply to the user even if persistence fails.
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
            
            # Transcribe audio — returns {"text": ..., "language": ...},
            # where "language" is what Whisper itself detected from the
            # AUDIO (e.g. "english", "hindi"). Short/accented clips can
            # make Whisper mis-transcribe into the wrong Indic script, so
            # we carry this detected language through separately and let
            # it override the (possibly garbled) transcribed text when
            # deciding which language to reply in.
            transcription_result = await transcribe_audio(media_bytes, audio_format="ogg")
            
            if not transcription_result or not transcription_result.get("text"):
                await send_text_message(sender, "❌ Could not transcribe audio. Please try again.")
                return

            transcription = transcription_result["text"]
            whisper_language = transcription_result.get("language") or None
            
            logger.info(f"📝 Transcription: {transcription[:100]}... | detected_language={whisper_language}")

            # Generate context-aware response FIRST — only persist anything
            # (user message, audio path, reply) once we know Gemini actually
            # answered. If Gemini is down, don't save to DB, but still let
            # the user know instead of leaving them hanging after the
            # "Processing audio..." message.
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

            # Cross-question the user based on this exchange.
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
        
        try:
            media_bytes, mime_type = await download_media(msg.image.id)
            logger.info(f"Downloaded image: {len(media_bytes)} bytes, type: {mime_type}")

            try:
                image_description = await process_image_with_vision(media_bytes, mime_type)
                logger.info(f"Image description: {image_description[:100]}...")

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

            # Cross-question the user based on this exchange.
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

    else:
        await send_text_message(sender, "I can handle text, audio, and images. Please send one of those formats.")

    if background_tasks:
        await asyncio.gather(*background_tasks, return_exceptions=True)


# ============ EXISTING ENDPOINTS (Ingestion, Testing, Health) ============

@app.post("/ingest/text", tags=["Ingestion"])
async def ingest_text_endpoint(req: IngestTextRequest):
    result = await ingest_text(
        text=req.text,
        source=req.source,
        chunk_tokens=req.chunk_tokens,
        overlap=req.overlap_tokens,
    )
    return {"status": "ingested", **result}


@app.post("/ingest/file", tags=["Ingestion"])
async def ingest_file_endpoint(
    file: UploadFile = File(...),
    source: str = Form(...),
    chunk_tokens: int = Form(300),
    overlap_tokens: int = Form(50),
):
    filename = file.filename or "upload"
    content = await file.read()

    try:
        text = read_file(filename, content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    result = await ingest_text(
        text=text,
        source=source or filename,
        chunk_tokens=chunk_tokens,
        overlap=overlap_tokens,
    )
    return {"status": "ingested", "filename": filename, **result}


@app.delete("/ingest/source", tags=["Ingestion"])
async def delete_source_endpoint(req: DeleteSourceRequest):
    result = await delete_source(req.source)
    return {"status": "deleted", **result}


@app.get("/ingest/stats", tags=["Ingestion"])
async def ingest_stats():
    return await get_index_stats()


@app.post("/test/send-template", tags=["Testing"])
async def test_send_template(req: TestTemplateRequest):
    result = await send_template_message(req.to, req.template_name)
    return {"status": "sent", "meta_response": result}


@app.post("/test/rag", tags=["Testing"])
async def test_rag(req: TestRAGRequest):
    chunks = await retrieve_context(req.user_message, top_k=req.top_k)
    reply = await get_llm_response(
        user_message=req.user_message,
        context_chunks=chunks if chunks else None,
    )
    return {
        "user_message": req.user_message,
        "chunks_retrieved": len(chunks),
        "chunks": chunks,
        "llm_reply": reply,
        "rag_used": len(chunks) > 0,
    }


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


@app.get("/health", tags=["Health"])
async def health():
    return {
        "status": "ok",
        "bot": "Steve's AI Lab Assistant (ENHANCED)",
        "phone_number_id": settings.phone_number_id,
        "pinecone_index": settings.pinecone_index_name,
        "supported_media": ["text", "audio", "image"],
        "features": [
            "Context-aware responses",
            "Audio transcription",
            "Message persistence",
            "Conversation history",
            "Multiple AI models"
        ],
    }


@app.get("/debug", tags=["Health"])
async def debug():
    token_check = await verify_token_valid()
    try:
        index = get_pinecone_index()
        # describe_index_stats() is a blocking Pinecone SDK call — offload
        # it so it doesn't stall the event loop (and every in-flight
        # WhatsApp request) while this admin/debug endpoint is hit.
        stats = await asyncio.to_thread(index.describe_index_stats)
        pinecone_ok = True
        vector_count = stats.total_vector_count
    except Exception as e:
        pinecone_ok = False
        vector_count = f"error: {e}"

    return {
        "config": {
            "phone_number_id": settings.phone_number_id,
            "pinecone_index": settings.pinecone_index_name,
            "rag_top_k": settings.rag_top_k,
            "openai_key_set": bool(settings.openai_api_key),
            "pinecone_key_set": bool(settings.pinecone_api_key),
            "verify_token_set": bool(settings.verify_token),
        },
        "token_check": token_check,
        "pinecone": {
            "connected": pinecone_ok,
            "total_vectors": vector_count,
        },
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