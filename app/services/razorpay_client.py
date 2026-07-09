"""
Razorpay integration for the 21-day premium subscription upsell.

Uses Razorpay's Payment Links API (https://razorpay.com/docs/api/payments/payment-links/)
— a single API call returns a hosted checkout URL we can drop straight into
a WhatsApp message. No frontend/widget needed on our side.

Two responsibilities live here:
1. create_payment_link()  — called when we want to offer the user premium.
2. verify_webhook_signature() — called by the /razorpay/webhook endpoint in
   main.py to confirm a webhook payload genuinely came from Razorpay (HMAC
   signature check) before we trust it and activate anything.
"""

import hashlib
import hmac
import logging
from functools import lru_cache

import razorpay

from app.core.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_client() -> razorpay.Client:
    settings = get_settings()
    client = razorpay.Client(auth=(settings.RAZORPAY_KEY_ID, settings.RAZORPAY_KEY_SECRET))
    return client


async def create_payment_link(phone_number: str, amount_rupees: int, description: str) -> dict:
    """
    Create a Razorpay Payment Link for a given user.

    Args:
        phone_number: WhatsApp sender ID, e.g. "919876543210" — used both
            to prefill the contact on Razorpay's checkout page and, more
            importantly, tucked into `notes` so the webhook handler can
            recover it later without trusting anything from the request URL.
        amount_rupees: plan price in whole rupees (e.g. 499).
        description: shown to the user on the Razorpay checkout page.

    Returns:
        dict with at least "id" (payment_link_id) and "short_url" (the link
        to send the user). Raises on API failure — caller should catch and
        decide how to degrade (e.g. skip sending the offer this time rather
        than crash the whole message handler).

    The razorpay SDK is synchronous (blocking HTTP under the hood) — the
    actual client.payment_link.create() call is offloaded to a thread via
    asyncio.to_thread so it doesn't stall the event loop for every other
    concurrent user's request, same pattern used for sqlite calls elsewhere
    in this codebase.
    """
    import asyncio

    client = _get_client()

    payload = {
        "amount": amount_rupees * 100,  # Razorpay wants paise, not rupees
        "currency": "INR",
        "description": description,
        "customer": {
            "contact": f"+{phone_number}" if not phone_number.startswith("+") else phone_number,
        },
        "notify": {"sms": False, "email": False},  # we notify via WhatsApp ourselves
        "reminder_enable": False,
        "notes": {
            "phone_number": phone_number,
            "plan": "premium_21day",
        },
    }

    logger.info(f"Creating Razorpay payment link for {phone_number} | amount=₹{amount_rupees}")
    link = await asyncio.to_thread(client.payment_link.create, payload)
    logger.info(f"Created payment link id={link.get('id')} url={link.get('short_url')}")
    return link


def verify_webhook_signature(request_body: bytes, received_signature: str) -> bool:
    """
    Verify a Razorpay webhook payload genuinely came from Razorpay, using
    the webhook secret configured in the Razorpay dashboard (Settings ->
    Webhooks — NOT the same as your API key secret).

    Razorpay signs the raw request body with HMAC-SHA256 using the webhook
    secret and sends it in the X-Razorpay-Signature header. We must recompute
    it over the exact raw bytes (not a re-serialized/parsed version, which
    can differ in whitespace/key order and break the comparison) and do a
    constant-time comparison to avoid timing attacks.
    """
    settings = get_settings()
    if not settings.RAZORPAY_WEBHOOK_SECRET:
        logger.error("razorpay_webhook_secret is not configured — rejecting webhook.")
        return False

    expected_signature = hmac.new(
        key=settings.RAZORPAY_WEBHOOK_SECRET.encode("utf-8"),
        msg=request_body,
        digestmod=hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(expected_signature, received_signature or "")