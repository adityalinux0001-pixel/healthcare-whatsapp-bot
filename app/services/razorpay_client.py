
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