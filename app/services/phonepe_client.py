"""
PhonePe integration for the 21-day premium subscription upsell.

Uses PhonePe PG Standard Checkout v2
(https://developer.phonepe.com/payment-gateway/website-integration/standard-checkout)
— one API call returns a hosted checkout URL we can drop straight into a
WhatsApp message. No frontend/widget needed on our side.

Three responsibilities live here:
1. create_payment_link()  — called when we want to offer the user premium.
2. verify_webhook_authorization() — called by the /phonepe/webhook endpoint in
   main.py to confirm a webhook payload genuinely came from PhonePe before we
   trust it and activate anything.
3. _get_access_token() — PhonePe (unlike Razorpay's basic-auth API keys) needs
   a short-lived OAuth token on every call, so we fetch and cache one here.

Deliberately built on httpx — already a dependency of this project — rather
than PhonePe's Python SDK, which isn't reliably published on PyPI.

NOTE ON LINK LIFETIME: a PhonePe checkout URL expires (see
PHONEPE_LINK_EXPIRE_AFTER_SECONDS), unlike a Razorpay payment link which
lived indefinitely. Callers that cache and re-send a previously created URL
must check its age first — see link_is_still_valid() below.
"""

import asyncio
import hashlib
import hmac
import logging
import time
import uuid

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# PhonePe host per environment. The /checkout/v2 paths are identical on both;
# only the host and the "pg-sandbox" path prefix differ.
_HOSTS = {
    "production": {
        "oauth": "https://api.phonepe.com/apis/identity-manager/v1/oauth/token",
        "pay": "https://api.phonepe.com/apis/pg/checkout/v2/pay",
        "status": "https://api.phonepe.com/apis/pg/checkout/v2/order/{order_id}/status",
    },
    "sandbox": {
        "oauth": "https://api-preprod.phonepe.com/apis/pg-sandbox/v1/oauth/token",
        "pay": "https://api-preprod.phonepe.com/apis/pg-sandbox/checkout/v2/pay",
        "status": "https://api-preprod.phonepe.com/apis/pg-sandbox/checkout/v2/order/{order_id}/status",
    },
}

# Refresh the token this many seconds before PhonePe's stated expiry, so an
# in-flight request can't be the one that discovers the token just died.
_TOKEN_REFRESH_MARGIN_SECONDS = 120

_HTTP_TIMEOUT_SECONDS = 20.0

# Cached OAuth token: (access_token, expires_at_epoch_seconds).
_token_cache: tuple[str, float] | None = None
_token_lock: asyncio.Lock | None = None


def _endpoints() -> dict:
    settings = get_settings()
    env = (settings.PHONEPE_ENV or "production").strip().lower()
    if env not in _HOSTS:
        logger.warning(f"Unknown PHONEPE_ENV={env!r} — falling back to 'production'.")
        env = "production"
    return _HOSTS[env]


def _get_token_lock() -> asyncio.Lock:
    """
    Lazily created so the Lock binds to the running event loop rather than
    whatever loop happened to be current at import time.
    """
    global _token_lock
    if _token_lock is None:
        _token_lock = asyncio.Lock()
    return _token_lock


async def _get_access_token(force_refresh: bool = False) -> str:
    """
    Fetch (or return a cached) PhonePe OAuth access token.

    PhonePe issues a client_credentials token from client_id/client_secret/
    client_version and returns `expires_at` as an epoch timestamp. We cache it
    in-process and refresh a couple of minutes early. The lock stops a burst of
    concurrent users all triggering their own token fetch on a cold start.

    This function is the ONLY place request authentication is decided. If this
    merchant account is ever moved to the legacy v1 salt-key/X-VERIFY scheme,
    this is the single function that changes (plus the header built in
    _auth_headers) — the payment logic below stays as-is.
    """
    global _token_cache

    if not force_refresh and _token_cache is not None:
        token, expires_at = _token_cache
        if time.time() < expires_at - _TOKEN_REFRESH_MARGIN_SECONDS:
            return token

    async with _get_token_lock():
        # Another coroutine may have refreshed it while we waited for the lock.
        if not force_refresh and _token_cache is not None:
            token, expires_at = _token_cache
            if time.time() < expires_at - _TOKEN_REFRESH_MARGIN_SECONDS:
                return token

        settings = get_settings()
        if not (settings.PHONEPE_CLIENT_ID and settings.PHONEPE_CLIENT_SECRET):
            raise RuntimeError(
                "PhonePe credentials are not configured "
                "(PHONEPE_CLIENT_ID / PHONEPE_CLIENT_SECRET)."
            )

        form = {
            "client_id": settings.PHONEPE_CLIENT_ID,
            "client_version": str(settings.PHONEPE_CLIENT_VERSION),
            "client_secret": settings.PHONEPE_CLIENT_SECRET,
            "grant_type": "client_credentials",
        }

        logger.info("Fetching a fresh PhonePe OAuth token.")
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_SECONDS) as client:
            resp = await client.post(
                _endpoints()["oauth"],
                data=form,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"PhonePe OAuth failed: HTTP {resp.status_code} — {resp.text[:500]}"
            )

        data = resp.json()
        token = data.get("access_token")
        if not token:
            raise RuntimeError(f"PhonePe OAuth response had no access_token: {data}")

        # Prefer PhonePe's absolute `expires_at`; fall back to `expires_in`,
        # and finally to a conservative 15 minutes if neither is present.
        expires_at = data.get("expires_at")
        if not isinstance(expires_at, (int, float)):
            expires_in = data.get("expires_in")
            expires_at = (
                time.time() + float(expires_in)
                if isinstance(expires_in, (int, float))
                else time.time() + 900
            )

        _token_cache = (token, float(expires_at))
        logger.info(f"PhonePe OAuth token cached (expires_at={expires_at}).")
        return token


async def _auth_headers() -> dict:
    token = await _get_access_token()
    return {
        "Content-Type": "application/json",
        "Authorization": f"O-Bearer {token}",
    }


async def create_payment_link(phone_number: str, amount_rupees: int, description: str) -> dict:
    """
    Create a PhonePe Standard Checkout session for a given user.

    Args:
        phone_number: WhatsApp sender ID, e.g. "919876543210" — tucked into
            `metaInfo.udf1` so the webhook handler can recover it without
            trusting anything from the request URL.
        amount_rupees: plan price in whole rupees (e.g. 499).
        description: shown to the user on the PhonePe checkout page.

    Returns:
        dict with at least "id" (our merchantOrderId — the key we store and
        that PhonePe echoes back in the webhook) and "short_url" (the checkout
        link to send the user). Raises on API failure — caller should catch and
        decide how to degrade (e.g. skip sending the offer this time rather
        than crash the whole message handler).

    The return shape intentionally mirrors what the previous Razorpay payment
    link returned, so callers need no special-casing.

    Unlike Razorpay, PhonePe does not mint the order id for us: we generate
    merchantOrderId ourselves, which is what makes the webhook lookup a direct
    primary-key hit on payment_links.
    """
    settings = get_settings()

    merchant_order_id = f"pp_{uuid.uuid4().hex}"

    payload = {
        "merchantOrderId": merchant_order_id,
        "amount": amount_rupees * 100,  # PhonePe wants paise, not rupees
        "expireAfter": settings.PHONEPE_LINK_EXPIRE_AFTER_SECONDS,
        "metaInfo": {
            "udf1": phone_number,
            "udf2": "premium_21day",
        },
        "paymentFlow": {
            "type": "PG_CHECKOUT",
            "message": description,
            "merchantUrls": {
                "redirectUrl": settings.PHONEPE_REDIRECT_URL,
            },
        },
    }

    logger.info(
        f"Creating PhonePe checkout for {phone_number} | amount=₹{amount_rupees} "
        f"| merchantOrderId={merchant_order_id}"
    )

    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_SECONDS) as client:
        resp = await client.post(
            _endpoints()["pay"], json=payload, headers=await _auth_headers()
        )

        # A cached token can go stale early (e.g. rotated credentials). Retry
        # exactly once with a forced refresh before giving up.
        if resp.status_code in (401, 403):
            logger.warning(
                f"PhonePe returned HTTP {resp.status_code} — refreshing token and retrying once."
            )
            await _get_access_token(force_refresh=True)
            resp = await client.post(
                _endpoints()["pay"], json=payload, headers=await _auth_headers()
            )

    if resp.status_code >= 400:
        raise RuntimeError(
            f"PhonePe create-payment failed: HTTP {resp.status_code} — {resp.text[:500]}"
        )

    data = resp.json()
    redirect_url = data.get("redirectUrl")
    if not redirect_url:
        raise RuntimeError(f"PhonePe create-payment response had no redirectUrl: {data}")

    logger.info(
        f"Created PhonePe checkout merchantOrderId={merchant_order_id} "
        f"orderId={data.get('orderId')} url={redirect_url}"
    )

    return {
        "id": merchant_order_id,
        "short_url": redirect_url,
        "phonepe_order_id": data.get("orderId"),
        "state": data.get("state"),
        "expire_at": data.get("expireAt"),
    }


def link_is_still_valid(created_at_age_seconds: float) -> bool:
    """
    Whether a checkout URL created `created_at_age_seconds` ago can still be
    re-sent to a user, or whether a fresh one must be created.

    Razorpay payment links never expired, so re-sending a stored URL was always
    safe. PhonePe checkout URLs do expire, so callers that cache a link (see
    _maybe_send_premium_offer in main.py) must gate the reuse on this. A small
    safety margin means we don't hand someone a link that dies while they're
    reading the message.
    """
    settings = get_settings()
    usable_window = settings.PHONEPE_LINK_EXPIRE_AFTER_SECONDS - _TOKEN_REFRESH_MARGIN_SECONDS
    return created_at_age_seconds < max(usable_window, 0)


def verify_webhook_authorization(received_authorization: str) -> bool:
    """
    Verify a PhonePe webhook genuinely came from PhonePe.

    PhonePe authenticates callbacks differently from Razorpay: instead of an
    HMAC over the raw request body, it sends

        Authorization: SHA256(username:password)

    where username/password are the webhook credentials you configured in the
    PhonePe dashboard. We recompute that digest from our own configured pair
    and compare in constant time to avoid timing attacks.

    Because the digest doesn't cover the body, the body itself is not
    authenticated by this check — which is exactly why the webhook handler
    looks the order up in our own DB by merchantOrderId and never trusts an
    amount or phone number straight off the wire.
    """
    settings = get_settings()
    username = settings.PHONEPE_WEBHOOK_USERNAME
    password = settings.PHONEPE_WEBHOOK_PASSWORD

    if not (username and password):
        logger.error(
            "PHONEPE_WEBHOOK_USERNAME / PHONEPE_WEBHOOK_PASSWORD are not configured "
            "— rejecting webhook."
        )
        return False

    expected = hashlib.sha256(f"{username}:{password}".encode("utf-8")).hexdigest()

    # PhonePe sends the bare hex digest. Tolerate a "SHA256 " prefix and case
    # differences, both of which have been seen in the wild, without weakening
    # the comparison itself.
    received = (received_authorization or "").strip()
    if received.upper().startswith("SHA256 "):
        received = received[len("SHA256 "):].strip()

    return hmac.compare_digest(expected, received.lower())
