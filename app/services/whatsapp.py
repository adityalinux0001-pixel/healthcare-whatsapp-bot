"""
Updated WhatsApp Module with Voice Message Support
Handles sending voice messages and audio responses
"""

import logging
import json
import httpx
from app.core.config import get_settings

logger = logging.getLogger(__name__)

BASE_URL = "https://graph.facebook.com/v25.0"


_http_client: httpx.AsyncClient | None = None


def _client() -> httpx.AsyncClient:
    global _http_client
    # Recreate the client if it doesn't exist yet OR if it was ever closed.
    # This makes the module self-healing if anything closes the shared client.
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
        )
    return _http_client


async def send_text_message(to: str, text: str, reply_to: str = None) -> dict:
    """
    Send a text message via WhatsApp.

    Args:
        to: Recipient phone number (international format)
        text: Message text (no markdown)
        reply_to: Message ID to reply to (optional)

    Returns:
        Meta API response
    """
    settings = get_settings()

    headers = {
        "Authorization": f"Bearer {settings.WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to,
        "type": "text",
        "text": {"body": text},
    }

    if reply_to:
        payload["context"] = {"message_id": reply_to}

    try:
        url = f"{BASE_URL}/{settings.PHONE_NUMBER_ID}/messages"
        resp = await _client().post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()

        result = resp.json()
        logger.info(f"Text message sent to {to}: {result}")
        return result

    except Exception as e:
        logger.error(f"Failed to send text message: {e}", exc_info=True)
        raise


async def send_audio_message(
    to: str,
    audio_bytes: bytes,
    caption: str = None,
    reply_to: str = None,
) -> dict:
    """
    Send an audio message via WhatsApp.

    Uploads audio file to WhatsApp Cloud API and sends as message.

    Args:
        to: Recipient phone number (international format)
        audio_bytes: Raw MP3 audio bytes
        caption: Optional caption for the audio
        reply_to: Message ID to reply to (optional)

    Returns:
        Meta API response

    Raises:
        Exception: If upload or send fails
    """
    settings = get_settings()

    logger.info(f"Preparing to send {len(audio_bytes)} bytes of audio to {to}")

    try:
        # Step 1: Upload audio file to WhatsApp
        media_id = await _upload_media(audio_bytes, "audio/mpeg", "audio.mp3")
        logger.info(f"Audio uploaded with media_id: {media_id}")

        # Step 2: Send audio message with the media_id
        headers = {
            "Authorization": f"Bearer {settings.WHATSAPP_TOKEN}",
            "Content-Type": "application/json",
        }

        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "audio",
            "audio": {"media_object_id": media_id},
        }

        if caption:
            payload["caption"] = caption

        if reply_to:
            payload["context"] = {"message_id": reply_to}

        url = f"{BASE_URL}/{settings.PHONE_NUMBER_ID}/messages"
        resp = await _client().post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()

        result = resp.json()
        logger.info(f"Audio message sent to {to}: {result}")
        return result

    except Exception as e:
        logger.error(f"Failed to send audio message: {e}", exc_info=True)
        raise


async def _upload_media(
    file_bytes: bytes,
    mime_type: str,
    filename: str,
) -> str:
    """
    Upload media file to WhatsApp Cloud API.

    Args:
        file_bytes: Raw file bytes
        mime_type: MIME type (e.g., "audio/mpeg", "image/jpeg")
        filename: Original filename

    Returns:
        Media object ID for use in messages

    Raises:
        Exception: If upload fails
    """
    settings = get_settings()

    headers = {
        "Authorization": f"Bearer {settings.WHATSAPP_TOKEN}",
    }

    files = {
        "file": (filename, file_bytes, mime_type),
        "type": (None, mime_type),
    }

    try:
        url = f"{BASE_URL}/{settings.PHONE_NUMBER_ID}/media"
        resp = await _client().post(url, headers=headers, files=files, timeout=60)
        resp.raise_for_status()

        result = resp.json()
        media_id = result.get("h", result.get("id"))  # h or id depending on API version

        if not media_id:
            raise ValueError(f"No media ID in response: {result}")

        logger.info(f"Media uploaded successfully: {media_id}")
        return media_id

    except Exception as e:
        logger.error(f"Failed to upload media: {e}", exc_info=True)
        raise


async def send_document_message(
    to: str,
    file_bytes: bytes,
    filename: str,
    caption: str = None,
    reply_to: str = None,
) -> dict:
    """
    Send a document (PDF, Word, etc.) via WhatsApp.

    Args:
        to: Recipient phone number
        file_bytes: Raw file bytes
        filename: Filename with extension
        caption: Optional caption
        reply_to: Message ID to reply to (optional)

    Returns:
        Meta API response
    """
    settings = get_settings()

    logger.info(f"Preparing to send document '{filename}' to {to}")

    try:
        # Upload document
        mime_type = _get_mime_type(filename)
        media_id = await _upload_media(file_bytes, mime_type, filename)

        # Send document message
        headers = {
            "Authorization": f"Bearer {settings.WHATSAPP_TOKEN}",
            "Content-Type": "application/json",
        }

        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "document",
            "document": {
                "media_object_id": media_id,
                "filename": filename,
            },
        }

        if caption:
            payload["document"]["caption"] = caption

        if reply_to:
            payload["context"] = {"message_id": reply_to}

        url = f"{BASE_URL}/{settings.PHONE_NUMBER_ID}/messages"
        resp = await _client().post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()

        result = resp.json()
        logger.info(f"Document sent to {to}: {result}")
        return result

    except Exception as e:
        logger.error(f"Failed to send document: {e}", exc_info=True)
        raise


async def send_template_message(to: str, template_name: str, params: list = None) -> dict:
    """
    Send a template message (pre-approved by Meta).

    Args:
        to: Recipient phone number
        template_name: Name of approved template
        params: List of parameter values

    Returns:
        Meta API response
    """
    settings = get_settings()

    headers = {
        "Authorization": f"Bearer {settings.WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "template",
        "template": {"name": template_name},
    }

    if params:
        payload["template"]["parameters"] = {"body": {"parameters": [{"type": "text", "text": p} for p in params]}}

    try:
        url = f"{BASE_URL}/{settings.PHONE_NUMBER_ID}/messages"
        resp = await _client().post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()

        result = resp.json()
        logger.info(f"Template sent to {to}: {result}")
        return result

    except Exception as e:
        logger.error(f"Failed to send template: {e}", exc_info=True)
        raise


async def mark_as_read(message_id: str, show_typing: bool = False) -> dict:
    """
    Mark a received message as read via Meta Graph API v25.0.

    Args:
        message_id: the WhatsApp message ID (wamid...) to mark as read.
        show_typing: retained for signature compatibility.
    """
    if not message_id or not isinstance(message_id, str):
        logger.warning(f"⚠️ Skipping mark_as_read: invalid message_id '{message_id}'")
        return {}

    settings = get_settings()

    headers = {
        "Authorization": f"Bearer {settings.WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "messaging_product": "whatsapp",
        "status": "read",
        "message_id": message_id,
    }

    try:
        url = f"{BASE_URL}/{settings.PHONE_NUMBER_ID}/messages"
        # FIX: previously this used `async with _client() as client:`, which
        # calls __aexit__ -> aclose() on the SHARED module-level client as
        # soon as this function returned. That permanently closed the client
        # and broke every subsequent call in this module (send_text_message,
        # send_audio_message, etc.) with "client has been closed" errors.
        # Just reuse the shared client directly, same as every other function here.
        resp = await _client().post(url, json=payload, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"Failed to mark message as read ({message_id}): {e}")
        return {}


async def verify_token_valid() -> dict:
    """Verify that the WhatsApp token is valid."""
    settings = get_settings()

    headers = {
        "Authorization": f"Bearer {settings.WHATSAPP_TOKEN}",
    }

    try:
        url = f"{BASE_URL}/{settings.PHONE_NUMBER_ID}"
        resp = await _client().get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        return {"valid": True, "phone_number_id": settings.PHONE_NUMBER_ID}
    except Exception as e:
        logger.error(f"Token verification failed: {e}")
        return {"valid": False, "error": str(e)}


def _get_mime_type(filename: str) -> str:
    """Get MIME type from filename."""
    ext = filename.split(".")[-1].lower()
    mime_types = {
        "pdf": "application/pdf",
        "doc": "application/msword",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "xls": "application/vnd.ms-excel",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "ppt": "application/vnd.ms-powerpoint",
        "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "txt": "text/plain",
        "csv": "text/csv",
    }
    return mime_types.get(ext, "application/octet-stream")
