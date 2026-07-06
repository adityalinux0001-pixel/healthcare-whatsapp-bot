"""
Updated WhatsApp Module with Voice Message Support
Handles sending voice messages and audio responses
"""

import logging
import json
import httpx
from app.config import get_settings

logger = logging.getLogger(__name__)

BASE_URL = "https://graph.facebook.com/v25.0"

# Reuse a single pooled/keep-alive HTTP client instead of opening a brand new
# TCP+TLS connection (via `async with httpx.AsyncClient() as client:`) on
# every single WhatsApp API call. Handshake overhead was adding real,
# avoidable latency to every outbound message/read-receipt/upload.
_http_client: httpx.AsyncClient | None = None


def _client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
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
        "Authorization": f"Bearer {settings.whatsapp_token}",
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
        url = f"{BASE_URL}/{settings.phone_number_id}/messages"
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
            "Authorization": f"Bearer {settings.whatsapp_token}",
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
        
        url = f"{BASE_URL}/{settings.phone_number_id}/messages"
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
        "Authorization": f"Bearer {settings.whatsapp_token}",
    }
    
    files = {
        "file": (filename, file_bytes, mime_type),
        "type": (None, mime_type),
    }
    
    try:
        url = f"{BASE_URL}/{settings.phone_number_id}/media"
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
            "Authorization": f"Bearer {settings.whatsapp_token}",
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
        
        url = f"{BASE_URL}/{settings.phone_number_id}/messages"
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
        "Authorization": f"Bearer {settings.whatsapp_token}",
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
        url = f"{BASE_URL}/{settings.phone_number_id}/messages"
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
    Mark a received message as read.

    Args:
        message_id: the WhatsApp message ID to mark as read.
        show_typing: if True, also shows the native WhatsApp "typing…"
            indicator to the user (three animated dots) for a few seconds
            or until the actual reply is sent, whichever comes first —
            whichever comes first is handled entirely by WhatsApp itself,
            no extra calls needed on our side. Purely cosmetic; failure to
            show it is not critical and never raises.
    """
    settings = get_settings()
    
    headers = {
        "Authorization": f"Bearer {settings.whatsapp_token}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "messaging_product": "whatsapp",
        "status": "read",
        "message_id": message_id,
    }

    if show_typing:
        payload["typing_indicator"] = {"type": "text"}
    
    try:
        url = f"{BASE_URL}/{settings.phone_number_id}/messages"
        resp = await _client().post(url, json=payload, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"Failed to mark message as read: {e}")
        # Don't raise - not critical


async def verify_token_valid() -> dict:
    """Verify that the WhatsApp token is valid."""
    settings = get_settings()
    
    headers = {
        "Authorization": f"Bearer {settings.whatsapp_token}",
    }
    
    try:
        url = f"{BASE_URL}/{settings.phone_number_id}"
        resp = await _client().get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        return {"valid": True, "phone_number_id": settings.phone_number_id}
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