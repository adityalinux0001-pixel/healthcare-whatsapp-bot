"""
Outbound message queue (producer side).

Maps directly to the "[ Outbound Message Queue ] ---> [ WhatsApp Cloud API ]"
box in the target architecture. Instead of core engine code calling
app.whatsapp.send_text_message(...) directly (which does an inline HTTP
POST to Meta and blocks on it), it publishes a small JSON job onto the
`kafka_outbound_topic` here. A separate consumer process
(outbound_worker.py) drains that topic and performs the actual WhatsApp
Cloud API calls.

This is opt-in: enqueue_outbound_message() only actually queues via Kafka
when queue_backend == "kafka". Otherwise it transparently calls the normal
synchronous app.whatsapp functions, so existing behavior (RQ / Celery /
Huey / no queue at all) is completely unaffected — this keeps the change
additive rather than a rewrite of every call site.

Only text/template/mark-as-read are queued for now (the high-volume,
latency-insensitive paths). Audio/document sends, which carry raw bytes,
are intentionally NOT queued through Kafka by default — putting large
binary payloads through Kafka messages is usually the wrong tradeoff
(message size limits, broker disk pressure); if you need to queue those
too, upload the media to object storage first and queue a reference to it
instead of the raw bytes.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# Job "type" values understood by outbound_worker.py's dispatcher.
JOB_SEND_TEXT = "send_text"
JOB_SEND_TEMPLATE = "send_template"
JOB_MARK_AS_READ = "mark_as_read"


def _use_kafka() -> bool:
    return (get_settings().QUEUE_BACKEND or "rq").lower() == "kafka"


async def enqueue_send_text(to: str, text: str, reply_to: Optional[str] = None) -> dict[str, Any]:
    """Queue (or, if not using Kafka, immediately send) a text message."""
    if _use_kafka():
        from app.core.kafka_client import publish

        settings = get_settings()
        publish(
            settings.KAFKA_OUTBOUND_TOPIC,
            key=to,
            value={
                "job_type": JOB_SEND_TEXT,
                "to": to,
                "text": text,
                "reply_to": reply_to,
            },
        )
        return {"status": "queued", "topic": settings.KAFKA_OUTBOUND_TOPIC, "to": to}

    from app.services.whatsapp import send_text_message

    return await send_text_message(to, text, reply_to=reply_to)


async def enqueue_send_template(to: str, template_name: str, params: Optional[list] = None) -> dict[str, Any]:
    if _use_kafka():
        from app.core.kafka_client import publish

        settings = get_settings()
        publish(
            settings.KAFKA_OUTBOUND_TOPIC,
            key=to,
            value={
                "job_type": JOB_SEND_TEMPLATE,
                "to": to,
                "template_name": template_name,
                "params": params,
            },
        )
        return {"status": "queued", "topic": settings.KAFKA_OUTBOUND_TOPIC, "to": to}

    from app.services.whatsapp import send_template_message

    return await send_template_message(to, template_name, params=params)


async def enqueue_mark_as_read(message_id: str, to: str, show_typing: bool = False) -> dict[str, Any]:
    """
    Queue a read-receipt.

    Note: mark_as_read only needs the message_id per the WhatsApp API, but
    we still require `to` here purely so this job can be keyed/partitioned
    by phone number like every other outbound job for this user, keeping
    per-user ordering consistent across job types.
    """
    if _use_kafka():
        from app.core.kafka_client import publish

        settings = get_settings()
        publish(
            settings.KAFKA_OUTBOUND_TOPIC,
            key=to,
            value={
                "job_type": JOB_MARK_AS_READ,
                "message_id": message_id,
                "show_typing": show_typing,
            },
        )
        return {"status": "queued", "topic": settings.KAFKA_OUTBOUND_TOPIC}

    from app.services.whatsapp import mark_as_read

    return await mark_as_read(message_id, show_typing=show_typing)