"""
Outbound worker entrypoint.

Consumes app_settings.kafka_outbound_topic and performs the actual
WhatsApp Cloud API calls (the "[ Outbound Message Queue ] ---> [ WhatsApp
Cloud API ]" step in the architecture diagram). Run one or more of these
as a separate process/container from the inbound worker(s) — outbound
send failures (e.g. WhatsApp rate limits) then can't back up or block
inbound message processing, and you can scale the two independently.

Only meaningful when QUEUE_BACKEND=kafka; for other backends, outbound
sends happen inline (see app/outbound_queue.py) and this process has
nothing to consume.

Usage:
    python outbound_worker.py
"""

import asyncio
import logging
import sys

from app.core.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger("whatsapp_bot_outbound_worker")


async def _handle_outbound_job_async(payload: dict) -> None:
    from app.core.outbound_queue import JOB_MARK_AS_READ, JOB_SEND_TEMPLATE, JOB_SEND_TEXT
    from app.services.whatsapp import mark_as_read, send_template_message, send_text_message

    job_type = payload.get("job_type")

    if job_type == JOB_SEND_TEXT:
        to = payload["to"]
        try:
            result = await send_text_message(
                to, payload["text"], reply_to=payload.get("reply_to")
            )
            logger.info(f"✅ Outbound text delivered to {to}: {result.get('messages')}")
        except Exception:
            logger.error(f"❌ Outbound text send failed for {to}", exc_info=True)

            raise

    elif job_type == JOB_SEND_TEMPLATE:
        to = payload["to"]
        try:
            result = await send_template_message(
                to, payload["template_name"], params=payload.get("params")
            )
            logger.info(f"✅ Outbound template delivered to {to}: {result.get('messages')}")
        except Exception:
            logger.error(f"❌ Outbound template send failed for {to}", exc_info=True)
            raise

    elif job_type == JOB_MARK_AS_READ:
        try:
            await mark_as_read(
                payload["message_id"], show_typing=payload.get("show_typing", False)
            )
        except Exception:
            # Read receipts are cosmetic — log but don't force a
            # redelivery/retry storm over something this low-stakes.
            logger.warning("Outbound mark-as-read failed (non-critical)", exc_info=True)

    else:
        logger.warning(f"Unknown outbound job_type={job_type!r} — dropping.")


def main() -> None:
    settings = get_settings()
    backend = (settings.QUEUE_BACKEND or "rq").lower()

    if backend != "kafka":
        logger.error(
            f"outbound_worker.py only applies when QUEUE_BACKEND=kafka "
            f"(currently '{settings.QUEUE_BACKEND}'). Nothing to consume — exiting."
        )
        sys.exit(1)

    from app.core.kafka_client import ensure_topics, run_consumer_loop

    ensure_topics(
        [settings.KAFKA_OUTBOUND_TOPIC], num_partitions=settings.KAFKA_NUM_PARTITIONS
    )


    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _handle_outbound_message(payload: dict) -> None:
        loop.run_until_complete(_handle_outbound_job_async(payload))

    logger.info("Starting Kafka outbound worker for WhatsApp bot")
    try:
        run_consumer_loop(
            topic=settings.KAFKA_OUTBOUND_TOPIC,
            group_id=settings.KAFKA_OUTBOUND_GROUP_ID,
            handle_message=_handle_outbound_message,
        )
    finally:
        loop.close()


if __name__ == "__main__":
    main()