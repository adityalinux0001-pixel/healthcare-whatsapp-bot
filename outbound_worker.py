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

from app.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger("whatsapp_bot_outbound_worker")


async def _handle_outbound_job_async(payload: dict) -> None:
    from app.outbound_queue import JOB_MARK_AS_READ, JOB_SEND_TEMPLATE, JOB_SEND_TEXT
    from app.whatsapp import mark_as_read, send_template_message, send_text_message

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
            # Re-raise so kafka_client's consumer loop does NOT commit the
            # offset — this job will be redelivered and retried rather
            # than silently dropped. WhatsApp send calls are not
            # inherently idempotent (a retry could double-send a
            # message on a transient network error even though Meta's
            # own API call actually succeeded) — an at-least-once
            # outbound queue is a deliberate tradeoff toward "make sure
            # the user gets their reply" over "never possibly duplicate
            # a message". Add your own dedupe key here if double-sends
            # become a real problem in practice.
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
    backend = (settings.queue_backend or "rq").lower()

    if backend != "kafka":
        logger.error(
            f"outbound_worker.py only applies when QUEUE_BACKEND=kafka "
            f"(currently '{settings.queue_backend}'). Nothing to consume — exiting."
        )
        sys.exit(1)

    from app.kafka_client import ensure_topics, run_consumer_loop

    ensure_topics(
        [settings.kafka_outbound_topic], num_partitions=settings.kafka_num_partitions
    )

    # IMPORTANT: one persistent event loop for the whole process, not a
    # fresh asyncio.run() per message. app/whatsapp.py's _client() caches
    # a single module-level httpx.AsyncClient the first time it's used —
    # that client (and its underlying connections) is bound to whichever
    # event loop was active when it was created. asyncio.run() tears the
    # loop down after every single message, so the SECOND message would
    # try to reuse a client bound to an already-closed loop and crash
    # with "Event loop is closed" / "attached to a different loop" —
    # exactly the bug hit on the inbound worker side for the same reason
    # (see worker.py's matching comment for the full explanation).
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _handle_outbound_message(payload: dict) -> None:
        loop.run_until_complete(_handle_outbound_job_async(payload))

    logger.info("Starting Kafka outbound worker for WhatsApp bot")
    try:
        run_consumer_loop(
            topic=settings.kafka_outbound_topic,
            group_id=settings.kafka_outbound_group_id,
            handle_message=_handle_outbound_message,
        )
    finally:
        loop.close()


if __name__ == "__main__":
    main()