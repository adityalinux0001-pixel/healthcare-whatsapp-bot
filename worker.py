import logging
import sys

from app.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger("whatsapp_bot_worker")


def main() -> None:
    settings = get_settings()
    backend = (settings.queue_backend or "rq").lower()

    if backend == "rq":
        from redis import Redis
        from rq import Queue, Worker

        # NOTE: newer RQ versions (2.x) removed the `Connection` context
        # manager that older RQ (1.x) required — Queue/Worker now just
        # take `connection=` directly instead. This works with both old
        # and new installed RQ versions.
        redis_conn = Redis.from_url(settings.redis_url)
        queue = Queue(settings.queue_name or "default", connection=redis_conn)
        logger.info("Starting RQ worker for WhatsApp bot")
        worker = Worker([queue], connection=redis_conn, name="whatsapp_bot_worker")
        worker.work()
        return

    if backend == "celery":
        try:
            from app.queueing import get_celery_app
        except ImportError as exc:
            logger.error("Celery backend selected but queueing module cannot be imported: %s", exc)
            sys.exit(1)

        celery_app = get_celery_app()
        logger.info("Starting Celery worker for WhatsApp bot")
        celery_app.worker_main([
            "worker",
            "--loglevel=info",
            "--concurrency=1",
            "-Q",
            settings.queue_name or "default",
        ])
        return

    if backend == "huey":
        try:
            from app.queueing import get_huey
            from huey.consumer import Consumer
        except ImportError as exc:
            logger.error("Huey backend selected but Huey is not installed: %s", exc)
            sys.exit(1)

        huey = get_huey()
        logger.info("Starting Huey consumer for WhatsApp bot")
        consumer = Consumer(huey, workers=1)
        consumer.run()
        return

    if backend == "kafka":
        try:
            import asyncio

            from app.kafka_client import ensure_topics, run_consumer_loop
            from app.main import _handle_incoming
        except ImportError as exc:
            logger.error("Kafka backend selected but confluent-kafka is not installed: %s", exc)
            sys.exit(1)

        # Best-effort topic creation — no-op / warns harmlessly if the
        # topics already exist or if auto-creation is disabled and you're
        # provisioning topics another way.
        ensure_topics(
            [settings.kafka_inbound_topic], num_partitions=settings.kafka_num_partitions
        )

        # IMPORTANT: unlike RQ/Celery/Huey (which run _run_handle, doing a
        # fresh asyncio.run() per job — fine there because each job is
        # otherwise isolated), the Kafka consumer loop is one long-running
        # process handling many messages back to back on the SAME asyncio
        # event loop. asyncio.run() creates AND CLOSES a new loop every
        # single call. Shared async clients that get lazily created and
        # cached on first use — e.g. app/redis_client.py's module-level
        # redis.asyncio singleton, used by the idempotency guard and the
        # Gemini concurrency semaphore — bind their connections to
        # whichever loop was running when they were first created. Once
        # that loop closes after message #1, message #2 tries to reuse
        # the same cached Redis client against a NOW-CLOSED loop, raising
        # "Future attached to a different loop" / "Event loop is closed".
        # Fix: create ONE event loop for this worker process up front and
        # run every message's handling coroutine on that same loop via
        # run_until_complete, so the Redis client (and any other lazily-
        # cached async client) is always used from the loop it was born on.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        def _handle_kafka_message(raw_msg: dict) -> None:
            loop.run_until_complete(_handle_incoming(raw_msg))

        logger.info("Starting Kafka inbound consumer for WhatsApp bot")
        try:
            run_consumer_loop(
                topic=settings.kafka_inbound_topic,
                group_id=settings.kafka_inbound_group_id,
                handle_message=_handle_kafka_message,
            )
        finally:
            loop.close()
        return

    logger.error("Unsupported queue backend: %s", settings.queue_backend)
    sys.exit(1)


if __name__ == "__main__":
    main()