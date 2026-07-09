import logging
import sys

from app.core.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger("whatsapp_bot_worker")


def main() -> None:
    settings = get_settings()
    backend = (settings.QUEUE_BACKEND or "rq").lower()

    if backend == "rq":
        from redis import Redis
        from rq import Queue, Worker

        redis_conn = Redis.from_url(settings.REDIS_URL)
        queue = Queue(settings.QUEUE_NAME, connection=redis_conn)
        logger.info("Starting RQ worker for WhatsApp bot")
        worker = Worker([queue], connection=redis_conn, name="whatsapp_bot_worker")
        worker.work()
        return

    if backend == "celery":
        try:
            from app.core.queueing import get_celery_app
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
            settings.QUEUE_NAME,
        ])
        return

    if backend == "huey":
        try:
            from app.core.queueing import get_huey
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

            from app.core.kafka_client import ensure_topics, run_consumer_loop_async
            from app.api.main import _handle_incoming
        except ImportError as exc:
            logger.error("Kafka backend selected but confluent-kafka is not installed: %s", exc)
            sys.exit(1)

        # Best-effort topic creation — no-op / warns harmlessly if the
        # topics already exist or if auto-creation is disabled and you're
        # provisioning topics another way.
        ensure_topics(
            [settings.KAFKA_INBOUND_TOPIC], num_partitions=settings.KAFKA_NUM_PARTITIONS
        )

        # Run consumer loop concurrently
        logger.info("Starting Kafka inbound consumer for WhatsApp bot (concurrent)")
        try:
            asyncio.run(
                run_consumer_loop_async(
                    topic=settings.KAFKA_INBOUND_TOPIC,
                    group_id=settings.KAFKA_INBOUND_GROUP_ID,
                    handle_message=_handle_incoming,
                    max_concurrent=settings.KAFKA_WORKER_MAX_CONCURRENT,
                )
            )
        except KeyboardInterrupt:
            logger.info("Kafka worker interrupted — shutting down.")
        return

    logger.error("Unsupported queue backend: %s", settings.QUEUE_BACKEND)
    sys.exit(1)


if __name__ == "__main__":
    main()