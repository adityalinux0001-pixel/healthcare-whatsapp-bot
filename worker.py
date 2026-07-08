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

            from app.kafka_client import ensure_topics, run_consumer_loop_async
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

        # Run the whole consumer loop as one coroutine on one asyncio
        # loop for the lifetime of the process. Unlike the old
        # run_until_complete-per-message approach, this loop stays open
        # the entire time — so any lazily-cached async client (e.g.
        # app/redis_client.py's module-level redis.asyncio singleton)
        # only ever binds to this one loop, and multiple messages'
        # _handle_incoming() coroutines can genuinely run concurrently as
        # tasks instead of one fully finishing before the next starts.
        # This is what fixes "whoever messages first blocks everyone
        # else" — different users' messages land on different Kafka
        # partitions (keyed by phone number) and now actually get
        # processed in parallel instead of queueing behind each other on
        # a single blocking consumer loop.
        logger.info("Starting Kafka inbound consumer for WhatsApp bot (concurrent)")
        try:
            asyncio.run(
                run_consumer_loop_async(
                    topic=settings.kafka_inbound_topic,
                    group_id=settings.kafka_inbound_group_id,
                    handle_message=_handle_incoming,
                    max_concurrent=settings.kafka_worker_max_concurrent,
                )
            )
        except KeyboardInterrupt:
            logger.info("Kafka worker interrupted — shutting down.")
        return

    logger.error("Unsupported queue backend: %s", settings.queue_backend)
    sys.exit(1)


if __name__ == "__main__":
    main()