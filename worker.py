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
        from rq import Connection, Queue, Worker

        redis_conn = Redis.from_url(settings.redis_url)
        queue = Queue(settings.queue_name or "default", connection=redis_conn)
        logger.info("Starting RQ worker for WhatsApp bot")
        with Connection(redis_conn):
            worker = Worker([queue], name="whatsapp_bot_worker")
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

    logger.error("Unsupported queue backend: %s", settings.queue_backend)
    sys.exit(1)


if __name__ == "__main__":
    main()
