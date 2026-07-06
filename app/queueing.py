import asyncio
from typing import Any

from redis import Redis
from app.config import get_settings

try:
    from rq import Queue
except ImportError:  # RQ may not be installed when using another backend
    Queue = None  # type: ignore

try:
    from celery import Celery
except ImportError:
    Celery = None  # type: ignore

try:
    from huey import RedisHuey
except ImportError:
    RedisHuey = None  # type: ignore

QUEUE_NAME = "default"
JOB_TIMEOUT = 1200
RESULT_TTL = 0

_celery_app: Any = None
_huey: Any = None


def _get_settings():
    return get_settings()


def _get_redis_conn() -> Redis:
    settings = _get_settings()
    return Redis.from_url(settings.redis_url)


def _run_handle(raw_msg: dict) -> None:
    from app.main import _handle_incoming

    asyncio.run(_handle_incoming(raw_msg))


def get_celery_app() -> Any:
    global _celery_app
    settings = _get_settings()
    if Celery is None:
        raise RuntimeError("Celery is not installed")
    if _celery_app is not None:
        return _celery_app

    _celery_app = Celery(
        "whatsapp_bot",
        broker=settings.redis_url,
        backend=settings.redis_url,
    )
    _celery_app.conf.task_serializer = "json"
    _celery_app.conf.result_serializer = "json"
    _celery_app.conf.accept_content = ["json"]
    _celery_app.conf.task_default_queue = settings.queue_name or QUEUE_NAME
    _celery_app.conf.task_acks_late = True
    _celery_app.conf.worker_prefetch_multiplier = 1
    _celery_app.conf.broker_transport_options = {"visibility_timeout": 3600}

    @_celery_app.task(name="app.queueing.process_incoming_celery_job")
    def process_incoming_celery_job(message: dict) -> None:
        _run_handle(message)

    return _celery_app


def get_huey() -> Any:
    global _huey
    settings = _get_settings()
    if RedisHuey is None:
        raise RuntimeError("Huey is not installed")
    if _huey is not None:
        return _huey

    _huey = RedisHuey(settings.queue_name or QUEUE_NAME, url=settings.redis_url)

    @_huey.task(name="app.queueing.process_incoming_huey")
    def process_incoming_huey(message: dict) -> None:
        _run_handle(message)

    return _huey


def _enqueue_rq(raw_msg: dict) -> str:
    settings = _get_settings()
    if Queue is None:
        raise RuntimeError("RQ is not installed")
    queue = Queue(settings.queue_name or QUEUE_NAME, connection=_get_redis_conn())
    job = queue.enqueue(
        process_incoming_job,
        raw_msg,
        job_timeout=JOB_TIMEOUT,
        result_ttl=RESULT_TTL,
    )
    return job.id


def _enqueue_celery(raw_msg: dict) -> str:
    celery_app = get_celery_app()
    result = celery_app.send_task(
        "app.queueing.process_incoming_celery_job",
        args=[raw_msg],
        queue=_get_settings().queue_name or QUEUE_NAME,
    )
    return str(result.id)


def _enqueue_huey(raw_msg: dict) -> str:
    huey = get_huey()
    task = huey.enqueue(process_incoming_huey, raw_msg)
    return str(task.id)


def enqueue_incoming(raw_msg: dict) -> str:
    settings = _get_settings()
    backend = (settings.queue_backend or "rq").lower()
    if backend == "rq":
        return _enqueue_rq(raw_msg)
    if backend == "celery":
        return _enqueue_celery(raw_msg)
    if backend == "huey":
        return _enqueue_huey(raw_msg)

    raise ValueError(f"Unsupported queue backend: {settings.queue_backend}")


def process_incoming_job(raw_msg: dict) -> None:
    _run_handle(raw_msg)
