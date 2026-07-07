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
_huey_process_incoming_task: Any = None  # cached @_huey.task-wrapped function


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
    global _huey, _huey_process_incoming_task
    settings = _get_settings()
    if RedisHuey is None:
        raise RuntimeError("Huey is not installed")
    if _huey is not None:
        return _huey

    _huey = RedisHuey(settings.queue_name or QUEUE_NAME, url=settings.redis_url)

    @_huey.task(name="app.queueing.process_incoming_huey")
    def process_incoming_huey(message: dict) -> None:
        _run_handle(message)

    # process_incoming_huey only exists in this local scope — cache it at
    # module level so _enqueue_huey (called after get_huey() elsewhere) can
    # actually reach it instead of hitting an undefined-name error.
    _huey_process_incoming_task = process_incoming_huey

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
    get_huey()  # ensures _huey_process_incoming_task is populated
    task = _huey_process_incoming_task.enqueue(raw_msg)
    return str(task.id)


def _to_plain_dict(raw_msg: Any) -> dict:
    """Normalize raw_msg to a plain JSON-serializable dict.

    raw_msg is a WebhookMessage Pydantic model (see app/models.py) when it
    comes straight from the webhook handler's `value.messages` list — NOT
    a dict. RQ/Celery/Huey never hit this problem because they pickle
    arguments (which works fine on Pydantic objects), but Kafka publishing
    goes through json.dumps(), which cannot serialize a Pydantic model
    directly. model_dump(by_alias=True) converts it to a dict using the
    original WhatsApp field names (e.g. "from" instead of "from_"), which
    is exactly what IncomingMessage.from_raw() expects on the consumer
    side (see app/models.py's from_raw, which does the same conversion).
    """
    if isinstance(raw_msg, dict):
        return raw_msg
    if hasattr(raw_msg, "model_dump"):
        return raw_msg.model_dump(by_alias=True)
    raise TypeError(f"Cannot convert {type(raw_msg)!r} to dict for Kafka publish")


def _extract_phone_number(raw_msg: dict) -> str:
    """Best-effort phone number extraction for Kafka partition keying.

    Expects a plain dict (see _to_plain_dict) — falls back to 'unknown'
    (all such messages share one partition) rather than raising, so a
    malformed message still gets queued and the normal handler/idempotency
    logic can reject it, instead of it being lost before it ever reaches
    Kafka.
    """
    return str(raw_msg.get("from") or raw_msg.get("sender") or "unknown")


def _enqueue_kafka(raw_msg: Any) -> str:
    from app.kafka_client import publish

    settings = _get_settings()
    raw_msg_dict = _to_plain_dict(raw_msg)
    phone_number = _extract_phone_number(raw_msg_dict)
    publish(settings.kafka_inbound_topic, key=phone_number, value=raw_msg_dict)
    # Kafka doesn't hand back a job id the way RQ/Celery do — the
    # (topic, key) pair is the closest analogue and is useful in logs.
    return f"{settings.kafka_inbound_topic}:{phone_number}"


def enqueue_incoming(raw_msg: dict) -> str:
    settings = _get_settings()
    backend = (settings.queue_backend or "rq").lower()
    if backend == "rq":
        return _enqueue_rq(raw_msg)
    if backend == "celery":
        return _enqueue_celery(raw_msg)
    if backend == "huey":
        return _enqueue_huey(raw_msg)
    if backend == "kafka":
        return _enqueue_kafka(raw_msg)

    raise ValueError(f"Unsupported queue backend: {settings.queue_backend}")


def process_incoming_job(raw_msg: dict) -> None:
    _run_handle(raw_msg)