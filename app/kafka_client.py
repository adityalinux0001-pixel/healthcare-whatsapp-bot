"""
Kafka producer/consumer helpers.

Shared by:
- app/queueing.py        (inbound producer — webhook receiver side)
- worker.py               (inbound consumer — worker microservice side)
- app/outbound_queue.py   (outbound producer — wherever we'd otherwise
                           call whatsapp.send_* directly)
- outbound_worker.py      (outbound consumer — actually talks to the
                           WhatsApp Cloud API)

Design notes
------------
Messages are keyed by phone number (`key=phone_number.encode()`). Kafka
guarantees ordering only within a partition, and the default partitioner
hashes the key to pick a partition — so keying by phone number means every
message for a given user always lands on the same partition and is
therefore processed in order by whichever single consumer owns that
partition, while different users' messages fan out across partitions/
workers for concurrency. This mirrors "Partitioned by Phone Number" in the
target architecture diagram.

We use confluent-kafka (librdkafka bindings) for both producer and
consumer — it's the fastest/most complete Python client and is what's
usually meant by "production Kafka client" in this ecosystem.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Optional

from confluent_kafka import Consumer, KafkaError, KafkaException, Producer
from confluent_kafka.admin import AdminClient, NewTopic

from app.config import get_settings

logger = logging.getLogger(__name__)

_producer: Optional[Producer] = None


def _delivery_report(err, msg) -> None:
    if err is not None:
        logger.error(f"❌ Kafka delivery failed [{msg.topic()}]: {err}")
    else:
        logger.debug(
            f"✅ Kafka delivered to {msg.topic()}[{msg.partition()}]@{msg.offset()}"
        )


def get_producer() -> Producer:
    """Return a process-wide singleton Kafka producer.

    One Producer instance per process is the recommended usage pattern —
    it batches and manages its own background delivery thread internally,
    so there's no benefit (and real overhead) to creating a new one per
    call.
    """
    global _producer
    if _producer is None:
        settings = get_settings()
        _producer = Producer(
            {
                "bootstrap.servers": settings.kafka_bootstrap_servers,
                "acks": settings.kafka_producer_acks,
                # Retry transient broker errors instead of failing the
                # publish outright — the webhook ack has already been
                # returned to Meta by this point, so we can afford to
                # retry here without risking a duplicate delivery from
                # Meta's side.
                "retries": 5,
                "retry.backoff.ms": 200,
                "linger.ms": 20,
                "compression.type": "lz4",
                "enable.idempotence": True,
            }
        )
    return _producer


def publish(topic: str, key: str, value: dict[str, Any]) -> None:
    """Publish a JSON-serializable dict to `topic`, partitioned by `key`.

    Non-blocking: hands off to librdkafka's internal queue and returns.
    Call flush() (e.g. at shutdown, or after a burst in a script) if you
    need a synchronous guarantee that everything has actually gone out.
    """
    producer = get_producer()
    try:
        producer.produce(
            topic,
            key=key.encode("utf-8"),
            value=json.dumps(value).encode("utf-8"),
            callback=_delivery_report,
        )
        # Serve delivery-report callbacks / internal queue without
        # blocking for acks — this is what actually gets bytes onto the
        # wire promptly instead of waiting for linger.ms/batching alone.
        producer.poll(0)
    except BufferError:
        logger.warning("Kafka producer queue full — flushing and retrying once.")
        producer.flush(5)
        producer.produce(
            topic,
            key=key.encode("utf-8"),
            value=json.dumps(value).encode("utf-8"),
            callback=_delivery_report,
        )
        producer.poll(0)


def flush(timeout: float = 10.0) -> None:
    if _producer is not None:
        _producer.flush(timeout)


def ensure_topics(topics: list[str], num_partitions: int, replication_factor: int = 1) -> None:
    """Best-effort topic creation (idempotent — ignores 'already exists').

    Useful for local/dev/docker-compose where auto.create.topics.enable
    may be off. In a managed production cluster you'd normally provision
    topics via Terraform/kafka-topics.sh instead and can skip calling
    this, but it's safe to call either way.
    """
    settings = get_settings()
    admin = AdminClient({"bootstrap.servers": settings.kafka_bootstrap_servers})
    new_topics = [
        NewTopic(t, num_partitions=num_partitions, replication_factor=replication_factor)
        for t in topics
    ]
    futures = admin.create_topics(new_topics, request_timeout=15)
    for topic, future in futures.items():
        try:
            future.result()
            logger.info(f"✅ Kafka topic ready: {topic}")
        except KafkaException as e:
            # Code 36 = TOPIC_ALREADY_EXISTS — expected on every restart.
            if "already exists" in str(e).lower():
                logger.debug(f"Kafka topic already exists: {topic}")
            else:
                logger.warning(f"⚠️ Could not create Kafka topic {topic}: {e}")


def run_consumer_loop(
    topic: str,
    group_id: str,
    handle_message: Callable[[dict[str, Any]], None],
    poll_timeout: float = 1.0,
) -> None:
    """Blocking consume loop — call from a worker process's main().

    `handle_message` receives the decoded JSON dict for each message. Offsets
    are committed only after `handle_message` returns without raising, so a
    worker crash mid-processing results in at-least-once redelivery (the
    same message may be handled twice) rather than silently dropping work —
    downstream handlers should already be idempotent (see
    app/idempotency.py, already used for inbound WhatsApp messages).
    """
    settings = get_settings()
    consumer = Consumer(
        {
            "bootstrap.servers": settings.kafka_bootstrap_servers,
            "group.id": group_id,
            "auto.offset.reset": settings.kafka_consumer_auto_offset_reset,
            # Commit offsets ourselves, after successful processing, not
            # automatically in the background — see docstring above.
            "enable.auto.commit": False,
        }
    )
    consumer.subscribe([topic])
    logger.info(f"🟢 Kafka consumer started | topic={topic} group={group_id}")

    try:
        while True:
            msg = consumer.poll(poll_timeout)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                logger.error(f"Kafka consumer error: {msg.error()}")
                continue

            try:
                payload = json.loads(msg.value().decode("utf-8"))
                handle_message(payload)
                consumer.commit(msg, asynchronous=False)
            except Exception:
                logger.error(
                    f"❌ Failed to process Kafka message from {topic} "
                    f"[partition={msg.partition()} offset={msg.offset()}] "
                    "— NOT committing offset, will be redelivered.",
                    exc_info=True,
                )
    except KeyboardInterrupt:
        logger.info("Kafka consumer interrupted — shutting down.")
    finally:
        consumer.close()