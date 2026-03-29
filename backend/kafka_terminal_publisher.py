"""
Background publisher that drains terminal events from the outbox into Kafka.
"""
from __future__ import annotations

import json
import logging
import socket
import time
import uuid
from typing import Any

try:
    from kafka import KafkaProducer
except Exception:  # pragma: no cover - optional dependency during local dev
    KafkaProducer = None  # type: ignore[assignment]

from backend.task_event_outbox import TaskEventOutboxItem, TaskEventOutboxStore
from config import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_CLIENT_ID,
    KAFKA_ENABLED,
    KAFKA_MESSAGE_MAX_BYTES,
    KAFKA_PUBLISHER_BATCH_SIZE,
    KAFKA_PUBLISHER_POLL_SECONDS,
    KAFKA_SASL_MECHANISM,
    KAFKA_SASL_PASSWORD,
    KAFKA_SASL_USERNAME,
    KAFKA_SECURITY_PROTOCOL,
)

logger = logging.getLogger(__name__)


class KafkaTerminalPublisher:
    def __init__(self, *, store: TaskEventOutboxStore | None = None, producer: Any | None = None) -> None:
        self.store = store or TaskEventOutboxStore()
        self.instance_id = f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
        self.producer = producer or self._build_producer()

    def publish_once(self, batch_size: int = KAFKA_PUBLISHER_BATCH_SIZE) -> int:
        claimed = self.store.claim_batch(batch_size=max(1, batch_size), locked_by=self.instance_id)
        if not claimed:
            return 0
        published = 0
        for item in claimed:
            if self._publish_item(item):
                published += 1
        return published

    def publish_forever(self, poll_seconds: float = KAFKA_PUBLISHER_POLL_SECONDS) -> None:
        logger.info("Kafka terminal publisher started instance_id=%s", self.instance_id)
        while True:
            published = self.publish_once()
            if published == 0:
                time.sleep(max(0.5, poll_seconds))

    def close(self) -> None:
        try:
            self.producer.flush(timeout=10)
        finally:
            self.producer.close()

    def _publish_item(self, item: TaskEventOutboxItem) -> bool:
        payload_bytes = json.dumps(item.payload_json, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        try:
            future = self.producer.send(
                item.topic,
                key=item.event_key.encode("utf-8"),
                value=payload_bytes,
            )
            future.get(timeout=30)
        except Exception as exc:
            logger.exception("Kafka publish failed for task_id=%s outbox_id=%s", item.task_id, item.outbox_id)
            self.store.mark_retry(item.outbox_id, error=str(exc))
            return False

        self.store.mark_published(item.outbox_id)
        return True

    def _build_producer(self) -> Any:
        if not KAFKA_ENABLED:
            raise RuntimeError("Kafka terminal publisher 未启用")
        if KafkaProducer is None:
            raise RuntimeError("kafka-python 未安装，无法启动 Kafka terminal publisher")

        producer_kwargs = {
            "bootstrap_servers": list(KAFKA_BOOTSTRAP_SERVERS),
            "client_id": KAFKA_CLIENT_ID,
            "acks": "all",
            "retries": 3,
            "max_request_size": KAFKA_MESSAGE_MAX_BYTES,
            "value_serializer": lambda value: value,
            "key_serializer": lambda value: value,
        }
        if KAFKA_SECURITY_PROTOCOL:
            producer_kwargs["security_protocol"] = KAFKA_SECURITY_PROTOCOL
        if KAFKA_SASL_MECHANISM:
            producer_kwargs["sasl_mechanism"] = KAFKA_SASL_MECHANISM
        if KAFKA_SASL_USERNAME:
            producer_kwargs["sasl_plain_username"] = KAFKA_SASL_USERNAME
        if KAFKA_SASL_PASSWORD:
            producer_kwargs["sasl_plain_password"] = KAFKA_SASL_PASSWORD
        return KafkaProducer(**producer_kwargs)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    publisher = KafkaTerminalPublisher()
    try:
        publisher.publish_forever()
    finally:
        publisher.close()


if __name__ == "__main__":
    main()
