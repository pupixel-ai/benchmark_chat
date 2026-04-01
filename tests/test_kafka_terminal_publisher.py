from __future__ import annotations

import unittest

from backend.kafka_terminal_publisher import KafkaTerminalPublisher
from backend.task_event_outbox import TaskEventOutboxItem


class _FakeFuture:
    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = exc

    def get(self, timeout: int | None = None) -> None:
        del timeout
        if self.exc is not None:
            raise self.exc


class _FakeProducer:
    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = exc
        self.calls: list[dict] = []
        self.flushed = False
        self.closed = False

    def send(self, topic: str, *, key: bytes, value: bytes) -> _FakeFuture:
        self.calls.append({"topic": topic, "key": key, "value": value})
        return _FakeFuture(self.exc)

    def flush(self, timeout: int | None = None) -> None:
        del timeout
        self.flushed = True

    def close(self) -> None:
        self.closed = True


class _FakeConfluentProducer:
    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = exc
        self.calls: list[dict] = []
        self.flushed = False
        self.closed = False

    def publish(self, topic: str, *, key: bytes, value: bytes, timeout: float = 30.0) -> None:
        self.calls.append({"topic": topic, "key": key, "value": value, "timeout": timeout})
        if self.exc is not None:
            raise self.exc

    def flush(self, timeout: int | None = None) -> None:
        del timeout
        self.flushed = True

    def close(self) -> None:
        self.closed = True


class _FakeStore:
    def __init__(self, items: list[TaskEventOutboxItem]) -> None:
        self.items = items
        self.claimed = 0
        self.published: list[str] = []
        self.retried: list[tuple[str, str]] = []

    def claim_batch(self, *, batch_size: int, locked_by: str) -> list[TaskEventOutboxItem]:
        del batch_size, locked_by
        if self.claimed:
            return []
        self.claimed += 1
        return list(self.items)

    def mark_published(self, outbox_id: str) -> None:
        self.published.append(outbox_id)

    def mark_retry(self, outbox_id: str, *, error: str) -> None:
        self.retried.append((outbox_id, error))


class KafkaTerminalPublisherTests(unittest.TestCase):
    def _item(self) -> TaskEventOutboxItem:
        return TaskEventOutboxItem(
            outbox_id="outbox-1",
            event_id="event-1",
            topic="memory.task.terminal.v1",
            event_type="task.completed",
            task_id="task-1",
            event_key="task-1",
            dedupe_key="task:task-1:terminal:task.completed",
            payload_json={"event_id": "event-1", "event_type": "task.completed", "task_id": "task-1"},
            status="publishing",
            attempt_count=0,
            available_at="2026-03-29T00:00:00",
            locked_at="2026-03-29T00:00:00",
            locked_by="publisher-1",
            published_at=None,
            last_error=None,
            created_at="2026-03-29T00:00:00",
        )

    def test_publish_once_marks_rows_as_published(self) -> None:
        store = _FakeStore([self._item()])
        producer = _FakeProducer()
        publisher = KafkaTerminalPublisher(store=store, producer=producer)

        published = publisher.publish_once()

        self.assertEqual(published, 1)
        self.assertEqual(store.published, ["outbox-1"])
        self.assertEqual(store.retried, [])
        self.assertEqual(producer.calls[0]["topic"], "memory.task.terminal.v1")
        self.assertEqual(producer.calls[0]["key"], b"task-1")

    def test_publish_once_marks_rows_for_retry_when_send_fails(self) -> None:
        store = _FakeStore([self._item()])
        producer = _FakeProducer(exc=RuntimeError("broker unavailable"))
        publisher = KafkaTerminalPublisher(store=store, producer=producer)

        published = publisher.publish_once()

        self.assertEqual(published, 0)
        self.assertEqual(store.published, [])
        self.assertEqual(len(store.retried), 1)
        self.assertEqual(store.retried[0][0], "outbox-1")
        self.assertIn("broker unavailable", store.retried[0][1])

    def test_publish_once_supports_confluent_style_producer(self) -> None:
        store = _FakeStore([self._item()])
        producer = _FakeConfluentProducer()
        publisher = KafkaTerminalPublisher(store=store, producer=producer)

        published = publisher.publish_once()

        self.assertEqual(published, 1)
        self.assertEqual(store.published, ["outbox-1"])
        self.assertEqual(store.retried, [])
        self.assertEqual(producer.calls[0]["topic"], "memory.task.terminal.v1")
        self.assertEqual(producer.calls[0]["key"], b"task-1")
