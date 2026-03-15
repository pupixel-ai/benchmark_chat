"""External storage adapters for Redis / Neo4j / Milvus."""

from __future__ import annotations

import hashlib
import json
import math
import re
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import (
    MEMORY_EXTERNAL_SINKS_ENABLED,
    MEMORY_MILVUS_COLLECTION,
    MEMORY_MILVUS_DB_NAME,
    MEMORY_MILVUS_PASSWORD,
    MEMORY_MILVUS_TOKEN,
    MEMORY_MILVUS_URI,
    MEMORY_MILVUS_USER,
    MEMORY_MILVUS_VECTOR_DIM,
    MEMORY_NEO4J_DATABASE,
    MEMORY_NEO4J_PASSWORD,
    MEMORY_NEO4J_URI,
    MEMORY_NEO4J_USERNAME,
    MEMORY_REDIS_PREFIX,
    MEMORY_REDIS_URL,
)
from utils import save_json


NODE_GROUP_ID_FIELDS = {
    "user": "user_id",
    "persons": "person_uuid",
    "photos": "photo_uuid",
    "sessions": "session_uuid",
    "movements": "movement_uuid",
    "timelines": "timeline_uuid",
    "events": "event_uuid",
}


class MemoryStoragePublisher:
    """Publishes the materialized memory views to configured external sinks."""

    def __init__(self, task_dir: str | Path) -> None:
        self.task_dir = Path(task_dir)
        self.output_dir = self.task_dir / "output" / "memory"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_path = self.output_dir / "external_publish_report.json"

    def publish(self, storage: Dict[str, Any], *, user_id: str) -> Dict[str, Any]:
        report = {
            "enabled": MEMORY_EXTERNAL_SINKS_ENABLED,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "user_id": user_id,
            "redis": self._skip_report("disabled"),
            "neo4j": self._skip_report("disabled"),
            "milvus": self._skip_report("disabled"),
        }

        if MEMORY_EXTERNAL_SINKS_ENABLED:
            report["redis"] = RedisStorageAdapter(user_id=user_id).publish(storage.get("redis", {}))
            report["neo4j"] = Neo4jStorageAdapter().publish(storage.get("neo4j", {}))
            report["milvus"] = MilvusStorageAdapter(user_id=user_id).publish(storage.get("milvus", {}))

        save_json(report, str(self.report_path))
        report["report_path"] = str(self.report_path)
        return report

    def _skip_report(self, reason: str) -> Dict[str, Any]:
        return {"status": "skipped", "reason": reason}


class RedisStorageAdapter:
    def __init__(self, user_id: str) -> None:
        self.user_id = user_id

    def publish(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not MEMORY_REDIS_URL:
            return {"status": "skipped", "reason": "MEMORY_REDIS_URL not configured"}

        try:
            import redis

            client = redis.Redis.from_url(MEMORY_REDIS_URL, decode_responses=True)
            pipeline = client.pipeline()
            written_keys = []
            for key_name, item in payload.items():
                if not isinstance(item, dict):
                    continue
                redis_key = item.get("key") or f"{MEMORY_REDIS_PREFIX}:{self.user_id}:{key_name}"
                body = {sub_key: sub_value for sub_key, sub_value in item.items() if sub_key != "key"}
                pipeline.set(redis_key, json.dumps(body, ensure_ascii=False))
                written_keys.append(redis_key)
            pipeline.execute()
            return {
                "status": "published",
                "key_count": len(written_keys),
                "keys": written_keys,
            }
        except Exception as exc:
            return {
                "status": "failed",
                "reason": str(exc),
            }


class Neo4jStorageAdapter:
    def publish(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not MEMORY_NEO4J_URI:
            return {"status": "skipped", "reason": "MEMORY_NEO4J_URI not configured"}

        try:
            from neo4j import GraphDatabase

            auth = None
            if MEMORY_NEO4J_USERNAME:
                auth = (MEMORY_NEO4J_USERNAME, MEMORY_NEO4J_PASSWORD)
            driver = GraphDatabase.driver(MEMORY_NEO4J_URI, auth=auth)
            node_index: Dict[str, tuple[str, str]] = {}

            with driver.session(database=MEMORY_NEO4J_DATABASE or None) as session:
                nodes = payload.get("nodes", {})
                for group_name, records in nodes.items():
                    id_field = NODE_GROUP_ID_FIELDS.get(group_name)
                    if not id_field:
                        continue
                    for record in records:
                        if id_field not in record:
                            continue
                        labels = ":".join(self._sanitize_label(label) for label in record.get("labels", [])) or "MemoryNode"
                        props = dict(record.get("properties", {}))
                        props[id_field] = record[id_field]
                        session.run(
                            f"MERGE (n:{labels} {{{id_field}: $node_id}}) SET n += $props",
                            node_id=record[id_field],
                            props=props,
                        )
                        node_index[record[id_field]] = (labels, id_field)

                edge_count = 0
                skipped_edges = 0
                for edge in payload.get("edges", []):
                    left = node_index.get(edge.get("from_id"))
                    right = node_index.get(edge.get("to_id"))
                    if not left or not right:
                        skipped_edges += 1
                        continue
                    left_labels, left_id_field = left
                    right_labels, right_id_field = right
                    rel_type = self._sanitize_rel(edge.get("edge_type", "RELATED_TO"))
                    props = dict(edge.get("properties", {}))
                    props["edge_id"] = edge.get("edge_id")
                    session.run(
                        (
                            f"MATCH (a:{left_labels} {{{left_id_field}: $from_id}}) "
                            f"MATCH (b:{right_labels} {{{right_id_field}: $to_id}}) "
                            f"MERGE (a)-[r:{rel_type} {{edge_id: $edge_id}}]->(b) "
                            f"SET r += $props"
                        ),
                        from_id=edge.get("from_id"),
                        to_id=edge.get("to_id"),
                        edge_id=edge.get("edge_id"),
                        props=props,
                    )
                    edge_count += 1

            driver.close()
            return {
                "status": "published",
                "node_count": len(node_index),
                "edge_count": edge_count,
                "skipped_edges": skipped_edges,
            }
        except Exception as exc:
            return {"status": "failed", "reason": str(exc)}

    def _sanitize_label(self, value: str) -> str:
        cleaned = re.sub(r"[^0-9A-Za-z_]", "_", str(value or "MemoryNode"))
        if cleaned and cleaned[0].isdigit():
            cleaned = f"Node_{cleaned}"
        return cleaned or "MemoryNode"

    def _sanitize_rel(self, value: str) -> str:
        cleaned = re.sub(r"[^0-9A-Za-z_]", "_", str(value or "RELATED_TO").upper())
        if cleaned and cleaned[0].isdigit():
            cleaned = f"REL_{cleaned}"
        return cleaned or "RELATED_TO"


class MilvusStorageAdapter:
    def __init__(self, user_id: str) -> None:
        self.user_id = user_id

    def publish(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not MEMORY_MILVUS_URI:
            return {"status": "skipped", "reason": "MEMORY_MILVUS_URI not configured"}

        segments = payload.get("segments", [])
        if not segments:
            return {"status": "skipped", "reason": "no segments to publish"}

        client = None
        local_server = None
        try:
            from pymilvus import DataType, MilvusClient

            client, local_server = self._open_client(MilvusClient)
            if not client.has_collection(MEMORY_MILVUS_COLLECTION):
                schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
                schema.add_field("segment_uuid", DataType.VARCHAR, is_primary=True, max_length=64)
                schema.add_field("user_id", DataType.VARCHAR, max_length=128)
                schema.add_field("photo_uuid", DataType.VARCHAR, max_length=64)
                schema.add_field("event_uuid", DataType.VARCHAR, max_length=64)
                schema.add_field("person_uuid", DataType.VARCHAR, max_length=64)
                schema.add_field("session_uuid", DataType.VARCHAR, max_length=64)
                schema.add_field("segment_type", DataType.VARCHAR, max_length=64)
                schema.add_field("text", DataType.VARCHAR, max_length=8192)
                schema.add_field("sparse_terms", DataType.VARCHAR, max_length=2048)
                schema.add_field("embedding_source", DataType.VARCHAR, max_length=128)
                schema.add_field("importance_score", DataType.FLOAT)
                schema.add_field("evidence_refs_json", DataType.VARCHAR, max_length=8192)
                schema.add_field("vector", DataType.FLOAT_VECTOR, dim=MEMORY_MILVUS_VECTOR_DIM)
                index_params = MilvusClient.prepare_index_params()
                index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
                client.create_collection(
                    collection_name=MEMORY_MILVUS_COLLECTION,
                    schema=schema,
                    index_params=index_params,
                )

            rows = [self._segment_row(segment) for segment in segments]
            client.upsert(collection_name=MEMORY_MILVUS_COLLECTION, data=rows)
            return {
                "status": "published",
                "collection": MEMORY_MILVUS_COLLECTION,
                "segment_count": len(rows),
                "vector_dim": MEMORY_MILVUS_VECTOR_DIM,
                "mode": "local-db" if local_server is not None else "remote",
            }
        except Exception as exc:
            return {"status": "failed", "reason": str(exc)}
        finally:
            try:
                if client is not None:
                    client.close()
            except Exception:
                pass
            try:
                if local_server is not None:
                    local_server.stop()
            except Exception:
                pass

    def _open_client(self, client_cls: Any) -> Tuple[Any, Any | None]:
        if MEMORY_MILVUS_URI.endswith(".db"):
            from milvus_lite.server import Server

            db_path = Path(MEMORY_MILVUS_URI).expanduser().resolve()
            db_path.parent.mkdir(parents=True, exist_ok=True)
            address = f"127.0.0.1:{self._find_free_port()}"
            local_server = Server(str(db_path), address)
            if not local_server.init() or not local_server.start():
                raise RuntimeError(f"failed to start milvus-lite tcp bridge for {db_path}")

            client = client_cls(
                uri=f"http://{address}",
                timeout=10,
            )
            return client, local_server

        client = client_cls(
            uri=MEMORY_MILVUS_URI,
            user=MEMORY_MILVUS_USER,
            password=MEMORY_MILVUS_PASSWORD,
            token=MEMORY_MILVUS_TOKEN,
            db_name=MEMORY_MILVUS_DB_NAME,
        )
        return client, None

    def _find_free_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            return int(sock.getsockname()[1])

    def _segment_row(self, segment: Dict[str, Any]) -> Dict[str, Any]:
        text = str(segment.get("text") or "")
        sparse_terms = ",".join(segment.get("sparse_terms", [])[:64])
        evidence_refs_json = json.dumps(segment.get("evidence_refs", []), ensure_ascii=False)
        return {
            "segment_uuid": str(segment.get("segment_uuid")),
            "user_id": self.user_id,
            "photo_uuid": str(segment.get("photo_uuid") or ""),
            "event_uuid": str(segment.get("event_uuid") or ""),
            "person_uuid": str(segment.get("person_uuid") or ""),
            "session_uuid": str(segment.get("session_uuid") or ""),
            "segment_type": str(segment.get("segment_type") or ""),
            "text": text[:8192],
            "sparse_terms": sparse_terms[:2048],
            "embedding_source": str(segment.get("embedding_source") or "textual_stub"),
            "importance_score": float(segment.get("importance_score") or 0.0),
            "evidence_refs_json": evidence_refs_json[:8192],
            "vector": self._deterministic_vector(text, MEMORY_MILVUS_VECTOR_DIM),
        }

    def _deterministic_vector(self, text: str, dim: int) -> List[float]:
        if not text:
            return [0.0] * dim

        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values: List[float] = []
        state = digest
        while len(values) < dim:
            for byte in state:
                values.append((byte / 127.5) - 1.0)
                if len(values) >= dim:
                    break
            state = hashlib.sha256(state).digest()

        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [round(value / norm, 6) for value in values[:dim]]
