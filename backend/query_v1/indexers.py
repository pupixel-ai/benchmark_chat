"""Optional Milvus / Neo4j publishers for query v1."""

from __future__ import annotations

import json
import re
import socket
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from config import (
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
    MEMORY_QUERY_V1_EVIDENCE_COLLECTION,
    MEMORY_QUERY_V1_EVENT_COLLECTION,
)
from memory_module.embeddings import EmbeddingProvider


class MilvusQueryIndexer:
    """Publishes query-v1 event/evidence docs to Milvus when configured."""

    def __init__(self) -> None:
        self.embedder = EmbeddingProvider.from_config(dim=MEMORY_MILVUS_VECTOR_DIM)

    def publish(
        self,
        *,
        event_views: Iterable[Dict[str, Any]],
        evidence_docs: Iterable[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not MEMORY_MILVUS_URI:
            return {"status": "skipped", "reason": "MEMORY_MILVUS_URI not configured"}

        event_views = list(event_views or [])
        evidence_docs = list(evidence_docs or [])
        if not event_views and not evidence_docs:
            return {"status": "skipped", "reason": "no query-v1 docs to publish"}

        client = None
        local_server = None
        try:
            from pymilvus import MilvusClient

            client, local_server = self._open_client(MilvusClient)
            collections: List[Dict[str, Any]] = []
            if event_views:
                self._ensure_event_collection(client, MilvusClient)
                rows = [self._event_view_row(item) for item in event_views]
                client.upsert(collection_name=MEMORY_QUERY_V1_EVENT_COLLECTION, data=rows)
                collections.append(
                    {
                        "collection": MEMORY_QUERY_V1_EVENT_COLLECTION,
                        "record_type": "event_views_v1",
                        "record_count": len(rows),
                    }
                )
            if evidence_docs:
                self._ensure_evidence_collection(client, MilvusClient)
                rows = [self._evidence_row(item) for item in evidence_docs]
                client.upsert(collection_name=MEMORY_QUERY_V1_EVIDENCE_COLLECTION, data=rows)
                collections.append(
                    {
                        "collection": MEMORY_QUERY_V1_EVIDENCE_COLLECTION,
                        "record_type": "evidence_docs_v1",
                        "record_count": len(rows),
                    }
                )
            return {
                "status": "published",
                "collections": collections,
                "record_count": sum(int(item["record_count"]) for item in collections),
                "mode": "local-db" if local_server is not None else "remote",
                "vector_dim": MEMORY_MILVUS_VECTOR_DIM,
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
            client = client_cls(uri=f"http://{address}", timeout=10)
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

    def _ensure_event_collection(self, client: Any, milvus_cls: Any) -> None:
        if client.has_collection(MEMORY_QUERY_V1_EVENT_COLLECTION):
            return
        from pymilvus import DataType

        schema = milvus_cls.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("doc_id", DataType.VARCHAR, is_primary=True, max_length=160)
        schema.add_field("user_id", DataType.VARCHAR, max_length=128)
        schema.add_field("event_id", DataType.VARCHAR, max_length=128)
        schema.add_field("source_task_id", DataType.VARCHAR, max_length=128)
        schema.add_field("view_type", DataType.VARCHAR, max_length=64)
        schema.add_field("retrieval_text", DataType.VARCHAR, max_length=8192)
        schema.add_field("start_ts", DataType.VARCHAR, max_length=64)
        schema.add_field("end_ts", DataType.VARCHAR, max_length=64)
        schema.add_field("person_ids_json", DataType.VARCHAR, max_length=4096)
        schema.add_field("place_refs_json", DataType.VARCHAR, max_length=4096)
        schema.add_field("tag_keys_json", DataType.VARCHAR, max_length=4096)
        schema.add_field("cover_photo_id", DataType.VARCHAR, max_length=128)
        schema.add_field("supporting_photo_ids_json", DataType.VARCHAR, max_length=8192)
        schema.add_field("confidence", DataType.FLOAT)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=MEMORY_MILVUS_VECTOR_DIM)
        index_params = milvus_cls.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
        client.create_collection(collection_name=MEMORY_QUERY_V1_EVENT_COLLECTION, schema=schema, index_params=index_params)

    def _ensure_evidence_collection(self, client: Any, milvus_cls: Any) -> None:
        if client.has_collection(MEMORY_QUERY_V1_EVIDENCE_COLLECTION):
            return
        from pymilvus import DataType

        schema = milvus_cls.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("evidence_id", DataType.VARCHAR, is_primary=True, max_length=160)
        schema.add_field("user_id", DataType.VARCHAR, max_length=128)
        schema.add_field("event_id", DataType.VARCHAR, max_length=128)
        schema.add_field("photo_id", DataType.VARCHAR, max_length=128)
        schema.add_field("source_task_id", DataType.VARCHAR, max_length=128)
        schema.add_field("evidence_type", DataType.VARCHAR, max_length=64)
        schema.add_field("retrieval_text", DataType.VARCHAR, max_length=8192)
        schema.add_field("normalized_value", DataType.VARCHAR, max_length=8192)
        schema.add_field("numeric_value", DataType.FLOAT)
        schema.add_field("numeric_unit", DataType.VARCHAR, max_length=64)
        schema.add_field("person_ids_json", DataType.VARCHAR, max_length=4096)
        schema.add_field("place_refs_json", DataType.VARCHAR, max_length=4096)
        schema.add_field("confidence", DataType.FLOAT)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=MEMORY_MILVUS_VECTOR_DIM)
        index_params = milvus_cls.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
        client.create_collection(collection_name=MEMORY_QUERY_V1_EVIDENCE_COLLECTION, schema=schema, index_params=index_params)

    def _event_view_row(self, item: Dict[str, Any]) -> Dict[str, Any]:
        text = str(item.get("retrieval_text") or "")[:8192]
        embedding, _, _ = self.embedder.embed_text(text, task_type="document")
        return {
            "doc_id": str(item.get("doc_id") or ""),
            "user_id": str(item.get("user_id") or ""),
            "event_id": str(item.get("event_id") or ""),
            "source_task_id": str(item.get("source_task_id") or ""),
            "view_type": str(item.get("view_type") or ""),
            "retrieval_text": text,
            "start_ts": str(item.get("start_ts") or ""),
            "end_ts": str(item.get("end_ts") or ""),
            "person_ids_json": json.dumps(item.get("person_ids", []), ensure_ascii=False)[:4096],
            "place_refs_json": json.dumps(item.get("place_refs", []), ensure_ascii=False)[:4096],
            "tag_keys_json": json.dumps(item.get("tag_keys", []), ensure_ascii=False)[:4096],
            "cover_photo_id": str(item.get("cover_photo_id") or ""),
            "supporting_photo_ids_json": json.dumps(item.get("supporting_photo_ids", []), ensure_ascii=False)[:8192],
            "confidence": float(item.get("confidence") or 0.0),
            "vector": embedding,
        }

    def _evidence_row(self, item: Dict[str, Any]) -> Dict[str, Any]:
        text = str(item.get("retrieval_text") or "")[:8192]
        embedding, _, _ = self.embedder.embed_text(text, task_type="document")
        numeric_value = item.get("numeric_value")
        try:
            numeric_value = float(numeric_value) if numeric_value is not None else 0.0
        except Exception:
            numeric_value = 0.0
        return {
            "evidence_id": str(item.get("evidence_id") or ""),
            "user_id": str(item.get("user_id") or ""),
            "event_id": str(item.get("event_id") or ""),
            "photo_id": str(item.get("photo_id") or ""),
            "source_task_id": str(item.get("source_task_id") or ""),
            "evidence_type": str(item.get("evidence_type") or ""),
            "retrieval_text": text,
            "normalized_value": str(item.get("normalized_value") or "")[:8192],
            "numeric_value": numeric_value,
            "numeric_unit": str(item.get("numeric_unit") or "")[:64],
            "person_ids_json": json.dumps(item.get("person_ids", []), ensure_ascii=False)[:4096],
            "place_refs_json": json.dumps(item.get("place_refs", []), ensure_ascii=False)[:4096],
            "confidence": float(item.get("confidence") or 0.0),
            "vector": embedding,
        }


class Neo4jQueryIndexer:
    """Publishes the query-v1 structural graph to Neo4j when configured."""

    def publish(self, graph_payload: Dict[str, Any]) -> Dict[str, Any]:
        if not MEMORY_NEO4J_URI:
            return {"status": "skipped", "reason": "MEMORY_NEO4J_URI not configured"}
        try:
            from neo4j import GraphDatabase

            auth = None
            if MEMORY_NEO4J_USERNAME:
                auth = (MEMORY_NEO4J_USERNAME, MEMORY_NEO4J_PASSWORD)
            driver = GraphDatabase.driver(MEMORY_NEO4J_URI, auth=auth)
            with driver.session(database=MEMORY_NEO4J_DATABASE or None) as session:
                node_index: Dict[str, Tuple[str, str]] = {}
                for group_name, records in dict(graph_payload.get("nodes") or {}).items():
                    label = self._sanitize_label(group_name[:-1] if group_name.endswith("s") else group_name)
                    id_field = self._id_field_for_group(group_name)
                    for record in list(records or []):
                        node_id = str(record.get(id_field) or "")
                        if not node_id:
                            continue
                        props = self._sanitize_properties(record)
                        session.run(
                            f"MERGE (n:{label} {{{id_field}: $node_id}}) SET n += $props",
                            node_id=node_id,
                            props=props,
                        )
                        node_index[node_id] = (label, id_field)
                edge_count = 0
                skipped_edges = 0
                for edge in list(graph_payload.get("edges") or []):
                    left = node_index.get(str(edge.get("from_id") or ""))
                    right = node_index.get(str(edge.get("to_id") or ""))
                    if not left or not right:
                        skipped_edges += 1
                        continue
                    left_label, left_id = left
                    right_label, right_id = right
                    rel_type = self._sanitize_rel(str(edge.get("edge_type") or "RELATED_TO"))
                    props = self._sanitize_properties(dict(edge.get("properties") or {}))
                    session.run(
                        (
                            f"MATCH (a:{left_label} {{{left_id}: $from_id}}) "
                            f"MATCH (b:{right_label} {{{right_id}: $to_id}}) "
                            f"MERGE (a)-[r:{rel_type}]->(b) SET r += $props"
                        ),
                        from_id=str(edge.get("from_id") or ""),
                        to_id=str(edge.get("to_id") or ""),
                        props=props,
                    )
                    edge_count += 1
            driver.close()
            return {"status": "published", "edge_count": edge_count, "skipped_edges": skipped_edges}
        except Exception as exc:
            return {"status": "failed", "reason": str(exc)}

    def _id_field_for_group(self, group_name: str) -> str:
        return {
            "persons": "person_id",
            "events": "event_id",
            "relationships": "relationship_id",
            "groups": "group_id",
            "places": "place_ref",
        }.get(group_name, "id")

    def _sanitize_label(self, value: str) -> str:
        cleaned = re.sub(r"[^0-9A-Za-z_]", "_", value or "Node")
        if cleaned and cleaned[0].isdigit():
            cleaned = f"Node_{cleaned}"
        return cleaned or "Node"

    def _sanitize_rel(self, value: str) -> str:
        cleaned = re.sub(r"[^0-9A-Za-z_]", "_", value.upper())
        if cleaned and cleaned[0].isdigit():
            cleaned = f"REL_{cleaned}"
        return cleaned or "RELATED_TO"

    def _sanitize_properties(self, props: Dict[str, Any]) -> Dict[str, Any]:
        sanitized: Dict[str, Any] = {}
        for key, value in props.items():
            self._collect_property(sanitized, self._sanitize_label(key.lower()), value)
        return sanitized

    def _collect_property(self, target: Dict[str, Any], key: str, value: Any) -> None:
        if value is None:
            return
        if isinstance(value, datetime):
            target[key] = value.isoformat()
            return
        if isinstance(value, (str, int, float, bool)):
            target[key] = value
            return
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                self._collect_property(target, f"{key}_{self._sanitize_label(str(nested_key).lower())}", nested_value)
            return
        if isinstance(value, (list, tuple)):
            scalar_items: List[Any] = []
            for item in value:
                if isinstance(item, (str, int, float, bool)):
                    scalar_items.append(item)
                else:
                    target[f"{key}_json"] = json.dumps(value, ensure_ascii=False, default=str)
                    return
            target[key] = scalar_items
            return
        target[key] = str(value)
