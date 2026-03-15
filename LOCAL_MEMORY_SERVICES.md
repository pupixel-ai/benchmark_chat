# Local Memory Services

This project can run a practical local memory stack without Homebrew.

## What it installs

- `Redis`: built locally from the official source tarball
- `Neo4j`: prepared from the official community tarball if the download is reachable
- `Milvus`: runs as `milvus-lite` through `pymilvus`, so there is no separate daemon

## Install

```bash
scripts/install_local_memory_services.sh
```

The installer writes local endpoint hints to:

```text
runtime/local-services/local-memory.env
```

## Start

```bash
scripts/start_local_memory_services.sh
```

## Check

```bash
scripts/check_local_memory_services.sh
```

## Stop

```bash
scripts/stop_local_memory_services.sh
```

## Local `.env` values

Copy the generated values from `runtime/local-services/local-memory.env` into `.env`.

Typical local values:

```env
MEMORY_EXTERNAL_SINKS_ENABLED=true
MEMORY_REDIS_URL=redis://127.0.0.1:6379/0
MEMORY_REDIS_PREFIX=memory
MEMORY_NEO4J_URI=neo4j://127.0.0.1:7687
MEMORY_NEO4J_USERNAME=neo4j
MEMORY_NEO4J_PASSWORD=memory_local_neo4j
MEMORY_NEO4J_DATABASE=neo4j
MEMORY_MILVUS_URI=/absolute/path/to/runtime/local-services/milvus/memory_milvus.db
MEMORY_MILVUS_COLLECTION=memory_segments
MEMORY_MILVUS_VECTOR_DIM=32
```

## Notes

- `Milvus Lite` is a development topology. It is enough for local adapter validation, but it is not a drop-in production replacement for a standalone Milvus cluster.
- On this machine, the adapter uses a short-lived local TCP bridge for `milvus-lite` instead of its default Unix socket path, because the Unix socket mode is not reliable in the current Python/gRPC environment.
- If the Neo4j tarball cannot be downloaded from the official CDN on this network, the rest of the stack still works and the code can point to any reachable Neo4j instance later.
