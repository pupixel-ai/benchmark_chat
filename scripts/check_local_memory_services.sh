#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="$ROOT_DIR/.tooling/local-services"
STATE_DIR="$ROOT_DIR/runtime/local-services"
REDIS_SRC_DIR="$(find "$TOOLS_DIR/redis-src" -maxdepth 1 -type d -name 'redis-*' | sort | tail -n 1)"
REDIS_CLI="$REDIS_SRC_DIR/src/redis-cli"
JDK_HOME="$TOOLS_DIR/jdk/current"
NEO4J_HOME="$(find "$TOOLS_DIR/neo4j" -maxdepth 1 -type d -name 'neo4j-community-*' | sort | tail -n 1 || true)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

echo "[redis]"
if [ -x "$REDIS_CLI" ]; then
  "$REDIS_CLI" -h 127.0.0.1 -p 6379 ping
else
  echo "redis-cli not installed"
fi

echo
echo "[neo4j]"
if [ -n "$NEO4J_HOME" ] && [ -x "$JDK_HOME/bin/java" ]; then
  export JAVA_HOME="$JDK_HOME"
  if [ -f "$ROOT_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$ROOT_DIR/.env"
    set +a
  fi
  "$PYTHON_BIN" - <<'PY'
import os
try:
    from neo4j import GraphDatabase
except Exception as exc:
    print(f"neo4j driver unavailable: {exc}")
    raise SystemExit(0)

uri = os.getenv("MEMORY_NEO4J_URI", "neo4j://127.0.0.1:7687")
username = os.getenv("MEMORY_NEO4J_USERNAME", "neo4j")
password = os.getenv("MEMORY_NEO4J_PASSWORD", "memory_local_neo4j")
driver = GraphDatabase.driver(uri, auth=(username, password))
with driver.session() as session:
    result = session.run("RETURN 1 AS ok").single()
    print(f"neo4j ok={result['ok']}")
driver.close()
PY
else
  echo "neo4j not installed"
fi

echo
echo "[milvus-lite]"
if [ -x "$PYTHON_BIN" ]; then
  export MEMORY_LOCAL_MILVUS_DB="$STATE_DIR/milvus/memory_milvus.db"
  "$PYTHON_BIN" - <<'PY'
import os
import socket
from pathlib import Path

try:
    from pymilvus import MilvusClient
    from milvus_lite.server import Server
except Exception as exc:
    print(f"milvus client unavailable: {exc}")
    raise SystemExit(0)

db_path = Path(os.environ["MEMORY_LOCAL_MILVUS_DB"]).resolve()
db_path.parent.mkdir(parents=True, exist_ok=True)
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    port = sock.getsockname()[1]

server = Server(str(db_path), f"127.0.0.1:{port}")
if not server.init() or not server.start():
    raise SystemExit("failed to start milvus-lite tcp bridge")

client = MilvusClient(uri=f"http://127.0.0.1:{port}", timeout=5)
collections = client.list_collections(timeout=5)
print(f"milvus-lite ok collections={collections}")
try:
    client.close()
except Exception:
    pass
server.stop()
PY
else
  echo ".venv python not found"
fi
