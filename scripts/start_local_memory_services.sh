#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="$ROOT_DIR/.tooling/local-services"
STATE_DIR="$ROOT_DIR/runtime/local-services"
REDIS_SRC_DIR="$(find "$TOOLS_DIR/redis-src" -maxdepth 1 -type d -name 'redis-*' | sort | tail -n 1)"
REDIS_BIN="$REDIS_SRC_DIR/src/redis-server"
REDIS_CLI="$REDIS_SRC_DIR/src/redis-cli"
REDIS_CONF="$STATE_DIR/redis/redis.conf"
REDIS_PID="$STATE_DIR/redis/redis.pid"
REDIS_LOG="$STATE_DIR/redis/redis.log"

NEO4J_HOME="$(find "$TOOLS_DIR/neo4j" -maxdepth 1 -type d -name 'neo4j-community-*' | sort | tail -n 1 || true)"
JDK_HOME="$TOOLS_DIR/jdk/current"
NEO4J_PASSWORD="${MEMORY_LOCAL_NEO4J_PASSWORD:-memory_local_neo4j}"

mkdir -p "$STATE_DIR/redis" "$STATE_DIR/neo4j" "$STATE_DIR/milvus"

cat >"$REDIS_CONF" <<EOF
bind 127.0.0.1
port 6379
daemonize yes
dir $STATE_DIR/redis
pidfile $REDIS_PID
logfile $REDIS_LOG
save ""
appendonly no
EOF

if [ -x "$REDIS_BIN" ]; then
  if [ -f "$REDIS_PID" ] && kill -0 "$(cat "$REDIS_PID")" 2>/dev/null; then
    echo "Redis is already running"
  else
    "$REDIS_BIN" "$REDIS_CONF"
    echo "Redis started on redis://127.0.0.1:6379/0"
  fi
else
  echo "Redis binary not found. Run scripts/install_local_memory_services.sh first." >&2
fi

if [ -n "$NEO4J_HOME" ] && [ -x "$NEO4J_HOME/bin/neo4j" ] && [ -x "$JDK_HOME/bin/java" ]; then
  export JAVA_HOME="$JDK_HOME"
  export NEO4J_AUTH="neo4j/$NEO4J_PASSWORD"
  "$NEO4J_HOME/bin/neo4j" start || true
  echo "Neo4j start requested on neo4j://127.0.0.1:7687"
else
  echo "Neo4j is not installed locally yet"
fi

echo "Milvus local mode uses $STATE_DIR/milvus/memory_milvus.db and starts on demand via pymilvus."

if [ -x "$REDIS_CLI" ]; then
  "$REDIS_CLI" -h 127.0.0.1 -p 6379 ping || true
fi
