#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="$ROOT_DIR/.tooling/local-services"
STATE_DIR="$ROOT_DIR/runtime/local-services"
REDIS_SRC_DIR="$(find "$TOOLS_DIR/redis-src" -maxdepth 1 -type d -name 'redis-*' | sort | tail -n 1)"
REDIS_CLI="$REDIS_SRC_DIR/src/redis-cli"
REDIS_PID="$STATE_DIR/redis/redis.pid"
NEO4J_HOME="$(find "$TOOLS_DIR/neo4j" -maxdepth 1 -type d -name 'neo4j-community-*' | sort | tail -n 1 || true)"
JDK_HOME="$TOOLS_DIR/jdk/current"

if [ -x "$REDIS_CLI" ]; then
  "$REDIS_CLI" -h 127.0.0.1 -p 6379 shutdown nosave || true
elif [ -f "$REDIS_PID" ]; then
  kill "$(cat "$REDIS_PID")" || true
fi

if [ -n "$NEO4J_HOME" ] && [ -x "$NEO4J_HOME/bin/neo4j" ] && [ -x "$JDK_HOME/bin/java" ]; then
  export JAVA_HOME="$JDK_HOME"
  "$NEO4J_HOME/bin/neo4j" stop || true
fi

echo "Requested local memory services shutdown."
