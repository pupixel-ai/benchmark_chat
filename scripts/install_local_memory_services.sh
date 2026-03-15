#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="$ROOT_DIR/.tooling/local-services"
DOWNLOAD_DIR="$TOOLS_DIR/downloads"
REDIS_SRC_DIR="$TOOLS_DIR/redis-src"
JDK_DIR="$TOOLS_DIR/jdk"
NEO4J_DIR="$TOOLS_DIR/neo4j"
STATE_DIR="$ROOT_DIR/runtime/local-services"
SKIP_NEO4J_INSTALL="${SKIP_NEO4J_INSTALL:-false}"

REDIS_VERSION="${REDIS_VERSION:-7.4.2}"
REDIS_ARCHIVE="redis-${REDIS_VERSION}.tar.gz"
REDIS_URL="https://download.redis.io/releases/${REDIS_ARCHIVE}"

NEO4J_VERSION="${NEO4J_VERSION:-5.26.0}"
NEO4J_ARCHIVE="neo4j-community-${NEO4J_VERSION}-unix.tar.gz"
NEO4J_URL="https://neo4j.com/artifact.php?name=${NEO4J_ARCHIVE}"

mkdir -p "$DOWNLOAD_DIR" "$REDIS_SRC_DIR" "$JDK_DIR" "$NEO4J_DIR" "$STATE_DIR"/{redis,neo4j,milvus}

download_jdk() {
  local jdk_archive="$DOWNLOAD_DIR/temurin21-aarch64-mac.tar.gz"
  local jdk_current="$JDK_DIR/current"

  if [ -x "$jdk_current/bin/java" ]; then
    echo "JDK already installed at $jdk_current"
    return 0
  fi

  echo "Resolving Temurin 21 aarch64 macOS archive..."
  local release_json
  release_json="$(curl -fsSL https://api.github.com/repos/adoptium/temurin21-binaries/releases/latest)"
  local asset_url
  asset_url="$(printf '%s' "$release_json" | python3 -c 'import json, sys
data = json.load(sys.stdin)
for asset in data.get("assets", []):
    name = asset.get("name", "")
    if "aarch64_mac" in name and name.endswith(".tar.gz") and "jdk" in name and "debugimage" not in name and "testimage" not in name and "staticlibs" not in name:
        print(asset["browser_download_url"])
        break
else:
    raise SystemExit("No matching Temurin JDK asset found")
')"

  echo "Downloading JDK from $asset_url"
  if ! curl --connect-timeout 20 --max-time 300 -fL "$asset_url" -o "$jdk_archive"; then
    echo "JDK download failed; skipping local Neo4j installation for now." >&2
    rm -f "$jdk_archive"
    return 1
  fi

  rm -rf "$JDK_DIR"/jdk-* "$jdk_current"
  tar -xzf "$jdk_archive" -C "$JDK_DIR"
  local extracted
  extracted="$(find "$JDK_DIR" -maxdepth 1 -type d -name 'jdk-*' | head -n 1)"
  if [ -z "$extracted" ]; then
    echo "JDK extraction failed" >&2
    return 1
  fi

  ln -sfn "$extracted/Contents/Home" "$jdk_current"
  "$jdk_current/bin/java" -version
}

install_redis() {
  local archive_path="$DOWNLOAD_DIR/$REDIS_ARCHIVE"
  local source_path="$REDIS_SRC_DIR/redis-$REDIS_VERSION"

  if [ ! -f "$archive_path" ]; then
    echo "Downloading Redis $REDIS_VERSION"
    curl -fL "$REDIS_URL" -o "$archive_path"
  fi

  if [ ! -d "$source_path" ]; then
    tar -xzf "$archive_path" -C "$REDIS_SRC_DIR"
  fi

  echo "Building Redis $REDIS_VERSION"
  make -C "$source_path" MALLOC=libc BUILD_TLS=no -j"$(sysctl -n hw.ncpu)"
}

install_neo4j() {
  local jdk_current="$JDK_DIR/current"
  local archive_path="$DOWNLOAD_DIR/$NEO4J_ARCHIVE"
  local install_path="$NEO4J_DIR/neo4j-community-$NEO4J_VERSION"

  if [ ! -x "$jdk_current/bin/java" ]; then
    echo "Skipping Neo4j install because JDK is not available" >&2
    return 1
  fi

  if [ ! -f "$archive_path" ]; then
    echo "Attempting Neo4j $NEO4J_VERSION download"
    if ! curl -fL -A "Mozilla/5.0" "$NEO4J_URL" -o "$archive_path"; then
      echo "Neo4j download failed from official artifact endpoint. Leaving Neo4j uninstalled for now." >&2
      rm -f "$archive_path"
      return 1
    fi
  fi

  rm -rf "$install_path"
  mkdir -p "$NEO4J_DIR"
  tar -xzf "$archive_path" -C "$NEO4J_DIR"

  local conf_path="$install_path/conf/neo4j.conf"
  local data_dir="$STATE_DIR/neo4j/data"
  local logs_dir="$STATE_DIR/neo4j/logs"
  local run_dir="$STATE_DIR/neo4j/run"
  local import_dir="$STATE_DIR/neo4j/import"
  local password="${MEMORY_LOCAL_NEO4J_PASSWORD:-memory_local_neo4j}"

  mkdir -p "$data_dir" "$logs_dir" "$run_dir" "$import_dir"

  cat >"$conf_path" <<EOF
server.default_listen_address=127.0.0.1
server.http.listen_address=:7474
server.bolt.listen_address=:7687
server.directories.data=$data_dir
server.directories.logs=$logs_dir
server.directories.run=$run_dir
server.directories.import=$import_dir
server.memory.heap.initial_size=512m
server.memory.heap.max_size=512m
dbms.security.auth_enabled=true
EOF

  export JAVA_HOME="$jdk_current"
  "$install_path/bin/neo4j-admin" dbms set-initial-password "$password" >/dev/null 2>&1 || true
  echo "Neo4j prepared at $install_path"
}

install_python_bits() {
  local python_bin="$ROOT_DIR/.venv/bin/python"
  local pip_bin="$ROOT_DIR/.venv/bin/pip"
  local pip_args=(
    --trusted-host pypi.org
    --trusted-host files.pythonhosted.org
  )

  if [ ! -x "$python_bin" ]; then
    echo "Creating .venv"
    python3 -m venv "$ROOT_DIR/.venv"
  fi

  "$python_bin" -m pip install "${pip_args[@]}" --upgrade pip || true
  "$pip_bin" install "${pip_args[@]}" -r "$ROOT_DIR/requirements.txt" milvus-lite
}

write_env_hint() {
  local neo4j_password="${MEMORY_LOCAL_NEO4J_PASSWORD:-memory_local_neo4j}"
  cat >"$STATE_DIR/local-memory.env" <<EOF
# Local memory service endpoints generated by scripts/install_local_memory_services.sh
MEMORY_EXTERNAL_SINKS_ENABLED=true
MEMORY_REDIS_URL=redis://127.0.0.1:6379/0
MEMORY_REDIS_PREFIX=memory
MEMORY_NEO4J_URI=neo4j://127.0.0.1:7687
MEMORY_NEO4J_USERNAME=neo4j
MEMORY_NEO4J_PASSWORD=$neo4j_password
MEMORY_NEO4J_DATABASE=neo4j
MEMORY_MILVUS_URI=$STATE_DIR/milvus/memory_milvus.db
MEMORY_MILVUS_COLLECTION=memory_segments
MEMORY_MILVUS_VECTOR_DIM=32
EOF
  echo "Wrote $STATE_DIR/local-memory.env"
}

install_python_bits
install_redis
if [ "$SKIP_NEO4J_INSTALL" != "true" ]; then
  download_jdk || true
  install_neo4j || true
else
  echo "Skipping local Neo4j install because SKIP_NEO4J_INSTALL=true"
fi
write_env_hint

echo
echo "Local memory service prerequisites are ready."
echo "Generated env hints: $STATE_DIR/local-memory.env"
