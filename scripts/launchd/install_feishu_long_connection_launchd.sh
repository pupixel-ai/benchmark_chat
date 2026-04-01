#!/bin/zsh
set -euo pipefail

SOURCE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LABEL="com.benchmark_chat.feishu_long_connection"
PLIST_PATH="${HOME}/Library/LaunchAgents/${LABEL}.plist"
UID_VALUE="$(id -u)"

is_protected_root() {
  case "$1" in
    "${HOME}/Desktop/"*|\
    "${HOME}/Documents/"*|\
    "${HOME}/Downloads/"*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

if is_protected_root "${SOURCE_ROOT}"; then
  ROOT_MODE="runtime_mirror"
  ROOT_DIR="${HOME}/.codex_local/runtime/benchmark_chat"
  mkdir -p "${ROOT_DIR}"
  rsync -a --delete \
    --exclude '.git' \
    --exclude '.codex_local/logs' \
    --exclude '.DS_Store' \
    "${SOURCE_ROOT}/" "${ROOT_DIR}/"

  if [[ -d "${SOURCE_ROOT}/.codex_local/venvs/feishu_long_connection" ]]; then
    mkdir -p "${ROOT_DIR}/.codex_local/venvs"
    rsync -a \
      "${SOURCE_ROOT}/.codex_local/venvs/feishu_long_connection/" \
      "${ROOT_DIR}/.codex_local/venvs/feishu_long_connection/"
  fi
else
  ROOT_MODE="inplace"
  ROOT_DIR="${SOURCE_ROOT}"
fi

RUNNER_PATH="${ROOT_DIR}/scripts/launchd/run_feishu_long_connection_daemon.sh"
LOG_DIR="${ROOT_DIR}/.codex_local/logs"

mkdir -p "${HOME}/Library/LaunchAgents" "${LOG_DIR}"

cat > "${PLIST_PATH}" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/zsh</string>
    <string>${RUNNER_PATH}</string>
  </array>
  <key>WorkingDirectory</key>
  <string>${HOME}</string>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <dict>
    <key>SuccessfulExit</key>
    <false/>
  </dict>
  <key>ThrottleInterval</key>
  <integer>10</integer>
  <key>StandardOutPath</key>
  <string>${LOG_DIR}/launchd.feishu_long_connection.out.log</string>
  <key>StandardErrorPath</key>
  <string>${LOG_DIR}/launchd.feishu_long_connection.err.log</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>PYTHONUNBUFFERED</key>
    <string>1</string>
    <key>FEISHU_WORKER_ROOT</key>
    <string>${ROOT_DIR}</string>
  </dict>
</dict>
</plist>
EOF

chmod 755 "${RUNNER_PATH}"

if launchctl print "gui/${UID_VALUE}/${LABEL}" >/dev/null 2>&1; then
  launchctl bootout "gui/${UID_VALUE}" "${PLIST_PATH}" || true
fi

launchctl bootstrap "gui/${UID_VALUE}" "${PLIST_PATH}"
launchctl enable "gui/${UID_VALUE}/${LABEL}" || true
launchctl kickstart -k "gui/${UID_VALUE}/${LABEL}"

echo "[installed] ${LABEL}"
echo "[plist] ${PLIST_PATH}"
echo "[runner] ${RUNNER_PATH}"
echo "[root_mode] ${ROOT_MODE}"
echo "[source_root] ${SOURCE_ROOT}"
echo "[effective_root] ${ROOT_DIR}"
echo "[logs] ${LOG_DIR}/launchd.feishu_long_connection.out.log"
echo "[logs] ${LOG_DIR}/launchd.feishu_long_connection.err.log"
