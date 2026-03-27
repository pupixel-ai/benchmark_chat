#!/bin/zsh
set -euo pipefail

LABEL="com.benchmark_chat.feishu_long_connection"
PLIST_PATH="${HOME}/Library/LaunchAgents/${LABEL}.plist"
UID_VALUE="$(id -u)"

if launchctl print "gui/${UID_VALUE}/${LABEL}" >/dev/null 2>&1; then
  launchctl bootout "gui/${UID_VALUE}" "${PLIST_PATH}" || true
fi

launchctl disable "gui/${UID_VALUE}/${LABEL}" || true
rm -f "${PLIST_PATH}"

echo "[removed] ${LABEL}"
echo "[plist] ${PLIST_PATH}"

