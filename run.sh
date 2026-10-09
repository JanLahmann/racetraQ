#!/usr/bin/env bash
# One-command start: create/refresh a venv, install racetraq, launch the server.
#   RACETRAQ_EXTRAS=hardware ./run.sh   also installs the Hardware-mode extra
#   RACETRAQ_OFFLINE=1 ./run.sh         skips the install (venue without Wi-Fi)
# The install step is skipped automatically when it fails but racetraq is
# already installed, so a booth laptop starts without network.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
VENV=".venv"
EXTRAS="${RACETRAQ_EXTRAS:-}"

if [ ! -d "$VENV" ]; then
  "$PYTHON" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

SPEC="."
[ -n "$EXTRAS" ] && SPEC=".[$EXTRAS]"
if [ -n "${RACETRAQ_OFFLINE:-}" ]; then
  echo "racetraQ: offline start, skipping the install"
elif ! pip install --quiet --disable-pip-version-check --timeout 15 -e "$SPEC"; then
  if python -c "import racetraq" 2>/dev/null; then
    echo "racetraQ: install failed (no network?) — starting the installed version"
  else
    echo "racetraQ: install failed and racetraq is not installed yet" >&2
    exit 1
  fi
fi

URL="http://127.0.0.1:${RACETRAQ_PORT:-8000}"
echo "racetraQ starting at $URL"
(command -v open >/dev/null && sleep 2 && open "$URL" &) 2>/dev/null || true
(command -v xdg-open >/dev/null && sleep 2 && xdg-open "$URL" &) 2>/dev/null || true

exec python -m racetraq "$@"
