#!/usr/bin/env bash
# Copy the correct requirements file to requirements.txt for Streamlit Cloud.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

usage() {
  echo "Usage: $0 dispatcher|management|filter"
  echo ""
  echo "  dispatcher  -> requirements-dispatcher.txt  (app.py, lightweight)"
  echo "  management  -> requirements-management.txt (management.py, full stack)"
  echo "  filter      -> requirements-filter.txt     (filter_app.py, Excel prep)"
  exit 1
}

TARGET="${1:-}"
case "$TARGET" in
  dispatcher|app)
    SRC="requirements-dispatcher.txt"
    ;;
  management|mgmt)
    SRC="requirements-management.txt"
    ;;
  filter|prep)
    SRC="requirements-filter.txt"
    ;;
  *)
    usage
    ;;
esac

if [[ ! -f "$SRC" ]]; then
  echo "Missing $SRC" >&2
  exit 1
fi

cp "$SRC" requirements.txt
echo "Updated requirements.txt from $SRC"
