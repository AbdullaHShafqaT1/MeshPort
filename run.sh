#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PORT=8000
while [ "$#" -gt 0 ]; do
  case "$1" in
    --port|-p)
      PORT="${2:?Port number required}"
      shift 2
      ;;
    *)
      PORT="$1"
      shift
      ;;
  esac
done

exec .venv/bin/python -m uvicorn app.studio:app --host 127.0.0.1 --port "$PORT"
