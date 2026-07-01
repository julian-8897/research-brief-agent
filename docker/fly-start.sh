#!/bin/sh
set -eu

export QDRANT__STORAGE__STORAGE_PATH="${QDRANT__STORAGE__STORAGE_PATH:-/qdrant/storage}"
mkdir -p "$QDRANT__STORAGE__STORAGE_PATH"

(
  cd /qdrant
  ./qdrant
) &

for _attempt in $(seq 1 30); do
  if python - <<'PY'
import urllib.request

try:
    urllib.request.urlopen("http://127.0.0.1:6333/", timeout=1)
except Exception:
    raise SystemExit(1)
PY
  then
    break
  fi
  sleep 1
done

exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000
