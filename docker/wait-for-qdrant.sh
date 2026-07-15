#!/bin/sh
set -eu

for _attempt in $(seq 1 60); do
  if python -c 'import urllib.request; urllib.request.urlopen("http://qdrant:6333/readyz", timeout=1)' >/dev/null 2>&1; then
    exec "$@"
  fi
  sleep 1
done

echo "Qdrant did not become ready within 60 seconds" >&2
exit 1
