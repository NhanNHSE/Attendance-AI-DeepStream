#!/usr/bin/env bash
set -euo pipefail

: "${IF_REST_URL:=http://ifr:18081}"
# mặc định chạy script Python của bạn; đổi sang deepstream-app -c nếu muốn
: "${DS_CMD:=python3 /workspace/app/ds_to_insightface.py}"

echo "[ds] Waiting for InsightFace-REST at ${IF_REST_URL} ..."
for i in {1..90}; do
  if curl -fsS "${IF_REST_URL}/docs" >/dev/null 2>&1; then
    echo "[ds] IFR is up."
    break
  fi
  sleep 1
  if [[ $i -eq 90 ]]; then
    echo "[ds] Timeout waiting for IFR." >&2
    exit 1
  fi
done

echo "[ds] Starting: ${DS_CMD}"
exec bash -lc "${DS_CMD}"
