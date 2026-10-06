#!/usr/bin/env bash
# Compare a base model against suppressed variants served by LM Studio, one
# variant at a time so two models never have to share VRAM for long.
#
#   BASE=my-model PREFIX=my-model-supp SCALES="015 025 050 075" \
#   API=http://127.0.0.1:1234/v1 N=100 scripts/lmstudio_scale_compare.sh
#
# Variants are expected to be installed in LM Studio as ${PREFIX}${scale}
# (scripts/install_to_lmstudio.sh --tag names them that way). Results land in
# runs/compare-s<scale>/ via compare_models.py.
set -euo pipefail

BASE="${BASE:?set BASE to the LM Studio identifier of the unmodified model}"
PREFIX="${PREFIX:?set PREFIX, e.g. my-model-supp}"
SCALES="${SCALES:-015 025 050 075}"
API="${API:-http://127.0.0.1:1234/v1}"
N="${N:-100}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ping_model() {
  curl -sf -X POST "${API}/chat/completions" -H "Content-Type: application/json" \
    -d "{\"model\": \"$1\", \"messages\": [{\"role\": \"user\", \"content\": \"ping\"}], \"max_tokens\": 1}" > /dev/null
}
wait_loaded() {
  for _ in $(seq 1 600); do
    curl -sf "${API}/models" | grep -q "\"id\": *\"$1\"" && return 0
    sleep 1
  done
  echo "timed out waiting for $1 to load" >&2; return 1
}

for scale in ${SCALES}; do
  target="${PREFIX}${scale}"
  echo "=== ${target} vs ${BASE} ==="
  # LM Studio loads models just-in-time on first request.
  ping_model "${BASE}"; ping_model "${target}"
  wait_loaded "${BASE}"; wait_loaded "${target}"
  python "${here}/compare_models.py" \
    --target "base=${API}@${BASE}" \
    --target "s${scale}=${API}@${target}" \
    --n "${N}" --out "runs/compare-s${scale}"
  # Evict the variant before loading the next one.
  curl -s -X POST "${API}/models/unload" -H "Content-Type: application/json" \
    -d "{\"model\": \"${target}\"}" > /dev/null || true
done
echo "done; see runs/compare-s*/"
