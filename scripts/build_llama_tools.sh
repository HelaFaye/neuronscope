#!/usr/bin/env bash
# Build llama.cpp with NeuronScope's cett-dump extractor.
#
#   scripts/build_llama_tools.sh                     # CPU, ~/llama.cpp
#   scripts/build_llama_tools.sh --backend vulkan    # or cuda, metal, hip
#   scripts/build_llama_tools.sh --dir /opt/llama.cpp --ref b6500
#
# Clones (or updates) llama.cpp, copies llama-tools/cett-dump into tools/,
# registers it, and builds llama-cett-dump, llama-server, llama-quantize and
# llama-eval-callback. Prints the env exports for env.local.sh at the end.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIR="${HOME}/llama.cpp"; BACKEND="cpu"; REF=""; JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir) DIR="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --ref) REF="$2"; shift 2 ;;
    -j) JOBS="$2"; shift 2 ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 1 ;;
  esac
done
case "$BACKEND" in
  cpu) FLAGS=() ;;
  vulkan) FLAGS=(-DGGML_VULKAN=ON) ;;
  cuda) FLAGS=(-DGGML_CUDA=ON) ;;
  metal) FLAGS=(-DGGML_METAL=ON) ;;
  hip|rocm) FLAGS=(-DGGML_HIP=ON) ;;
  *) echo "backend must be cpu, vulkan, cuda, metal or hip" >&2; exit 1 ;;
esac
if [[ ! -d "$DIR/.git" ]]; then
  git clone https://github.com/ggml-org/llama.cpp "$DIR"
fi
if [[ -n "$REF" ]]; then git -C "$DIR" fetch --depth 1 origin "$REF" && git -C "$DIR" checkout FETCH_HEAD; fi
rm -rf "$DIR/tools/cett-dump"
cp -r "$here/llama-tools/cett-dump" "$DIR/tools/"
grep -q 'add_subdirectory(cett-dump)' "$DIR/tools/CMakeLists.txt" || echo 'add_subdirectory(cett-dump)' >> "$DIR/tools/CMakeLists.txt"
cmake -S "$DIR" -B "$DIR/build" -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF "${FLAGS[@]}"
cmake --build "$DIR/build" --config Release -j "$JOBS" \
  --target llama-cett-dump llama-server llama-quantize llama-eval-callback
cat <<MSG

Built in $DIR/build/bin. Add to env.local.sh:
  export NS_LLAMA="$DIR"
  export NS_CETT="$DIR/build/bin/llama-cett-dump"
  export NS_LLAMA_SERVER="$DIR/build/bin/llama-server"
MSG
