ORIGINAL_MODEL="huihui-ornith-1.5-9b-abliterated"
API_BASE="http://localhost:1234/v1"

# 2. BENCHMARK LOOP: Pre-load both models to bypass the 400 error
for scale in 015 025 050 075; do
  TARGET_ID="ornith-supp${scale}"

  echo "=== Triggering JIT Load for Scale ${scale} ==="

  # Send dummy requests to trigger just-in-time loading for both models
  curl -s -X POST "${API_BASE}/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$ORIGINAL_MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"ping\"}], \"max_tokens\": 1}" > /dev/null

  curl -s -X POST "${API_BASE}/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$TARGET_ID\", \"messages\": [{\"role\": \"user\", \"content\": \"ping\"}], \"max_tokens\": 1}" > /dev/null

  echo "Waiting for models to settle in VRAM..."

  # Block execution until both models appear in the active models list
  while ! curl -s "${API_BASE}/models" | grep -q "\"id\": \"$ORIGINAL_MODEL\""; do sleep 1; done
  while ! curl -s "${API_BASE}/models" | grep -q "\"id\": \"$TARGET_ID\""; do sleep 1; done

  echo "Models verified in VRAM. Starting benchmark..."

  python scripts/compare_models.py \
    --target "base=${API_BASE}@${ORIGINAL_MODEL}" \
    --target "s${scale}=${API_BASE}@${TARGET_ID}" \
    --n 100 \
    --out "runs/compare-s${scale}"

  echo "Finished scale ${scale}. Results saved to runs/compare-s${scale}"

  # 3. POST-TEST CLEANUP: Evict the scaled model immediately after testing
  echo "Unloading ${TARGET_ID} to prevent VRAM overflow on the next loop..."
  curl -s -X POST "${API_BASE}/models/unload" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$TARGET_ID\"}" > /dev/null

  echo "-----------------------------------"
done

echo "Automated tuning complete!"
