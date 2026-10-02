#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
URL="${URL:-http://192.168.41.171:1234/v1}"
MODEL="${MODEL:-huihui-ornith-1.5-9b-abliterated}"
DATA="${DATA:-data/TriviaQA/rc.nocontext/train-00000-of-00001.parquet}"
OUT="${OUT:-data/consistency_samples.jsonl}"
STEP="${STEP:-200}"; LIMIT="${LIMIT:-3000}"; CONC="${CONC:-1}"
mkdir -p logs
TSV="logs/throughput-$(date +%Y%m%d).tsv"
[[ -f $TSV ]] || printf 'time\tscanned\tattempted\tkept_t\tkept_f\tsecs\ts_per_q\n' > "$TSV"
att() { [[ -f "$OUT.attempted" ]] && wc -l < "$OUT.attempted" || echo 0; }
lab() { [[ -f "$OUT" ]] && grep -c "\"judge\": \"$1\"" "$OUT" || echo 0; }
say() { printf '\033[1;36m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
SCAN=$(att)
while (( SCAN < LIMIT )); do
  SCAN=$(( SCAN + STEP ))
  curl -sf --max-time 10 "$URL/models" >/dev/null || { say "server down, wait 120s"; sleep 120; continue; }
  b=$(att); t0=$(date +%s)
  say "scanning first $SCAN of the dataset ($b already attempted)"
  python -u scripts/collect_responses_lmstudio.py --base_url "$URL" --model "$MODEL" \
    --data_path "$DATA" --output_path "$OUT" --sample_num 5 \
    --max_questions "$SCAN" --concurrency "$CONC" --task trivia
  t1=$(date +%s); n=$(att); did=$(( n - b )); s=$(( t1 - t0 ))
  per=$(awk -v s=$s -v x=$did 'BEGIN{printf "%.1f",(x>0)?s/x:0}')
  printf '%s\t%d\t%d\t%d\t%d\t%d\t%s\n' "$(date +%H:%M:%S)" "$SCAN" "$n" \
    "$(lab true)" "$(lab false)" "$s" "$per" >> "$TSV"
  kt=$(lab true); kf=$(lab false)
  pairs=$(( kt < kf ? kt : kf ))
  say "+$did in ${s}s (${per}s/q) | kept ${kt}/${kf} | $pairs balanced pairs"
done
say "finished: $(att) attempted, $(lab true)/$(lab false)"
