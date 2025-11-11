#!/usr/bin/env bash
set -euo pipefail

# Faster preparation pipeline: quick data generation with greedy/random only
# Avoids timeouts by using simpler opponents and shorter time limits

REPO_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$REPO_ROOT"

# Fast settings
GAMES_PER_SIZE="${GAMES_PER_SIZE:-5}"
SIZES="small"                       # Just small boards
MATCHUPS="greedy random"            # Fast opponents only (no self-play)
TIME_CAP="0.10"                     # Very fast moves (100ms)
TEACHER_SECONDS="0.5"               # Fast teacher eval
TEACHER_MAX="2000"                  # Limit teacher rows
TIMEOUT="600"                       # 10-minute timeout per game

LOG_DIR="$REPO_ROOT/logs"
MODELS_DIR="$REPO_ROOT/models"
mkdir -p "$LOG_DIR" "$MODELS_DIR"

gen_log="$LOG_DIR/generation.log"
teacher_log="$LOG_DIR/teacher.log"
train_log="$LOG_DIR/training.log"

echo "=== Quick Prep: Fast data generation ===" | tee "$gen_log"
echo "SIZES=$SIZES MATCHUPS=$MATCHUPS GAMES=$GAMES_PER_SIZE TIME_CAP=$TIME_CAP" | tee -a "$gen_log"

echo "=== Step 1: Build ===" | tee -a "$gen_log"
make cmake-rebuild >>"$gen_log" 2>&1 || { echo "Build failed" >&2; exit 1; }

echo "=== Step 2: Backup old data ===" | tee -a "$gen_log"
if [[ -f "$LOG_DIR/value_data.jsonl" ]]; then
  mv "$LOG_DIR/value_data.jsonl" "$LOG_DIR/value_data.jsonl.bak.$(date +%s)" || true
fi

echo "=== Step 3: Generate data (greedy/random opponents, fast) ===" | tee -a "$gen_log"
RS_TIME_CAP="$TIME_CAP" python3 scripts/arena.py \
  --games-per-size "$GAMES_PER_SIZE" \
  --sizes $SIZES \
  --matchups $MATCHUPS \
  --timeout "$TIMEOUT" \
  --log \
  --rotate-port \
  --port 8080 >>"$gen_log" 2>&1 || true

DATA_FILE="$LOG_DIR/value_data.jsonl"
if [[ ! -s "$DATA_FILE" ]]; then
  echo "[ERROR] No data at $DATA_FILE" >&2
  exit 2
fi

# Check diversity
echo "=== Data check ===" | tee -a "$gen_log"
python3 - "$DATA_FILE" << 'PY'
import json, sys
from collections import Counter
c = Counter()
with open(sys.argv[1]) as f:
    for i, line in enumerate(f):
        if i > 1000: break
        try:
            t = json.loads(line).get('meta', {}).get('turn', -1)
            c[t] += 1
        except: pass
print(f'Turn counts (sample): {dict(c)}')
print(f'Total turns > 0: {sum(c.get(t, 0) for t in range(1, 100))}')
PY

echo "=== Step 4: Teacher pass ===" | tee "$teacher_log"
python3 scripts/teacher_value_fixed.py \
  --input "$DATA_FILE" \
  --output "$LOG_DIR/value_teacher.jsonl" \
  --seconds "$TEACHER_SECONDS" \
  --max "$TEACHER_MAX" >>"$teacher_log" 2>&1 || true

TEACHER_FILE="$LOG_DIR/value_teacher.jsonl"
if [[ ! -s "$TEACHER_FILE" ]]; then
  echo "[ERROR] Teacher output missing: $TEACHER_FILE" >&2
  exit 3
fi

# Check target diversity
python3 - "$TEACHER_FILE" << 'PY'
import json, sys
vals = set()
with open(sys.argv[1]) as f:
    for i, line in enumerate(f):
        if i > 500: break
        try:
            v = float(json.loads(line).get('target', 0.0))
            vals.add(round(v, 3))
        except: pass
print(f'Unique targets: {len(vals)}')
print(f'Sample: {sorted(list(vals))[:15]}')
PY

echo "=== Step 5: Train ===" | tee "$train_log"
python3 ml/train_value.py \
  --input "$TEACHER_FILE" \
  --out "$MODELS_DIR/value_v1.txt" >>"$train_log" 2>&1 || true

MODEL_FILE="$MODELS_DIR/value_v1.txt"
if [[ ! -s "$MODEL_FILE" ]]; then
  echo "[ERROR] Model missing: $MODEL_FILE" >&2
  exit 4
fi

# Check weights
nonzero=$(awk '{if ($1+0 != 0) c++} END{print c+0}' "$MODEL_FILE")
total=$(wc -l < "$MODEL_FILE" | awk '{print $1+0}')
echo "Weights non-zero: $nonzero / $total" | tee -a "$train_log"

if [[ "${nonzero:-0}" -eq 0 ]]; then
  echo "[WARN] All weights zero - targets may be homogeneous" | tee -a "$train_log"
  exit 5
fi

echo "=== SUCCESS ===" | tee -a "$train_log"
echo "Model: $MODEL_FILE"
echo "Logs: $gen_log | $teacher_log | $train_log"
