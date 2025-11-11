#!/usr/bin/env bash
set -euo pipefail

# Automated preparation pipeline:
#  1) Generate diverse self-play/baseline games with logging
#  2) Run teacher pass to compute value targets
#  3) Train value model and verify non-zero weights
#
# Usage (defaults are sane):
#   chmod +x scripts/prepare_for_learning.sh
#   ./scripts/prepare_for_learning.sh
#
# Tunables via env vars:
#   GAMES_PER_SIZE   (default: 10)
#   SIZES            (default: "small medium")
#   MATCHUPS         (default: "self greedy random")
#   TIME_CAP         (default: 0.30)          # RS_TIME_CAP for agent
#   TEACHER_SECONDS  (default: 0.30)          # seconds per position in teacher
#   TEACHER_MAX      (default: 5000)          # number of teacher rows
#   PORT_BASE        (default: 8080)
#
# Outputs:
#   logs/generation.log
#   logs/value_data.jsonl
#   logs/value_teacher.jsonl
#   logs/training.log
#   models/value_v1.txt

REPO_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$REPO_ROOT"

GAMES_PER_SIZE="${GAMES_PER_SIZE:-10}"
SIZES="${SIZES:-small medium}"
MATCHUPS="${MATCHUPS:-self greedy random}"
TIME_CAP="${TIME_CAP:-0.30}"
TEACHER_SECONDS="${TEACHER_SECONDS:-0.30}"
TEACHER_MAX="${TEACHER_MAX:-5000}"
PORT_BASE="${PORT_BASE:-8080}"

LOG_DIR="$REPO_ROOT/logs"
MODELS_DIR="$REPO_ROOT/models"
mkdir -p "$LOG_DIR" "$MODELS_DIR"

gen_log="$LOG_DIR/generation.log"
teacher_log="$LOG_DIR/teacher.log"
train_log="$LOG_DIR/training.log"

echo "=== Step 0: Build student agent (C++) ===" | tee "$gen_log"
make cmake-rebuild >>"$gen_log" 2>&1 || { echo "Build failed" >&2; exit 1; }

echo "=== Step 1: Generate diverse game data (this can take time) ===" | tee -a "$gen_log"
echo "SIZES=$SIZES MATCHUPS=$MATCHUPS GAMES_PER_SIZE=$GAMES_PER_SIZE RS_TIME_CAP=$TIME_CAP" | tee -a "$gen_log"

# Optional: ensure duplicate matchup uses C++ agent instead of Python baseline
# Set USE_DUPLICATE_CPP=1 and include 'duplicate' in MATCHUPS to enable.
DUP_FLAG=""
if [[ "${USE_DUPLICATE_CPP:-0}" == "1" ]]; then
  case " $MATCHUPS " in *" duplicate "*) DUP_FLAG="--use-duplicate-cpp" ;; esac
fi

# Use rotate-port to avoid socket reuse issues
RS_TIME_CAP="$TIME_CAP" python3 scripts/arena.py \
  --games-per-size "$GAMES_PER_SIZE" \
  --sizes $SIZES \
  --matchups $MATCHUPS \
  $DUP_FLAG \
  --log \
  --rotate-port \
  --port "$PORT_BASE" >>"$gen_log" 2>&1 || true

# Quick diversity check — ensure we logged something and not only turn 0
DATA_FILE="$LOG_DIR/value_data.jsonl"
if [[ ! -s "$DATA_FILE" ]]; then
  echo "[ERROR] No data logged at $DATA_FILE. Check generation logs: $gen_log" >&2
  exit 2
fi

echo "=== Step 1b: Turn diversity check (sample of 5000 lines) ===" | tee -a "$gen_log"
python3 - "$DATA_FILE" << 'PY' 2>>"$gen_log"
import json,sys
from collections import Counter
path=sys.argv[1]
c=Counter()
with open(path,'r') as f:
    for i,l in enumerate(f):
        if i>5000: break
        try:
            t=json.loads(l).get('meta',{}).get('turn',-1)
            c[t]+=1
        except: pass
print('Turn counts (sample):', dict(c))
mid = sum(c.get(t,0) for t in range(10,31))
print('Mid-game (10..30) sample count:', mid)
PY

echo "=== Step 2: Teacher pass (compute targets) ===" | tee "$teacher_log"
python3 scripts/teacher_value_fixed.py \
  --input "$DATA_FILE" \
  --output "$LOG_DIR/value_teacher.jsonl" \
  --seconds "$TEACHER_SECONDS" \
  --max "$TEACHER_MAX" >>"$teacher_log" 2>&1 || true

TEACHER_FILE="$LOG_DIR/value_teacher.jsonl"
if [[ ! -s "$TEACHER_FILE" ]]; then
  echo "[ERROR] Teacher output missing or empty: $TEACHER_FILE" >&2
  echo "See $teacher_log" >&2
  exit 3
fi

echo "=== Step 2b: Target diversity check (sample of 1000) ===" | tee -a "$teacher_log"
python3 - "$TEACHER_FILE" << 'PY' 2>>"$teacher_log"
import json,sys
vals=set()
path=sys.argv[1]
with open(path,'r') as f:
    for i,l in enumerate(f):
        if i>1000: break
        try:
            v=float(json.loads(l).get('target',0.0))
            vals.add(round(v,3))
        except: pass
print('Unique targets (sample):', len(vals))
print('Sample values:', sorted(list(vals))[:20])
PY

echo "=== Step 3: Train value model ===" | tee "$train_log"
python3 ml/train_value.py \
  --input "$TEACHER_FILE" \
  --out "$MODELS_DIR/value_v1.txt" >>"$train_log" 2>&1 || true

MODEL_FILE="$MODELS_DIR/value_v1.txt"
if [[ ! -s "$MODEL_FILE" ]]; then
  echo "[ERROR] Model file missing: $MODEL_FILE" >&2
  echo "See $train_log" >&2
  exit 4
fi

echo "=== Step 3b: Non-zero weights check ===" | tee -a "$train_log"
nonzero=$(awk '{if ($1+0 != 0) c++} END{print c+0}' "$MODEL_FILE")
total=$(wc -l < "$MODEL_FILE" | awk '{print $1+0}')
echo "Weights non-zero: $nonzero / $total" | tee -a "$train_log"
if [[ "${nonzero:-0}" -eq 0 ]]; then
  echo "[WARN] All weights are zero — likely homogeneous targets. Generate more diverse data and rerun." | tee -a "$train_log"
  exit 5
fi

echo "=== SUCCESS: Value model trained with non-zero weights ==="
echo "Model: $MODEL_FILE"
echo "Logs:  $gen_log | $teacher_log | $train_log"
