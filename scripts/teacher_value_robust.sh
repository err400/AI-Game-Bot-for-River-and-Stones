#!/usr/bin/env bash
# Robust per-row teacher value generator.
# - Runs the C++ agent's evaluate() on each JSONL row in its own Python process
# - Survives native crashes on bad rows and keeps going
# - Avoids zsh quoting issues by keeping everything inside bash

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bash scripts/teacher_value_robust.sh \
  --input  <path/to/input.jsonl> \
  --output <path/to/output.jsonl> \
  [--seconds 0.3] [--k0 5] [--k1 1.2] [--c-puct 1.2] [--limit N]

Notes:
- Run from repo root so PYTHONPATH=. finds student_agent_cpp
- Creates/overwrites --output; appends one JSON per successfully labeled row
USAGE
}

INPUT=""
OUTPUT=""
SECONDS_BUDGET="0.3"
K0="5"
K1="1.2"
C_PUCT="1.2"
LIMIT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)  INPUT="$2"; shift 2;;
    --output) OUTPUT="$2"; shift 2;;
    --seconds) SECONDS_BUDGET="$2"; shift 2;;
    --k0)     K0="$2"; shift 2;;
    --k1)     K1="$2"; shift 2;;
    --c-puct) C_PUCT="$2"; shift 2;;
    --limit)  LIMIT="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

if [[ -z "$INPUT" || -z "$OUTPUT" ]]; then
  usage
  exit 1
fi

if [[ ! -f "$INPUT" ]]; then
  echo "[error] Input not found: $INPUT" >&2
  exit 1
fi

IN_COUNT=$(wc -l < "$INPUT" | tr -d ' ')
if [[ "${IN_COUNT}" -eq 0 ]]; then
  echo "[warn] Input has 0 lines: $INPUT" >&2
fi

mkdir -p "$(dirname "$OUTPUT")"
> "$OUTPUT"

# Create a one-row Python evaluator in a temp file
PY_ONCE="$(mktemp -t teacher_once_XXXX.py)"
cat > "$PY_ONCE" <<'PY'
import sys, os, json
from student_agent_cpp import StudentAgent as SA
s = sys.stdin.read()
if not s:
    sys.exit(0)
try:
    r = json.loads(s)
    board = r.get('board')
    rows = int(r.get('rows', 0))
    cols = int(r.get('cols', 0))
    sc = list(map(int, r.get('score_cols', [])))
    side = r.get('side')
    if not board or not rows or not cols or not sc or not side:
        sys.exit(0)
    ag = SA(side)
    seconds = float(os.environ.get('TV_SECONDS', '0.3'))
    v = ag.evaluate({'board': board, 'rows': rows, 'cols': cols, 'score_cols': sc},
                    rows, cols, sc, side, seconds)
    r['target'] = float(v)
    sys.stdout.write(json.dumps(r) + "\n")
except Exception:
    pass
PY

# Verify we can import the module before looping
if ! PYTHONPATH=. python3 -c "import student_agent_cpp" >/dev/null 2>&1; then
  echo "[error] Cannot import student_agent_cpp. Run 'make cmake-rebuild' from repo root." >&2
  rm -f "$PY_ONCE"
  exit 1
fi

echo "[info] Input:  $INPUT ($IN_COUNT rows)"
echo "[info] Output: $OUTPUT"
echo "[info] seconds=$SECONDS_BUDGET k0=$K0 k1=$K1 c_puct=$C_PUCT limit=${LIMIT:-none}"

export TV_SECONDS="$SECONDS_BUDGET"
export RS_K0="$K0"
export RS_K1="$K1"
export RS_C_PUCT="$C_PUCT"

INPUT_FOR_LOOP="$INPUT"
TEMP_IN=""
if [[ -n "$LIMIT" ]]; then
  TEMP_IN="$(mktemp -t teacher_in_XXXX.jsonl)"
  head -n "$LIMIT" "$INPUT" > "$TEMP_IN"
  INPUT_FOR_LOOP="$TEMP_IN"
fi

LINES_DONE=0
while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  printf '%s\n' "$line" | PYTHONPATH=. python3 -u "$PY_ONCE" >> "$OUTPUT" || true
  LINES_DONE=$((LINES_DONE+1))
done < "$INPUT_FOR_LOOP"

OUT_COUNT=$(wc -l < "$OUTPUT" | tr -d ' ')
echo "[info] Processed $LINES_DONE rows; wrote $OUT_COUNT rows to $OUTPUT"

rm -f "$PY_ONCE"
[[ -n "$TEMP_IN" ]] && rm -f "$TEMP_IN"

