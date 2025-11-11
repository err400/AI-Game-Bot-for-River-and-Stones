#!/usr/bin/env python3
"""
Fixed teacher value pass using choose() workaround.

The evaluate() method crashes due to a segfault, so we use choose()
and then extract the root value via get_last_root_value().

Usage:
  PYTHONPATH=. python scripts/teacher_value_fixed.py \
    --input logs/value_all.jsonl \
    --output logs/value_teacher.jsonl \
    --seconds 0.3 \
    --max 5000
"""
import argparse
import json
import sys
import os
from pathlib import Path

# Add paths for imports
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'build'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'build_no_lto'))

# Override module path if set
_OVERRIDE_DIR = os.environ.get("RS_STUDENT_MODULE_PATH")
if _OVERRIDE_DIR and _OVERRIDE_DIR not in sys.path:
    sys.path.insert(0, _OVERRIDE_DIR)


def read_jsonl(path: str):
    with open(path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception:
                continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--seconds', type=float, default=0.3)
    ap.add_argument('--max', type=int, default=0)
    args = ap.parse_args()

    # Import after path setup
    try:
        from student_agent_cpp import StudentAgent
    except Exception as e:
        print(f"Failed to import StudentAgent: {e}")
        sys.exit(1)

    Path(os.path.dirname(args.output) or '.').mkdir(parents=True, exist_ok=True)
    n_in = n_out = n_err = 0

    with open(args.output, 'w') as fw:
        for row in read_jsonl(args.input):
            n_in += 1
            if args.max and n_out >= args.max:
                break

            # Validate required fields
            board = row.get('board')
            rows = row.get('rows')
            cols = row.get('cols')
            score_cols = row.get('score_cols')
            side = row.get('side')

            if not board or not rows or not cols or not score_cols or not side:
                n_err += 1
                continue

            # Create game state
            gs = {
                'board': board,
                'rows': int(rows),
                'cols': int(cols),
                'score_cols': list(map(int, score_cols)),
            }

            # WORKAROUND: Use choose() instead of evaluate() to avoid segfault
            # Then extract root value via get_last_root_value()
            try:
                agent = StudentAgent(str(side))
                # Run MCTS via choose
                _ = agent.choose(gs, args.seconds, args.seconds)
                # Extract root value from last search
                value = agent.get_last_root_value()

                # Write output with target
                out = dict(row)
                out['target'] = value
                fw.write(json.dumps(out) + "\n")
                fw.flush()  # Flush to see progress
                n_out += 1

                if n_out % 100 == 0:
                    print(f"Processed {n_out} / {n_in} rows...")

            except Exception as e:
                n_err += 1
                if n_err <= 5:  # Only print first few errors
                    print(f"Error on row {n_in}: {e}")
                continue

    print(f"\nDone! Processed {n_in} rows, wrote {n_out} with teacher targets to {args.output}")
    print(f"Errors: {n_err}")


if __name__ == '__main__':
    main()
