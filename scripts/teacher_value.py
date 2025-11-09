#!/usr/bin/env python3
"""
Teacher value pass: read logged positions (with 'board') and compute deeper
MCTS value targets using the C++ agent's evaluate() method.

Input JSONL rows should contain at least: 'board', 'rows', 'cols', 'score_cols', 'side'.
Writes a new JSONL with an added 'target' field (value in roughly [-10,10],
root-side perspective).

Usage:
  python scripts/teacher_value.py \
    --input logs/value_data.jsonl \
    --output logs/value_teacher.jsonl \
    --seconds 1.5 \
    --max 5000

Set RS_ROOT_NOISE to a small value (e.g., 0.15) to decorrelate search a bit, if desired.
"""
import argparse
import json
import os
from typing import Iterator, Dict, Any

from pathlib import Path


def read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
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
    ap.add_argument('--input', required=True, help='Path to JSONL produced by logging (must include board)')
    ap.add_argument('--output', required=True, help='Where to write JSONL with targets appended')
    ap.add_argument('--seconds', type=float, default=1.5, help='Per-position MCTS time budget (seconds)')
    ap.add_argument('--max', type=int, default=0, help='Optional limit of rows (0 => all)')
    ap.add_argument('--module-path', default='', help='Optional override path for student_agent_module (snapshot)')
    args = ap.parse_args()

    if args.module_path:
        os.environ['RS_STUDENT_MODULE_PATH'] = args.module_path

    # Import agent wrapper
    from student_agent_cpp import StudentAgent

    n_in = 0
    n_out = 0

    Path(os.path.dirname(args.output) or '.').mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as fw:
        for row in read_jsonl(args.input):
            n_in += 1
            if args.max and n_out >= args.max:
                break
            board = row.get('board')
            rows = int(row.get('rows', 0))
            cols = int(row.get('cols', 0))
            score_cols = list(map(int, row.get('score_cols', [])))
            side = row.get('side')
            if not board or not rows or not cols or not score_cols or not side:
                continue
            try:
                agent = StudentAgent(side)
                val = agent.evaluate({'board': board, 'rows': rows, 'cols': cols, 'score_cols': score_cols}, rows, cols, score_cols, side, float(args.seconds))
                out = dict(row)
                out['target'] = float(val)
                fw.write(json.dumps(out) + "\n")
                n_out += 1
            except Exception as e:
                # Skip on failure
                continue

    print(f"Processed {n_in} rows, wrote {n_out} with teacher targets to {args.output}")


if __name__ == '__main__':
    main()

