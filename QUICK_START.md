# Quick Start — Prepare for Learning

Use this sequence to generate diverse data, produce teacher targets, and train a non‑trivial value model. Do not start overnight learning until these steps succeed.

## TL;DR (Recommended defaults)

1) Generate diverse self‑play and vs baseline games (logs/value_data.jsonl):

```
python3 scripts/arena.py \
  --games-per-size 10 \
  --sizes small medium \
  --matchups self greedy random \
  --log \
  --rotate-port
```

Tip: Cap per‑move time for faster data generation

```
RS_TIME_CAP=0.25 python3 scripts/arena.py ...
```

2) Run teacher pass to compute targets (logs/value_teacher.jsonl):

```
python3 scripts/teacher_value_fixed.py \
  --input logs/value_data.jsonl \
  --output logs/value_teacher.jsonl \
  --seconds 0.3 \
  --max 5000
```

3) Train value model (writes models/value_v1.txt):

```
python3 ml/train_value.py \
  --input logs/value_teacher.jsonl \
  --out models/value_v1.txt
```

4) Verify model is non‑trivial (weights not all zeros):

```
head models/value_v1.txt
```

If values are all `0.0`: your dataset isn’t diverse enough (likely too many turn‑0 positions). Generate more games and repeat.

## One‑button runner

To automate the flow above with safety checks:

```
chmod +x scripts/prepare_for_learning.sh
./scripts/prepare_for_learning.sh
```

The script will:
- Generate diverse games across sizes/matchups (logs to `logs/generation.log`)
- Run teacher evaluation with a robust path (logs to `logs/teacher.log`)
- Train and write `models/value_v1.txt` (logs to `logs/training.log`)
- Validate that weights are non‑zero and targets are diverse

## Notes

- You can lower time per move during data generation with `RS_TIME_CAP`, e.g. `0.20–0.30` to speed up.
- The teacher pass script uses `choose()` + `get_last_root_value()` to avoid the C++ evaluate() segfault.
- Illegal river pushes are disabled in the move generator; if your environment allows them, re‑enable the call to `explore_river_push()` in `student_agent.cpp` and add appropriate validation.

