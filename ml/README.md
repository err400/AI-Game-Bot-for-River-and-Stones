Value Training Setup
====================

Goal
-----
Train a lightweight value function V(s) on top of the 16‑dimensional features already computed in C++ (StateFeatures). The model predicts normalized final tournament score from the acting side’s perspective.

Data Format
-----------
Use JSONL (one object per line), each with at least:

{
  "features": [f0..f15],          # 16 numbers in the same order as StateFeatures
  "final_score_from_side": 12.0,  # tournament score from the acting side’s perspective
  "rows": 13,
  "cols": 12,
  "score_cols": [3,4,5,6,7,8],
  "side": "circle",
  "meta": {"game_id": "...", "turn": 23}
}

Target normalization: target = clip(final_score_from_side / 50.0, -1, +1).

Training
--------
Run ridge regression (closed form) to compute weights:

python ml/train_value.py \
  --input logs/value_data.jsonl \
  --out models/value_v1.txt \
  --l2 1e-3

This writes models/value_v1.txt with 16 numbers (one per line). The C++ agent auto-loads this file (or from the alternative search paths defined in load_value_model()) and uses it in leaf_value().

Feature Order (StateFeatures)
-----------------------------
Index -> Meaning
 0 -> n_self (stones in own scoring area)
 1 -> n_opp
 2 -> m_self (1‑move reachability estimate)
 3 -> m_opp
 4 -> min_dist_self (to own scoring)
 5 -> min_dist_opp
 6 -> avg_dist_self
 7 -> avg_dist_opp
 8 -> rivers_self
 9 -> rivers_opp
10 -> mobility_self (adjacent empties)
11 -> mobility_opp
12 -> center_pressure_self (0..1, weight in middle columns)
13 -> center_pressure_opp
14 -> threat_self (binary one‑move threat)
15 -> threat_opp

Integrating Logging
-------------------
The simplest place to log is on the Python side (student_agent_cpp.py), right after you call the C++ agent’s choose(). For each move:

1) Build the same 16‑D feature vector in Python (mirror the C++ functions or call a helper exposed by C++ if you add one later).
2) Append a JSONL line with features, side, rows/cols/score_cols. Tag with a game_id.
3) When the game finishes, compute the official tournament score from the engine and append it back to all rows for that game (or create a second JSONL with final labels and join later).

Tips
----
- Keep data local (no network). The agent already handles missing weights (falls back to heuristic V).
- Start with a few thousand samples across board sizes (0.5–0.8s per move). Balance circle/square.
- Keep terminal overrides in C++: non‑terminals are clamped to ±8; terminals are ±50.
- After integrating a model, re‑tune c_puct (0.8–2.0) and prior temperature.

