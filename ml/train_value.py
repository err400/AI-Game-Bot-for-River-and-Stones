#!/usr/bin/env python3
"""
Train a simple value function (ridge regression) for the Stones & Rivers agent.

Input: JSONL with one row per position (ideally from played games), e.g.

  {
    "features": [f0, f1, ..., f15],   # 16-dim StateFeatures order
    "final_score_from_side": 12.0,    # tournament score from the acting side's perspective
    "rows": 13, "cols": 12,
    "score_cols": [3,4,5,6,7,8],
    "side": "circle",
    "meta": {"game_id": "...", "turn": 23}
  }

Target: we normalize final_score to [-1, 1] via clip(score/50, -1, 1).
Output: models/value_v1.txt – space/newline separated weights (16 numbers).

Run:
  python ml/train_value.py \
    --input logs/value_data.jsonl \
    --out models/value_v1.txt \
    --l2 1e-3
"""
import argparse
import json
import math
import os
from typing import List

import numpy as np


def load_jsonl(path: str):
    X = []
    y = []
    n_bad = 0
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                feats = obj.get('features')
                if feats is None or len(feats) != 16:
                    n_bad += 1
                    continue
                # Prefer explicit normalized target if present
                if 'target' in obj:
                    t = float(obj['target'])
                else:
                    # Use final tournament score (from side perspective) and normalize
                    fs = float(obj.get('final_score_from_side', 0.0))
                    t = max(-1.0, min(1.0, fs / 50.0))
                X.append([float(v) for v in feats])
                y.append(t)
            except Exception:
                n_bad += 1
                continue
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return X, y, n_bad


def ridge_regression(X: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    """Closed-form ridge: w = (X^T X + λI)^{-1} X^T y"""
    n_features = X.shape[1]
    XtX = X.T @ X
    reg = l2 * np.eye(n_features, dtype=np.float64)
    Xty = X.T @ y
    w = np.linalg.solve(XtX + reg, Xty)
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='Path to JSONL with feature rows')
    ap.add_argument('--out', default='models/value_v1.txt', help='Where to save weights')
    ap.add_argument('--l2', type=float, default=1e-3, help='Ridge regularization strength')
    args = ap.parse_args()

    X, y, n_bad = load_jsonl(args.input)
    if X.size == 0:
        raise SystemExit(f'No valid rows in {args.input}')

    # Fit
    w = ridge_regression(X, y, args.l2)

    # Report basic metrics
    y_hat = X @ w
    mae = float(np.mean(np.abs(y_hat - y)))
    # Pearson r
    if X.shape[0] > 1:
        r = float(np.corrcoef(y_hat, y)[0, 1])
    else:
        r = float('nan')
    print(f'Trained on {len(y)} rows (bad lines: {n_bad})')
    print(f'MAE={mae:.4f}, Pearson r={r:.4f}, L2={args.l2}')

    # Save
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        for i, val in enumerate(w.tolist()):
            sep = '\n' if i + 1 < len(w) else ''
            f.write(f'{val}{sep}')
    print(f'Wrote weights to {args.out} (len={len(w)})')


if __name__ == '__main__':
    main()

