#!/usr/bin/env python3
"""
Train a simple linear policy scorer over per-move features using softmax
cross-entropy. Input JSONL (one row per root position):

{
  "move_features": [[f0..fD-1], ...],
  "target_idx": 3,               # or provide "target_p": [p0..pK-1]
  "rows": 13, "cols": 12,
  "score_cols": [..], "side": "circle",
  "meta": {...}
}

Outputs models/policy_v1.txt with D+1 numbers (weights then bias), which the
agent uses at the root to compute priors via softmax(Xw + b).
"""
import argparse
import json
import os
from typing import List, Tuple

import numpy as np


def read_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                out.append(json.loads(s))
            except Exception:
                continue
    return out


def build_dataset(rows: List[dict]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    X_list = []  # list of (K,D)
    T_list = []  # list of (K,) one-hot or (K,) probs
    for r in rows:
        mvf = r.get('move_features')
        if not mvf:
            continue
        M = np.asarray(mvf, dtype=np.float64)
        if M.ndim != 2 or M.shape[1] == 0:
            continue
        K = M.shape[0]
        if 'target_p' in r:
            p = np.asarray(r['target_p'], dtype=np.float64)
            if p.shape[0] != K:
                continue
            t = p / (np.sum(p) + 1e-9)
        else:
            ti = int(r.get('target_idx', -1))
            if ti < 0 or ti >= K:
                continue
            t = np.zeros(K, dtype=np.float64)
            t[ti] = 1.0
        X_list.append(M)
        T_list.append(t)
    return X_list, T_list


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)
    e = np.exp(z)
    s = e / (np.sum(e) + 1e-12)
    return s


def train_linear(X_list: List[np.ndarray], T_list: List[np.ndarray], l2: float, lr: float, epochs: int) -> Tuple[np.ndarray, float]:
    D = X_list[0].shape[1]
    w = np.zeros(D, dtype=np.float64)
    b = 0.0
    for ep in range(epochs):
        g_w = np.zeros_like(w)
        g_b = 0.0
        loss = 0.0
        nsum = 0
        for X, T in zip(X_list, T_list):
            logits = X @ w + b  # (K,)
            P = softmax(logits)
            # Cross-entropy with target distribution T
            loss += -np.sum(T * np.log(P + 1e-12))
            nsum += 1
            # Gradients: dL/dw = X^T (P - T) + l2*w, dL/db = sum(P - T)
            diff = (P - T)  # (K,)
            g_w += X.T @ diff
            g_b += np.sum(diff)
        # L2 reg
        loss = loss / max(1, nsum) + 0.5 * l2 * float(np.sum(w*w))
        g_w = g_w / max(1, nsum) + l2 * w
        g_b = g_b / max(1, nsum)
        w -= lr * g_w
        b -= lr * g_b
        if (ep+1) % max(1, epochs//10) == 0:
            print(f"epoch {ep+1}/{epochs} loss={loss:.4f}")
    return w, float(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='JSONL with move_features and target_idx or target_p')
    ap.add_argument('--out', default='models/policy_v1.txt', help='Where to save weights (D+1 numbers)')
    ap.add_argument('--l2', type=float, default=1e-3)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--epochs', type=int, default=200)
    args = ap.parse_args()

    rows = read_jsonl(args.input)
    X_list, T_list = build_dataset(rows)
    if not X_list:
        raise SystemExit('No valid rows in input')
    w, b = train_linear(X_list, T_list, args.l2, args.lr, args.epochs)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        for v in w.tolist():
            f.write(f"{v}\n")
        f.write(f"{b}\n")
    print(f"Saved policy weights to {args.out} (D={len(w)})")


if __name__ == '__main__':
    main()

