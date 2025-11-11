#!/usr/bin/env python3
"""
Run self-play games locally without the web server, using the same C++ student
agent on both sides. Optionally logs value training rows compatible with the
teacher pipeline.

Examples:
  RS_TIME_CAP=0.25 python3 scripts/local_arena.py --games 5 --size small --log

This avoids HTTP entirely and uses client_server/gameEngine.py for rules.
"""
import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from client_server.gameEngine import (
    DEFAULT_ROWS, DEFAULT_COLS, default_start_board,
    score_cols_for, compute_final_scores, validate_and_apply_move,
)
from student_agent_cpp import StudentAgent as StudentAgentCpp


def in_bounds(x: int, y: int, rows: int, cols: int) -> bool:
    return 0 <= x < cols and 0 <= y < rows


def _feature_vector(board: list, rows: int, cols: int, score_cols: List[int], side: str) -> List[float]:
    # Minimal copy from bot_client for value features (16-dim)
    from client_server.gameEngine import opponent

    def top_score_row() -> int: return 2
    def bottom_score_row() -> int: return rows - 3

    def mdist_to_own_score(owner: str, r: int, c: int) -> int:
        target_row = top_score_row() if owner == "circle" else bottom_score_row()
        row_d = abs(r - target_row)
        col_d = min((abs(c - sc) for sc in score_cols), default=0)
        return row_d + col_d

    def min_avg_dist(owner: str) -> Tuple[float, float]:
        mn = 1e9; tot = 0.0; cnt = 0
        for y in range(rows):
            for x in range(cols):
                p = board[y][x]
                if p and getattr(p, 'side', '') == 'stone' and getattr(p, 'owner', '') == owner:
                    d = mdist_to_own_score(owner, y, x)
                    mn = min(mn, float(d)); tot += d; cnt += 1
        if cnt == 0: return 0.0, 0.0
        return (0.0 if mn >= 1e8 else float(mn)), float(tot / cnt)

    def count_rivers(owner: str) -> int:
        c = 0
        for y in range(rows):
            for x in range(cols):
                p = board[y][x]
                if p and getattr(p, 'owner', '') == owner and getattr(p, 'side', '') == 'river':
                    c += 1
        return c

    def approx_mob(owner: str) -> int:
        mob = 0
        for y in range(rows):
            for x in range(cols):
                p = board[y][x]
                if p and getattr(p, 'owner', '') == owner:
                    for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny, nx = y+dy, x+dx
                        if 0 <= ny < rows and 0 <= nx < cols and (board[ny][nx] is None):
                            mob += 1
        return mob

    owner = side; opp = opponent(owner)
    # Count scored
    sr_top = top_score_row(); sr_bot = bottom_score_row()
    scored_self = sum(1 for c in score_cols if board[sr_top][c] and getattr(board[sr_top][c], 'owner', '') == 'circle') if owner=='circle' else \
                  sum(1 for c in score_cols if board[sr_bot][c] and getattr(board[sr_bot][c], 'owner', '') == 'square')
    scored_opp  = sum(1 for c in score_cols if board[sr_top][c] and getattr(board[sr_top][c], 'owner', '') == 'circle') if owner=='square' else \
                  sum(1 for c in score_cols if board[sr_bot][c] and getattr(board[sr_bot][c], 'owner', '') == 'square')

    # Reachability in one (approx)
    def reach(owner: str) -> int:
        cnt=0
        for y in range(rows):
            for x in range(cols):
                p = board[y][x]
                if p and getattr(p, 'side','')=='stone' and getattr(p,'owner','')==owner:
                    for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny, nx = y+dy, x+dx
                        if 0 <= ny < rows and 0 <= nx < cols and (board[ny][nx] is None):
                            cnt += 1; break
        return cnt

    m_self = reach(owner); m_opp = reach(opp)
    min_self, avg_self = min_avg_dist(owner); min_opp, avg_opp = min_avg_dist(opp)
    rivers_self, rivers_opp = count_rivers(owner), count_rivers(opp)
    mob_self, mob_opp = approx_mob(owner), approx_mob(opp)
    center_self = sum(1 for y in range(rows) for x in score_cols if board[y][x] and getattr(board[y][x],'owner','')==owner)
    center_opp  = sum(1 for y in range(rows) for x in score_cols if board[y][x] and getattr(board[y][x],'owner','')==opp)

    return [
        float(scored_self), float(scored_opp),
        float(m_self), float(m_opp),
        float(min_self), float(min_opp),
        float(avg_self), float(avg_opp),
        float(rivers_self), float(rivers_opp),
        float(mob_self), float(mob_opp),
        float(center_self/(len(score_cols) or 1)), float(center_opp/(len(score_cols) or 1)),
        0.0, 0.0,
    ]


def piece_board_to_json(board: List[List[Any]]) -> List[List[Optional[Dict[str,Any]]]]:
    out=[]
    for row in board:
        orow=[]
        for cell in row:
            if cell is None: orow.append(None)
            else:
                orow.append({
                    'owner': getattr(cell,'owner',''),
                    'side': getattr(cell,'side','stone'),
                    'orientation': getattr(cell,'orientation','horizontal'),
                })
        out.append(orow)
    return out


def run_game(rows: int, cols: int, log: bool) -> Dict[str,Any]:
    board = default_start_board(rows, cols)
    score_cols = score_cols_for(cols)
    agents = {
        'circle': StudentAgentCpp('circle'),
        'square': StudentAgentCpp('square'),
    }
    time_left = {'circle': 60.0, 'square': 60.0}
    current = 'circle'
    turn = 0
    win_count = 4 if cols <= 12 else (5 if cols <= 14 else 6)

    # Logging
    enable_log = log or (os.environ.get('RS_LOG_VALUE','0') == '1')
    log_path = os.environ.get('RS_LOG_PATH', str(Path(__file__).resolve().parents[1]/'logs'/'value_data.jsonl'))
    if enable_log:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    while turn < 400:
        agent = agents[current]
        # Optional logging row
        if enable_log:
            feats = _feature_vector(board, rows, cols, score_cols, current)
            row = {
                'features': feats,
                'rows': rows,
                'cols': cols,
                'score_cols': score_cols,
                'side': current,
                'board': piece_board_to_json(board),
                'meta': {'turn': turn, 'strategy':'student_cpp'},
            }
            with open(log_path,'a') as f: f.write(json.dumps(row)+"\n")

        mv = agent.choose(board, rows, cols, score_cols, time_left[current], time_left['square' if current=='circle' else 'circle'])
        ok, msg = validate_and_apply_move(board, mv or {}, current, rows, cols, score_cols)
        if not ok:
            # fall back to a no-op move (skip) or terminate
            break

        # Win check
        scores = compute_final_scores(board, None, rows, cols, score_cols, None)
        if scores.get('circle',0) >= win_count or scores.get('square',0) >= win_count:
            break

        current = 'square' if current=='circle' else 'circle'
        turn += 1

    winner = 'circle' if scores.get('circle',0) >= win_count else ('square' if scores.get('square',0) >= win_count else 'draw')
    return {'turns': turn, 'winner': winner}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', type=int, default=5)
    ap.add_argument('--size', choices=['small','medium','large'], default='small')
    ap.add_argument('--log', action='store_true')
    args = ap.parse_args()

    if args.size == 'small': rows, cols = 13, 12
    elif args.size == 'medium': rows, cols = 13, 14
    else: rows, cols = 13, 16

    results=[]
    for i in range(args.games):
        r = run_game(rows, cols, args.log)
        results.append(r)
        print(f"Game {i+1}/{args.games}: winner={r['winner']} turns={r['turns']}")

    print("Done.")


if __name__ == '__main__':
    main()

