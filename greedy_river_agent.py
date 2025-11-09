"""Python wrapper for Greedy River Agent with legality fallback.

Ensures any move returned by the C++ module passes the authoritative engine's
validation. If not, falls back to the first legal move generated via
compute_valid_targets + validate_and_apply_move.
"""
import sys
import os
import copy
from typing import List, Dict, Any

# Add build_greedy directory to path (module lives here)
_REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
_BUILD_DIR = os.path.join(_REPO_ROOT, "build_greedy")
if _BUILD_DIR not in sys.path:
    sys.path.insert(0, _BUILD_DIR)

try:
    from greedy_river_module import GreedyRiverAgent as _GreedyAgent
except ImportError as e:
    raise ImportError(f"Failed to import greedy_river_module: {e}")


def _to_str(x):
    try:
        return str(getattr(x, "name"))
    except Exception:
        return str(x)


def _cell_to_dict(cell):
    """Convert engine cell (None/dict/Piece) to a plain dict for C++ wrapper."""
    if cell is None:
        return {}
    if isinstance(cell, dict):
        # Normalize keys and values to strings
        return {
            "owner": _to_str(cell.get("owner", "")),
            "side": _to_str(cell.get("side", "")),
            "orientation": _to_str(cell.get("orientation", "")),
        }
    # Likely a gameEngine.Piece instance
    try:
        owner = _to_str(getattr(cell, "owner"))
        side = _to_str(getattr(cell, "side"))
        try:
            orientation = _to_str(getattr(cell, "orientation"))
        except Exception:
            orientation = ""
        return {"owner": owner, "side": side, "orientation": orientation}
    except Exception:
        return {}


def _normalize_board(board):
    return [[_cell_to_dict(c) for c in row] for row in board]


class GreedyRiverAgent:
    def __init__(self, player):
        self.player = player
        self.agent = _GreedyAgent(player)

    def choose(self, board, rows, cols, score_cols, time_left, opp_time):
        """Choose a move using greedy river strategy"""
        b = _normalize_board(board)
        r = int(rows)
        c = int(cols)
        sc = [int(x) for x in score_cols]
        tl = float(time_left)
        ot = float(opp_time)

        mv = self.agent.choose(b, r, c, sc, tl, ot)
        # Validate against authoritative engine. If invalid, fallback to first legal move.
        try:
            try:
                from client_server.gameEngine import validate_and_apply_move, compute_valid_targets
            except Exception:
                from gameEngine import validate_and_apply_move, compute_valid_targets

            ok, _ = validate_and_apply_move(copy.deepcopy(board), mv, self.player, r, c, sc)
            if ok:
                return mv

            # Fallback: enumerate legal moves for this side
            legal: List[Dict[str, Any]] = []
            for y in range(r):
                for x in range(c):
                    cell = board[y][x]
                    if not cell or getattr(cell, 'owner', None) != self.player:
                        continue
                    info = compute_valid_targets(board, x, y, self.player, r, c, sc)
                    for (tx, ty) in info.get('moves', set()):
                        mv2 = {"action": "move", "from": [x, y], "to": [tx, ty]}
                        ok2, _ = validate_and_apply_move(copy.deepcopy(board), mv2, self.player, r, c, sc)
                        if ok2:
                            return mv2
                    for (to_xy, pushed_xy) in info.get('pushes', []):
                        tx, ty = to_xy
                        px, py = pushed_xy
                        mv2 = {"action": "push", "from": [x, y], "to": [tx, ty], "pushed_to": [px, py]}
                        ok2, _ = validate_and_apply_move(copy.deepcopy(board), mv2, self.player, r, c, sc)
                        if ok2:
                            return mv2
                    # Try safe flips/rotates and pick the first valid one
                    for ori in ("horizontal", "vertical"):
                        mv2 = {"action": "flip", "from": [x, y], "orientation": ori}
                        ok2, _ = validate_and_apply_move(copy.deepcopy(board), mv2, self.player, r, c, sc)
                        if ok2:
                            return mv2
                    mv2 = {"action": "flip", "from": [x, y]}
                    ok2, _ = validate_and_apply_move(copy.deepcopy(board), mv2, self.player, r, c, sc)
                    if ok2:
                        return mv2
                    mv2 = {"action": "rotate", "from": [x, y]}
                    ok2, _ = validate_and_apply_move(copy.deepcopy(board), mv2, self.player, r, c, sc)
                    if ok2:
                        return mv2
        except Exception:
            # If engine not importable, trust the greedy move
            pass
        return mv


# Alias for compatibility with agent factory
StudentAgent = GreedyRiverAgent
