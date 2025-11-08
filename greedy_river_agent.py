"""Python wrapper for Greedy River Agent"""
import sys
import os

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
        return self.agent.choose(b, r, c, sc, tl, ot)


# Alias for compatibility with agent factory
StudentAgent = GreedyRiverAgent
