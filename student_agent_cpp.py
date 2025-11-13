import sys, os, importlib

# Make sure our local build/ is preferred over any pip 'build' package
_REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
_BUILD_DIR = os.path.join(_REPO_ROOT, "build")

# Optional override: prefer a provided module search path (for snapshots/leagues)
_OVERRIDE_DIR = os.environ.get("RS_STUDENT_MODULE_PATH")
search_paths = []
if _OVERRIDE_DIR:
    search_paths.append(_OVERRIDE_DIR)
search_paths.extend([_BUILD_DIR, _REPO_ROOT])
for _p in search_paths:
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

# Try to import the compiled extension from build/
try:
    import student_agent_module as _sam
except Exception:
    # Fallback: import as build.student_agent_module but ensure our repo_root is first in sys.path
    try:
        _sam = importlib.import_module("build.student_agent_module")
    except Exception as _e:
        raise ImportError(f"Failed to import compiled student_agent_module from build/: {_e}")
import copy
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple

def _to_str(x):
    try:
        return str(getattr(x, "name"))
    except Exception:
        return str(x)

def _cell_to_dict(cell):
    if cell is None:
        return {}
    if isinstance(cell, dict):
        return {
            "owner": _to_str(cell.get("owner", "")),
            "side": _to_str(cell.get("side", "")),
            "orientation": _to_str(cell.get("orientation", "")),
        }
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
    return [[ _cell_to_dict(c) for c in row ] for row in board]

def _to_int_list(x):
    return [int(v) for v in x]


def get_opponent(player: str) -> str:
    return "square" if player == "circle" else "circle"

def in_bounds(x: int, y: int, rows: int, cols: int) -> bool:
    """Check if coordinates are within board boundaries."""
    return 0 <= x < cols and 0 <= y < rows

def score_cols_for(cols: int) -> List[int]:
    """Get the column indices for scoring areas."""
    w = 4
    start = max(0, (cols - w) // 2)
    return list(range(start, start + w))

def top_score_row() -> int:
    """Get the row index for Circle's scoring area."""
    return 2

def bottom_score_row(rows: int) -> int:
    """Get the row index for Square's scoring area."""
    return rows - 3

def is_opponent_score_cell(x: int, y: int, player: str, rows: int, cols: int, score_cols: List[int]) -> bool:
    """Check if a cell is in the opponent's scoring area."""
    if player == "circle":
        return (y == bottom_score_row(rows)) and (x in score_cols)
    else:
        return (y == top_score_row()) and (x in score_cols)

def is_own_score_cell(x: int, y: int, player: str, rows: int, cols: int, score_cols: List[int]) -> bool:
    """Check if a cell is in the player's own scoring area."""
    if player == "circle":
        return (y == top_score_row()) and (x in score_cols)
    else:
        return (y == bottom_score_row(rows)) and (x in score_cols)

class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    """
    def __init__(self, player: str):
        """Initialize agent with player identifier."""
        self.player = player
        self.opponent = get_opponent(player)
    
    @abstractmethod
    def choose(self, board: List[List[Any]], rows: int, cols: int, score_cols: List[int], current_player_time: float, opponent_time: float) -> Optional[Dict[str, Any]]:
        pass

class StudentAgent:
    def __init__(self, side: str):
        self._impl = _sam.StudentAgent(side)
        self._side = side

    def choose(self, *args):
        # Support both call styles:
        #  - choose(board, rows, cols, score_cols, cur_t, opp_t)
        #  - choose(game_state_dict, cur_t, opp_t)
        if not args:
            raise TypeError("choose() missing arguments")

        if isinstance(args[0], dict):
            gs, cur_t, opp_t = args
            board = gs["board"]
            rows, cols = int(gs["rows"]), int(gs["cols"])
            score_cols = _to_int_list(gs["score_cols"])
        else:
            if len(args) != 6:
                raise TypeError("choose(board, rows, cols, score_cols, cur_t, opp_t) expected")
            board, rows, cols, score_cols, cur_t, opp_t = args
            rows, cols = int(rows), int(cols)
            score_cols = _to_int_list(score_cols)

        board_norm = _normalize_board(board)
        m = self._impl.choose(board_norm, rows, cols, score_cols, float(cur_t), float(opp_t))

        move_dict = {
            "action": getattr(m, "action"),
            "from":  list(getattr(m, "from_pos")),
            "to":    list(getattr(m, "to_pos")),
            "pushed_to": list(getattr(m, "pushed_to")),
            "orientation": getattr(m, "orientation"),
        }
        # Note: Smoke validation removed to avoid any dependency on client_server or TA engine modules.
        # We now trust the compiled agent's move and return it as-is.
        return move_dict

    def evaluate(self, game_state_or_board, rows=None, cols=None, score_cols=None, side=None, time_limit: float = 1.5) -> float:
        """Evaluate a state using the underlying MCTS (returns root-side value).

        Accepts either a full game_state dict with keys {board, rows, cols, score_cols}
        or the tuple (board, rows, cols, score_cols).
        """
        if isinstance(game_state_or_board, dict):
            gs = game_state_or_board
            board = gs["board"]
            rows = int(gs["rows"]) if rows is None else int(rows)
            cols = int(gs["cols"]) if cols is None else int(cols)
            score_cols = _to_int_list(gs["score_cols"]) if score_cols is None else _to_int_list(score_cols)
            side = side or self._side
        else:
            board = game_state_or_board
            rows = int(rows)
            cols = int(cols)
            score_cols = _to_int_list(score_cols)
            side = side or self._side

        board_norm = _normalize_board(board)
        # The C++ binding expects legacy format (vector<map>) and side explicitly
        val = self._impl.evaluate(board_norm, rows, cols, score_cols, str(side), float(time_limit))
        return float(val)

def test_student_agent():
    """
    Basic test to verify the student agent can be created and make moves.
    """
    print("Testing StudentAgent...")
    
    try:
        from gameEngine import default_start_board, DEFAULT_ROWS, DEFAULT_COLS
        
        rows, cols = DEFAULT_ROWS, DEFAULT_COLS
        score_cols = score_cols_for(cols)
        board = default_start_board(rows, cols)
        
        agent = StudentAgent("circle")
        move = agent.choose(board, rows, cols, score_cols,1.0,1.0)
        
        if move:
            print("Agent successfully generated a move")
        else:
            print("Agent returned no move")
    
    except ImportError:
        agent = StudentAgent("circle")
        print("StudentAgent created successfully")

if __name__ == "__main__":
    # Run basic test when file is executed directly
    test_student_agent()
