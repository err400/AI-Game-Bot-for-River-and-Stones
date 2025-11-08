import build.student_agent_module as _sam
import copy
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

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

        # Smoke-validate with authoritative engine. If invalid, fallback to a legal move.
        try:
            try:
                from client_server.gameEngine import validate_and_apply_move, generate_all_moves
            except Exception:
                from gameEngine import validate_and_apply_move, generate_all_moves  # fallback import
            board_copy = copy.deepcopy(board)
            ok, _msg = validate_and_apply_move(board_copy, move_dict, self._side, rows, cols, score_cols)
        except Exception:
            ok = True  # if engine import fails, trust the move

        if not ok:
            try:
                # Get all legal moves from engine and pick the first
                legal_moves = generate_all_moves(board, self._side, rows, cols, score_cols)
                for mv in legal_moves:
                    try:
                        ok2, _ = validate_and_apply_move(copy.deepcopy(board), mv, self._side, rows, cols, score_cols)
                        if ok2:
                            return mv
                    except Exception:
                        continue
            except Exception:
                pass
        return move_dict

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
