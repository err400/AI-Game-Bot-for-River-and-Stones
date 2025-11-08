#!/usr/bin/env python3
"""Play main MCTS agent vs greedy river agent"""
import sys
import os
import time

def create_starting_board():
    """13x12 board with standard starting positions"""
    board = [[{} for _ in range(12)] for _ in range(13)]

    # Circle pieces (top)
    board[4][4] = {"owner": "circle", "side": "stone"}
    board[4][5] = {"owner": "circle", "side": "stone"}
    board[4][6] = {"owner": "circle", "side": "stone"}
    board[4][7] = {"owner": "circle", "side": "stone"}
    board[5][5] = {"owner": "circle", "side": "stone"}
    board[5][6] = {"owner": "circle", "side": "stone"}

    # Square pieces (bottom)
    board[8][4] = {"owner": "square", "side": "stone"}
    board[8][5] = {"owner": "square", "side": "stone"}
    board[8][6] = {"owner": "square", "side": "stone"}
    board[8][7] = {"owner": "square", "side": "stone"}
    board[7][5] = {"owner": "square", "side": "stone"}
    board[7][6] = {"owner": "square", "side": "stone"}

    return board

def apply_move(board, move):
    """Apply a move and return new board"""
    import copy
    new_board = copy.deepcopy(board)

    action = move.get('action', '')
    from_pos = move.get('from', [])
    to_pos = move.get('to', [])
    pushed_to = move.get('pushed_to', [])
    orientation = move.get('orientation', '')

    if not from_pos or len(from_pos) < 2:
        return new_board

    from_r, from_c = from_pos[1], from_pos[0]

    if from_r < 0 or from_r >= len(new_board) or from_c < 0 or from_c >= len(new_board[0]):
        return new_board

    if action == 'move':
        if to_pos and len(to_pos) >= 2:
            to_r, to_c = to_pos[1], to_pos[0]
            if 0 <= to_r < len(new_board) and 0 <= to_c < len(new_board[0]):
                new_board[to_r][to_c] = new_board[from_r][from_c]
                new_board[from_r][from_c] = {}

    elif action == 'push':
        if to_pos and pushed_to and len(to_pos) >= 2 and len(pushed_to) >= 2:
            to_r, to_c = to_pos[1], to_pos[0]
            push_r, push_c = pushed_to[1], pushed_to[0]
            if (0 <= to_r < len(new_board) and 0 <= to_c < len(new_board[0]) and
                0 <= push_r < len(new_board) and 0 <= push_c < len(new_board[0])):
                pusher = new_board[from_r][from_c]
                pushed = new_board[to_r][to_c]
                new_board[push_r][push_c] = pushed
                new_board[to_r][to_c] = pusher
                if pusher.get('side') == 'river':
                    new_board[to_r][to_c]['side'] = 'stone'
                    new_board[to_r][to_c].pop('orientation', None)
                new_board[from_r][from_c] = {}

    elif action == 'flip':
        piece = new_board[from_r][from_c]
        if piece.get('side') == 'stone':
            piece['side'] = 'river'
            piece['orientation'] = orientation if orientation else 'horizontal'
        elif piece.get('side') == 'river':
            piece['side'] = 'stone'
            piece.pop('orientation', None)

    elif action == 'rotate':
        piece = new_board[from_r][from_c]
        if piece.get('side') == 'river':
            current_ori = piece.get('orientation', 'horizontal')
            if orientation:
                piece['orientation'] = orientation
            else:
                piece['orientation'] = 'vertical' if current_ori == 'horizontal' else 'horizontal'

    return new_board

def count_scoring(board, owner):
    """Count stones in scoring row"""
    score_row = 2 if owner == 'circle' else 10
    score_cols = [3, 4, 5, 6, 7, 8]
    count = 0

    for c in score_cols:
        if 0 <= score_row < len(board) and 0 <= c < len(board[0]):
            cell = board[score_row][c]
            if cell.get('side') == 'stone' and cell.get('owner') == owner:
                count += 1

    return count

def print_board_compact(board):
    """Print compact board view"""
    print("   ", end="")
    for c in range(3, 9):
        print(f"{c}", end=" ")
    print()
    for r in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        print(f"{r:2}", end=" ")
        for c in range(3, 9):
            cell = board[r][c]
            if not cell:
                print("·", end=" ")
            else:
                owner_char = cell['owner'][0].upper()
                side_char = cell['side'][0]
                if side_char == 's':
                    print(f"{owner_char}", end=" ")
                else:  # river
                    ori = cell.get('orientation', 'h')
                    if ori == 'horizontal':
                        print(f"{owner_char.lower()}━", end="")
                    else:
                        print(f"{owner_char.lower()}┃", end="")
        print()

def main():
    print("="*70)
    print("MCTS (Main) vs Greedy River Agent - Head to Head")
    print("="*70)

    # Load both agents
    print("\n[1/2] Loading MAIN MCTS agent...")
    sys.path.insert(0, '/Users/yuvikachaudhary/Desktop/Semester 5/COL333/COL333_A5')
    from student_agent_cpp import StudentAgent as MCTSAgent

    print("[2/2] Loading GREEDY RIVER agent...")
    from greedy_river_agent import StudentAgent as GreedyAgent

    print("\n✓ Both agents loaded successfully\n")

    # Play 1 game
    board = create_starting_board()

    # MCTS = circle, Greedy = square
    circle_agent = MCTSAgent("circle")
    square_agent = GreedyAgent("square")

    score_cols = [3, 4, 5, 6, 7, 8]
    circle_time = 30.0
    square_time = 30.0

    max_moves = 50
    move_num = 0

    print("Starting position:")
    print_board_compact(board)
    print(f"\nCircle (MCTS) vs Square (GREEDY RIVER)")
    print(f"Need 4 stones to win\n")

    move_log = []

    while move_num < max_moves:
        move_num += 1
        current_player = "circle" if move_num % 2 == 1 else "square"
        agent = circle_agent if current_player == "circle" else square_agent
        agent_name = "MCTS" if current_player == "circle" else "GREEDY"
        time_left = circle_time if current_player == "circle" else square_time

        print(f"\n--- Move {move_num}: {current_player.upper()} ({agent_name}) ---")

        # Make move
        start = time.time()
        try:
            move = agent.choose(board, 13, 12, score_cols, time_left, time_left)
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            break

        elapsed = time.time() - start

        if current_player == "circle":
            circle_time -= elapsed
        else:
            square_time -= elapsed

        # Print move
        move_str = f"{move['action']}"
        if move.get('from'):
            move_str += f" ({move['from'][0]},{move['from'][1]})"
        if move.get('to'):
            move_str += f"→({move['to'][0]},{move['to'][1]})"
        if move.get('pushed_to'):
            move_str += f"→({move['pushed_to'][0]},{move['pushed_to'][1]})"
        if move.get('orientation'):
            move_str += f" [{move['orientation']}]"

        print(f"{move_str} [{elapsed:.3f}s]")

        move_log.append({
            'move_num': move_num,
            'player': current_player,
            'agent': agent_name,
            'move': move_str,
            'time': elapsed
        })

        # Apply move
        board = apply_move(board, move)

        # Check scoring
        circle_score = count_scoring(board, 'circle')
        square_score = count_scoring(board, 'square')

        if move_num % 4 == 0 or circle_score >= 4 or square_score >= 4:
            print_board_compact(board)
            print(f"Score: Circle={circle_score}/4, Square={square_score}/4")

        if circle_score >= 4:
            print(f"\n{'='*70}")
            print(f"🎉 CIRCLE (MCTS) WINS after {move_num} moves!")
            print(f"{'='*70}")
            break
        if square_score >= 4:
            print(f"\n{'='*70}")
            print(f"🎉 SQUARE (GREEDY) WINS after {move_num} moves!")
            print(f"{'='*70}")
            break

        if circle_time <= 0:
            print(f"\n⏰ Circle ran out of time! Square wins.")
            break
        if square_time <= 0:
            print(f"\n⏰ Square ran out of time! Circle wins.")
            break

    if move_num >= max_moves:
        print(f"\n⏱️  Game reached move limit ({max_moves})")
        print(f"Final: Circle={circle_score}/4, Square={square_score}/4")
        if circle_score > square_score:
            print("Circle (MCTS) wins on points!")
        elif square_score > circle_score:
            print("Square (GREEDY) wins on points!")
        else:
            print("Draw!")

    # Analysis
    print(f"\n{'='*70}")
    print("DETAILED ANALYSIS")
    print(f"{'='*70}")

    # Time analysis
    mcts_times = [m['time'] for m in move_log if m['agent'] == 'MCTS']
    greedy_times = [m['time'] for m in move_log if m['agent'] == 'GREEDY']

    print(f"\nTime per move:")
    if mcts_times:
        print(f"  MCTS:   avg={sum(mcts_times)/len(mcts_times):.3f}s, max={max(mcts_times):.3f}s, total={sum(mcts_times):.1f}s")
    if greedy_times:
        print(f"  GREEDY: avg={sum(greedy_times)/len(greedy_times):.3f}s, max={max(greedy_times):.3f}s, total={sum(greedy_times):.1f}s")

    # Move type analysis
    print(f"\nMove history (last 15):")
    for m in move_log[-15:]:
        print(f"  {m['move_num']:2}. {m['agent']:6} {m['player']:6}: {m['move']}")

    # River usage
    mcts_flips = sum(1 for m in move_log if m['agent'] == 'MCTS' and 'flip' in m['move'])
    greedy_flips = sum(1 for m in move_log if m['agent'] == 'GREEDY' and 'flip' in m['move'])

    print(f"\nRiver usage:")
    print(f"  MCTS flips:   {mcts_flips}")
    print(f"  GREEDY flips: {greedy_flips}")

    # Winner
    print(f"\nResult:")
    if circle_score >= 4:
        print(f"  ✓ MCTS WON (Circle scored {circle_score}/4)")
        print(f"  ❌ Greedy LOST (Square scored {square_score}/4)")
    elif square_score >= 4:
        print(f"  ❌ MCTS LOST (Circle scored {circle_score}/4)")
        print(f"  ✓ Greedy WON (Square scored {square_score}/4)")
    else:
        print(f"  Game incomplete")

    print(f"\n{'='*70}")

if __name__ == "__main__":
    main()
