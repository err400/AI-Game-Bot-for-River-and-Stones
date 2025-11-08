#!/usr/bin/env python3
"""
Script to play against student_cpp AI and analyze its strategy
"""
import subprocess
import re
import time

def play_game():
    """Play a game against student_cpp and provide strategic moves"""

    # Start the game process
    proc = subprocess.Popen(
        ['python3', 'client_server/gameEngine.py', '--mode', 'hvai', '--circle', 'student_cpp', '--nogui', '--time', '5.0'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    turn = 0
    my_moves = []
    ai_moves = []

    print("=" * 60)
    print("GAME ANALYSIS - Playing as Square vs student_cpp (Circle)")
    print("=" * 60)

    try:
        output_buffer = ""
        while True:
            line = proc.stdout.readline()
            if not line:
                break

            print(line, end='')
            output_buffer += line

            # Check if AI made a move
            if "AI circle ->" in line:
                move_match = re.search(r"AI circle -> ({.*})", line)
                if move_match:
                    move = move_match.group(1)
                    ai_moves.append(move)
                    print(f"\n>>> AI MOVE #{len(ai_moves)}: {move}")

            # My turn to move
            if "square move JSON:" in line:
                turn += 1

                # Analyze board state and make strategic move
                move = decide_move(output_buffer, turn, ai_moves)
                my_moves.append(move)

                print(f"\n>>> MY MOVE #{len(my_moves)}: {move}")
                print(">>> REASONING: {}")

                # Send move
                proc.stdin.write(move + '\n')
                proc.stdin.flush()

                # Clear buffer for next turn
                output_buffer = ""

            # Press enter to continue
            if "Press Enter to continue..." in line:
                proc.stdin.write('\n')
                proc.stdin.flush()

            # Check for game end
            if "WINNER:" in line or "Game ended in a draw" in line:
                # Wait for final output
                time.sleep(1)
                break

    except KeyboardInterrupt:
        print("\n\nGame interrupted!")
    finally:
        proc.terminate()
        proc.wait()

    # Analyze the game
    print("\n\n" + "=" * 60)
    print("GAME ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Total turns: {turn}")
    print(f"AI moves made: {len(ai_moves)}")
    print(f"My moves made: {len(my_moves)}")

    return ai_moves, my_moves


def decide_move(board_output, turn, ai_moves):
    """
    Decide what move to make based on current board state and game analysis

    Strategy priorities:
    1. Early game (turns 1-10): Build rivers to create scoring paths
    2. Mid game (turns 11-30): Mix of building rivers and moving stones toward scoring
    3. Late game (turns 31+): Aggressively move stones into scoring zone
    """

    # Parse board to understand current state
    # For now, use simple heuristic-based moves

    if turn <= 3:
        # Early game: Create vertical river to enable scoring path
        # Square scores at row 10, so we want to build path from row 3/4 down to row 10
        if turn == 1:
            # Build first vertical river at column 6 (middle of our formation)
            return '{"action":"flip","from":[5,6],"orientation":"vertical"}'
        elif turn == 2:
            # Build another vertical river
            return '{"action":"flip","from":[7,5],"orientation":"vertical"}'
        else:
            # Start moving stones toward scoring zone
            return '{"action":"move","from":[3,5],"to":[5,5]}'

    elif turn <= 10:
        # Mid-early game: Mix rivers and strategic stone positioning
        if turn % 2 == 0:
            # Build more rivers
            return '{"action":"flip","from":[6,6],"orientation":"vertical"}'
        else:
            # Move stones closer to scoring
            return '{"action":"move","from":[4,5],"to":[6,5]}'

    elif turn <= 20:
        # Mid game: Focus on moving stones into scoring zone
        # Try to move stones from row 3-4 toward row 10
        if turn % 3 == 0:
            # Occasionally build horizontal rivers for cross-connections
            return '{"action":"flip","from":[8,5],"orientation":"horizontal"}'
        else:
            # Aggressively move stones toward row 10
            return '{"action":"move","from":[4,6],"to":[8,6]}'

    else:
        # Late game: All-in on scoring
        # Move stones into row 10 scoring zone
        return '{"action":"move","from":[8,5],"to":[10,5]}'


if __name__ == "__main__":
    ai_moves, my_moves = play_game()

    print("\n\nAI STRATEGY ANALYSIS:")
    print("-" * 60)
    print("AI moves:", ai_moves[:10])  # Show first 10 moves
    print("\nObservations:")
    print("1. What type of moves does AI make early?")
    print("2. Does AI prioritize rivers or stone movement?")
    print("3. Does AI react to my moves or follow fixed strategy?")
