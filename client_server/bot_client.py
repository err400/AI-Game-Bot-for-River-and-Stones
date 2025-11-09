#!/usr/bin/env python3
"""
Bot Client for River and Stones Game
Connects to the web server and plays the game using AI agents
"""

import requests
import time
import json
import sys
import os
import argparse
from typing import Dict, Any, Optional, List, Tuple
import uuid
from datetime import datetime
import pathlib
import logging

# Add parent directory to path for student_agent_cpp import
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

# Import game components
from agent import get_agent
from gameEngine import Piece

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BotClient:
    """Client that connects a bot to the game server"""
    
    def __init__(self, player: str, server_url: str = "http://localhost:8080", strategy: str = "random"):
        """
        Initialize bot client
        
        Args:
            player: "circle" or "square"
            server_url: URL of the game server
            strategy: Bot strategy ("random", "student", etc.)
        """
        self.player = player
        self.server_url = server_url.rstrip('/')
        self.strategy = strategy
        self.agent = get_agent(player, strategy)
        self.connected = False
        self.game_active = False
        
        logger.info(f"Initialized {strategy} bot for player {player}")
    
    def connect(self) -> bool:
        """Connect to the game server"""
        try:
            # Send connection request
            response = requests.post(
                f"{self.server_url}/bot/connect/{self.player}",
                json={
                    "name": f"{self.strategy}_{self.player}",
                    "strategy": self.strategy,
                    "version": "1.0"
                },
                timeout=10
            )
            
            if response.status_code == 200:
                self.connected = True
                logger.info(f"Successfully connected as {self.player}")
                return True
            else:
                logger.error(f"Failed to connect: {response.json()}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Connection error: {e}")
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from the game server"""
        try:
            response = requests.post(
                f"{self.server_url}/bot/disconnect/{self.player}",
                timeout=10
            )
            
            if response.status_code == 200:
                self.connected = False
                logger.info(f"Disconnected {self.player}")
                return True
            else:
                logger.error(f"Failed to disconnect: {response.json()}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Disconnect error: {e}")
            return False
    
    def get_game_state(self) -> Optional[Dict[str, Any]]:
        """Get current game state from server"""
        try:
            response = requests.get(
                f"{self.server_url}/bot/game_state/{self.player}",
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get game state: {response.json()}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting game state: {e}")
            return None
    
    def make_move(self, move: Dict[str, Any]) -> bool:
        """Send move to server"""
        try:
            response = requests.post(
                f"{self.server_url}/bot/move/{self.player}",
                json={"move": move},
                timeout=30
            )
            
            result = response.json()
            
            if result.get("success"):
                logger.info(f"Move successful: {move}")
                if result.get("winner"):
                    logger.info(f"Game finished! Winner: {result['winner']}")
                return True
            else:
                logger.error(f"Move failed: {result.get('error', 'Unknown error')}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making move: {e}")
            return False
    
    def convert_board_format(self, bot_board: list) -> list:
        """Convert board from bot format to game engine format"""
        converted_board = []
        for row in bot_board:
            converted_row = []
            for cell in row:
                if cell is None:
                    converted_row.append(None)
                else:
                    # Convert dict to Piece object
                    piece = Piece(
                        owner=cell["owner"],
                        side=cell.get("side", "stone"),
                        orientation=cell.get("orientation", "horizontal")
                    )
                    converted_row.append(piece)
            converted_board.append(converted_row)
        return converted_board
    
    def _feature_vector(self, board: list, rows: int, cols: int, score_cols: List[int], side: str) -> List[float]:
        """Compute 16-D feature vector mirroring C++ StateFeatures."""
        from gameEngine import opponent, count_scoring_pieces, count_reachable_in_one

        def top_score_row() -> int:
            return 2

        def bottom_score_row() -> int:
            return rows - 3

        def mdist_to_own_score(owner: str, r: int, c: int) -> int:
            target_row = top_score_row() if owner == "circle" else bottom_score_row()
            row_d = abs(r - target_row)
            col_d = min((abs(c - sc) for sc in score_cols), default=0)
            return row_d + col_d

        def min_avg_dist(owner: str) -> Tuple[float, float]:
            mn = 1e9
            tot = 0.0
            cnt = 0
            for y in range(rows):
                for x in range(cols):
                    p = board[y][x]
                    if p and getattr(p, 'side', '') == 'stone' and getattr(p, 'owner', '') == owner:
                        d = mdist_to_own_score(owner, y, x)
                        mn = min(mn, float(d))
                        tot += d
                        cnt += 1
            if cnt == 0:
                return 0.0, 0.0
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

        def center_pressure(owner: str) -> float:
            start = max(0, cols//2 - 1)
            endc = min(cols-1, start+1)
            own = 0
            ownTot = 0
            for y in range(rows):
                for x in range(cols):
                    p = board[y][x]
                    if p and getattr(p, 'owner', '') == owner:
                        ownTot += 1
                        if start <= x <= endc:
                            own += 1
            return 0.0 if ownTot == 0 else float(own)/float(ownTot)

        def threat_flag(owner: str) -> int:
            need = 4 if cols <= 12 else (5 if cols <= 14 else 6)
            n = float(count_scoring_pieces(board, owner, rows, cols, score_cols))
            if n >= need - 1:
                return 1
            m = float(count_reachable_in_one(board, owner, rows, cols, score_cols))
            return 1 if m > 0 else 0

        opp = 'square' if side == 'circle' else 'circle'
        from gameEngine import count_scoring_pieces, count_reachable_in_one
        n_self = float(count_scoring_pieces(board, side, rows, cols, score_cols))
        n_opp  = float(count_scoring_pieces(board, opp,  rows, cols, score_cols))
        m_self = float(count_reachable_in_one(board, side, rows, cols, score_cols))
        m_opp  = float(count_reachable_in_one(board, opp,  rows, cols, score_cols))
        md_s_min, md_s_avg = min_avg_dist(side)
        md_o_min, md_o_avg = min_avg_dist(opp)
        r_self = float(count_rivers(side))
        r_opp  = float(count_rivers(opp))
        mob_s  = float(approx_mob(side))
        mob_o  = float(approx_mob(opp))
        cp_s   = float(center_pressure(side))
        cp_o   = float(center_pressure(opp))
        th_s   = float(threat_flag(side))
        th_o   = float(threat_flag(opp))

        return [
            n_self, n_opp,
            m_self, m_opp,
            md_s_min, md_o_min,
            md_s_avg, md_o_avg,
            r_self, r_opp,
            mob_s, mob_o,
            cp_s, cp_o,
            th_s, th_o,
        ]

    def play_game(self):
        """Main game loop"""
        logger.info(f"Starting game loop for {self.player} using {self.strategy} strategy")
        
        # Connect to server
        if not self.connect():
            logger.error("Failed to connect to server")
            return
        
        try:
            # Prepare logging (optional). Set RS_LOG_VALUE=1 to enable.
            enable_log = os.environ.get('RS_LOG_VALUE', '0') == '1'
            log_path = os.environ.get('RS_LOG_PATH', os.path.join(os.path.dirname(_PARENT_DIR), 'logs', 'value_data.jsonl'))
            pending_rows: List[Dict[str, Any]] = []
            game_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            last_board_pieces = None

            # Wait for game to start
            logger.info("Waiting for game to start...")
            while self.connected:
                game_state = self.get_game_state()
                
                if not game_state:
                    logger.warning("No game state received, waiting...")
                    time.sleep(2)
                    continue
                
                if game_state["game_status"] == "waiting":
                    logger.info("Waiting for opponent to connect...")
                    time.sleep(2)
                    continue
                
                if game_state["game_status"] == "finished":
                    logger.info("Game finished!")
                    # Fetch final board for labeling (use last seen if needed)
                    final_board = last_board_pieces if last_board_pieces is not None else self.convert_board_format(game_state["board"])
                    if enable_log and pending_rows:
                        try:
                            # Compute final tournament scores
                            from gameEngine import compute_final_scores
                            scores = compute_final_scores(final_board, None, game_state["rows"], game_state["cols"], game_state["score_cols"])
                            # Assign labels and flush
                            os.makedirs(os.path.dirname(log_path), exist_ok=True)
                            with open(log_path, 'a') as f:
                                for row in pending_rows:
                                    row['final_score_from_side'] = float(scores.get(row['side'], 0.0))
                                    f.write(json.dumps(row) + "\n")
                            logger.info(f"Wrote {len(pending_rows)} rows to {log_path}")
                        except Exception as e:
                            logger.error(f"Failed to write training rows: {e}")
                        finally:
                            pending_rows.clear()
                    break
                
                if not game_state["your_turn"]:
                    logger.debug("Waiting for opponent's turn...")
                    time.sleep(1)
                    continue
                
                # It's our turn!
                logger.info(f"My turn! Time left: {game_state['time_left']:.1f}s")
                
                # Convert board format for agent (and store for logging/labels)
                converted_board = self.convert_board_format(game_state["board"])
                last_board_pieces = converted_board
                
                # Optionally log features (one row per our move)
                if enable_log:
                    feats = self._feature_vector(converted_board, game_state["rows"], game_state["cols"], game_state["score_cols"], self.player)
                    try:
                        # Log raw serializable board from server so we can run teacher evaluation later
                        serial_board = game_state["board"]
                    except Exception:
                        # Fallback: convert back to dicts
                        serial_board = [[cell.to_dict() if cell else None for cell in row] for row in converted_board]
                    pending_rows.append({
                        'features': feats,
                        'rows': int(game_state['rows']),
                        'cols': int(game_state['cols']),
                        'score_cols': list(map(int, game_state['score_cols'])),
                        'side': self.player,
                        'board': serial_board,
                        'meta': {
                            'game_id': game_id,
                            'turn': int(game_state.get('turn', 0)),
                            'strategy': self.strategy,
                        }
                    })

                # Get move from agent
                start_time = time.time()
                move = self.agent.choose(
                    converted_board,
                    game_state["rows"],
                    game_state["cols"],
                    game_state["score_cols"],
                    game_state["time_left"],
                    game_state["opponent_time"]
                )
                thinking_time = time.time() - start_time
                
                if move is None:
                    logger.warning("Agent returned no move")
                    time.sleep(1)
                    continue
                
                logger.info(f"Agent chose move (thinking time: {thinking_time:.2f}s): {move}")
                
                # Send move to server
                if self.make_move(move):
                    logger.info("Move sent successfully")
                else:
                    logger.error("Failed to send move")
                
                # Small delay before next iteration
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            logger.info("Bot interrupted by user")
        except Exception as e:
            logger.error(f"Unexpected error in game loop: {e}")
        finally:
            # Disconnect from server
            self.disconnect()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Bot client for River and Stones game")
    parser.add_argument("player", choices=["circle", "square"], help="Player side")
    parser.add_argument("port", type=int, help="Server port (8181 for circle, 8182 for square)")
    parser.add_argument("--server", default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--strategy", default="random", choices=["random", "student", "student_cpp", "greedy_river"],
                       help="Bot strategy (default: random)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Construct server URL
    server_url = f"http://{args.server}:{args.port}"
    
    # Validate port assignment
    expected_port = 8181 if args.player == "circle" else 8182
    if args.port != expected_port:
        logger.warning(f"Warning: {args.player} player typically uses port {expected_port}, but using {args.port}")
    
    # Create and run bot
    bot = BotClient(args.player, server_url, args.strategy)
    
    print(f"""
🤖 Starting {args.strategy} bot for {args.player} player
Server: {server_url}
Strategy: {args.strategy}

Press Ctrl+C to stop the bot.
""")
    
    bot.play_game()

if __name__ == "__main__":
    main()
