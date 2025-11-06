#!/usr/bin/env python3
"""
Bot Client for River and Stones Game
Connects to the web server and plays the game using AI agents
"""

import requests
import time
import json
import sys
import argparse
from typing import Dict, Any, Optional
import logging

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
    
    def play_game(self):
        """Main game loop"""
        logger.info(f"Starting game loop for {self.player} using {self.strategy} strategy")
        
        # Connect to server
        if not self.connect():
            logger.error("Failed to connect to server")
            return
        
        try:
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
                    break
                
                if not game_state["your_turn"]:
                    logger.debug("Waiting for opponent's turn...")
                    time.sleep(1)
                    continue
                
                # It's our turn!
                logger.info(f"My turn! Time left: {game_state['time_left']:.1f}s")
                
                # Convert board format for agent
                converted_board = self.convert_board_format(game_state["board"])
                
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
    parser.add_argument("--strategy", default="random", choices=["random", "student", "student_cpp"],
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