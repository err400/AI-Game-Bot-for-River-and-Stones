"""
Web Server for River and Stones Game
Serves web GUI and handles bot connections
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import threading
from datetime import datetime

try:
    from flask import Flask, render_template, request, jsonify
    from flask_socketio import SocketIO, emit
except ImportError:
    print("Flask and Flask-SocketIO required. Install with: pip install flask flask-socketio")
    exit(1)

from gameEngine import (
    default_start_board, score_cols_for, get_win_count, 
    validate_and_apply_move, check_win, board_to_ascii,
    compute_final_scores, Piece
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GameCoordinator:
    """Manages game state and coordinates between web GUI and bot players"""
    
    def __init__(self):
        self.games: Dict[str, Dict[str, Any]] = {}  # game_id -> game_state
        self.current_game_id = "main"
        self.bot_connections = {}  # port -> bot_info
        self.socketio = None
        
    def create_game(self, board_size: str = "small") -> str:
        """Create a new game with specified board size"""
        game_id = f"game_{int(time.time())}"
        
        # Determine board dimensions
        if board_size == "small":
            rows, cols = 13, 12
        elif board_size == "medium":
            rows, cols = 15, 14
        elif board_size == "large":
            rows, cols = 17, 16
        else:
            rows, cols = 13, 12
            
        # Initialize game state
        board = default_start_board(rows, cols)
        score_cols = score_cols_for(cols)
        win_count = get_win_count(cols)
        
        game_state = {
            "id": game_id,
            "board_size": board_size,
            "rows": rows,
            "cols": cols,
            "board": board,
            "score_cols": score_cols,
            "win_count": win_count,
            "current_player": "circle",
            "players": {
                "circle": {"connected": False, "port": 8181, "time_left": 60.0},
                "square": {"connected": False, "port": 8182, "time_left": 60.0}
            },
            "game_status": "waiting",  # waiting, active, finished
            "winner": None,
            "turn_count": 0,
            "last_move_time": time.time(),
            "game_log": []
        }
        
        self.games[game_id] = game_state
        self.current_game_id = game_id
        logger.info(f"Created new game {game_id} with board size {board_size}")
        return game_id
    
    def get_game(self, game_id: str = None) -> Optional[Dict[str, Any]]:
        """Get game state by ID"""
        if game_id is None:
            game_id = self.current_game_id
        return self.games.get(game_id)
    
    def connect_bot(self, player: str, port: int, bot_info: Dict[str, Any]) -> bool:
        """Connect a bot player to the game"""
        game = self.get_game()
        if not game:
            return False
            
        if player not in ["circle", "square"]:
            return False
            
        # Update player connection status
        game["players"][player]["connected"] = True
        game["players"][player]["bot_info"] = bot_info
        self.bot_connections[port] = {"player": player, "game_id": game["id"], "info": bot_info}
        
        logger.info(f"Bot connected: {player} on port {port}")
        
        # Check if both players are connected
        if all(p["connected"] for p in game["players"].values()):
            game["game_status"] = "active"
            game["last_move_time"] = time.time()
            logger.info(f"Game {game['id']} started - both players connected")
            self._broadcast_game_update(game)
            
        return True
    
    def disconnect_bot(self, port: int) -> bool:
        """Disconnect a bot player"""
        if port not in self.bot_connections:
            return False
            
        bot_info = self.bot_connections[port]
        game = self.get_game(bot_info["game_id"])
        
        if game:
            player = bot_info["player"]
            game["players"][player]["connected"] = False
            if game["game_status"] == "active":
                game["game_status"] = "paused"
                logger.info(f"Game paused - {player} disconnected")
        
        del self.bot_connections[port]
        logger.info(f"Bot disconnected from port {port}")
        return True
    
    def make_move(self, player: str, move: Dict[str, Any]) -> Dict[str, Any]:
        """Process a move from a bot player"""
        game = self.get_game()
        if not game:
            return {"success": False, "error": "No active game"}
            
        if game["game_status"] != "active":
            return {"success": False, "error": "Game not active"}
            
        if game["current_player"] != player:
            return {"success": False, "error": "Not your turn"}
        
        # Calculate elapsed time for current player
        elapsed_time = time.time() - game["last_move_time"]
        game["players"][player]["time_left"] -= elapsed_time
        
        # Check for timeout
        if game["players"][player]["time_left"] <= 0:
            winner = "square" if player == "circle" else "circle"
            game["winner"] = winner
            game["game_status"] = "finished"
            result = {"success": True, "timeout": True, "winner": winner}
            self._broadcast_game_update(game)
            return result
        
        # Validate and apply move
        success, message = validate_and_apply_move(
            game["board"], move, player, 
            game["rows"], game["cols"], game["score_cols"]
        )
        
        if success:
            # Log the move
            game["game_log"].append({
                "turn": game["turn_count"] + 1,
                "player": player,
                "move": move,
                "time": datetime.now().isoformat(),
                "time_left": game["players"][player]["time_left"]
            })
            
            # Check for win condition
            winner = check_win(game["board"], game["rows"], game["cols"], game["score_cols"])
            
            if winner:
                game["winner"] = winner
                game["game_status"] = "finished"
                # Calculate final scores
                final_scores = compute_final_scores(
                    game["board"], winner, game["rows"], game["cols"], game["score_cols"],
                    remaining_times={
                        "circle": game["players"]["circle"]["time_left"],
                        "square": game["players"]["square"]["time_left"]
                    }
                )
                game["final_scores"] = final_scores
                logger.info(f"Game finished! Winner: {winner}")
            else:
                # Switch to next player
                game["current_player"] = "square" if player == "circle" else "circle"
                game["turn_count"] += 1
                game["last_move_time"] = time.time()
            
            # Broadcast update to web clients
            self._broadcast_game_update(game)
            
            return {"success": True, "message": message, "winner": winner}
        else:
            return {"success": False, "error": message}
    
    def _broadcast_game_update(self, game: Dict[str, Any]):
        """Broadcast game state update to web clients"""
        if self.socketio:
            # Convert board to serializable format
            serialized_board = []
            for row in game["board"]:
                serialized_row = []
                for cell in row:
                    if cell:
                        serialized_row.append(cell.to_dict())
                    else:
                        serialized_row.append(None)
                serialized_board.append(serialized_row)
            
            update_data = {
                "board": serialized_board,
                "current_player": game["current_player"],
                "game_status": game["game_status"],
                "turn_count": game["turn_count"],
                "players": game["players"],
                "winner": game.get("winner"),
                "final_scores": game.get("final_scores"),
                "board_size": game["board_size"],
                "rows": game["rows"],
                "cols": game["cols"],
                "score_cols": game["score_cols"],
                "win_count": game["win_count"]
            }
            
            self.socketio.emit('game_update', update_data)
    
    def get_game_state_for_bot(self, player: str) -> Optional[Dict[str, Any]]:
        """Get game state formatted for bot consumption"""
        game = self.get_game()
        if not game:
            return None
            
        # Convert board to bot format
        bot_board = []
        for row in game["board"]:
            bot_row = []
            for cell in row:
                if cell:
                    bot_row.append(cell.to_dict())
                else:
                    bot_row.append(None)
            bot_board.append(bot_row)
        
        return {
            "board": bot_board,
            "rows": game["rows"],
            "cols": game["cols"],
            "score_cols": game["score_cols"],
            "current_player": game["current_player"],
            "your_turn": game["current_player"] == player,
            "time_left": game["players"][player]["time_left"],
            "opponent_time": game["players"]["square" if player == "circle" else "circle"]["time_left"],
            "game_status": game["game_status"],
            "turn_count": game["turn_count"]
        }

# Initialize game coordinator
coordinator = GameCoordinator()

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'river_stones_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")
coordinator.socketio = socketio

@app.route('/')
def index():
    """Serve the main game interface"""
    return render_template('index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return app.send_static_file(filename)

@app.route('/api/create_game', methods=['POST'])
def create_game():
    """API endpoint to create a new game"""
    data = request.get_json() or {}
    board_size = data.get('board_size', 'small')
    
    if board_size not in ['small', 'medium', 'large']:
        return jsonify({"error": "Invalid board size"}), 400
    
    game_id = coordinator.create_game(board_size)
    game = coordinator.get_game(game_id)
    
    return jsonify({
        "success": True,
        "game_id": game_id,
        "board_size": board_size,
        "rows": game["rows"],
        "cols": game["cols"],
        "ports": {
            "circle": 8181,
            "square": 8182
        }
    })

@app.route('/api/game_state')
def get_game_state():
    """Get current game state for web interface"""
    game = coordinator.get_game()
    if not game:
        return jsonify({"error": "No active game"}), 404
    
    # Convert board for web display
    serialized_board = []
    for row in game["board"]:
        serialized_row = []
        for cell in row:
            if cell:
                serialized_row.append(cell.to_dict())
            else:
                serialized_row.append(None)
        serialized_board.append(serialized_row)
    
    return jsonify({
        "id": game["id"],
        "board": serialized_board,
        "board_size": game["board_size"],
        "rows": game["rows"],
        "cols": game["cols"],
        "score_cols": game["score_cols"],
        "win_count": game["win_count"],
        "current_player": game["current_player"],
        "players": game["players"],
        "game_status": game["game_status"],
        "turn_count": game["turn_count"],
        "winner": game.get("winner"),
        "final_scores": game.get("final_scores")
    })

# Bot connection endpoints
@app.route('/bot/connect/<player>', methods=['POST'])
def connect_bot(player):
    """Endpoint for bots to connect"""
    data = request.get_json() or {}
    bot_info = {
        "name": data.get("name", f"{player}_bot"),
        "strategy": data.get("strategy", "random"),
        "version": data.get("version", "1.0")
    }
    
    # Create game if it doesn't exist
    if not coordinator.get_game():
        board_size = data.get("board_size", "small")
        coordinator.create_game(board_size)
        logger.info(f"Auto-created game with board size: {board_size}")
    
    port = 8181 if player == "circle" else 8182
    success = coordinator.connect_bot(player, port, bot_info)
    
    if success:
        return jsonify({"success": True, "message": f"Bot connected as {player}"})
    else:
        return jsonify({"error": "Failed to connect bot"}), 400

@app.route('/bot/disconnect/<player>', methods=['POST'])
def disconnect_bot(player):
    """Endpoint for bots to disconnect"""
    port = 8181 if player == "circle" else 8182
    success = coordinator.disconnect_bot(port)
    
    if success:
        return jsonify({"success": True, "message": f"Bot disconnected"})
    else:
        return jsonify({"error": "Bot not found"}), 404

@app.route('/bot/game_state/<player>')
def get_bot_game_state(player):
    """Get game state for bot player"""
    game_state = coordinator.get_game_state_for_bot(player)
    if game_state:
        return jsonify(game_state)
    else:
        return jsonify({"error": "No active game"}), 404

@app.route('/bot/move/<player>', methods=['POST'])
def bot_make_move(player):
    """Endpoint for bots to make moves"""
    data = request.get_json()
    if not data or 'move' not in data:
        return jsonify({"error": "No move provided"}), 400
    
    result = coordinator.make_move(player, data['move'])
    return jsonify(result)

# WebSocket events
@socketio.on('connect')
def on_connect():
    """Handle web client connection"""
    logger.info("Web client connected")
    game = coordinator.get_game()
    if game:
        coordinator._broadcast_game_update(game)

@socketio.on('disconnect')
def on_disconnect():
    """Handle web client disconnection"""
    logger.info("Web client disconnected")

@socketio.on('create_game')
def on_create_game(data):
    """Handle game creation from web client"""
    board_size = data.get('board_size', 'small')
    game_id = coordinator.create_game(board_size)
    game = coordinator.get_game(game_id)
    coordinator._broadcast_game_update(game)
    emit('game_created', {"game_id": game_id, "board_size": board_size})

if __name__ == '__main__':
    import sys
    
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    
    print(f"""
🎮 River and Stones Web Server Starting...

Web Interface: http://localhost:{port}
Bot Connections:
  - Circle Player: http://localhost:{port}/bot/connect/circle
  - Square Player: http://localhost:{port}/bot/connect/square

Bot API Endpoints:
  - Connect: POST /bot/connect/<player>
  - Game State: GET /bot/game_state/<player>
  - Make Move: POST /bot/move/<player>
  - Disconnect: POST /bot/disconnect/<player>
    """)
    
    socketio.run(app, host='0.0.0.0', port=port, debug=True, allow_unsafe_werkzeug=True)