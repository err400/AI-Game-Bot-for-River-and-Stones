# River and Stones - Web Server

Multi-player web-based game server for the River and Stones assignment.

## Quick Start

### 1. Start the Server

```bash
bash start_server.sh [port]
```

Default port is 8080.

### 2. Connect Bots

In separate terminals:

```bash
# Circle bot
python bot_client.py circle 8080

# Square bot
python bot_client.py square 8080
```

### 3. View Game

Open browser: `http://localhost:8080`

## Board Sizes

The game supports three board sizes:
- **Small**: 13×12 (12 pieces per player)
- **Medium**: 15×14 (14 pieces per player)  
- **Large**: 17×16 (16 pieces per player)

Configure via command line:
```bash
python gameEngine.py --board-size medium
```

## Dependencies

Install via:
```bash
pip install -r web_requirements.txt
```

Required:
- Flask 3.1.2
- Flask-SocketIO 5.5.1
- requests 2.32.3
- pygame 2.6.1

## Files

- `web_server.py` - Flask web server with game coordination
- `bot_client.py` - Bot client for connecting AI agents
- `gameEngine.py` - Core game logic and rules
- `agent.py` - Agent interface
- `student_agent.py` - Student-implemented strategy
- `templates/index.html` - Web interface
- `start_server.sh` - Server startup script
- `web_requirements.txt` - Python dependencies

## Bot Strategies

```bash
# Random strategy (default)
python bot_client.py circle 8080 --strategy random

# Student strategy
python bot_client.py square 8080 --strategy student
```

## Troubleshooting

**Port already in use:**
```bash
lsof -i :8080 | grep LISTEN | awk '{print $2}' | xargs kill -9
```


