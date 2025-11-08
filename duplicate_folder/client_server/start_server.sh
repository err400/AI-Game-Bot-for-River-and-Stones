#!/bin/bash
# Web Server Startup Script for River and Stones Game
# Usage: ./start_server.sh [port]
# Default port: 8080

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Set default port
PORT=${1:-8080}



echo -e "${BLUE}🎮 River and Stones Web Server Startup${NC}"
echo -e "${BLUE}=====================================${NC}"
echo -e "${GREEN}Directory: $(pwd)${NC}"
echo ""

# Initialize conda for bash shell
eval "$(conda shell.bash hook)"


# Check if required packages are installed
echo -e "${YELLOW}📦 Checking dependencies...${NC}"
python -c "import flask, flask_socketio, requests" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}⚠️  Missing dependencies. Installing...${NC}"
    pip install -r web_requirements.txt
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Failed to install dependencies${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Dependencies installed${NC}"
fi

echo -e "${GREEN}✓ All dependencies satisfied${NC}"
echo ""
echo -e "${GREEN}Starting River and Stones Web Server on port $PORT${NC}"
echo -e "${GREEN}Access the game at: http://localhost:$PORT${NC}"
echo ""
echo -e "${YELLOW}To connect bots:${NC}"
echo -e "  Circle bot: ${BLUE}eval \"\$(conda shell.bash hook)\" && conda activate col333_TA && python bot_client.py circle $PORT${NC}"
echo -e "  Square bot: ${BLUE}eval \"\$(conda shell.bash hook)\" && conda activate col333_TA && python bot_client.py square $PORT${NC}"
echo ""

# Start the web server
echo -e "${GREEN}Starting web server...${NC}"
python web_server.py $PORT
