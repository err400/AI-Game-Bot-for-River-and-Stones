# Stones and Rivers Game

This repository contains the code supporting the code for Agent and Engine for the River and Stones.

## Details
This is a course assignment for the graduate-level Artificial Intelligence course taught by [**Prof. Mausam**](https://www.cse.iitd.ac.in/~mausam/). The assignment documentation can be found on the Course website.

## Rules
You can find the documentation to get all the rules of the game.

## Dependencies
- Python 3.9
- Pygame
- Numpy 
- Scipy
- flask
- python-socketio


## Setting up the Environment
```sh
conda create --name stones_river python=3.9
conda activate stones_river
pip install -r requirements.txt
```

## Run Instructions
Here are the instructions used to match ai or human players against each other.


## Main Files
- `gameEngine.py`: It is an instance of the game. It can be run locally on your environment. You can run in GUI or CLI mode.
- `agent.py`: It consists of the implementations of the Random Agent. 
- `student_agent.py` : You need to implement your agent here. Some predefined function has been given.
- `web_server.py` : Use it to start the webserver.
- `bot_client.py`: It calls student_agent to get the moves while interacting with the web server. 

Note: Details for running the C++ agent will be shared later. The same game will be used in the second phase in Assigment 5. And seperate details will be shared for the Assigment 5.

### Human vs Human
```sh
python gameEngine.py --mode hvh
```
### Human vs AI

```sh
python gameEngine.py --mode hvai --circle random
```
### AI vs AI

```sh
python gameEngine.py --mode aivai --circle random --square student
```

### No GUI
```sh
python gameEngine.py --mode aivai --circle random --square student --nogui
```

### Create server
```sh
bash start_server.sh 8080
```
Ensure that conda env is set before you start the server. Once server starts, you can navigate to http://localhost:8080" or whatever port you choose. Once to go to webpage, select boardsize and then click start game. Then we move to starting bots using following commands.
#### Starting first bot
```sh
python bot_client.py circle 8080 --strategy student
```
#### Running Second bot
```sh
python bot_client.py square 8080 --strategy student --server 10.10.10.10
```
--server will require the IP of system on which server is running. You can use ipconfig/ifconfig to identify the IP address. Provide that IP and it should connect. 
