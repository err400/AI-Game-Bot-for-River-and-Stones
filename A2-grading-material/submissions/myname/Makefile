##############################################
# Makefile: Build student_agent_module and run
# Supports both CMake-based and direct (pybind) builds
##############################################

# Compilers and tools
PY ?= python3
CMAKE ?= cmake
CC ?= gcc
CXX ?= g++

# Direct (pybind) build flags
CXXFLAGS ?= -std=c++17 -O3 -fPIC -Wall -Wextra
LDFLAGS ?= -shared
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  # Let unresolved Python symbols bind at runtime on macOS
  LDFLAGS += -undefined dynamic_lookup
endif

# Paths
BUILD_DIR := build
SRC := student_agent.cpp

.PHONY: all clean \
        pybind lib \
        cmake-config cmake-build cmake-rebuild \
        run run-aivai run-hvai run-hvh \
        run-aivai-gui run-hvai-gui run-hvh-gui \
        run-hvai-circle run-hvai-circle-gui \
        run-hvai-square run-hvai-square-gui \
        rebuild-run rebuild-run-aivai rebuild-run-hvai rebuild-run-hvh

##############################################
# Default: CMake rebuild
##############################################
all: cmake-rebuild

clean:
	rm -rf $(BUILD_DIR)

##############################################
# Direct pybind build (no CMake)
##############################################
# Build Python extension into build/student_agent_module<ext>
pybind: $(BUILD_DIR)/student_agent_module$(shell $(PY) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')")

$(BUILD_DIR)/student_agent_module%: $(SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $$($(PY) -m pybind11 --includes) -o $@ $< $(LDFLAGS)
	@echo "Built: $@"

# Optional legacy plain .so
lib: $(BUILD_DIR)/libstudent_agent.so

$(BUILD_DIR)/libstudent_agent.so: $(SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

##############################################
# CMake-based build (uses CMakeLists.txt)
##############################################
PYBIND11_DIR := $(shell $(PY) -m pybind11 --cmakedir 2>/dev/null)

cmake-config:
	rm -rf $(BUILD_DIR)
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(CMAKE) .. -Dpybind11_DIR="$(PYBIND11_DIR)" -DPYBIND11_FINDPYTHON=ON -DCMAKE_C_COMPILER="$(CC)" -DCMAKE_CXX_COMPILER="$(CXX)" -DCMAKE_BUILD_TYPE=Release

cmake-build: $(BUILD_DIR)/CMakeCache.txt
	$(CMAKE) --build $(BUILD_DIR) --config Release -j

$(BUILD_DIR)/CMakeCache.txt:
	$(MAKE) cmake-config

cmake-rebuild: cmake-config cmake-build
	@echo "Built CMake module in $(BUILD_DIR)/"

##############################################
# Run helpers (Game Engine)
##############################################
# Defaults (override via: make run MODE=hvai CIRCLE=student_cpp SQUARE=random)
MODE ?= aivai
CIRCLE ?= random
SQUARE ?= student_cpp
BOARD_SIZE ?= small
TIME_MIN ?= 1.0
NOGUI ?= 1

define RUN_ENGINE
PYTHONPATH=. $(PY) client_server/gameEngine.py \
  --mode $(MODE) \
  --circle $(CIRCLE) \
  --square $(SQUARE) \
  --board-size $(BOARD_SIZE) \
  --time $(TIME_MIN) \
  $(if $(filter 1,$(NOGUI)),--nogui,)
endef

run: cmake-rebuild
	$(RUN_ENGINE)

run-aivai:
	$(MAKE) run MODE=aivai

run-hvai:
	$(MAKE) run MODE=hvai

run-hvh:
	$(MAKE) run MODE=hvh

# GUI variants (set NOGUI=0)
run-aivai-gui:
	$(MAKE) run-aivai NOGUI=0

run-hvai-gui:
	$(MAKE) run-hvai NOGUI=0

run-hvh-gui:
	$(MAKE) run-hvh NOGUI=0

# HVAI convenience: choose which side is AI based on engine's rule
# Note: In this engine, HVAI assigns AI to circle if CIRCLE==random, otherwise to square.
run-hvai-circle:
	$(MAKE) run MODE=hvai CIRCLE=random SQUARE=student_cpp

run-hvai-circle-gui:
	$(MAKE) run-hvai-circle NOGUI=0

run-hvai-square:
	$(MAKE) run MODE=hvai CIRCLE=student_cpp SQUARE=random

run-hvai-square-gui:
	$(MAKE) run-hvai-square NOGUI=0

# One-shot: rebuild with CMake then run
rebuild-run: cmake-rebuild run
rebuild-run-aivai: cmake-rebuild run-aivai
rebuild-run-hvai: cmake-rebuild run-hvai
rebuild-run-hvh: cmake-rebuild run-hvh

##############################################
# Greedy River Agent build (direct pybind)
##############################################
.PHONY: greedy-build greedy-test-import greedy-clean

# Build Python extension into build_greedy/greedy_river_module<ext>
GREEDY_BUILD_DIR := build_greedy
GREEDY_SRC := greedy_river_agent.cpp

greedy-build: $(GREEDY_BUILD_DIR)/greedy_river_module$(shell $(PY) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')")

$(GREEDY_BUILD_DIR)/greedy_river_module%: $(GREEDY_SRC) | $(GREEDY_BUILD_DIR)
	$(CXX) $(CXXFLAGS) $$($(PY) -m pybind11 --includes) -o $@ $< $(LDFLAGS)
	@echo "Built: $@"

$(GREEDY_BUILD_DIR):
	mkdir -p $(GREEDY_BUILD_DIR)

greedy-test-import: greedy-build
	@$(PY) -c "import sys,importlib; sys.path.insert(0, '$(GREEDY_BUILD_DIR)'); m=importlib.import_module('greedy_river_module'); print('Greedy module OK:', m)"

greedy-clean:
	rm -rf $(GREEDY_BUILD_DIR)

##############################################
# Web server + Greedy bot helpers
##############################################
.PHONY: web-server bot-greedy-circle bot-greedy-square

PORT ?= 8080
HOST ?= localhost

web-server:
	PYTHONPATH=. $(PY) client_server/web_server.py $(PORT)

bot-greedy-circle: greedy-build
	PYTHONPATH=. $(PY) client_server/bot_client.py circle $(PORT) --server $(HOST) --strategy greedy_river

bot-greedy-square: greedy-build
	PYTHONPATH=. $(PY) client_server/bot_client.py square $(PORT) --server $(HOST) --strategy greedy_river
