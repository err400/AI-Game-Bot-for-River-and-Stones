CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -fPIC
LDFLAGS ?= -shared

TARGET_DIR := build
TARGET := $(TARGET_DIR)/libstudent_agent.so
SRC := student_agent.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC) | $(TARGET_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

$(TARGET_DIR):
	mkdir -p $(TARGET_DIR)

clean:
	rm -rf $(TARGET_DIR)

