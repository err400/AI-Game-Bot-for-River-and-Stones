#!/usr/bin/env bash
set -euo pipefail

if command -v g++ >/dev/null 2>&1; then
  CXX=g++
else
  echo "g++ not found" >&2
  exit 1
fi

make CXX="$CXX" all

echo "Built: build/libstudent_agent.so"

