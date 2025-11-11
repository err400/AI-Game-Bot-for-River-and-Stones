cmake_minimum_required(VERSION 3.15)
project(greedy_river_agent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 REQUIRED)

pybind11_add_module(greedy_river_module greedy_river_agent.cpp)

target_compile_options(greedy_river_module PRIVATE
    $<$<CONFIG:Release>:-O3>
)
