cmake_minimum_required(VERSION 3.10)

project(QTSynth)

find_package(fmt CONFIG REQUIRED)

add_executable(QTSynth main.cpp)

target_link_libraries(QTSynth PRIVATE fmt::fmt)