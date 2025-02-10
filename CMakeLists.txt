cmake_minimum_required(VERSION 3.10)

project(QTSynth VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(fmt CONFIG REQUIRED)
find_package(Qt6 COMPONENTS Core Widgets REQUIRED)
find_package(RtAudio CONFIG REQUIRED)

add_executable(QTSynth main.cpp)

target_link_libraries(QTSynth PRIVATE 
    fmt::fmt
    Qt6::Core
    Qt6::Widgets
    RtAudio::rtaudio
)

set_target_properties(QTSynth PROPERTIES
    AUTOMOC ON
    AUTOUIC ON
    AUTORCC ON
    WIN32_EXECUTABLE ON
    MACOSX_BUNDLE ON
)
