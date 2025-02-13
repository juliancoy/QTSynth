cmake_minimum_required(VERSION 3.10)

project(QTSynth VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
list(APPEND CMAKE_PREFIX_PATH "/home/julian/Documents/QTSynth/vcpkg_installed/x64-linux")

find_package(fmt CONFIG REQUIRED)
find_package(Qt6 COMPONENTS Core Widgets REQUIRED)
find_package(RtAudio CONFIG REQUIRED)
find_package(RtMidi CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(Tracy CONFIG REQUIRED)

# Add the include directory for cpp-base64 using target_include_directories
include_directories(/home/julian/Documents/QTSynth/vcpkg_installed/x64-linux/include/cpp-base64)

add_executable(QTSynth 
    main.cpp
    sample_compute.cpp
    /home/julian/Documents/QTSynth/vcpkg_installed/x64-linux/include/cpp-base64/base64.cpp
)


target_link_libraries(QTSynth PRIVATE 
    fmt::fmt
    Qt6::Core
    Qt6::Widgets
    RtAudio::rtaudio
    Tracy::TracyClient
    RtMidi::rtmidi
    nlohmann_json::nlohmann_json
)

# Add dr_wav and dr_mp3 as header-only libraries
target_include_directories(QTSynth PRIVATE ${CMAKE_SOURCE_DIR}/external)

set_target_properties(QTSynth PROPERTIES
    AUTOMOC ON
    AUTOUIC ON
    AUTORCC ON
    WIN32_EXECUTABLE ON
    MACOSX_BUNDLE ON
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -g -fno-omit-frame-pointer")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")