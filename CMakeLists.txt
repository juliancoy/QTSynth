cmake_minimum_required(VERSION 3.10)

project(QTSynth VERSION 0.1 LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find system packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(FMT REQUIRED fmt)
find_package(Qt6 COMPONENTS Core Widgets REQUIRED)
pkg_check_modules(RTAUDIO REQUIRED rtaudio)
pkg_check_modules(RTMIDI REQUIRED rtmidi)
pkg_check_modules(NLOHMANN_JSON REQUIRED nlohmann_json)

# Tracy profiler setup
set(TRACY_ENABLE ON)
add_subdirectory(external/tracy)

# Add the executable
add_executable(QTSynth 
    main.cpp
    sample_compute.cpp
    external/cpp-base64/base64.cpp
)

# Link system libraries
target_link_libraries(QTSynth PRIVATE 
    ${FMT_LIBRARIES}
    Qt6::Core
    Qt6::Widgets
    ${RTAUDIO_LIBRARIES}
    TracyClient
    ${RTMIDI_LIBRARIES}
    ${NLOHMANN_JSON_LIBRARIES}
)

# Include directories
target_include_directories(QTSynth PRIVATE 
    ${CMAKE_SOURCE_DIR}/external/cpp-base64
    ${FMT_INCLUDE_DIRS}
    ${RTAUDIO_INCLUDE_DIRS}
    ${RTMIDI_INCLUDE_DIRS}
    ${NLOHMANN_JSON_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/external
    ${CMAKE_SOURCE_DIR}/external/dr_libs
)

# Qt specific settings
set_target_properties(QTSynth PROPERTIES
    AUTOMOC ON
    AUTOUIC ON
    AUTORCC ON
    WIN32_EXECUTABLE ON
    MACOSX_BUNDLE ON
)

# Compiler and linker flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fno-omit-frame-pointer")
# ! Temporarily disabled for Valgrind profiling
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -g -fno-omit-frame-pointer")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
