cmake_minimum_required(VERSION 3.15...4.4)

project(SmallToText LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(compile_audios src/compile_audio_to_weights.cpp)
target_compile_definitions(compile_audios PRIVATE
    DATASET_METADATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)

target_compile_options(compile_audios PRIVATE
    -O3 -ffast-math -g
)
