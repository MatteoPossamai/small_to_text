cmake_minimum_required(VERSION 3.15...4.4)

project(SmallToText LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# -----------------
# Tests
# -----------------
include(CTest)
if(BUILD_TESTING)
  add_executable(encoding_tests tests/encoding.cpp)
  target_compile_options(encoding_tests PRIVATE -O3 -g)
  add_test(NAME encoding COMMAND encoding_tests)

  add_executable(network_tests tests/network.cpp)
  add_test(NAME network COMMAND network_tests)
endif()

# -----------------
# Compile audio target
# -----------------
add_executable(compile_audios src/compile_audio_to_weights.cpp)
target_compile_definitions(compile_audios PRIVATE
    DATASET_METADATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_compile_options(compile_audios PRIVATE
    -O3 -ffast-math -g
)

# -----------------
# Inference target
# -----------------
add_executable(inference src/inference.cpp)
target_compile_options(inference PRIVATE
    -O3 -ffast-math -g
)
