cmake_minimum_required(VERSION 3.15...4.4)

project(SmallToText LANGUAGES CXX)

set(NETWORK_MODEL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/network/speech_commands.txt")

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

  add_executable(mask_tests tests/mask.cpp)
  target_compile_options(mask_tests PRIVATE -UNDEBUG)
  add_test(NAME mask COMMAND mask_tests)

  add_executable(network_tests tests/network.cpp)
  add_test(NAME network COMMAND network_tests)

  add_executable(cnn_tests tests/cnn.cpp)
  add_test(NAME cnn COMMAND cnn_tests)
  add_test(NAME cnn_cli COMMAND ${CMAKE_COMMAND}
      -DCNN_EXECUTABLE=$<TARGET_FILE:cnn>
      -DTEST_OUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/cnn_cli.cmake)
  add_test(NAME model_profiles COMMAND ${CMAKE_COMMAND}
      -DCNN16_EXE=$<TARGET_FILE:cnn16>
      -DCNN5_EXE=$<TARGET_FILE:cnn5>
      -DCNN8_EXE=$<TARGET_FILE:cnn8>
      -DCNN_DEFAULT_EXE=$<TARGET_FILE:cnn>
      -DTEST_OUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/model_profiles.cmake)
endif()

# Build the same source for each saved model shape. No neural-network code changes.
function(add_cnn_variant name neurons window model)
  add_executable(${name} src/cnn_main.cpp)
  target_compile_definitions(${name} PRIVATE CNN_NEURONS=${neurons} CNN_WINDOW=${window})
  target_compile_options(${name} PRIVATE -O3 -g)

  add_executable(inference_${name} src/inference_cnn.cpp)
  target_compile_definitions(inference_${name} PRIVATE
      CNN_NEURONS=${neurons} CNN_WINDOW=${window}
      CNN_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/network/${model}")
  target_compile_options(inference_${name} PRIVATE -O3 -g)
endfunction()

add_cnn_variant(cnn16 16 5 2_cnn16_window5_20band_500.txt)
add_cnn_variant(cnn5 32 5 3_cnn32_window5_20band_500.txt)
add_cnn_variant(cnn8 32 8 4_cnn32_window8_20band_500.txt)
# Preserve the existing command names as the default eight-row build.
add_cnn_variant(cnn 32 8 4_cnn32_window8_20band_500.txt)

# Experimental trainer uses the separate CNN implementation.
add_executable(cnn2 src/cnn_main.cpp)
target_compile_definitions(cnn2 PRIVATE CNN_EXPERIMENTAL CNN_NEURONS=32 CNN_WINDOW=8)
target_compile_options(cnn2 PRIVATE -O3 -g)

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
# Train target
# -----------------
add_executable(train src/train.cpp)
target_compile_definitions(train PRIVATE
    NETWORK_MODEL_PATH="${NETWORK_MODEL_PATH}"
)
target_compile_options(train PRIVATE
    -O3 -ffast-math -g
)

# -----------------
# Scoring target
# -----------------
add_executable(scoring src/scoring.cpp)
target_compile_definitions(scoring PRIVATE
    NETWORK_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/network/1_mlp_timeshift_20band_300.txt"
)
target_compile_options(scoring PRIVATE
    -O3 -ffast-math -g
)

# -----------------
# Inference target
# -----------------
add_executable(inference src/inference.cpp)
target_compile_definitions(inference PRIVATE
    NETWORK_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/network/1_mlp_timeshift_20band_300.txt"
)
target_compile_options(inference PRIVATE
    -O3 -ffast-math -g
)

