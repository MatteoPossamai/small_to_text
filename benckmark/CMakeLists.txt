cmake_minimum_required(VERSION 3.15...4.4)

project(SmallToTextBench LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(trigonometry trigonometric_function.cpp)

target_compile_options(trigonometry PRIVATE
    -O3 -ffast-math -g
    $<$<CXX_COMPILER_ID:Clang>:-Rpass=loop-vectorize>
    $<$<CXX_COMPILER_ID:Clang>:-Rpass-missed=loop-vectorize>
    $<$<CXX_COMPILER_ID:Clang>:-Rpass-analysis=loop-vectorize>
    $<$<CXX_COMPILER_ID:Clang>:-save-temps=obj>
)

find_package(benchmark REQUIRED)
target_link_libraries(trigonometry PRIVATE benchmark::benchmark)
