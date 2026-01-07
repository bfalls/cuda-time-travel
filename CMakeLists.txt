cmake_minimum_required(VERSION 3.24)
project(cuda_time_travel LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_library(ttrecorder STATIC
    src/ttrecorder.cpp
)

target_include_directories(ttrecorder
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(ttrecorder
    PUBLIC
        CUDA::cudart
)

add_executable(tt_demo
    demo/tt_demo.cpp
)

target_link_libraries(tt_demo PRIVATE ttrecorder)

add_executable(tt_tests
    tests/tt_tests.cpp
)

target_link_libraries(tt_tests PRIVATE ttrecorder)
