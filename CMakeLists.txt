cmake_minimum_required(VERSION 3.24)
project(cuda_time_travel LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
option(TT_ENABLE_CUPTI "Enable CUPTI kernel tracing" OFF)

add_library(ttrecorder STATIC
    src/ttrecorder.cu
    src/tt_graph.cpp
    src/tt_trace.cpp
)

target_include_directories(ttrecorder
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(ttrecorder
    PUBLIC
        CUDA::cudart
)

if(TT_ENABLE_CUPTI)
    target_sources(ttrecorder PRIVATE src/tt_cupti.cpp)
    target_compile_definitions(ttrecorder PUBLIC TT_ENABLE_CUPTI=1)
    get_target_property(CUPTI_INCLUDES CUDA::cupti INTERFACE_INCLUDE_DIRECTORIES)
    if(CUPTI_INCLUDES)
        target_include_directories(ttrecorder PRIVATE ${CUPTI_INCLUDES})
    endif()
endif()

add_executable(tt_demo
    demo/tt_demo.cpp
)

target_link_libraries(tt_demo PRIVATE ttrecorder)

add_executable(tt_demo_graph
    demo/tt_demo_graph.cu
)

target_link_libraries(tt_demo_graph PRIVATE ttrecorder)

add_executable(tt_tests
    tests/tt_tests.cu
)

target_link_libraries(tt_tests PRIVATE ttrecorder)
