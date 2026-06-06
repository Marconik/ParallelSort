cmake_minimum_required(VERSION 3.18)
project(ParallelSort LANGUAGES CXX)

# ---------- 选项 ----------
option(ENABLE_CUDA "Build with CUDA GPU acceleration" ON)

# C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

# 头文件路径
include_directories(${CMAKE_SOURCE_DIR}/include)

# ---------- 串行排序库 (纯 C++) ----------
add_library(sort_serial STATIC
    src/sort_serial.cpp
)
target_include_directories(sort_serial PUBLIC include)

# ---------- GPU 并行排序库 + 可执行文件 ----------
if(ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_FLAGS_RELEASE "-O2 -DNDEBUG")

    add_library(sort_parallel STATIC
        src/sort_parallel.cu
    )
    target_include_directories(sort_parallel PUBLIC include)
    target_link_libraries(sort_parallel PUBLIC
        CUDA::cudart
        sort_serial
    )

    add_executable(sort_test src/main.cu)
    target_link_libraries(sort_test PRIVATE
        sort_serial
        sort_parallel
    )

    # CUDA 架构 (根据实际 GPU 调整)
    # 60=Pascal, 70=Volta, 75=Turing, 80=Ampere, 89=Ada, 90=Hopper, 120=Blackwell
    set_target_properties(sort_parallel PROPERTIES
        CUDA_ARCHITECTURES "60;70;75;80;89;90;120"
    )
else()
    # 纯串行可执行文件
    add_executable(sort_test src/main.cu)
    target_link_libraries(sort_test PRIVATE sort_serial)
    target_compile_definitions(sort_test PRIVATE DISABLE_CUDA)
    message(STATUS "CUDA disabled — building serial-only version")
endif()
