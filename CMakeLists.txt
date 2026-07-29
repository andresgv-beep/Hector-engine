# CMakeLists.txt
# HELIOS ENGINE - Build Configuration
# ====================================
#
# Build:
#   mkdir build && cd build
#   cmake ..
#   make -j$(nproc)
#
# Run tests:
#   ./test_tensor
#

cmake_minimum_required(VERSION 3.18)
project(helios-engine LANGUAGES CXX CUDA)

# ============================================================================
# C++ / CUDA Standards
# ============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ============================================================================
# CUDA Architecture
# ============================================================================

# "native" detecta la GPU local (RTX 5070 = Blackwell sm_120, requiere CUDA 12.8+)
# Para builds portables usar lista explícita, p.ej. "86;89;120"
# (ojo: CUDA 13 eliminó soporte para < sm_75)
set(CMAKE_CUDA_ARCHITECTURES "native")

# ============================================================================
# Compiler Flags
# ============================================================================

# Release by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# C++ flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")

# CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0 -DDEBUG")

# ============================================================================
# Source Files
# ============================================================================

set(ENGINE_SOURCES
    src/dtype.cpp
    src/optype.cpp
    src/tensor.cpp
    src/memory.cpp
    src/command.cpp
    src/engine.cpp
    src/hnf_loader.cpp
    src/graph_builder.cpp
    src/sampler.cpp
    src/htf_tokenizer.cpp
)

# CUDA kernels
set(KERNEL_SOURCES
    kernels/elementwise.cu
    kernels/activations.cu
    kernels/normalization.cu
    kernels/linear.cu
    kernels/matmul_hqs.cu
    kernels/matmul_hqs_compact.cu
    kernels/matmul_cublas.cu
    kernels/attention.cu
    kernels/sampling.cu
    kernels/fused_add_rmsnorm.cu
    kernels/register_kernels.cpp
)

# ============================================================================
# Find CUDA Toolkit
# ============================================================================

find_package(CUDAToolkit REQUIRED)

# Force CUDA include path (fix for some systems)
set(CUDA_INCLUDE_DIR "/usr/local/cuda/include")
if(NOT EXISTS "${CUDA_INCLUDE_DIR}/cuda_runtime.h")
    set(CUDA_INCLUDE_DIR "/usr/local/cuda-12.6/include")
endif()

message(STATUS "CUDA include dir: ${CUDA_INCLUDE_DIR}")

# ============================================================================
# Library
# ============================================================================

add_library(helios-engine STATIC
    ${ENGINE_SOURCES}
    ${KERNEL_SOURCES}
)

target_include_directories(helios-engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
    ${CUDA_INCLUDE_DIR}
)

target_link_libraries(helios-engine PUBLIC
    CUDA::cudart
    CUDA::cublas
)

# ============================================================================
# Tests
# ============================================================================

# TensorRegistry test
add_executable(test_tensor tests/test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE helios-engine)

# MemoryPool test
add_executable(test_memory tests/test_memory.cpp)
target_link_libraries(test_memory PRIVATE helios-engine)

# CommandBuffer test
add_executable(test_command tests/test_command.cpp)
target_link_libraries(test_command PRIVATE helios-engine)

# Engine test
add_executable(test_engine tests/test_engine.cpp)
target_link_libraries(test_engine PRIVATE helios-engine)

# Kernel test (CUDA)
add_executable(test_kernels tests/test_kernels.cu)
target_link_libraries(test_kernels PRIVATE helios-engine)

# HNF Loader test
add_executable(test_hnf_loader tests/test_hnf_loader.cpp)
target_link_libraries(test_hnf_loader PRIVATE helios-engine)

# Real HNF inspector (with block loading test)
add_executable(inspect_hnf test_real_hnf.cpp)
target_link_libraries(inspect_hnf PRIVATE helios-engine)

# Enable testing
enable_testing()
add_test(NAME TensorRegistryTest COMMAND test_tensor)
add_test(NAME MemoryPoolTest COMMAND test_memory)
add_test(NAME CommandBufferTest COMMAND test_command)
add_test(NAME EngineTest COMMAND test_engine)
add_test(NAME KernelTest COMMAND test_kernels)
add_test(NAME HnfLoaderTest COMMAND test_hnf_loader)

# GraphBuilder test
add_executable(test_graph_builder test_graph_builder.cpp)
target_link_libraries(test_graph_builder PRIVATE helios-engine)

# Generation test
add_executable(test_generate test_generate.cpp)
target_link_libraries(test_generate PRIVATE helios-engine)

# Generation test with KV cache
add_executable(test_generate_kv test_generate_kv.cpp)
target_link_libraries(test_generate_kv PRIVATE helios-engine)

# Tokenizer test
add_executable(test_tokenizer test_tokenizer.cpp)
target_link_libraries(test_tokenizer PRIVATE helios-engine)

# Full generation test (tokenizer + forward + sampling + decode)
add_executable(test_generate_full test_generate_full.cpp)
target_link_libraries(test_generate_full PRIVATE helios-engine)

# Multimodal test (CORTEX, CODE)
add_executable(test_multimodal test_multimodal.cpp)
target_link_libraries(test_multimodal PRIVATE helios-engine)

# Modal generation test (any modal)
add_executable(test_generate_modal test_generate_modal.cpp)
target_link_libraries(test_generate_modal PRIVATE helios-engine)

# ========== SMOKE TEST (progresivo, 4 fases) ==========
add_executable(test_smoke test_smoke.cpp)
target_link_libraries(test_smoke PRIVATE helios-engine)

# ========== CHAT TEST (interactive with templates) ==========
add_executable(test_chat test_chat.cpp)
target_link_libraries(test_chat PRIVATE helios-engine)

# ============================================================================
# Info
# ============================================================================

message(STATUS "")
message(STATUS "=== HELIOS ENGINE ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "")
