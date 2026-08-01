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
    src/gemma4_ple.cpp
    src/gemma4_validator.cpp
    src/gemma4_vision_validator.cpp
    src/gemma4_vision_preprocess.cpp
    src/graph_builder.cpp
    src/sampler.cpp
    src/htf_tokenizer.cpp
    src/chat_template.cpp
    src/hexos_bridge.cpp
)

# CUDA kernels
set(KERNEL_SOURCES
    kernels/elementwise.cu
    kernels/activations.cu
    kernels/normalization.cu
    kernels/linear.cu
    kernels/matmul_hq31k.cu
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

# HQ3.1K additive dtype/layout and CUDA GEMV
add_executable(test_hq31k tests/test_hq31k.cu)
target_link_libraries(test_hq31k PRIVATE helios-engine)

# HNF Loader test
add_executable(test_hnf_loader tests/test_hnf_loader.cpp)
target_link_libraries(test_hnf_loader PRIVATE helios-engine)

# Gemma 4 HNF/GM4X metadata contract (no model weights or CUDA allocations)
add_executable(test_gemma4_metadata tests/test_gemma4_metadata.cpp)
target_link_libraries(test_gemma4_metadata PRIVATE helios-engine)

# Gemma 4 GM4V, visual inventory and optional real block load/unload
add_executable(test_gemma4_vision_metadata tests/test_gemma4_vision_metadata.cpp)
target_link_libraries(test_gemma4_vision_metadata PRIVATE helios-engine)

# Gemma 4 RGB resize/patchify/position preprocessing (no image decoder)
add_executable(test_gemma4_vision_preprocess tests/test_gemma4_vision_preprocess.cpp)
target_link_libraries(test_gemma4_vision_preprocess PRIVATE helios-engine)

# Gemma 4 numerical primitives, isolated from graph construction
add_executable(test_gemma4_primitives tests/test_gemma4_primitives.cu)
target_link_libraries(test_gemma4_primitives PRIVATE helios-engine)

# Gemma 4 complete PLE preparation path and real-weight numerical reference
add_executable(test_gemma4_ple tests/test_gemma4_ple.cu)
target_link_libraries(test_gemma4_ple PRIVATE helios-engine)

# Gemma 4 heterogeneous scratch and explicit first-layer graph
add_executable(test_gemma4_graph tests/test_gemma4_graph.cu)
target_link_libraries(test_gemma4_graph PRIVATE helios-engine)

# Gemma 4 heterogeneous/shared KV layout and local/global cached attention
add_executable(test_gemma4_kv tests/test_gemma4_kv.cu)
target_link_libraries(test_gemma4_kv PRIVATE helios-engine)

# Gemma 4 HTF3 tokenizer contract and reference vectors
add_executable(test_gemma4_tokenizer tests/test_gemma4_tokenizer.cpp)
target_link_libraries(test_gemma4_tokenizer PRIVATE helios-engine)

# Canonical text-only Gemma 4 chat framing
add_executable(test_chat_template tests/test_chat_template.cpp)
target_link_libraries(test_chat_template PRIVATE helios-engine)

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
add_test(NAME HQ31KTest COMMAND test_hq31k)
add_test(NAME HnfLoaderTest COMMAND test_hnf_loader)
add_test(NAME Gemma4MetadataTest COMMAND test_gemma4_metadata)
add_test(NAME Gemma4VisionMetadataTest COMMAND test_gemma4_vision_metadata)
add_test(NAME Gemma4VisionPreprocessTest COMMAND test_gemma4_vision_preprocess)
add_test(NAME Gemma4PrimitivesTest COMMAND test_gemma4_primitives)
add_test(NAME Gemma4PleTest COMMAND test_gemma4_ple)
add_test(NAME Gemma4GraphTest COMMAND test_gemma4_graph)
add_test(NAME Gemma4KVTest COMMAND test_gemma4_kv)
add_test(NAME Gemma4TokenizerTest COMMAND test_gemma4_tokenizer)
add_test(NAME ChatTemplateTest COMMAND test_chat_template)

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

# Gemma 4 complete greedy/sampled generation through heterogeneous/shared KV
add_executable(test_gemma4_generate tests/test_gemma4_generate.cu)
target_link_libraries(test_gemma4_generate PRIVATE helios-engine)

# Gemma 4 raw-logit dump for converter quantization experiments
add_executable(gemma4_dump_logits tools/gemma4_dump_logits.cu)
target_link_libraries(gemma4_dump_logits PRIVATE helios-engine)

# Gemma 4 token-by-token quality calibration (NLL/entropy/top-1)
add_executable(gemma4_calibrate tools/gemma4_calibrate.cu)
target_link_libraries(gemma4_calibrate PRIVATE helios-engine)

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

# ========== HELIOS CHAT (multi-turn, KV continuity, dynamic output) ==========
add_executable(helios_chat helios_chat.cpp)
target_link_libraries(helios_chat PRIVATE helios-engine)

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
