#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace helios {
namespace kernels {

// Process-wide cuBLAS context shared by text matmuls and the one-shot visual
// tower. Keeping a single handle avoids a second internal workspace and keeps
// stream selection consistent with the existing matmul path.
cublasHandle_t cublas_handle_for_stream(cudaStream_t stream);

} // namespace kernels
} // namespace helios
