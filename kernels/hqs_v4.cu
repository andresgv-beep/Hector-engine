// HQS v4 (hq44k_g16) prefill: dequantize to FP16 and use cuBLAS. GEMV lives beside the tuner.
#include "kernels.hpp"
#include "cublas_context.hpp"
#include "hqs_v4_gemv.cuh"
#include <stdexcept>
namespace helios { namespace kernels {
namespace {
void checked(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }

// One thread per 8 weights: same split as the GEMV lanes.
__global__ void dequant_hq44(const uint8_t* weights, half* output, int K, int N) {
    const int row = blockIdx.x;
    const int first = (blockIdx.y * blockDim.x + threadIdx.x) * 8;
    if (row >= N || first >= K) return;
    const uint8_t* block = weights + (size_t(row) * (K / 256) + first / 256) * hqs_v4::BLOCK_BYTES;
    const int lane = (first % 256) / 8, group = lane >> 1;
    const float a = hqs_v4::half_at(block) * float(hqs_v4::field5(block, group * 5));
    const float m = hqs_v4::half_at(block + 2) * float(hqs_v4::field5(block, 80 + group * 5));
    const uint32_t packed = *reinterpret_cast<const uint32_t*>(block + hqs_v4::HEADER_BYTES + lane * 4);
    uint32_t out[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned byte = (packed >> (i * 8)) & 0xffu;
        const half lo = __float2half(a * float(byte & 15u) - m), hi = __float2half(a * float(byte >> 4) - m);
        out[i] = uint32_t(__half_as_ushort(lo)) | (uint32_t(__half_as_ushort(hi)) << 16);
    }
    *reinterpret_cast<uint4*>(output + size_t(row) * K + first) = make_uint4(out[0], out[1], out[2], out[3]);
}
} // anonymous namespace

void launch_dequant_hq44k(const uint8_t* w, half* y, int K, int N, cudaStream_t s) {
    if (K <= 0 || N <= 0) return;
    if (K % 256) throw std::runtime_error("hq44k requires K divisible by 256");
    dim3 grid(N, (K / 8 + 255) / 256);
    dequant_hq44<<<grid, 256, 0, s>>>(w, y, K, N);
}

void launch_matmul_hq44k_cublas(const half* x, const uint8_t* w, half* y, int M, int K, int N, cudaStream_t s) {
    static half* scratch = nullptr; static size_t capacity = 0;
    const size_t count = size_t(N) * K;
    if (count > capacity) {
        if (scratch) checked(cudaFree(scratch));
        scratch = nullptr; capacity = 0;
        checked(cudaMalloc(&scratch, count * sizeof(half))); capacity = count;
    }
    const auto handle = cublas_handle_for_stream(s);
    if (!handle) throw std::runtime_error("hq44k cuBLAS handle unavailable");
    launch_dequant_hq44k(w, scratch, K, N, s);
    const half alpha = __float2half(1.f), beta = __float2half(0.f);
    const auto result = cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, scratch, K, x, K, &beta, y, N);
    if (result != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("hq44k cublasHgemm failed");
}
} } // helios::kernels
