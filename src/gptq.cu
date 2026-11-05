// GPTQ en FP64 con cuSOLVER y cuBLAS. Ver gptq.hpp.
#include "gptq.hpp"

#include <cuda_fp16.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace helios { namespace gptq {
namespace {

constexpr int BLOCK = 128;

void check(cudaError_t e, const char* where) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("gptq ") + where + ": " + cudaGetErrorString(e));
}
void check(cublasStatus_t s, const char* where) {
    if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string("gptq ") + where + ": cuBLAS " + std::to_string(int(s)));
}
void check(cusolverStatus_t s, const char* where) {
    if (s != CUSOLVER_STATUS_SUCCESS) throw std::runtime_error(std::string("gptq ") + where + ": cuSOLVER " + std::to_string(int(s)));
}

// Un hilo por fila. u es el factor superior de H⁻¹ en orden de columnas:
// U(i, j) = u[i + j·n], que es hinv[i, j] de la referencia.
__global__ void gptq_block(double* __restrict__ w, const double* __restrict__ u, const float* __restrict__ a,
                           const float* __restrict__ m, uint8_t* __restrict__ codes, double* __restrict__ err,
                           int rows, int cols, int first, int width, int levels) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    double local[BLOCK];
    double* row_w = w + size_t(row) * cols + first;
    for (int j = 0; j < width; ++j) local[j] = row_w[j];
    const float* row_a = a + size_t(row) * cols + first;
    const float* row_m = m + size_t(row) * cols + first;
    for (int i = 0; i < width; ++i) {
        const double aa = row_a[i], mm = row_m[i];
        double q = 0.0;
        if (aa > 0.0) q = fmin(fmax(rint((local[i] + mm) / aa), 0.0), double(levels));
        codes[size_t(row) * cols + first + i] = uint8_t(q);
        const int col = first + i;
        const double e = (local[i] - (aa * q - mm)) / u[size_t(col) + size_t(col) * cols];
        for (int j = i; j < width; ++j) local[j] -= e * u[size_t(col) + size_t(first + j) * cols];
        err[size_t(row) * BLOCK + i] = e;
    }
}

}  // namespace

Quantizer::Quantizer(int max_rows, int max_cols, cudaStream_t stream)
    : max_rows_(max_rows), max_cols_(max_cols), stream_(stream) {
    check(cublasCreate(&cublas_), "cublasCreate");
    check(cublasSetStream(cublas_, stream_), "cublasSetStream");
    // Reproducible: sin atomics en cuBLAS.
    check(cublasSetAtomicsMode(cublas_, CUBLAS_ATOMICS_NOT_ALLOWED), "atomics");
    check(cusolverDnCreate(&solver_), "cusolverDnCreate");
    check(cusolverDnSetStream(solver_, stream_), "cusolverDnSetStream");
    check(cudaMalloc(&err_, sizeof(double) * size_t(max_rows) * BLOCK), "err");
    check(cudaMalloc(&info_, sizeof(int)), "info");
    int potrf = 0, potri = 0;
    check(cusolverDnDpotrf_bufferSize(solver_, CUBLAS_FILL_MODE_UPPER, max_cols, nullptr, max_cols, &potrf), "potrf size");
    check(cusolverDnDpotri_bufferSize(solver_, CUBLAS_FILL_MODE_UPPER, max_cols, nullptr, max_cols, &potri), "potri size");
    work_size_ = potrf > potri ? potrf : potri;
    check(cudaMalloc(&work_, sizeof(double) * size_t(work_size_)), "work");
}

Quantizer::~Quantizer() {
    cudaFree(err_);
    cudaFree(work_);
    cudaFree(info_);
    if (solver_) cusolverDnDestroy(solver_);
    if (cublas_) cublasDestroy(cublas_);
}

void Quantizer::run_device(double* w, double* h, const float* a, const float* m,
                           int rows, int cols, double damp, uint8_t* codes, int levels) {
    if (rows <= 0 || cols <= 0 || rows > max_rows_ || cols > max_cols_) throw std::invalid_argument("gptq: dimensiones");
    // Diagonal en el host: columnas muertas a 1 y amortiguación con la media resultante.
    std::vector<double> diag(cols);
    check(cudaMemcpy2DAsync(diag.data(), sizeof(double), h, sizeof(double) * (cols + 1), sizeof(double), cols,
                            cudaMemcpyDeviceToHost, stream_), "leer diagonal");
    check(cudaStreamSynchronize(stream_), "diagonal");
    double mean = 0.0;
    for (double& v : diag) { if (v == 0.0) v = 1.0; mean += v; }
    mean /= cols;
    for (double& v : diag) v += damp * mean;
    check(cudaMemcpy2DAsync(h, sizeof(double) * (cols + 1), diag.data(), sizeof(double), sizeof(double), cols,
                            cudaMemcpyHostToDevice, stream_), "escribir diagonal");

    // U superior con U'U = H⁻¹: Cholesky de H, inversa y Cholesky de la inversa.
    int info = 0;
    for (int step = 0; step < 3; ++step) {
        if (step == 1) check(cusolverDnDpotri(solver_, CUBLAS_FILL_MODE_UPPER, cols, h, cols, work_, work_size_, info_), "potri");
        else check(cusolverDnDpotrf(solver_, CUBLAS_FILL_MODE_UPPER, cols, h, cols, work_, work_size_, info_), "potrf");
        check(cudaMemcpyAsync(&info, info_, sizeof(int), cudaMemcpyDeviceToHost, stream_), "info");
        check(cudaStreamSynchronize(stream_), "factorización");
        if (info != 0) throw std::runtime_error("gptq: H no es definida positiva (paso " + std::to_string(step) + ", info " + std::to_string(info) + ")");
    }

    const int threads = 128, grid = (rows + threads - 1) / threads;
    for (int first = 0; first < cols; first += BLOCK) {
        const int width = cols - first < BLOCK ? cols - first : BLOCK;
        gptq_block<<<grid, threads, 0, stream_>>>(w, h, a, m, codes, err_, rows, cols, first, width, levels);
        check(cudaGetLastError(), "gptq_block");
        const int rest = cols - first - width;
        if (rest > 0) {
            // W[:, first+width:] -= err · U[first:first+width, first+width:], en orden de columnas:
            // (W')[rest, rows] -= (U_blk)'[rest, width] · (err')[width, rows].
            const double alpha = -1.0, beta = 1.0;
            check(cublasDgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N, rest, rows, width, &alpha,
                              h + first + size_t(first + width) * cols, cols, err_, BLOCK, &beta,
                              w + first + width, cols), "dgemm");
        }
    }
    check(cudaStreamSynchronize(stream_), "gptq");
}

std::vector<uint8_t> Quantizer::run_host(const std::vector<double>& w, const std::vector<double>& h,
                                         const std::vector<float>& a, const std::vector<float>& m,
                                         int rows, int cols, double damp, int levels) {
    double *dw, *dh; float *da, *dm; uint8_t* dc;
    const size_t n = size_t(rows) * cols;
    check(cudaMalloc(&dw, n * 8), "w"); check(cudaMalloc(&dh, size_t(cols) * cols * 8), "h");
    check(cudaMalloc(&da, n * 4), "a"); check(cudaMalloc(&dm, n * 4), "m"); check(cudaMalloc(&dc, n), "codes");
    check(cudaMemcpy(dw, w.data(), n * 8, cudaMemcpyHostToDevice), "w");
    check(cudaMemcpy(dh, h.data(), size_t(cols) * cols * 8, cudaMemcpyHostToDevice), "h");
    check(cudaMemcpy(da, a.data(), n * 4, cudaMemcpyHostToDevice), "a");
    check(cudaMemcpy(dm, m.data(), n * 4, cudaMemcpyHostToDevice), "m");
    run_device(dw, dh, da, dm, rows, cols, damp, dc, levels);
    std::vector<uint8_t> codes(n);
    check(cudaMemcpy(codes.data(), dc, n, cudaMemcpyDeviceToHost), "codes");
    cudaFree(dw); cudaFree(dh); cudaFree(da); cudaFree(dm); cudaFree(dc);
    return codes;
}

namespace {
float half_at(const uint8_t* p) { return __half2float(__ushort_as_half(uint16_t(p[0] | (p[1] << 8)))); }
uint32_t field(const uint8_t* block, int bit) {
    const uint32_t word = uint32_t(block[4 + bit / 8]) | (uint32_t(block[5 + bit / 8]) << 8);
    return (word >> (bit % 8)) & 31u;
}
}  // namespace

void hq44_grid(const uint8_t* blocks, int rows, int physical, float* a, float* m) {
    const int per_row = physical / 256;
    for (int r = 0; r < rows; ++r)
        for (int b = 0; b < per_row; ++b) {
            const uint8_t* block = blocks + (size_t(r) * per_row + b) * HQ44_BLOCK_BYTES;
            const float d = half_at(block), dm = half_at(block + 2);
            for (int g = 0; g < 16; ++g) {
                const float ag = d * float(field(block, 5 * g)), mg = dm * float(field(block, 80 + 5 * g));
                for (int i = 0; i < 16; ++i) {
                    const size_t at = size_t(r) * physical + b * 256 + g * 16 + i;
                    a[at] = ag; m[at] = mg;
                }
            }
        }
}

void hq44_codes(const uint8_t* blocks, int rows, int physical, uint8_t* codes) {
    const size_t count = size_t(rows) * physical;
    for (size_t k = 0; k < count; ++k) {
        const uint8_t byte = blocks[(k / 256) * HQ44_BLOCK_BYTES + 24 + (k % 256) / 2];
        codes[k] = (k % 2 == 0) ? (byte & 15) : (byte >> 4);
    }
}

void hq44_set_codes(uint8_t* blocks, const uint8_t* codes, int rows, int physical, int logical) {
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < logical; ++c) {
            const size_t k = size_t(r) * physical + c;
            uint8_t& byte = blocks[(k / 256) * HQ44_BLOCK_BYTES + 24 + (k % 256) / 2];
            const uint8_t code = codes[size_t(r) * logical + c] & 15;
            byte = (k % 2 == 0) ? uint8_t((byte & 0xF0) | code) : uint8_t((byte & 0x0F) | (code << 4));
        }
}

} } // helios::gptq
