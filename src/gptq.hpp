// GPTQ dentro de una rejilla afín fija: elige los códigos de cada peso, no la rejilla.
//
// Mismo algoritmo y misma aritmética (FP64) que gptq_codes() de
// hector-gemma4-moe/captures/gemma4-26b-gptq-final-v1/scripts/gptq_full.py:
//   H += amortiguación · media(diag H) (las columnas muertas llevan 1 en la diagonal),
//   U = Cholesky superior de H⁻¹, y columna a columna, en bloques de 128,
//   código = clamp(rint((w + m) / a), 0, niveles), error = (w − (a·código − m)) / U[i,i],
//   y el error se reparte sobre las columnas que faltan con la fila i de U.
// Lo usan las herramientas de refinado; el motor de inferencia no depende de esto.
#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <cstdint>
#include <vector>

namespace helios { namespace gptq {

class Quantizer {
public:
    Quantizer(int max_rows, int max_cols, cudaStream_t stream = nullptr);
    ~Quantizer();
    Quantizer(const Quantizer&) = delete;
    Quantizer& operator=(const Quantizer&) = delete;

    // Todo en el dispositivo y fila a fila. w [rows, cols] se modifica; h [cols, cols]
    // se destruye; a y m [rows, cols] dan valor = a·código − m. Escribe codes [rows, cols].
    void run_device(double* w, double* h, const float* a, const float* m,
                    int rows, int cols, double damp, uint8_t* codes, int levels = 15);

    // Igual con punteros de host, para pruebas.
    std::vector<uint8_t> run_host(const std::vector<double>& w, const std::vector<double>& h,
                                  const std::vector<float>& a, const std::vector<float>& m,
                                  int rows, int cols, double damp, int levels = 15);

    cublasHandle_t cublas() const { return cublas_; }
    cudaStream_t stream() const { return stream_; }

private:
    int max_rows_, max_cols_;
    cudaStream_t stream_;
    cublasHandle_t cublas_ = nullptr;
    cusolverDnHandle_t solver_ = nullptr;
    double* err_ = nullptr;       // [max_rows, 128]
    double* work_ = nullptr;
    int work_size_ = 0;
    int* info_ = nullptr;
};

// HQ44 (hq44k_g16, kernels/hqs_v4_gemv.cuh): bloques de 256 pesos y 152 bytes.
constexpr int HQ44_BLOCK_BYTES = 152;
// a y m de cada peso de una matriz [rows, physical] de bloques.
void hq44_grid(const uint8_t* blocks, int rows, int physical, float* a, float* m);
// Códigos 0..15 de cada peso.
void hq44_codes(const uint8_t* blocks, int rows, int physical, uint8_t* codes);
// Sustituye los códigos de las primeras `logical` columnas; el relleno conserva los suyos.
void hq44_set_codes(uint8_t* blocks, const uint8_t* codes, int rows, int physical, int logical);

} } // helios::gptq
