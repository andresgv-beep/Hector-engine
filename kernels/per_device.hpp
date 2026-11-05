#pragma once

#include <cuda_runtime.h>

namespace helios {
namespace kernels {

// CUDA guarda la GPU activa por hilo, y un handle de cuBLAS o un buffer de una
// GPU no valen en otra: lo que antes era global de proceso va por dispositivo.
template <typename T>
class PerDevice {
public:
    static constexpr int MAX_DEVICES = 16;

    T& current() { return at(current_device()); }
    T& at(int device) { return slots_[device]; }

    static int current_device() {
        int device = 0;
        cudaGetDevice(&device);
        return device;
    }

private:
    T slots_[MAX_DEVICES]{};
};

}  // namespace kernels
}  // namespace helios
