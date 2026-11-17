// tools/qwen35_vision_dump.cu
// La torre visual de Qwen3.5 sobre una imagen, volcando su salida a .npy para compararla con
// tools/reference/qwen35_vision_ref.py --dump (salida.npy).
//
//     qwen35_vision_dump <modelo.hnf> <imagen.ppm> <salida.npy>
//
// La imagen va en PPM binario (P6, 8 bits): decodificar PNG o JPEG queda fuera del motor.
#include "hnf_loader.hpp"
#include "qwen35_vision.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace helios;

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "uso: " << argv[0] << " <modelo.hnf> <imagen.ppm> <salida.npy>" << std::endl;
        return 1;
    }
    try {
        std::ifstream f(argv[2], std::ios::binary);
        std::string magic;
        uint32_t width = 0, height = 0, maximum = 0;
        f >> magic >> width >> height >> maximum;
        f.get();
        if (!f || magic != "P6" || maximum != 255) throw std::runtime_error("se espera un PPM binario de 8 bits");
        std::vector<uint8_t> pixels(size_t(width) * height * 3);
        f.read(reinterpret_cast<char*>(pixels.data()), std::streamsize(pixels.size()));
        if (!f) throw std::runtime_error("PPM incompleto");

        HnfLoader loader;
        if (!loader.open(argv[1])) throw std::runtime_error("no abro el HNF: " + loader.last_error());
        Qwen35VisionConfig config;
        if (!config.read(loader.config())) throw std::runtime_error("este HNF no trae la torre visual de Qwen3.5");
        Qwen35VisionRunner runner(loader, config);
        runner.keep_output = true;
        cudaStream_t stream;
        if (cudaStreamCreate(&stream) != cudaSuccess) throw std::runtime_error("no pude crear el stream");
        half* output = nullptr;
        if (cudaMalloc(&output, size_t(config.max_tokens()) * config.out_hidden * sizeof(half)) != cudaSuccess) {
            throw std::runtime_error("sin memoria para la salida");
        }
        Qwen35ImageGrid grid;
        std::string error;
        const Gemma4RgbView view{pixels.data(), width, height, 0};
        for (int pass = 0; pass < 2; ++pass) {          // la segunda, ya con el fichero en caché
            const auto start = std::chrono::steady_clock::now();
            if (!runner.run(view, output, grid, stream, &error)) throw std::runtime_error(error);
            std::cerr << "pasada " << pass + 1 << ": " << std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count() << " ms" << std::endl;
        }
        const std::vector<float>& values = runner.last_output();
        std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + std::to_string(grid.tokens()) + ", " +
                             std::to_string(config.out_hidden) + "), }";
        while ((10 + header.size() + 1) % 64 != 0) header += ' ';
        header += '\n';
        std::ofstream out(argv[3], std::ios::binary);
        out.write("\x93NUMPY\x01\x00", 8);
        const uint16_t length = uint16_t(header.size());
        out.write(reinterpret_cast<const char*>(&length), 2);
        out.write(header.data(), std::streamsize(header.size()));
        out.write(reinterpret_cast<const char*>(values.data()), std::streamsize(values.size() * sizeof(float)));
        std::cerr << "rejilla " << grid.rows << "×" << grid.columns << ": " << grid.tokens() << " vectores de ancho "
                  << config.out_hidden << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
