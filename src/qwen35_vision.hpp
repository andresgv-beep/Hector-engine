// src/qwen35_vision.hpp
// ============================================================================
// La torre visual de la familia Qwen3.5
// ============================================================================
//
// Convierte una imagen RGB8 en vectores del ancho del modelo de texto, uno por cada bloque
// de 2×2 parches. Es lo que hace tools/reference/qwen35_vision_ref.py, que es su vara de
// medir: redimensionado a múltiplos de 32 píxeles, parches de 3×2×16×16 en orden de bloques
// de fusión, posición aprendida interpolada, bloques con atención completa y rotación en dos
// ejes, y la fusión final.
//
// Los pesos no viven en la VRAM. La torre son 0,9 GB en el 35B y se usa una vez por imagen,
// así que se leen del fichero capa a capa a una ventana pequeña de la tarjeta mientras se
// calcula y no queda nada residente: ni los pesos ni la memoria de trabajo, que se reserva
// al empezar una imagen y se suelta al acabar. El fichero queda en la caché de disco.
#pragma once

#include "gemma4_vision_preprocess.hpp"
#include "hnf_loader.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace helios {

struct Qwen35VisionConfig {
    uint32_t depth = 0, hidden = 0, heads = 0, intermediate = 0;
    uint32_t patch = 0, merge = 0, temporal = 0, out_hidden = 0, grid_side = 0;
    // Tope de parches por imagen: decide la memoria de trabajo (las puntuaciones de la
    // atención son cabezas × parches² en FP16) y cuántos puestos ocupa una imagen en el
    // contexto, parches / merge². HELIOS_VISION_MAX_PARCHES lo cambia.
    uint32_t max_patches = 2304;
    int32_t image_token = -1, start_token = -1, end_token = -1;

    // Lee la geometría de las propiedades que deja el conversor. Falso si el HNF no trae
    // una torre de esta familia.
    bool read(const ModelConfig& config);
    uint32_t patch_values() const { return 3 * temporal * patch * patch; }
    uint32_t max_tokens() const { return max_patches / (merge * merge); }
};

// La rejilla de vectores que sale de una imagen: filas × columnas, fila a fila.
struct Qwen35ImageGrid {
    uint32_t rows = 0, columns = 0;
    uint32_t tokens() const { return rows * columns; }
};

// El tamaño al que se redimensiona una imagen: múltiplos de patch·merge, con el área entre
// 256×256 y lo que permita el tope de parches. Como smart_resize de Qwen2-VL.
bool qwen35_vision_target_size(uint32_t width, uint32_t height, const Qwen35VisionConfig& config,
                               uint32_t& target_width, uint32_t& target_height, std::string* error = nullptr);

class Qwen35VisionRunner {
public:
    // `loader` tiene que seguir vivo mientras exista el runner.
    Qwen35VisionRunner(const HnfLoader& loader, const Qwen35VisionConfig& config);

    const Qwen35VisionConfig& config() const { return config_; }

    // Pasa la imagen por la torre en la GPU activa y deja en `output` (memoria de esa GPU,
    // con sitio para max_tokens() × out_hidden valores FP16) los vectores de la rejilla.
    bool run(const Gemma4RgbView& image, half* output, Qwen35ImageGrid& grid,
             cudaStream_t stream, std::string* error = nullptr);

    // Solo para comparar con la referencia: la salida de la última imagen, en FP32.
    const std::vector<float>& last_output() const { return last_output_; }
    bool keep_output = false;

private:
    const TensorEntry& tensor(const std::string& name) const;

    const HnfLoader& loader_;
    Qwen35VisionConfig config_;
    std::vector<float> last_output_;
};

}  // namespace helios
