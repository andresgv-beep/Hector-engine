#include "gemma4_vision_preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace helios {
namespace {

void set_error(std::string* error, const std::string& message) {
    if (error) *error = message;
}

bool checked_mul(size_t a, size_t b, size_t& result) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) return false;
    result = a * b;
    return true;
}

bool checked_mul_u64(uint64_t a, uint64_t b, uint64_t& result) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) return false;
    result = a * b;
    return true;
}

double cubic_filter(double value) {
    constexpr double a = -0.5; // Keys cubic, PyTorch/Pillow antialias path
    const double x = std::abs(value);
    if (x < 1.0) {
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    }
    if (x < 2.0) {
        return (((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a);
    }
    return 0.0;
}

struct AxisWeights {
    uint32_t input_size = 0;
    uint32_t output_size = 0;
    uint32_t max_taps = 0;
    uint32_t precision = 0;
    std::vector<uint32_t> first;
    std::vector<uint32_t> count;
    std::vector<int16_t> weights;
};

bool make_axis_weights(uint32_t input_size, uint32_t output_size,
                       AxisWeights& axis) {
    if (input_size == 0 || output_size == 0) return false;
    axis = AxisWeights{};
    axis.input_size = input_size;
    axis.output_size = output_size;
    const double scale = static_cast<double>(input_size) / output_size;
    const double support = scale >= 1.0 ? 2.0 * scale : 2.0;
    axis.max_taps = static_cast<uint32_t>(std::ceil(support)) * 2 + 1;
    axis.first.resize(output_size);
    axis.count.resize(output_size);
    std::vector<double> floating(
        static_cast<size_t>(output_size) * axis.max_taps, 0.0);

    double global_max = 0.0;
    const double inverse_scale = scale >= 1.0 ? 1.0 / scale : 1.0;
    for (uint32_t output = 0; output < output_size; ++output) {
        const double center = scale * (static_cast<double>(output) + 0.5);
        int64_t first = static_cast<int64_t>(center - support + 0.5);
        first = std::max<int64_t>(first, 0);
        int64_t count = std::min<int64_t>(
            static_cast<int64_t>(center + support + 0.5), input_size) - first;
        count = std::clamp<int64_t>(count, 0, axis.max_taps);
        if (count == 0) return false;
        axis.first[output] = static_cast<uint32_t>(first);
        axis.count[output] = static_cast<uint32_t>(count);

        double total = 0.0;
        double* row = floating.data() +
            static_cast<size_t>(output) * axis.max_taps;
        for (int64_t tap = 0; tap < count; ++tap) {
            const double distance =
                (static_cast<double>(tap + first) - center + 0.5) * inverse_scale;
            row[tap] = cubic_filter(distance);
            total += row[tap];
        }
        if (total == 0.0) return false;
        for (int64_t tap = 0; tap < count; ++tap) {
            row[tap] /= total;
            global_max = std::max(global_max, row[tap]);
        }
    }

    for (axis.precision = 0; axis.precision < 22; ++axis.precision) {
        const int next = static_cast<int>(
            0.5 + global_max * static_cast<double>(uint32_t{1} <<
                                                   (axis.precision + 1)));
        if (next >= (1 << 15)) break;
    }
    if (axis.precision == 0 || axis.precision > 22) return false;
    axis.weights.resize(floating.size());
    const double multiplier =
        static_cast<double>(uint32_t{1} << axis.precision);
    for (size_t index = 0; index < floating.size(); ++index) {
        const double scaled = floating[index] * multiplier;
        axis.weights[index] = static_cast<int16_t>(
            scaled < 0.0 ? static_cast<int>(scaled - 0.5)
                         : static_cast<int>(scaled + 0.5));
    }
    return true;
}

uint8_t fixed_point_sample(const uint8_t* source, size_t stride,
                           const AxisWeights& axis, uint32_t output) {
    int value = 1 << (axis.precision - 1);
    const int16_t* weights = axis.weights.data() +
        static_cast<size_t>(output) * axis.max_taps;
    for (uint32_t tap = 0; tap < axis.count[output]; ++tap) {
        value += source[static_cast<size_t>(axis.first[output] + tap) * stride] *
                 weights[tap];
    }
    value >>= axis.precision;
    return static_cast<uint8_t>(std::clamp(value, 0, 255));
}

bool resize_bicubic_aa_rgb(const Gemma4RgbView& source,
                           uint32_t target_width, uint32_t target_height,
                           std::vector<uint8_t>& output) {
    const size_t source_stride = source.row_stride_bytes != 0
        ? source.row_stride_bytes : static_cast<size_t>(source.width) * 3;
    size_t output_bytes = 0;
    if (!checked_mul(target_width, target_height, output_bytes) ||
        !checked_mul(output_bytes, 3, output_bytes)) return false;
    output.resize(output_bytes);

    if (source.width == target_width && source.height == target_height) {
        const size_t row_bytes = static_cast<size_t>(source.width) * 3;
        for (uint32_t row = 0; row < source.height; ++row) {
            std::copy_n(source.data + static_cast<size_t>(row) * source_stride,
                        row_bytes,
                        output.data() + static_cast<size_t>(row) * row_bytes);
        }
        return true;
    }

    AxisWeights horizontal;
    AxisWeights vertical;
    if (!make_axis_weights(source.width, target_width, horizontal) ||
        !make_axis_weights(source.height, target_height, vertical)) return false;

    size_t intermediate_bytes = 0;
    if (!checked_mul(source.height, target_width, intermediate_bytes) ||
        !checked_mul(intermediate_bytes, 3, intermediate_bytes)) return false;
    std::vector<uint8_t> intermediate(intermediate_bytes);

    for (uint32_t row = 0; row < source.height; ++row) {
        const uint8_t* input_row =
            source.data + static_cast<size_t>(row) * source_stride;
        uint8_t* output_row = intermediate.data() +
            static_cast<size_t>(row) * target_width * 3;
        for (uint32_t column = 0; column < target_width; ++column) {
            for (uint32_t channel = 0; channel < 3; ++channel) {
                output_row[static_cast<size_t>(column) * 3 + channel] =
                    fixed_point_sample(input_row + channel, 3, horizontal, column);
            }
        }
    }

    const size_t intermediate_stride = static_cast<size_t>(target_width) * 3;
    for (uint32_t row = 0; row < target_height; ++row) {
        for (uint32_t column = 0; column < target_width; ++column) {
            for (uint32_t channel = 0; channel < 3; ++channel) {
                const uint8_t* first = intermediate.data() +
                    static_cast<size_t>(column) * 3 + channel;
                output[(static_cast<size_t>(row) * target_width + column) * 3 +
                       channel] = fixed_point_sample(
                           first, intermediate_stride, vertical, row);
            }
        }
    }
    return true;
}

} // namespace

bool gemma4_vision_target_size(
    uint32_t source_width, uint32_t source_height,
    const Gemma4VisionPreprocessConfig& config,
    uint32_t& target_width, uint32_t& target_height,
    std::string* error) {
    target_width = 0;
    target_height = 0;
    if (source_width == 0 || source_height == 0 || config.patch_size == 0 ||
        config.pooling_kernel_size == 0 || config.max_soft_tokens == 0) {
        set_error(error, "source dimensions and Gemma 4 geometry must be non-zero");
        return false;
    }

    uint64_t side_multiple = 0;
    uint64_t max_patches = 0;
    uint64_t target_pixels = 0;
    if (!checked_mul_u64(config.patch_size, config.pooling_kernel_size,
                         side_multiple) ||
        !checked_mul_u64(config.max_soft_tokens, config.pooling_kernel_size,
                         max_patches) ||
        !checked_mul_u64(max_patches, config.pooling_kernel_size,
                         max_patches) ||
        !checked_mul_u64(max_patches, config.patch_size, target_pixels) ||
        !checked_mul_u64(target_pixels, config.patch_size, target_pixels) ||
        max_patches > std::numeric_limits<uint32_t>::max()) {
        set_error(error, "Gemma 4 preprocessing geometry overflows");
        return false;
    }
    const double source_pixels =
        static_cast<double>(source_width) * source_height;
    const double factor = std::sqrt(
        static_cast<double>(target_pixels) / source_pixels);
    uint64_t height = static_cast<uint64_t>(std::floor(
        factor * source_height / side_multiple)) * side_multiple;
    uint64_t width = static_cast<uint64_t>(std::floor(
        factor * source_width / side_multiple)) * side_multiple;

    if (height == 0 && width == 0) {
        set_error(error, "Gemma 4 resize rounded both dimensions to zero");
        return false;
    }
    const uint64_t max_side =
        (max_patches / (static_cast<uint64_t>(config.pooling_kernel_size) *
                        config.pooling_kernel_size)) * side_multiple;
    if (height == 0) {
        height = side_multiple;
        width = std::min<uint64_t>(
            (static_cast<uint64_t>(source_width) / source_height) * side_multiple,
            max_side);
    } else if (width == 0) {
        width = side_multiple;
        height = std::min<uint64_t>(
            (static_cast<uint64_t>(source_height) / source_width) * side_multiple,
            max_side);
    }
    if (height == 0 || width == 0 || height * width > target_pixels ||
        height > std::numeric_limits<uint32_t>::max() ||
        width > std::numeric_limits<uint32_t>::max()) {
        set_error(error, "Gemma 4 target geometry is invalid or exceeds its patch budget");
        return false;
    }
    target_width = static_cast<uint32_t>(width);
    target_height = static_cast<uint32_t>(height);
    return true;
}

bool gemma4_vision_preprocess_rgb(
    const Gemma4RgbView& image,
    const Gemma4VisionPreprocessConfig& config,
    Gemma4VisionPreprocessResult& result,
    std::string* error) {
    result = Gemma4VisionPreprocessResult{};
    if (!image.data) {
        set_error(error, "RGB image pointer is null");
        return false;
    }
    const size_t tight_stride = static_cast<size_t>(image.width) * 3;
    if (image.row_stride_bytes != 0 && image.row_stride_bytes < tight_stride) {
        set_error(error, "RGB row stride is smaller than width * 3");
        return false;
    }
    if (!std::isfinite(config.rescale_factor) || config.rescale_factor <= 0.0f) {
        set_error(error, "Gemma 4 rescale factor must be finite and positive");
        return false;
    }

    uint32_t resized_width = 0;
    uint32_t resized_height = 0;
    if (!gemma4_vision_target_size(image.width, image.height, config,
                                   resized_width, resized_height, error)) {
        return false;
    }
    std::vector<uint8_t> resized;
    if (!resize_bicubic_aa_rgb(image, resized_width, resized_height, resized)) {
        set_error(error, "Gemma 4 bicubic resize failed");
        return false;
    }

    result.source_width = image.width;
    result.source_height = image.height;
    result.resized_width = resized_width;
    result.resized_height = resized_height;
    result.patch_columns = resized_width / config.patch_size;
    result.patch_rows = resized_height / config.patch_size;
    result.real_patches = result.patch_columns * result.patch_rows;
    result.max_patches = config.max_soft_tokens * config.pooling_kernel_size *
                         config.pooling_kernel_size;
    result.soft_tokens = result.real_patches /
        (config.pooling_kernel_size * config.pooling_kernel_size);
    const uint64_t patch_values = static_cast<uint64_t>(config.patch_size) *
                                  config.patch_size * 3;
    if (patch_values > std::numeric_limits<uint32_t>::max()) {
        set_error(error, "Gemma 4 patch width overflows");
        result = Gemma4VisionPreprocessResult{};
        return false;
    }
    result.patch_values = static_cast<uint32_t>(patch_values);
    if (result.real_patches > result.max_patches ||
        result.real_patches % (config.pooling_kernel_size *
                               config.pooling_kernel_size) != 0) {
        set_error(error, "Gemma 4 resized image violates its patch budget");
        result = Gemma4VisionPreprocessResult{};
        return false;
    }

    size_t value_count = 0;
    if (!checked_mul(result.max_patches, result.patch_values, value_count)) {
        set_error(error, "Gemma 4 patch tensor size overflows");
        result = Gemma4VisionPreprocessResult{};
        return false;
    }
    result.pixel_values.assign(value_count, 0.0f);
    result.position_ids.assign(static_cast<size_t>(result.max_patches) * 2, -1);

    for (uint32_t patch_y = 0; patch_y < result.patch_rows; ++patch_y) {
        for (uint32_t patch_x = 0; patch_x < result.patch_columns; ++patch_x) {
            const uint32_t patch_index = patch_y * result.patch_columns + patch_x;
            result.position_ids[static_cast<size_t>(patch_index) * 2] = patch_x;
            result.position_ids[static_cast<size_t>(patch_index) * 2 + 1] = patch_y;
            float* destination = result.pixel_values.data() +
                static_cast<size_t>(patch_index) * result.patch_values;
            size_t value = 0;
            for (uint32_t inner_y = 0; inner_y < config.patch_size; ++inner_y) {
                const uint32_t pixel_y = patch_y * config.patch_size + inner_y;
                for (uint32_t inner_x = 0; inner_x < config.patch_size; ++inner_x) {
                    const uint32_t pixel_x = patch_x * config.patch_size + inner_x;
                    const uint8_t* pixel = resized.data() +
                        (static_cast<size_t>(pixel_y) * resized_width + pixel_x) * 3;
                    destination[value++] = pixel[0] * config.rescale_factor;
                    destination[value++] = pixel[1] * config.rescale_factor;
                    destination[value++] = pixel[2] * config.rescale_factor;
                }
            }
        }
    }
    return true;
}

} // namespace helios
