#include "gemma4_vision_preprocess.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace helios;

namespace {

struct Fixture {
    const char* name;
    uint32_t width;
    uint32_t height;
    uint32_t target_width;
    uint32_t target_height;
    uint32_t soft_tokens;
};

constexpr Fixture kFixtures[] = {
    {"aligned", 960, 672, 960, 672, 280},
    {"square", 512, 512, 768, 768, 256},
    {"portrait", 333, 1000, 432, 1344, 252},
    {"panorama", 1600, 300, 1824, 336, 266},
    {"extreme", 5000, 1, 13440, 48, 280},
};

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        std::exit(1);
    }
}

std::vector<uint8_t> make_fixture(uint32_t width, uint32_t height,
                                  size_t row_padding = 0) {
    const size_t stride = static_cast<size_t>(width) * 3 + row_padding;
    std::vector<uint8_t> result(stride * height, 0xa5);
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            uint8_t* pixel = result.data() + static_cast<size_t>(y) * stride + x * 3;
            pixel[0] = static_cast<uint8_t>(17u * x + 3u * y + (x ^ y));
            pixel[1] = static_cast<uint8_t>(5u * x + 29u * y + ((x * y) >> 5));
            pixel[2] = static_cast<uint8_t>(11u * x + 7u * y + ((x >> 2) ^ (y << 1)));
        }
    }
    return result;
}

void check_positions_and_padding(const Gemma4VisionPreprocessResult& result) {
    require(result.position_ids.size() == static_cast<size_t>(result.max_patches) * 2,
            "position tensor size");
    for (uint32_t patch = 0; patch < result.real_patches; ++patch) {
        require(result.position_ids[static_cast<size_t>(patch) * 2] ==
                    static_cast<int32_t>(patch % result.patch_columns),
                "position x ordering");
        require(result.position_ids[static_cast<size_t>(patch) * 2 + 1] ==
                    static_cast<int32_t>(patch / result.patch_columns),
                "position y ordering");
    }
    for (uint32_t patch = result.real_patches; patch < result.max_patches; ++patch) {
        require(result.position_ids[static_cast<size_t>(patch) * 2] == -1 &&
                    result.position_ids[static_cast<size_t>(patch) * 2 + 1] == -1,
                "padded positions must be -1");
        const float* values = result.pixel_values.data() +
            static_cast<size_t>(patch) * result.patch_values;
        for (uint32_t value = 0; value < result.patch_values; ++value) {
            require(values[value] == 0.0f, "padded patches must be zero");
        }
    }
}

Gemma4VisionPreprocessResult run_fixture(const Fixture& fixture,
                                         size_t row_padding = 0) {
    const std::vector<uint8_t> rgb =
        make_fixture(fixture.width, fixture.height, row_padding);
    Gemma4RgbView view;
    view.data = rgb.data();
    view.width = fixture.width;
    view.height = fixture.height;
    view.row_stride_bytes = static_cast<size_t>(fixture.width) * 3 + row_padding;
    Gemma4VisionPreprocessResult result;
    std::string error;
    require(gemma4_vision_preprocess_rgb(
                view, Gemma4VisionPreprocessConfig{}, result, &error),
            fixture.name + std::string(": ") + error);
    require(result.resized_width == fixture.target_width &&
                result.resized_height == fixture.target_height,
            fixture.name + std::string(": target dimensions"));
    require(result.soft_tokens == fixture.soft_tokens,
            fixture.name + std::string(": dynamic soft-token count"));
    require(result.real_patches == result.patch_columns * result.patch_rows,
            fixture.name + std::string(": real patch geometry"));
    require(result.max_patches == 2520 && result.patch_values == 768,
            fixture.name + std::string(": fixed output geometry"));
    require(result.pixel_values.size() == static_cast<size_t>(2520) * 768,
            fixture.name + std::string(": padded patch tensor size"));
    check_positions_and_padding(result);
    return result;
}

void test_geometry_and_layout() {
    for (const Fixture& fixture : kFixtures) run_fixture(fixture);

    // A padded source row must produce the same aligned, no-resize patches.
    const Gemma4VisionPreprocessResult tight = run_fixture(kFixtures[0]);
    const Gemma4VisionPreprocessResult padded = run_fixture(kFixtures[0], 17);
    require(tight.pixel_values == padded.pixel_values,
            "RGB row stride must not alter patches");
    require(tight.position_ids == padded.position_ids,
            "RGB row stride must not alter positions");

    Gemma4VisionPreprocessResult invalid_result;
    std::string error;
    Gemma4RgbView invalid{};
    require(!gemma4_vision_preprocess_rgb(
                invalid, Gemma4VisionPreprocessConfig{}, invalid_result, &error),
            "null RGB input must be rejected");
    require(!error.empty(), "invalid RGB input must report an error");
    std::cout << "PASS: Gemma 4 geometry, HWC patches, XY positions and padding"
              << std::endl;
}

std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    require(input.is_open(), "cannot open oracle fixture: " + path);
    const std::streamsize size = input.tellg();
    require(size >= 0, "cannot size oracle fixture: " + path);
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(bytes.data()), size);
    require(input.good(), "cannot read oracle fixture: " + path);
    return bytes;
}

void compare_oracle(const std::string& directory) {
    const float scale = Gemma4VisionPreprocessConfig{}.rescale_factor;
    for (const Fixture& fixture : kFixtures) {
        const Gemma4VisionPreprocessResult result = run_fixture(fixture);
        const std::string path = directory + "/" + fixture.name + ".rgb";
        const std::vector<uint8_t> expected = read_file(path);
        require(expected.size() ==
                    static_cast<size_t>(fixture.target_width) * fixture.target_height * 3,
                fixture.name + std::string(": oracle RGB byte count"));

        size_t mismatches = 0;
        int maximum_byte_error = 0;
        for (uint32_t patch = 0; patch < result.real_patches; ++patch) {
            const uint32_t patch_x = patch % result.patch_columns;
            const uint32_t patch_y = patch / result.patch_columns;
            const float* values = result.pixel_values.data() +
                static_cast<size_t>(patch) * result.patch_values;
            size_t value = 0;
            for (uint32_t inner_y = 0; inner_y < 16; ++inner_y) {
                for (uint32_t inner_x = 0; inner_x < 16; ++inner_x) {
                    const uint32_t x = patch_x * 16 + inner_x;
                    const uint32_t y = patch_y * 16 + inner_y;
                    const uint8_t* pixel = expected.data() +
                        (static_cast<size_t>(y) * result.resized_width + x) * 3;
                    for (uint32_t channel = 0; channel < 3; ++channel) {
                        const int actual_byte = static_cast<int>(
                            std::lround(values[value++] / scale));
                        const int difference = std::abs(actual_byte - pixel[channel]);
                        maximum_byte_error = std::max(maximum_byte_error, difference);
                        mismatches += difference != 0;
                    }
                }
            }
        }
        require(mismatches == 0,
                std::string(fixture.name) + ": resize differs from ATen at " +
                    std::to_string(mismatches) + " channels; max error=" +
                    std::to_string(maximum_byte_error));
        std::cout << "PASS: " << fixture.name << " resize byte-identical to ATen"
                  << std::endl;
    }
}

} // namespace

int main(int argc, char** argv) {
    test_geometry_and_layout();
    if (argc == 2) compare_oracle(argv[1]);
    if (argc > 2) {
        std::cerr << "Usage: test_gemma4_vision_preprocess [oracle_directory]"
                  << std::endl;
        return 2;
    }
    return 0;
}
