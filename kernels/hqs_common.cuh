// kernels/hqs_common.cuh
// ============================================================================
// HQS COMMON - Constantes y tipos para kernels CUDA
// ============================================================================
// Matching HQS v6 NUCLEAR format from Rust converter
// + HQ3.1K/HQ4.1K/HQ5.1K compact header formats
//

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace helios {
namespace hqs {

// ============================================================================
// CONSTANTS (matching common.rs)
// ============================================================================

constexpr int SUPER_BLOCK_SIZE = 256;
constexpr int GROUP_SIZE = 8;
constexpr int NUM_GROUPS = 32;


constexpr int PAYLOAD_4BIT = 128;

constexpr int PAYLOAD_5BIT = 160;

constexpr float Q_MAX_4BIT = 15.0f;
constexpr float Q_MAX_5BIT = 31.0f;
constexpr float Q_MAX_6BIT = 63.0f;
constexpr float Q_MAX_3BIT = 7.0f;

constexpr float EPS = 1e-7f;

// ============================================================================
// HQ4.1K / HQ5.1K COMPACT HEADER
// ============================================================================
// Layout (40 bytes):
//   [0..2)   d_scale:  fp16
//   [2..4)   d_min:    fp16
//   [4..6)   min_base: fp16
//   [6..8)   _pad
//   [8..24)  q_scale[32]: 4-bit packed (16 bytes)
//   [24..40) q_min[32]:   4-bit packed (16 bytes)

constexpr int COMPACT_HEADER_SIZE = 40;

constexpr int HQ31K_PAYLOAD = 96;
constexpr int HQ31K_BLOCK_SIZE = COMPACT_HEADER_SIZE + HQ31K_PAYLOAD;  // 136
constexpr int HQ41K_BLOCK_SIZE = COMPACT_HEADER_SIZE + PAYLOAD_4BIT;  // 168
constexpr int HQ51K_BLOCK_SIZE = COMPACT_HEADER_SIZE + PAYLOAD_5BIT;  // 200

// HQ6.2K layout (264 bytes): fp16 d_scale/d_min/min_base, two padding
// bytes, 32 byte-aligned q_scale values, 32 byte-aligned q_min values and
// 192 bytes of 6-bit payload.
constexpr int HQ62K_HEADER_SIZE = 72;
constexpr int HQ62K_PAYLOAD = 192;
constexpr int HQ62K_BLOCK_SIZE = HQ62K_HEADER_SIZE + HQ62K_PAYLOAD;  // 264

// ============================================================================
// DEVICE FUNCTIONS - Compact Header decoding (40-byte)
// ============================================================================

__device__ __forceinline__
void decode_compact_group(
    const uint8_t* block_ptr,
    int group_idx,
    float& min_f,
    float& scoeff,
    float q_max_inv
) {
    float d_scale = __half2float(__ushort_as_half(
        block_ptr[0] | (block_ptr[1] << 8)));
    float d_min = __half2float(__ushort_as_half(
        block_ptr[2] | (block_ptr[3] << 8)));
    float min_base = __half2float(__ushort_as_half(
        block_ptr[4] | (block_ptr[5] << 8)));
    
    int s_byte = 8 + group_idx / 2;
    uint8_t s_packed = block_ptr[s_byte];
    uint8_t q_s = (group_idx % 2 == 0) ? (s_packed >> 4) : (s_packed & 0x0F);
    
    int m_byte = 24 + group_idx / 2;
    uint8_t m_packed = block_ptr[m_byte];
    uint8_t q_m = (group_idx % 2 == 0) ? (m_packed >> 4) : (m_packed & 0x0F);
    
    float scale = d_scale * (float(q_s) * (1.0f / 15.0f));
    min_f = min_base + d_min * (float(q_m) * (1.0f / 15.0f));
    scoeff = scale * q_max_inv;
}

__device__ __forceinline__
void decode_hq62k_group(
    const uint8_t* block_ptr,
    int group_idx,
    float& min_f,
    float& scoeff
) {
    float d_scale = __half2float(__ushort_as_half(
        block_ptr[0] | (block_ptr[1] << 8)));
    float d_min = __half2float(__ushort_as_half(
        block_ptr[2] | (block_ptr[3] << 8)));
    float min_base = __half2float(__ushort_as_half(
        block_ptr[4] | (block_ptr[5] << 8)));
    float scale = d_scale * (float(block_ptr[8 + group_idx]) * (1.0f / 255.0f));
    min_f = min_base + d_min *
        (float(block_ptr[40 + group_idx]) * (1.0f / 255.0f));
    scoeff = scale * (1.0f / Q_MAX_6BIT);
}

// ============================================================================
// HELPER - Block/element indexing
// ============================================================================

__device__ __forceinline__
int element_to_block(int elem_idx, int block_size) {
    return elem_idx / SUPER_BLOCK_SIZE;
}

__device__ __forceinline__
int element_to_group(int elem_idx) {
    return (elem_idx % SUPER_BLOCK_SIZE) / GROUP_SIZE;
}

__device__ __forceinline__
int element_in_group(int elem_idx) {
    return elem_idx % GROUP_SIZE;
}

} // namespace hqs
} // namespace helios
