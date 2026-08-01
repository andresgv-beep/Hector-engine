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

constexpr int HEADER_SIZE = 128;       // HQ4K/HQ5K: 32 groups × 4 bytes

constexpr int HQ4K_PAYLOAD = 128;
constexpr int HQ4K_BLOCK_SIZE = 256;

constexpr int HQ5K_PAYLOAD = 160;
constexpr int HQ5K_BLOCK_SIZE = 288;

constexpr float HQ4K_Q_MAX = 15.0f;
constexpr float HQ5K_Q_MAX = 31.0f;
constexpr float HQ3K_Q_MAX = 7.0f;

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
constexpr int HQ41K_BLOCK_SIZE = COMPACT_HEADER_SIZE + HQ4K_PAYLOAD;  // 168
constexpr int HQ51K_BLOCK_SIZE = COMPACT_HEADER_SIZE + HQ5K_PAYLOAD;  // 200

// ============================================================================
// GROUP PARAMS (in registers)
// ============================================================================

struct __align__(4) GroupParams {
    half min;
    half scale;
};

// ============================================================================
// DEVICE FUNCTIONS - HQ4K/HQ5K Header decoding (128-byte)
// ============================================================================

__device__ __forceinline__ 
GroupParams decode_group(const uint8_t* header, int group_idx) {
    const uint8_t* ptr = header + group_idx * 4;
    GroupParams gp;
    gp.min = __ushort_as_half(ptr[0] | (ptr[1] << 8));
    gp.scale = __ushort_as_half(ptr[2] | (ptr[3] << 8));
    return gp;
}

__device__ __forceinline__
void decode_all_groups(const uint8_t* header, GroupParams* params) {
    #pragma unroll
    for (int g = 0; g < NUM_GROUPS; g++) {
        params[g] = decode_group(header, g);
    }
}

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

// ============================================================================
// DEVICE FUNCTIONS - HQ4K dequantization
// ============================================================================

__device__ __forceinline__
void unpack_hq4k_byte(uint8_t packed, uint8_t& q0, uint8_t& q1) {
    q0 = (packed >> 4) & 0x0F;
    q1 = packed & 0x0F;
}

__device__ __forceinline__
float dequant_hq4k_element(uint8_t q, GroupParams gp) {
    float min_f = __half2float(gp.min);
    float scale_f = __half2float(gp.scale);
    return min_f + (float(q) / HQ4K_Q_MAX) * scale_f;
}

__device__ __forceinline__
void dequant_hq4k_group(
    const uint8_t* payload, int group_idx, GroupParams gp, float* out
) {
    int byte_offset = group_idx * 4;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint8_t packed = payload[byte_offset + i];
        uint8_t q0, q1;
        unpack_hq4k_byte(packed, q0, q1);
        out[i * 2] = dequant_hq4k_element(q0, gp);
        out[i * 2 + 1] = dequant_hq4k_element(q1, gp);
    }
}

// ============================================================================
// DEVICE FUNCTIONS - HQ5K dequantization
// ============================================================================

__device__ __forceinline__
void unpack_hq5k_chunk(const uint8_t* bytes, uint8_t* q) {
    uint64_t bits = uint64_t(bytes[0]) 
                  | (uint64_t(bytes[1]) << 8)
                  | (uint64_t(bytes[2]) << 16)
                  | (uint64_t(bytes[3]) << 24)
                  | (uint64_t(bytes[4]) << 32);
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        q[k] = (bits >> (k * 5)) & 0x1F;
    }
}

__device__ __forceinline__
float dequant_hq5k_element(uint8_t q, GroupParams gp) {
    float min_f = __half2float(gp.min);
    float scale_f = __half2float(gp.scale);
    return min_f + (float(q) / HQ5K_Q_MAX) * scale_f;
}

__device__ __forceinline__
void dequant_hq5k_group(
    const uint8_t* payload, int group_idx, GroupParams gp, float* out
) {
    int byte_offset = group_idx * 5;
    uint8_t q[8];
    unpack_hq5k_chunk(payload + byte_offset, q);
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        out[i] = dequant_hq5k_element(q[i], gp);
    }
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
