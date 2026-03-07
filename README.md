# HELIOS Engine — Héctor v1

Custom CUDA inference engine for transformer models. Built from scratch in C++/CUDA without PyTorch, TensorFlow, or llama.cpp. Everything from memory management to attention kernels is handwritten.

## What is this

Héctor is the core inference engine of the HELIOS project. It loads models in a proprietary binary format (HNF — HELIOS Neural Format), builds a compute graph from the weight tensors it finds, and runs the forward pass entirely on GPU using custom CUDA kernels.

The engine is architecture-agnostic: it detects the model structure (attention type, MLP variant, normalization, RoPE style) from the tensor names in the HNF file and builds the correct graph automatically. No architecture-specific code paths — one generic pipeline handles everything.

## Features

**Engine core:**
- Custom tensor registry with named tensors on GPU
- Scratch memory pool with bump allocation (zero fragmentation)
- Command buffer pattern — build graph once, execute many times
- CUDA Graph capture and replay for autoregressive decode
- Device-side cache position for true single-capture graphs

**CUDA kernels (23 registered ops):**
- Elementwise: add, multiply, scale, copy, bias
- Activations: SiLU, GELU, GELU-new, fused SiLU×mul, fused GELU×mul
- Normalization: RMSNorm, LayerNorm, fused add+RMSNorm
- Linear: matmul via cuBLAS (FP16, FP32) and custom HQS quantized matmul
- Attention: full multi-head attention, cached attention with KV update
- Positional: RoPE (standard, LLaMA-3 scaled, LongRoPE, YaRN, dynamic NTK)
- Sampling: argmax, temperature+top-k+top-p on GPU
- Utilities: embedding lookup, QKV split, half split

**Quantization (HQS — HELIOS Quantization System):**
- HQ4K — 4-bit quantized with per-block scales (32 elements/block)
- HQ5K — 5-bit quantized with per-block scales
- HQ41K / HQ51K — 1K-block variants
- Fused dequant-matmul kernels (weights stay quantized in memory)

**Model format (HNF v9):**
- Binary container with 16 block slots (text, vision, audio, cortex, code, tokenizer, etc.)
- Block table at fixed offset — O(1) block lookup
- Embedded tokenizer (HTF — HELIOS Tokenizer Format) with BPE, special tokens, and chat templates
- Multimodal support: combine text + vision + code models in one file
- Execution hints (JSON or binary) for architecture metadata

**Architecture support:**
- Qwen2 (ChatML template)
- Phi-3 / Phi-4 (LongRoPE, partial rotary)
- DeepSeek Coder
- Falcon (MQA)
- Any standard transformer that follows the detected patterns (fused/separate QKV, gated/plain MLP, RMSNorm/LayerNorm)

**Tokenizer (HTF):**
- BPE with byte-fallback
- Special token handling (BOS, EOS, pad, chat markers)
- Multi-domain vocabulary support

**KV Cache:**
- Pre-allocated for max sequence length
- GQA support (num_kv_heads ≠ num_heads)
- Per-layer cache with batch dimension

## What's not done yet

- No HTTP/API server — runs as test binaries only
- No multi-GPU / tensor parallelism
- No continuous batching
- No streaming output
- No GGUF/safetensors import (requires conversion to HNF)
- CUDA Graph replay limited to fixed-topology decode steps
- No Windows support (Linux only)

## Requirements

- Linux (tested on Ubuntu 22.04/24.04)
- NVIDIA GPU (Turing or newer — sm_75+)
- CUDA Toolkit 12.x
- CMake 3.18+
- g++ with C++17 support

## Build

```bash
git clone https://github.com/YOUR_USER/helios-engine.git
cd helios-engine
mkdir build && cd build
cmake ..
make -j$(nproc)
```

Default CUDA architectures: `sm_75` (Turing/2080Ti), `sm_86` (Ampere/3090), `sm_89` (Ada/4070). Edit `CMAKE_CUDA_ARCHITECTURES` in CMakeLists.txt if needed.

## Usage

The engine requires models converted to HNF format. Test binaries are built automatically:

```bash
# Run unit tests
./test_tensor
./test_memory
./test_command
./test_engine
./test_kernels
./test_hnf_loader

# Smoke test (progressive 4-phase validation)
./test_smoke <path_to_model.hnf>

# Interactive chat
./test_chat <path_to_model.hnf> text "What is the meaning of life?"

# Generation with KV cache
./test_generate_kv <path_to_model.hnf> text "Once upon a time" 128

# Multimodal generation
./test_generate_modal <path_to_model.hnf> cortex "Explain quantum computing"
```

## Project structure

```
helios-engine/
├── CMakeLists.txt
├── src/
│   ├── engine.hpp/cpp          — Core engine (execution, kernel registry, CUDA graphs)
│   ├── tensor.hpp/cpp          — Tensor registry (named tensors on GPU)
│   ├── memory.hpp/cpp          — Scratch memory pool (bump allocator)
│   ├── command.hpp/cpp         — Command buffer (op graph)
│   ├── graph_builder.hpp/cpp   — Auto-detect architecture, build forward pass
│   ├── hnf_loader.hpp/cpp      — HNF v9 model loader
│   ├── htf_tokenizer.hpp/cpp   — BPE tokenizer (HTF format)
│   ├── kv_cache.hpp            — KV cache for autoregressive decode
│   ├── sampler.hpp/cpp         — GPU sampling (temperature, top-k, top-p)
│   ├── dtype.hpp/cpp           — Data type registry (FP16, FP32, HQ4K, HQ5K)
│   └── optype.hpp/cpp          — Operation type registry
├── kernels/
│   ├── kernels.hpp             — Kernel declarations
│   ├── register_kernels.cpp    — All 23 kernel registrations
│   ├── elementwise.cu          — Add, mul, scale, copy
│   ├── activations.cu          — SiLU, GELU, fused variants
│   ├── normalization.cu        — RMSNorm, LayerNorm
│   ├── fused_add_rmsnorm.cu    — Fused residual + RMSNorm
│   ├── linear.cu               — Linear projection dispatch
│   ├── matmul_cublas.cu        — cuBLAS FP16/FP32 matmul
│   ├── matmul_hqs.cu           — HQS quantized matmul (4-bit, 5-bit)
│   ├── matmul_hqs_compact.cu   — Compact HQS variant
│   ├── attention.cu            — Multi-head attention + cached attention
│   ├── sampling.cu             — Argmax, temperature, top-k/p
│   ├── hqs_common.cuh          — Shared HQS dequant helpers
│   └── quantize_q8.cuh         — Q8 quantization utilities
├── tests/                      — Unit tests (tensor, memory, command, engine, kernels, HNF)
├── test_*.cpp                  — Integration tests (generation, chat, multimodal, smoke)
└── verify_*.py                 — Python weight verification scripts
```

## License

MIT
