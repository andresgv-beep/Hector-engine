// kernels/register_kernels.cpp
// ============================================================================
// KERNEL REGISTRATION - Connects kernels to Engine
// ============================================================================
// Diseño polimórfico: kernels seleccionados según dtype de tensores
//

#include "kernels.hpp"
#include "dtype.hpp"
#include "optype.hpp"
#include <stdexcept>
#include <cmath>

namespace helios {
namespace kernels {

// ============================================================================
// HELPER: Get tensor as typed pointer WITH VALIDATION
// ============================================================================

inline half* as_fp16(TensorInfo* t) {
    if (!t) return nullptr;
    if (t->dtype != dtype::FP16()) {
        throw std::runtime_error(
            "Expected FP16 tensor but got " + std::string(dtype_name(t->dtype)) +
            " — check model dtype vs kernel dispatch"
        );
    }
    return static_cast<half*>(t->ptr);
}

inline const half* as_fp16_const(TensorInfo* t) {
    if (!t) return nullptr;
    if (t->dtype != dtype::FP16()) {
        throw std::runtime_error(
            "Expected FP16 tensor but got " + std::string(dtype_name(t->dtype)) +
            " — check model dtype vs kernel dispatch"
        );
    }
    return static_cast<const half*>(t->ptr);
}

// Quantized tensors: no dtype check (handled by matmul dispatch)
inline uint8_t* as_u8(TensorInfo* t) {
    if (!t) return nullptr;
    return static_cast<uint8_t*>(t->ptr);
}

inline const uint8_t* as_u8_const(TensorInfo* t) {
    if (!t) return nullptr;
    return static_cast<const uint8_t*>(t->ptr);
}

inline int32_t* as_i32(TensorInfo* t) {
    if (!t) return nullptr;
    return static_cast<int32_t*>(t->ptr);
}

// ============================================================================
// ELEMENTWISE KERNEL REGISTRATION
// ============================================================================

void register_elementwise_kernels(Engine& engine) {
    
    // ADD
    engine.register_kernel(op::ADD(), [](ExecContext& ctx, const Command& cmd) {
        TensorInfo* a = ctx.in(0);
        TensorInfo* b = ctx.in(1);
        TensorInfo* c = ctx.output;
        
        if (!a || !b || !c) {
            throw std::runtime_error("ADD: missing tensors");
        }
        
        launch_add_fp16(
            as_fp16_const(a), as_fp16_const(b), as_fp16(c),
            a->numel(),
            ctx.stream
        );
    });
    
    // ADD_BIAS (broadcast 1D bias to 2D/3D tensor)
    engine.register_kernel(op::ADD_BIAS(), [](ExecContext& ctx, const Command& cmd) {
        TensorInfo* input = ctx.in(0);
        TensorInfo* bias = ctx.in(1);
        TensorInfo* output = ctx.output;
        
        if (!input || !bias || !output) {
            throw std::runtime_error("ADD_BIAS: missing tensors");
        }
        
        if (!input->ptr || !bias->ptr || !output->ptr) {
            throw std::runtime_error("ADD_BIAS: null GPU pointer");
        }
        
        size_t dim = bias->numel();
        size_t total = input->numel();
        size_t input_last_dim = input->shape.back();
        
        if (dim != input_last_dim) {
            throw std::runtime_error("ADD_BIAS: dim mismatch - bias=" + 
                std::to_string(dim) + " input_last_dim=" + std::to_string(input_last_dim));
        }
        
        launch_add_bias_fp16(
            as_fp16_const(input), as_fp16_const(bias), as_fp16(output),
            total, dim,
            ctx.stream
        );
    });
    
    // MUL
    engine.register_kernel(op::MUL(), [](ExecContext& ctx, const Command& cmd) {
        TensorInfo* a = ctx.in(0);
        TensorInfo* b = ctx.in(1);
        TensorInfo* c = ctx.output;
        
        if (!a || !b || !c) {
            throw std::runtime_error("MUL: missing tensors");
        }
        
        launch_mul_fp16(
            as_fp16_const(a), as_fp16_const(b), as_fp16(c),
            a->numel(),
            ctx.stream
        );
    });
    
    // SCALE
    engine.register_kernel(op::SCALE(), [](ExecContext& ctx, const Command& cmd) {
        TensorInfo* input = ctx.in(0);
        TensorInfo* output = ctx.output;
        float scalar = cmd.get<float>("scalar", 1.0f);
        
        if (!input || !output) {
            throw std::runtime_error("SCALE: missing tensors");
        }
        
        launch_scale_fp16(
            as_fp16_const(input), as_fp16(output),
            scalar,
            input->numel(),
            ctx.stream
        );
    });
    
    // COPY
    engine.register_kernel(op::COPY(), [](ExecContext& ctx, const Command& cmd) {
        TensorInfo* src = ctx.in(0);
        TensorInfo* dst = ctx.output;
        
        if (!src || !dst) {
            throw std::runtime_error("COPY: missing tensors");
        }
        
        launch_copy_fp16(
            as_fp16_const(src), as_fp16(dst),
            src->numel(),
            ctx.stream
        );
    });
}

// ============================================================================
// ACTIVATION KERNEL REGISTRATION
// ============================================================================

void register_activation_kernels(Engine& engine) {
    
    // SILU
    engine.register_kernel(op::SILU(), [](ExecContext& ctx, const Command& cmd) {
        TensorInfo* input = ctx.in(0);
        TensorInfo* output = ctx.output;
        
        if (!input || !output) {
            throw std::runtime_error("SILU: missing tensors");
        }
        
        launch_silu_fp16(
            as_fp16_const(input), as_fp16(output),
            input->numel(),
            ctx.stream
        );
    });
    
    // GELU
    engine.register_kernel(op::GELU(), [](ExecContext& ctx, const Command& cmd) {
        TensorInfo* input = ctx.in(0);
        TensorInfo* output = ctx.output;
        
        if (!input || !output) {
            throw std::runtime_error("GELU: missing tensors");
        }
        
        launch_gelu_fp16(
            as_fp16_const(input), as_fp16(output),
            input->numel(),
            ctx.stream
        );
    });
    
    // SILU_MUL — fused silu(gate) * up
    {
        auto silu_mul_id = OpTypeRegistry::Builder("silu_mul")
            .category("activation").inputs(2, 2).requires_output(true).build();
        engine.register_kernel(silu_mul_id, [](ExecContext& ctx, const Command& cmd) {
            TensorInfo* gate = ctx.in(0);
            TensorInfo* up = ctx.in(1);
            TensorInfo* output = ctx.output;
            if (!gate || !up || !output) {
                throw std::runtime_error("SILU_MUL: missing tensors");
            }
            launch_silu_mul_fp16(
                as_fp16_const(gate), as_fp16_const(up), as_fp16(output),
                gate->numel(), ctx.stream
            );
        });
    }
    
    // GELU_MUL — fused gelu(gate) * up
    {
        auto gelu_mul_id = OpTypeRegistry::Builder("gelu_mul")
            .category("activation").inputs(2, 2).requires_output(true).build();
        engine.register_kernel(gelu_mul_id, [](ExecContext& ctx, const Command& cmd) {
            TensorInfo* gate = ctx.in(0);
            TensorInfo* up = ctx.in(1);
            TensorInfo* output = ctx.output;
            if (!gate || !up || !output) {
                throw std::runtime_error("GELU_MUL: missing tensors");
            }
            launch_gelu_mul_fp16(
                as_fp16_const(gate), as_fp16_const(up), as_fp16(output),
                gate->numel(), ctx.stream
            );
        });
    }
    
    // SOFTMAX
    engine.register_kernel(op::SOFTMAX(), [](ExecContext& ctx, const Command& cmd) {
        TensorInfo* input = ctx.in(0);
        TensorInfo* output = ctx.output;
        
        if (!input || !output) {
            throw std::runtime_error("SOFTMAX: missing tensors");
        }
        
        int batch = 1;
        for (size_t i = 0; i < input->shape.size() - 1; i++) {
            batch *= input->shape[i];
        }
        int seq_len = input->shape.back();
        
        launch_softmax_fp16(
            as_fp16_const(input), as_fp16(output),
            batch, seq_len,
            ctx.stream
        );
    });
}

// ============================================================================
// NORMALIZATION KERNEL REGISTRATION
// ============================================================================

void register_norm_kernels(Engine& engine) {
    
    // RMSNORM
    engine.register_kernel(op::RMSNORM(), [](ExecContext& ctx, const Command& cmd) {
        TensorInfo* input = ctx.in(0);
        TensorInfo* weight = ctx.in(1);
        TensorInfo* output = ctx.output;
        float eps = cmd.get<float>("eps", 1e-5f);
        
        if (!input || !weight || !output) {
            throw std::runtime_error("RMSNORM: missing tensors");
        }

        // "dim" explícito permite normalizar sub-filas (p.ej. QK-norm por-head:
        // tensor [1, H*HD] con peso [HD] → H filas de HD)
        int dim = (int)cmd.get<uint32_t>("dim", 0);
        if (dim == 0) dim = input->shape.back();
        int batch = input->numel() / dim;
        
        launch_rmsnorm_fp16(
            as_fp16_const(input), as_fp16_const(weight), as_fp16(output),
            batch, dim, eps,
            ctx.stream
        );
    });
    

    // ADD_RMSNORM — fused add + rmsnorm
    {
        auto add_rmsnorm_id = OpTypeRegistry::Builder("add_rmsnorm")
            .build();
        engine.register_kernel(add_rmsnorm_id, [](ExecContext& ctx, const Command& cmd) {
            TensorInfo* a = ctx.in(0);
            TensorInfo* b = ctx.in(1);
            TensorInfo* weight = ctx.in(2);
            TensorInfo* normed = ctx.output;
            float eps = cmd.get<float>("eps", 1e-5f);
            std::string res_name = cmd.get<std::string>("output_residual", "");
            TensorInfo* residual = ctx.registry->get(res_name);
            if (!a || !b || !weight || !normed || !residual)
                throw std::runtime_error("ADD_RMSNORM: missing tensors");
            int dim = a->shape.back();
            int batch = a->numel() / dim;
            launch_fused_add_rmsnorm_fp16(
                as_fp16_const(a), as_fp16_const(b), as_fp16_const(weight),
                as_fp16(residual), as_fp16(normed),
                batch, dim, eps, ctx.stream);
        });
    }

    // LAYERNORM
    engine.register_kernel(op::LAYERNORM(), [](ExecContext& ctx, const Command& cmd) {
        TensorInfo* input = ctx.in(0);
        TensorInfo* weight = ctx.in(1);
        TensorInfo* bias = ctx.num_inputs() > 2 ? ctx.in(2) : nullptr;
        TensorInfo* output = ctx.output;
        float eps = cmd.get<float>("eps", 1e-5f);
        
        if (!input || !weight || !output) {
            throw std::runtime_error("LAYERNORM: missing tensors");
        }
        
        int dim = input->shape.back();
        int batch = input->numel() / dim;
        
        launch_layernorm_fp16(
            as_fp16_const(input), as_fp16_const(weight),
            bias ? as_fp16_const(bias) : nullptr,
            as_fp16(output),
            batch, dim, eps,
            ctx.stream
        );
    });
}

// ============================================================================
// LINEAR (MATMUL) KERNEL REGISTRATION
// ============================================================================

void register_linear_kernels(Engine& engine) {
    
    // MATMUL - polymorphic based on weight dtype
    // Standard transformer convention: weight is [out_features, in_features] = [N, K]
    // So: input [M, K] @ weight.T [K, N] = output [M, N]
    engine.register_kernel(op::MATMUL(), [](ExecContext& ctx, const Command& cmd) {
        TensorInfo* input = ctx.in(0);
        TensorInfo* weight = ctx.in(1);
        TensorInfo* output = ctx.output;
        
        if (!input || !weight || !output) {
            throw std::runtime_error("MATMUL: missing tensors");
        }
        
        // Deduce dimensions
        int N = weight->shape[0];
        int K_weight = weight->shape.size() > 1 ? weight->shape[1] : weight->shape[0];
        
        int M, K_input;
        if (input->shape.size() <= 2) {
            M = input->shape[0];
            K_input = input->shape.size() > 1 ? input->shape[1] : 1;
        } else {
            M = 1;
            for (size_t i = 0; i < input->shape.size() - 1; i++) {
                M *= input->shape[i];
            }
            K_input = input->shape.back();
        }
        
        // Override M with actual seq_len if provided (scratch tensors have max shape)
        uint32_t seq_len = cmd.get<uint32_t>("seq_len", 0);
        if (seq_len > 0) {
            M = static_cast<int>(seq_len);
        }

        int K = K_input;

        // row_offset: procesar solo desde esa fila del input. Lo usa el
        // lm_head en prefill — de S posiciones solo importa la última, y
        // calcular vocab×S logits para tirar S-1 es el mayor desperdicio del
        // prefill (además obliga a descuantizar la matriz entera a un búfer
        // temporal enorme). Con esto, el lm_head del prefill es un GEMV.
        const half* in_ptr = as_fp16_const(input);
        uint32_t row_offset = cmd.get<uint32_t>("row_offset", 0);
        if (row_offset > 0) in_ptr += (size_t)row_offset * K;

        // Select kernel based on weight dtype
        if (weight->dtype == dtype::HQ4K()) {
            launch_matmul_hq4k(
                in_ptr, as_u8_const(weight), as_fp16(output),
                M, K, N,
                ctx.stream
            );
        } else if (weight->dtype == dtype::HQ5K()) {
            launch_matmul_hq5k(
                in_ptr, as_u8_const(weight), as_fp16(output),
                M, K, N,
                ctx.stream
            );
        } else if (weight->dtype == dtype::HQ41K()) {
            launch_matmul_hq41k(
                in_ptr, as_u8_const(weight), as_fp16(output),
                M, K, N,
                ctx.stream
            );
        } else if (weight->dtype == dtype::HQ51K()) {
            launch_matmul_hq51k(
                in_ptr, as_u8_const(weight), as_fp16(output),
                M, K, N,
                ctx.stream
            );
        } else {
            launch_matmul_fp16(
                in_ptr, as_fp16_const(weight), as_fp16(output),
                M, K, N,
                ctx.stream
            );
        }
    });
    
    // MATMUL_T (B transposed) - reserved
    engine.register_kernel(op::MATMUL_T(), [](ExecContext& ctx, const Command& cmd) {
        throw std::runtime_error("MATMUL_T not yet implemented");
    });
}

// ============================================================================
// MEMORY KERNEL REGISTRATION
// ============================================================================

void register_memory_kernels(Engine& engine) {
    
    // EMBEDDING
    engine.register_kernel(op::EMBEDDING(), [](ExecContext& ctx, const Command& cmd) {
        TensorInfo* indices = ctx.in(0);
        TensorInfo* table = ctx.in(1);
        TensorInfo* output = ctx.output;
        
        if (!indices || !table || !output) {
            throw std::runtime_error("EMBEDDING: missing tensors");
        }
        
        int batch = indices->shape[0];
        int seq = indices->shape.size() > 1 ? indices->shape[1] : 1;
        int vocab = table->shape[0];
        int dim = table->shape[1];
        
        launch_embedding_fp16(
            as_i32(indices), as_fp16_const(table), as_fp16(output),
            batch, seq, vocab, dim,
            ctx.stream
        );
    });
    
    // SPLIT_QKV - Zero-copy for decode (batch_seq=1), kernel for prefill
    engine.register_kernel(op::SPLIT_QKV(), [](ExecContext& ctx, const Command& cmd) {
        TensorInfo* qkv_fused = ctx.in(0);
        TensorInfo* q_out = ctx.output;
        
        if (!qkv_fused || !q_out) {
            throw std::runtime_error("SPLIT_QKV: missing tensors");
        }
        
        std::string k_name = cmd.get<std::string>("output_k", "");
        std::string v_name = cmd.get<std::string>("output_v", "");
        uint32_t q_size = cmd.get<uint32_t>("q_size", 0);
        uint32_t k_size = cmd.get<uint32_t>("k_size", 0);
        uint32_t v_size = cmd.get<uint32_t>("v_size", 0);
        
        TensorInfo* k_out = ctx.registry->get(k_name);
        TensorInfo* v_out = ctx.registry->get(v_name);
        
        if (!k_out || !v_out) {
            throw std::runtime_error("SPLIT_QKV: missing output tensors k=" + k_name + " v=" + v_name);
        }
        
        size_t fused_dim = q_size + k_size + v_size;
        uint32_t sl = cmd.get<uint32_t>("seq_len", 0);
        int batch_seq = (sl > 0) ? static_cast<int>(sl) : static_cast<int>(qkv_fused->numel() / fused_dim);
        
        if (batch_seq == 1) {
            // Zero-copy: point outputs into fused tensor
            half* base = as_fp16(qkv_fused);
            q_out->ptr = base;
            k_out->ptr = base + q_size;
            v_out->ptr = base + q_size + k_size;
        } else {
            // Prefill: need actual copy due to interleaved layout
            launch_split_3way(
                as_fp16_const(qkv_fused),
                as_fp16(q_out), as_fp16(k_out), as_fp16(v_out),
                batch_seq, q_size, k_size, v_size,
                ctx.stream
            );
        }
    });
    
    // SPLIT_HALF - Zero-copy for decode (batch_seq=1), kernel for prefill
    engine.register_kernel(op::SPLIT_HALF(), [](ExecContext& ctx, const Command& cmd) {
        TensorInfo* fused = ctx.in(0);
        TensorInfo* first = ctx.output;
        
        if (!fused || !first) {
            throw std::runtime_error("SPLIT_HALF: missing tensors");
        }
        
        std::string second_name = cmd.get<std::string>("output_second", "");
        uint32_t split_size = cmd.get<uint32_t>("split_size", 0);
        
        TensorInfo* second = ctx.registry->get(second_name);
        
        if (!second) {
            throw std::runtime_error("SPLIT_HALF: missing second output tensor: " + second_name);
        }
        
        size_t fused_dim = 2 * split_size;
        uint32_t sl = cmd.get<uint32_t>("seq_len", 0);
        int batch_seq = (sl > 0) ? static_cast<int>(sl) : static_cast<int>(fused->numel() / fused_dim);
        
        if (batch_seq == 1) {
            // Zero-copy: point outputs into fused tensor
            half* base = as_fp16(fused);
            first->ptr = base;
            second->ptr = base + split_size;
        } else {
            // Prefill: need actual copy
            launch_split_2way(
                as_fp16_const(fused),
                as_fp16(first), as_fp16(second),
                batch_seq, split_size, split_size,
                ctx.stream
            );
        }
    });
}

// ============================================================================
// ATTENTION KERNEL REGISTRATION
// ============================================================================

void register_attention_kernels(Engine& engine) {
    
    // ATTENTION - Full attention (prefill)
    engine.register_kernel(op::ATTENTION(), [](ExecContext& ctx, const Command& cmd) {
        TensorInfo* q = ctx.in(0);
        TensorInfo* k = ctx.in(1);
        TensorInfo* v = ctx.in(2);
        TensorInfo* output = ctx.output;
        
        if (!q || !k || !v || !output) {
            throw std::runtime_error("ATTENTION: missing tensors");
        }
        
        uint32_t num_heads = cmd.get<uint32_t>("num_heads", 32);
        uint32_t num_kv_heads = cmd.get<uint32_t>("num_kv_heads", num_heads);
        uint32_t head_dim = cmd.get<uint32_t>("head_dim", 128);
        float scale = cmd.get<float>("scale", 1.0f / std::sqrt(float(head_dim)));
        bool causal = cmd.get<bool>("causal", true);
        
        int batch = q->shape[0];
        int seq_q = cmd.get<uint32_t>("seq_q", 0);
        int seq_kv = cmd.get<uint32_t>("seq_kv", 0);
        
        if (seq_q == 0) seq_q = q->shape.size() > 1 ? q->shape[1] : 1;
        if (seq_kv == 0) seq_kv = k->shape.size() > 1 ? k->shape[1] : seq_q;
        
        launch_attention_fp16(
            as_fp16_const(q), as_fp16_const(k), as_fp16_const(v),
            as_fp16(output),
            batch, seq_q, seq_kv,
            num_heads, num_kv_heads, head_dim,
            scale, causal,
            ctx.stream
        );
    });
    
    // ROPE - Rotary Position Embedding
    engine.register_kernel(op::ROPE(), [&engine](ExecContext& ctx, const Command& cmd) {
        TensorInfo* qk = ctx.in(0);
        TensorInfo* output = ctx.output;
        
        if (!qk || !output) {
            throw std::runtime_error("ROPE: missing tensors");
        }
        
        float theta = cmd.get<float>("theta", 10000.0f);
        uint32_t head_dim = cmd.get<uint32_t>("dim", 128);
        uint32_t offset = cmd.get<uint32_t>("offset", 0);
        float partial_rotary = cmd.get<float>("partial_rotary", 1.0f);
        float scaling_factor = cmd.get<float>("rope_scaling_factor", 1.0f);
        
        int rotary_dim = (int)(head_dim * partial_rotary);
        rotary_dim = (rotary_dim / 2) * 2;
        if (rotary_dim <= 0) rotary_dim = head_dim;
        
        // Copy input to output if different pointers
        if (qk->ptr != output->ptr) {
            cudaMemcpyAsync(output->ptr, qk->ptr, qk->size_bytes, 
                           cudaMemcpyDeviceToDevice, ctx.stream);
        }
        
        // Prefer explicit params over shape-deduction (scratch shapes may be oversized)
        uint32_t explicit_heads = cmd.get<uint32_t>("num_heads", (uint32_t)0);
        uint32_t explicit_seq = cmd.get<uint32_t>("seq_len", (uint32_t)0);
        
        int batch = 1;
        int seq = 1;
        int num_heads;
        
        if (explicit_heads > 0) {
            num_heads = explicit_heads;
            if (explicit_seq > 0) {
                seq = explicit_seq;
            } else if (qk->shape.size() >= 3) {
                seq = qk->shape[1];
            } else if (qk->shape.size() == 2) {
                seq = qk->shape[0];
            }
            if (qk->shape.size() >= 3) batch = qk->shape[0];
        } else {
            if (qk->shape.size() >= 3) {
                batch = qk->shape[0];
                seq = qk->shape[1];
                num_heads = qk->shape[2] / head_dim;
            } else if (qk->shape.size() == 2) {
                batch = 1;
                num_heads = qk->shape[1] / head_dim;
                seq = qk->shape[0];
            } else {
                int total_elements = qk->size_bytes / sizeof(half);
                num_heads = total_elements / head_dim;
            }
        }
        
        // device_pos: offset leído de device (permite CUDA Graph capture-once)
        if (cmd.get<uint32_t>("device_pos", 0) && engine.has_device_cache_pos()) {
            launch_rope_inplace_fp16_dp(
                    as_fp16(output),
                    batch, seq, num_heads, head_dim,
                    rotary_dim,
                    engine.device_cache_pos(), theta,
                    scaling_factor,
                    ctx.stream
                );
        } else {
            launch_rope_inplace_fp16(
                    as_fp16(output),
                    batch, seq, num_heads, head_dim,
                    rotary_dim,
                    offset, theta,
                    scaling_factor,
                    ctx.stream
                );
        }
    });
    
    // QK_NORM_ROPE — fusión Qwen3: rmsnorm por-head (q,k) + rope (q,k) en 1 kernel
    {
        auto qk_norm_rope_id = OpTypeRegistry::Builder("qk_norm_rope").build();
        engine.register_kernel(qk_norm_rope_id, [&engine](ExecContext& ctx, const Command& cmd) {
            TensorInfo* q = ctx.in(0);
            TensorInfo* k = ctx.in(1);
            TensorInfo* qw = ctx.in(2);
            TensorInfo* kw = ctx.in(3);
            if (!q || !k || !qw || !kw) {
                throw std::runtime_error("QK_NORM_ROPE: missing tensors");
            }

            uint32_t H = cmd.get<uint32_t>("num_heads", 32);
            uint32_t KVH = cmd.get<uint32_t>("num_kv_heads", H);
            uint32_t HD = cmd.get<uint32_t>("dim", 128);
            uint32_t seq_len = cmd.get<uint32_t>("seq_len", 1);
            uint32_t offset = cmd.get<uint32_t>("offset", 0);
            float eps = cmd.get<float>("eps", 1e-6f);
            float theta = cmd.get<float>("theta", 10000.0f);
            float partial_rotary = cmd.get<float>("partial_rotary", 1.0f);
            float scaling_factor = cmd.get<float>("rope_scaling_factor", 1.0f);

            int rotary_dim = (int)(HD * partial_rotary);
            rotary_dim = (rotary_dim / 2) * 2;
            if (rotary_dim <= 0) rotary_dim = HD;

            if (cmd.get<uint32_t>("device_pos", 0) && engine.has_device_cache_pos()) {
                launch_qk_norm_rope_fp16_dp(
                    as_fp16(q), as_fp16(k), as_fp16_const(qw), as_fp16_const(kw),
                    1, seq_len, H, KVH, HD, rotary_dim,
                    engine.device_cache_pos(), eps, theta, scaling_factor,
                    ctx.stream);
            } else {
                launch_qk_norm_rope_fp16(
                    as_fp16(q), as_fp16(k), as_fp16_const(qw), as_fp16_const(kw),
                    1, seq_len, H, KVH, HD, rotary_dim,
                    offset, eps, theta, scaling_factor,
                    ctx.stream);
            }
        });
    }

    // ATTENTION_CACHED - Attention using KV cache (decode)
    engine.register_kernel(op::ATTENTION_CACHED(), [&engine](ExecContext& ctx, const Command& cmd) {
        TensorInfo* q = ctx.in(0);
        TensorInfo* k_cache = ctx.in(1);
        TensorInfo* v_cache = ctx.in(2);
        TensorInfo* output = ctx.output;
        
        if (!q || !k_cache || !v_cache || !output) {
            throw std::runtime_error("ATTENTION_CACHED: missing tensors");
        }
        
        uint32_t num_heads = cmd.get<uint32_t>("num_heads", 32);
        uint32_t num_kv_heads = cmd.get<uint32_t>("num_kv_heads", num_heads);
        uint32_t head_dim = cmd.get<uint32_t>("head_dim", 128);
        float scale = cmd.get<float>("scale", 1.0f / std::sqrt(float(head_dim)));
        uint32_t seq_len = cmd.get<uint32_t>("seq_len", 1);
        uint32_t max_seq_len = cmd.get<uint32_t>("max_seq_len", 2048);
        
        // device_pos: total_seq leído de device (permite CUDA Graph capture-once)
        if (cmd.get<uint32_t>("device_pos", 0) && engine.has_device_cache_pos()) {
            launch_attention_cached_fp16_dp(
                    as_fp16_const(q),
                    as_fp16_const(k_cache),
                    as_fp16_const(v_cache),
                    as_fp16(output),
                    1, engine.device_total_seq(),
                    num_heads, num_kv_heads, head_dim,
                    max_seq_len, scale,
                    ctx.stream
                );
        } else {
            launch_attention_cached_fp16(
                    as_fp16_const(q),
                    as_fp16_const(k_cache),
                    as_fp16_const(v_cache),
                    as_fp16(output),
                    1, seq_len,
                    num_heads, num_kv_heads, head_dim,
                    max_seq_len, scale,
                    ctx.stream
                );
        }
    });
    
    // ATTENTION_PREFILL_CACHED — S_new queries sobre el cache completo (causal)
    {
        auto prefill_id = OpTypeRegistry::Builder("attention_prefill_cached").build();
        engine.register_kernel(prefill_id, [](ExecContext& ctx, const Command& cmd) {
            TensorInfo* q = ctx.in(0);
            TensorInfo* k_cache = ctx.in(1);
            TensorInfo* v_cache = ctx.in(2);
            TensorInfo* output = ctx.output;
            if (!q || !k_cache || !v_cache || !output) {
                throw std::runtime_error("ATTENTION_PREFILL_CACHED: missing tensors");
            }

            uint32_t num_heads = cmd.get<uint32_t>("num_heads", 32);
            uint32_t num_kv_heads = cmd.get<uint32_t>("num_kv_heads", num_heads);
            uint32_t head_dim = cmd.get<uint32_t>("head_dim", 128);
            float scale = cmd.get<float>("scale", 1.0f / std::sqrt(float(head_dim)));
            uint32_t seq_new = cmd.get<uint32_t>("seq_len", 1);
            uint32_t past_len = cmd.get<uint32_t>("past_len", 0);
            uint32_t max_seq_len = cmd.get<uint32_t>("max_seq_len", 2048);

            launch_attention_prefill_cached_fp16(
                as_fp16_const(q), as_fp16_const(k_cache), as_fp16_const(v_cache),
                as_fp16(output),
                (int)seq_new, (int)past_len,
                (int)num_heads, (int)num_kv_heads, (int)head_dim,
                (int)max_seq_len, scale, ctx.stream);
        });
    }

    // KV_CACHE_UPDATE - Update KV cache with new K/V
    engine.register_kernel(op::KV_CACHE_UPDATE(), [&engine](ExecContext& ctx, const Command& cmd) {
        TensorInfo* k_new = ctx.in(0);
        TensorInfo* v_new = ctx.in(1);
        TensorInfo* k_cache = ctx.in(2);
        TensorInfo* v_cache = ctx.in(3);
        
        if (!k_new || !v_new || !k_cache || !v_cache) {
            throw std::runtime_error("KV_CACHE_UPDATE: missing tensors");
        }
        
        uint32_t position = cmd.get<uint32_t>("position", 0);
        uint32_t max_seq_len = cmd.get<uint32_t>("max_seq_len", 2048);
        uint32_t kv_heads = cmd.get<uint32_t>("num_kv_heads", 2);
        uint32_t head_dim = cmd.get<uint32_t>("head_dim", 128);
        
        // Use explicit seq_len from command (preferred — scratch tensors are oversized)
        int batch = 1;
        int seq_len = cmd.get<uint32_t>("seq_len", (uint32_t)0);
        
        if (seq_len == 0) {
            // Fallback: deduce from tensor shape
            if (k_new->shape.size() >= 3) {
                batch = k_new->shape[0];
                seq_len = k_new->shape[1];
            } else if (k_new->shape.size() == 2) {
                int total = k_new->size_bytes / sizeof(half);
                int per_token = kv_heads * head_dim;
                seq_len = total / (per_token * batch);
                if (seq_len < 1) seq_len = 1;
            }
        } else {
            if (k_new->shape.size() >= 3) batch = k_new->shape[0];
        }
        
        // device_pos: position leído de device (permite CUDA Graph capture-once)
        if (cmd.get<uint32_t>("device_pos", 0) && engine.has_device_cache_pos()) {
            launch_kv_cache_update_dp(
                    as_fp16_const(k_new),
                    as_fp16_const(v_new),
                    as_fp16(k_cache),
                    as_fp16(v_cache),
                    batch, seq_len, kv_heads, head_dim,
                    max_seq_len, engine.device_cache_pos(),
                    ctx.stream
                );
        } else {
            launch_kv_cache_update(
                    as_fp16_const(k_new),
                    as_fp16_const(v_new),
                    as_fp16(k_cache),
                    as_fp16(v_cache),
                    batch, seq_len, kv_heads, head_dim,
                    max_seq_len, position,
                    ctx.stream
                );
        }
    });
}

// ============================================================================
// REGISTER ALL
// ============================================================================

void register_all_kernels(Engine& engine) {
    register_elementwise_kernels(engine);
    register_activation_kernels(engine);
    register_norm_kernels(engine);
    register_linear_kernels(engine);
    register_memory_kernels(engine);
    register_attention_kernels(engine);
}

} // namespace kernels
} // namespace helios
