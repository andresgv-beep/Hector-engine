#!/usr/bin/env python3
"""
Compara pesos del safetensor original con el HNF para detectar corrupción.
Uso: python verify_weights.py <safetensor_path> <hnf_path>
"""
import sys
import struct
import json
import numpy as np

def read_safetensor_index(path):
    """Lee el header del safetensor y devuelve metadata de tensores."""
    with open(path, 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        header_json = f.read(header_size)
        header = json.loads(header_json)
        data_offset = 8 + header_size
    return header, data_offset

def read_safetensor_tensor(path, data_offset, meta):
    """Lee un tensor del safetensor."""
    offsets = meta['data_offsets']
    dtype_str = meta['dtype']
    shape = meta['shape']
    
    dtype_map = {
        'F16': np.float16,
        'BF16': np.uint16,  # Read as uint16, convert later
        'F32': np.float32,
    }
    
    np_dtype = dtype_map.get(dtype_str, np.float16)
    start, end = offsets
    
    with open(path, 'rb') as f:
        f.seek(data_offset + start)
        raw = f.read(end - start)
    
    arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape)
    
    # Convert BF16 to FP32
    if dtype_str == 'BF16':
        # BF16: take uint16, shift left 16 bits to make FP32
        arr_u32 = arr.astype(np.uint32) << 16
        arr = arr_u32.view(np.float32)
    elif dtype_str == 'F16':
        arr = arr.astype(np.float32)
    
    return arr

def read_hnf_manifest(hnf_path):
    """Lee el manifest del HNF para encontrar tensores."""
    with open(hnf_path, 'rb') as f:
        magic = f.read(4)
        if magic != b'HNF\x00' and magic != b'HNF\x01':
            # Try reading version
            pass
        # Read full file to find JSON manifest
        f.seek(0)
        data = f.read()
    
    # Find manifest JSON - it's typically after the magic header
    # HNF v9.1 format: magic(4) + version(4) + manifest_offset(8) + ...
    # Let's find it by searching for the JSON start
    # Actually, let's just use the tensor registry from the engine
    return data

def main():
    if len(sys.argv) < 3:
        print("Uso: python verify_weights.py <safetensor_path> <hnf_path>")
        sys.exit(1)
    
    st_path = sys.argv[1]
    hnf_path = sys.argv[2]
    
    print(f"Safetensor: {st_path}")
    print(f"HNF: {hnf_path}")
    
    # ── Read safetensor ──
    header, data_offset = read_safetensor_index(st_path)
    
    # Filter out __metadata__
    tensors = {k: v for k, v in header.items() if k != '__metadata__'}
    
    print(f"\n=== Safetensor: {len(tensors)} tensores ===")
    
    # Show a few tensors
    for name in sorted(tensors.keys())[:5]:
        meta = tensors[name]
        print(f"  {name}: shape={meta['shape']} dtype={meta['dtype']}")
    print("  ...")
    
    # ── Read key tensors from safetensor and show statistics ──
    test_tensors = [
        'model.embed_tokens.weight',
        'lm_head.weight', 
        'model.norm.weight',
        'model.layers.0.self_attn.q_proj.weight',
        'model.layers.0.mlp.gate_proj.weight',
        'model.layers.0.input_layernorm.weight',
    ]
    
    print(f"\n=== Estadísticas de pesos originales (safetensor) ===")
    original_stats = {}
    for name in test_tensors:
        if name not in tensors:
            print(f"  {name}: NO ENCONTRADO")
            continue
        
        arr = read_safetensor_tensor(st_path, data_offset, tensors[name])
        stats = {
            'shape': arr.shape,
            'dtype': tensors[name]['dtype'],
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'first_8': arr.flatten()[:8].tolist(),
            'abs_mean': float(np.mean(np.abs(arr))),
        }
        original_stats[name] = stats
        
        print(f"\n  {name} [{stats['dtype']}] shape={stats['shape']}")
        print(f"    mean={stats['mean']:.6f} std={stats['std']:.6f}")
        print(f"    min={stats['min']:.6f} max={stats['max']:.6f}")
        print(f"    abs_mean={stats['abs_mean']:.6f}")
        print(f"    first_8: {[f'{x:.6f}' for x in stats['first_8']]}")
    
    # ── Now read the FP16 tensors from HNF directly ──
    # The embedding and lm_head are stored as FP16 in HNF
    # We can find them by scanning the HNF binary
    print(f"\n=== Verificación directa: embedding FP16 del HNF ===")
    print("(Los tensores FP16 como embedding/lm_head/norms se pueden comparar directamente)")
    print("(Los tensores HQ4K/HQ5K necesitan dequantización)")
    
    # For FP16 tensors, convert original to FP16 and show
    for name in ['model.embed_tokens.weight', 'lm_head.weight', 'model.norm.weight']:
        if name not in original_stats:
            continue
        arr = read_safetensor_tensor(st_path, data_offset, tensors[name])
        # Convert to FP16 (same as what converter does: BF16→F32→FP16)
        arr_fp16 = arr.astype(np.float16)
        arr_back = arr_fp16.astype(np.float32)
        
        error = np.abs(arr - arr_back)
        print(f"\n  {name}: BF16→F32→FP16 conversion error:")
        print(f"    max_error={float(np.max(error)):.8f}")
        print(f"    mean_error={float(np.mean(error)):.8f}")
        print(f"    first_8 original:  {[f'{x:.6f}' for x in arr.flatten()[:8]]}")
        print(f"    first_8 as FP16:   {[f'{x:.6f}' for x in arr_back.flatten()[:8]]}")
    
    # ── KEY TEST: Check if embed_tokens row 0 matches what HELIOS loads ──
    if 'model.embed_tokens.weight' in tensors:
        arr = read_safetensor_tensor(st_path, data_offset, tensors['model.embed_tokens.weight'])
        arr_fp16 = arr.astype(np.float16)
        
        print(f"\n=== CLAVE: Primeros 16 bytes del embedding (como FP16 raw) ===")
        raw_bytes = arr_fp16.flatten()[:8].tobytes()
        hex_str = ' '.join(f'{b:02x}' for b in raw_bytes)
        print(f"  Hex (primeros 8 elementos = 16 bytes): {hex_str}")
        print(f"  Valores: {arr_fp16.flatten()[:8].astype(np.float32).tolist()}")
        
        print(f"\n  Para verificar en HELIOS, añade este código al test_chat.cpp:")
        print(f"  // Leer primeros 16 bytes del embedding")
        print(f"  uint8_t emb_bytes[16];")
        print(f"  cudaMemcpy(emb_bytes, engine.tensors().get(prefix + \".token_embedding.weight\")->ptr, 16, cudaMemcpyDeviceToHost);")
        print(f"  fprintf(stderr, \"EMB hex: \");")
        print(f"  for(int i=0;i<16;i++) fprintf(stderr, \"%02x \", emb_bytes[i]);")
        print(f"  fprintf(stderr, \"\\n\");")
        print(f"  // Should match: {hex_str}")

if __name__ == '__main__':
    main()
