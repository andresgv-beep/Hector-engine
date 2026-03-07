#!/usr/bin/env python3
import sys, struct, json, numpy as np

SUPER_BLOCK_SIZE = 256
HEADER_SIZE = 128
HQ5K_BLOCK_SIZE = 288
HQ5K_Q_MAX = 31.0

def dequant_hq5k(data, numel):
    num_blocks = len(data) // HQ5K_BLOCK_SIZE
    output = []
    for b in range(num_blocks):
        bs = b * HQ5K_BLOCK_SIZE
        header = data[bs:bs+HEADER_SIZE]
        payload = data[bs+HEADER_SIZE:bs+HQ5K_BLOCK_SIZE]
        for g in range(32):
            ho = g * 4
            mn = struct.unpack('<e', header[ho:ho+2])[0]
            sc = struct.unpack('<e', header[ho+2:ho+4])[0]
            po = g * 5
            bits = 0
            for i in range(5):
                bits |= payload[po+i] << (i*8)
            for i in range(8):
                q = (bits >> (i*5)) & 0x1F
                output.append(mn + (q/HQ5K_Q_MAX)*sc)
    return np.array(output[:numel], dtype=np.float32)

st_path = sys.argv[1]
hnf_path = sys.argv[2]

# Read safetensor
with open(st_path,'rb') as f:
    hs = struct.unpack('<Q', f.read(8))[0]
    hdr = json.loads(f.read(hs))
    doff = 8 + hs

# Read HNF manifest
with open(hnf_path,'rb') as f:
    raw = f.read()
for i in range(len(raw)-1, max(0,len(raw)-5000000), -1):
    if raw[i:i+1] == b'{':
        try:
            txt = raw[i:].decode('utf-8','ignore')
            bc = 0
            for j,c in enumerate(txt):
                if c=='{': bc+=1
                elif c=='}': bc-=1
                if bc==0:
                    manifest = json.loads(txt[:j+1])
                    break
            break
        except: continue

# Build tensor index from manifest
ti = {}
for bl in manifest.get('blocks',[]):
    for t in bl.get('tensors',[]):
        ti[t['name']] = t

# Compare q_proj layer 0
st_name = 'model.layers.0.self_attn.q_proj.weight'
hnf_name = 'code.layer0.attn.q_proj.weight'

meta = hdr[st_name]
s,e = meta['data_offsets']
with open(st_path,'rb') as f:
    f.seek(doff+s); orig_raw = f.read(e-s)
bf16 = np.frombuffer(orig_raw, dtype=np.uint16)
original = (bf16.astype(np.uint32)<<16).view(np.float32)

info = ti[hnf_name]
with open(hnf_path,'rb') as f:
    f.seek(info['offset']); qdata = f.read(info['size'])

recovered = dequant_hq5k(qdata, original.size).reshape(original.shape)

corr = np.corrcoef(original.flatten(), recovered.flatten())[0,1]
diff = np.abs(original - recovered)
print(f"Tensor: {st_name} -> {hnf_name}")
print(f"  Original:  mean={original.mean():.6f} std={original.std():.6f}")
print(f"  Recovered: mean={recovered.mean():.6f} std={recovered.std():.6f}")
print(f"  Correlation: {corr:.6f}")
print(f"  Max error: {diff.max():.6f}")
print(f"  Mean error: {diff.mean():.6f}")
print(f"  first_4 orig: {original.flatten()[:4]}")
print(f"  first_4 recv: {recovered.flatten()[:4]}")
if corr > 0.99:
    print("  ✅ Quantization OK")
else:
    print("  ❌ CORRUPTED!")
