"""Control FP16 de la cola MoE: separa el redondeo de FP16 de un error del puerto.

Repite la cola con numpy redondeando a FP16 tras cada operación, como hace el
motor, y compara motor y control contra el oráculo FP32. Si el motor falla un
umbral absoluto igual que el control, el fallo es de FP16.

uso: moe_fp16_control.py references/ references/layer-00-oracle-fp32 salidas/
salidas/ se genera con: test_gemma4_moe_layer references/ salidas/
"""
import json,struct,sys
from pathlib import Path
import numpy as np
ref,orc,mot,tol=Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]),2e-4
m=json.loads((orc/'manifest.json').read_text())['outputs']
O={k:np.fromfile(orc/v['file'],dtype=v['dtype']).reshape(v['shape']) for k,v in m.items()}
M={k:np.fromfile(mot/v['file'],dtype=v['dtype']).reshape(v['shape']) for k,v in m.items()}
f=open(ref/'layer-00-original/layer.safetensors','rb');n=struct.unpack('<Q',f.read(8))[0];hd=json.loads(f.read(n));base=8+n
def bf16(name,start=0,count=None):
    i=hd['model.language_model.layers.0.'+name];a,b=i['data_offsets']
    f.seek(base+a+start)
    raw=np.frombuffer(f.read((b-a-start) if count is None else count),dtype='<u2')
    return (raw.astype(np.uint32)<<16).view(np.float32).reshape(-1)
h16=lambda a:np.asarray(a,np.float32).astype(np.float16).astype(np.float32)
eps=1e-6
def W(name,shape=None):
    v=h16(bf16(name)); return v if shape is None else v.reshape(shape)
def rms(x,w=None):
    y=x*np.power(np.mean(np.square(x),-1,keepdims=True,dtype=np.float32)+eps,-0.5,dtype=np.float32)
    return h16(y if w is None else y*w)
H,I,E,K=2816,704,128,8
gelu=lambda x:0.5*x*(1+np.tanh(0.79788456*(x+0.044715*x*x*x),dtype=np.float32))
C={}
r=h16(O['residual'])
a=C['shared_input']=rms(r,W('pre_feedforward_layernorm.weight'))
g=h16(a@W('mlp.gate_proj.weight',(2112,H)).T);u=h16(a@W('mlp.up_proj.weight',(2112,H)).T)
sh=h16(h16(h16(gelu(g))*u)@W('mlp.down_proj.weight',(H,2112)).T)
C['shared_branch_post']=rms(sh,W('post_feedforward_layernorm_1.weight'))
ri=C['router_input']=h16(rms(r,W('router.scale'))*(H**-0.5))
lg=C['router_logits']=h16(ri@W('router.proj.weight',(E,H)).T)
b=C['expert_input']=rms(r,W('pre_feedforward_layernorm_2.weight'))
ids=O['expert_ids'];wt=M['expert_weights']
G=np.zeros((4,K,I),np.float32);U=G.copy();A=G.copy();Y=np.zeros((4,K,H),np.float32)
cache={}
for t in range(4):
    for s in range(K):
        e=int(ids[t,s])
        if e not in cache:
            gu=h16(bf16('experts.gate_up_proj',e*2*I*H*2,2*I*H*2)).reshape(2*I,H)
            dw=h16(bf16('experts.down_proj',e*H*I*2,H*I*2)).reshape(H,I)
            cache[e]=(gu,dw)
        gu,dw=cache[e]
        o=h16(b[t]@gu.T);G[t,s]=o[:I];U[t,s]=o[I:]
        A[t,s]=h16(gelu(o[:I])*o[I:]);Y[t,s]=h16(A[t,s]@dw.T)*wt[t,s]
C['gate']=G;C['up']=U;C['gelu_mul']=A;C['expert_contributions']=Y
C['expert_sum']=h16(Y.sum(1))
C['expert_branch_post']=rms(C['expert_sum'],W('post_feedforward_layernorm_2.weight'))
C['combined_before_common_norm']=h16(C['shared_branch_post']+C['expert_branch_post'])
C['common_norm']=rms(C['combined_before_common_norm'],W('post_feedforward_layernorm.weight'))
C['output']=h16(h16(r+C['common_norm'])*W('layer_scalar'))
def st(x,y):
    e=np.abs(x-y);bad=int((e>tol+tol*np.abs(y)).sum())
    return bad,float(np.sqrt((e.astype(np.float64)**2).sum()/(y.astype(np.float64)**2).sum())),float(e.max())
print(f"{'tensor':30}{'n':>7} | {'FALLOS motor':>12}{'error':>10} | {'FALLOS ctrl':>12}{'error':>10} | {'motor/ctrl':>10} {'max_abs':>9}")
worst=0
for k in ['shared_input','shared_branch_post','router_input','router_logits','expert_input','gate','up','gelu_mul','expert_contributions','expert_sum','expert_branch_post','combined_before_common_norm','common_norm','output']:
    bm,rm,mm=st(M[k],O[k]);bc,rc,mc=st(C[k],O[k]);ratio=rm/rc if rc else 0;worst=max(worst,ratio)
    print(f"{k:30}{O[k].size:7d} | {bm:12d}{rm:10.2e} | {bc:12d}{rc:10.2e} | {ratio:10.2f} {mm:9.2e}")
print('peor razon motor/control',round(worst,2))
print('umbral minimo por FP16: un ulp en [1,2) =',2**-10,'; mitad de ulp =',2**-11,'; atol+rtol*1.5 =',2e-4*2.5)
