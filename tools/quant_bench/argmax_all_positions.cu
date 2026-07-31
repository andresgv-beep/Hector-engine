// argmax en cada posicion con la ruta generica del grafo (Qwen3 y demas).
#include "src/hnf_loader.hpp"
#include "src/graph_builder.hpp"
#include "src/kv_cache.hpp"
#include "kernels/kernels.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
using namespace helios;
int main(int argc, char** argv) {
    std::vector<int32_t> toks; { std::string cur;
        for (const char* q=argv[2]; ; ++q){ if(*q==','||!*q){toks.push_back(atoi(cur.c_str()));cur.clear();if(!*q)break;} else cur+=*q; } }
    const uint32_t S=(uint32_t)toks.size();
    EngineConfig cfg; cfg.scratch_pool.auto_fraction=0.0f; cfg.scratch_pool.min_size_bytes=0;
    Engine engine(cfg);
    kernels::register_all_kernels(engine);
    HnfLoader L;
    if(!L.open(argv[1])||!L.load_block(BLOCK_TEXT_MODEL,engine)){printf("no carga\n");return 1;}
    const auto config=L.config();
    GraphBuilder b;
    const auto arch=b.detect_architecture(engine,"text",config);
    b.allocate_scratch(engine,config,arch,1,1);
    KVCacheConfig kc;
    kc.num_layers=config.num_hidden_layers(); kc.num_kv_heads=config.num_key_value_heads();
    kc.head_dim=config.head_dim(); kc.max_batch_size=1; kc.max_seq_len=S+8;
    KVCache kv; if(!kv.allocate(kc)){printf("kv fallo\n");return 1;}
    for(uint32_t l=0;l<kc.num_layers;l++){
        TensorInfo ki; ki.ptr=kv.k_cache(l);
        ki.shape={kc.max_batch_size,kc.max_seq_len,kc.num_kv_heads,kc.head_dim};
        ki.dtype=dtype::FP16(); ki.owns_memory=false;
        ki.size_bytes=(size_t)kc.max_batch_size*kc.max_seq_len*kc.num_kv_heads*kc.head_dim*2;
        TensorInfo vi=ki; vi.ptr=kv.v_cache(l);
        engine.tensors().register_tensor("_kv.layer"+std::to_string(l)+".k",ki);
        engine.tensors().register_tensor("_kv.layer"+std::to_string(l)+".v",vi);
    }
    const uint32_t V=config.vocab_size();
    void* tk=engine.tensors().allocate_and_register("t",{1,1},dtype::INT32());
    std::vector<int32_t> am(S); std::vector<half> h(V);
    for(uint32_t p=0;p<S;p++){
        cudaMemcpy(tk,&toks[p],4,cudaMemcpyHostToDevice);
        engine.execute(b.build_forward_cached(engine,config,arch,"t",1,1,"_kv",p,kc.max_seq_len));
        engine.sync();
        auto* lg=b.get_logits(engine);
        cudaMemcpy(h.data(),lg->ptr,(size_t)V*2,cudaMemcpyDeviceToHost);
        uint32_t best=0; float bv=-1e30f;
        for(uint32_t i=0;i<V;i++){float x=__half2float(h[i]); if(x>bv){bv=x;best=i;}}
        am[p]=(int32_t)best;
        if(p%200==0) fprintf(stderr,"\r  %u/%u",p,S);
    }
    fprintf(stderr,"\r  %u/%u\n",S,S);
    FILE* f=fopen(argv[3],"wb"); fwrite(am.data(),4,S,f); fclose(f);
    return 0;
}
