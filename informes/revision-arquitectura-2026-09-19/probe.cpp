// Read-only architecture probes against the production library.
#include "inference_session.hpp"
#include "hnf_loader.hpp"
#include "sampler.hpp"
#include "kernels.hpp"
#include <cuda_profiler_api.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now()-t).count();
}
static void check(cudaError_t e) {
    if(e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
static double used() {
    size_t f,t; check(cudaMemGetInfo(&f,&t)); return (t-f)/1048576.;
}
static std::string prompt(int repeats) {
    std::string p="Datos de referencia:\n";
    for(int i=0;i<repeats;++i)
        p+="El servidor registra eventos de red, memoria, disco y conexiones activas.\n";
    return p+"Escribe una explicación técnica extensa de al menos mil palabras sobre cómo funciona un sistema operativo. Empieza directamente y desarrolla todos los detalles.";
}
static void inventory(const char* path) {
    helios::HnfLoader l;
    if(!l.open(path)) throw std::runtime_error("metadata load");
    const auto& c=l.config(); const auto& g=l.gemma4_config();
    unsigned hd=0,intermediate=0,locals=0,globals=0;
    for(const auto& layer:g.layers) {
        hd=std::max(hd,layer.head_dim); intermediate=std::max(intermediate,layer.intermediate_size);
    }
    const size_t ple=g.has_flag(helios::GEMMA4_EXT_FLAG_PLE)?g.ple_hidden_size:0;
    const size_t row=5*c.hidden_size()+2*c.num_attention_heads()*hd+
        2*c.num_key_value_heads()*hd+3*intermediate+
        (ple?2*ple*g.layers.size()+2*ple:0);
    std::printf("META heads=%u kv_heads=%u hd_max=%u layers=%zu shared=%u vision_metadata=%d scratch512_MiB=%.3f scratch6144_MiB=%.3f\n",
        c.num_attention_heads(),c.num_key_value_heads(),hd,g.layers.size(),g.num_kv_shared_layers,
        l.has_gemma4_vision_config(),row*512*2/1048576.,row*6144*2/1048576.);
    for(int chunk:{512,6144}) {
        size_t bytes=0;
        for(size_t i=0;i<g.layers.size()-g.num_kv_shared_layers;++i) {
            const auto& layer=g.layers[i];
            size_t slots=layer.is_global_attention()?16384:std::min(16384u,layer.sliding_window+chunk);
            bytes+=slots*layer.kv_heads_or(c.num_key_value_heads())*layer.head_dim*4;
            if(chunk==512) { if(layer.is_global_attention())++globals;else ++locals; }
        }
        std::printf("KV_FORMULA context=16384 chunk=%d MiB=%.3f\n",chunk,bytes/1048576.);
    }
    std::printf("OWNED_KV local_layers=%u global_layers=%u\n",locals,globals);
}
static void sample_probe() {
    cudaStream_t stream;check(cudaStreamCreate(&stream));
    for(int vocab:{151936,262144}) {
        std::vector<half> data(vocab); unsigned state=97;
        for(auto& x:data) { state=1664525u*state+1013904223u; x=__float2half(float(state>>8)/16777216.f*16.f-8.f); }
        half* device;check(cudaMalloc(&device,data.size()*2));
        check(cudaMemcpy(device,data.data(),data.size()*2,cudaMemcpyHostToDevice));
        for(int k:{0,16,64,65,128}) {
            helios::Sampler sampler;sampler.set_seed(42);
            helios::SamplingConfig cfg;cfg.temperature=k?0.7f:0.f;cfg.top_k=k;cfg.top_p=.95f;
            for(int i=0;i<3;++i) sampler.sample(device,vocab,cfg,stream);
            std::vector<double> times;
            for(int i=0;i<11;++i) { auto t=Clock::now();sampler.sample(device,vocab,cfg,stream);times.push_back(elapsed(t)); }
            std::sort(times.begin(),times.end());
            std::printf("SAMPLER vocab=%d temperature=%.1f top_k=%d median_ms=%.6f\n",vocab,cfg.temperature,k,times[5]);
            std::fflush(stdout);
        }
        check(cudaFree(device));
    }
    check(cudaStreamDestroy(stream));
}
int main(int argc,char** argv) {
    try {
        if(argc==2 && std::string(argv[1])=="sampling") {sample_probe();return 0;}
        if(argc<3)return 2;
        std::string mode=argv[2]; inventory(argv[1]);
        if(mode=="metadata")return 0;
        std::string error,code;
        std::printf("MEM phase=initial MiB=%.3f\n",used());
        helios::Model::Config cfg;cfg.hnf_path=argv[1];cfg.max_seq_len=16384;
        auto model=helios::Model::load(cfg,&error);
        if(!model)throw std::runtime_error(error);
        std::printf("MEM phase=model_loaded MiB=%.3f multimodal=%d\n",used(),model->info().multimodal);
        {
            helios::InferenceSession session;
            if(!session.attach(model,&error))throw std::runtime_error(error);
            std::printf("MEM phase=main_attached MiB=%.3f\n",used());
            {
                helios::InferenceSession aux;
                if(!aux.attach(model,&error,1024))throw std::runtime_error(error);
                std::printf("MEM phase=aux_attached MiB=%.3f\n",used());
            }
            const int repeats=std::getenv("AUDIT_REPEAT")?std::atoi(std::getenv("AUDIT_REPEAT")):384;
            auto p=prompt(repeats);
            for(int trial=0;trial<2;++trial) {
                session.reset(); std::atomic<bool> cancel{mode=="cancel"};
                helios::InferenceSession::GenConfig gen;gen.temperature=0;
                gen.max_visible_tokens=128;gen.max_thinking_tokens=0;gen.close_turn=false;
                helios::InferenceSession::TurnStats st;helios::InferenceSession::FinishReason reason;
                const auto start=Clock::now(); double first_ms=-1;size_t bytes=0;
                if(trial==1 && mode=="profile-prefill")check(cudaProfilerStart());
                const bool ok=session.run_turn({{"user",p}},{},gen,
                    [&](const std::string& s){if(first_ms<0)first_ms=elapsed(start);bytes+=s.size();}, {},
                    [&](uint32_t,double){if(trial==1){
                        if(mode=="profile-prefill")check(cudaProfilerStop());
                        if(mode=="profile-decode")check(cudaProfilerStart());
                    }},cancel,&st,&reason,&code,&error);
                if(trial==1 && mode=="profile-decode")check(cudaProfilerStop());
                if(!ok)throw std::runtime_error(code+": "+error);
                std::printf("TURN mode=%s trial=%d prompt=%u tokens=%u prefill_ms=%.3f decode_ms=%.3f first_ms=%.3f total_ms=%.3f reason=%s cache=%u bytes=%zu\n",
                    mode.c_str(),trial,st.prefill_tokens,st.generated_tokens,st.prefill_ms,st.decode_ms,
                    first_ms,elapsed(start),helios::InferenceSession::finish_reason_name(reason),session.cache_position(),bytes);
                std::fflush(stdout);
            }
            std::printf("MEM phase=after_turn MiB=%.3f\n",used());
        }
        std::printf("MEM phase=after_session_destroy MiB=%.3f\n",used());
        model.reset();
        std::printf("MEM phase=after_model_destroy MiB=%.3f\n",used());
        // This process owns no live models/sessions now; isolate global BLAS storage.
        helios::kernels::cleanup_cublas();
        std::printf("MEM phase=after_blas_cleanup MiB=%.3f\n",used());
    }catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
