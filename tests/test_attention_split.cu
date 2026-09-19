#include "kernels.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace helios::kernels;
static void ck(cudaError_t e) { if(e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
static void require(bool v, const char* msg) { if(!v) throw std::runtime_error(msg); }
template<class T> struct Device {
    T* p=nullptr;
    explicit Device(size_t n) { ck(cudaMalloc(&p,n*sizeof(T))); }
    ~Device() { cudaFree(p); }
};
struct Stream { cudaStream_t s; Stream(){ck(cudaStreamCreate(&s));} ~Stream(){cudaStreamDestroy(s);} };
struct Graph {
    cudaGraph_t g=nullptr; cudaGraphExec_t x=nullptr;
    ~Graph(){if(x)cudaGraphExecDestroy(x);if(g)cudaGraphDestroy(g);}
    void finish(cudaStream_t s){ck(cudaStreamEndCapture(s,&g));ck(cudaGraphInstantiate(&x,g,nullptr,nullptr,0));}
};
static float random_value(uint32_t& state) {
    state=1664525u*state+1013904223u;
    return float((state>>8)&65535)/32768.f-1.f;
}
static float measure(cudaStream_t stream, const Graph& graph) {
    cudaEvent_t begin,end; ck(cudaEventCreate(&begin));ck(cudaEventCreate(&end));
    for(int i=0;i<3;++i)ck(cudaGraphLaunch(graph.x,stream));
    ck(cudaEventRecord(begin,stream));
    for(int i=0;i<5;++i)ck(cudaGraphLaunch(graph.x,stream));
    ck(cudaEventRecord(end,stream));ck(cudaEventSynchronize(end));
    float ms;ck(cudaEventElapsedTime(&ms,begin,end));
    ck(cudaEventDestroy(begin));ck(cudaEventDestroy(end));return ms*1000.f/100.f;
}

static void run(int batch,int heads,int kvh,int hd,int window,int slots,bool peaked,bool bench) {
    Stream stream;
    const size_t qn=size_t(batch)*heads*hd, kn=size_t(batch)*slots*kvh*hd;
    std::vector<half> q(qn),k(kn),v(kn),reference(qn),split(qn);
    uint32_t seed=97+hd+kvh;
    for(auto& x:q)x=__float2half(random_value(seed)*(peaked?8.f:0.25f));
    for(auto& x:k)x=__float2half(random_value(seed)*2.f);
    for(auto& x:v)x=__float2half(random_value(seed));
    Device<half> dq(qn),dk(kn),dv(kn),dr(qn),ds(qn);
    Device<int32_t> dn(1);
    Device<float> work(attention_cached_split_workspace_bytes(batch,heads)/sizeof(float));
    ck(cudaMemcpy(dq.p,q.data(),qn*2,cudaMemcpyHostToDevice));
    ck(cudaMemcpy(dk.p,k.data(),kn*2,cudaMemcpyHostToDevice));
    ck(cudaMemcpy(dv.p,v.data(),kn*2,cudaMemcpyHostToDevice));
    const float scale=peaked?1.f:1.f/std::sqrt(float(hd));
    auto launch=[&](bool use_split){
        if(use_split) launch_attention_cached_fp16_split_dp(dq.p,dk.p,dv.p,ds.p,work.p,
            batch,dn.p,heads,kvh,hd,16384,scale,window,stream.s,slots);
        else launch_attention_cached_fp16_dp(dq.p,dk.p,dv.p,dr.p,
            batch,dn.p,heads,kvh,hd,16384,scale,window,stream.s,slots);
    };
    int n=1; ck(cudaMemcpy(dn.p,&n,4,cudaMemcpyHostToDevice));
    Graph parity;
    ck(cudaStreamBeginCapture(stream.s,cudaStreamCaptureModeGlobal));
    launch(false);launch(true);parity.finish(stream.s);
    Graph timings[2];
    if(bench)for(int mode=0;mode<2;++mode){
        ck(cudaStreamBeginCapture(stream.s,cudaStreamCaptureModeGlobal));
        for(int i=0;i<20;++i)launch(mode!=0);
        timings[mode].finish(stream.s);
    }
    for(int length:{1,17,127,1023,1024,1025,1535,1536,1537,4096,8192,12000}) {
        n=length;ck(cudaMemcpy(dn.p,&n,4,cudaMemcpyHostToDevice));
        ck(cudaMemset(work.p,0xff,attention_cached_split_workspace_bytes(batch,heads)));
        ck(cudaGraphLaunch(parity.x,stream.s));ck(cudaStreamSynchronize(stream.s));
        ck(cudaMemcpy(reference.data(),dr.p,qn*2,cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(split.data(),ds.p,qn*2,cudaMemcpyDeviceToHost));
        float maxdiff=0;
        for(size_t i=0;i<qn;++i){
            require(std::isfinite(__half2float(split[i])),"nonfinite output");
            maxdiff=std::max(maxdiff,std::fabs(__half2float(split[i])-__half2float(reference[i])));
        }
        if(std::memcmp(reference.data(),split.data(),qn*2)!=0){
            std::printf("DIFF hd=%d kvh=%d n=%d max=%g\n",hd,kvh,n,maxdiff);
            throw std::runtime_error("split/reference outputs are not bitwise identical");
        }
        // Independent double-precision softmax for two heads and both batches.
        // Uses the same physical ring but no CUDA implementation/reduction.
        double cpu_error=0;
        for(int b=0;b<batch;++b) for(int h:{0,heads-1}) {
            const int first=window?std::max(0,n-window):0, kh=h/(heads/kvh);
            std::vector<double> scores(n-first);
            double maximum=-INFINITY;
            for(int p=first;p<n;++p){
                double dot=0;
                for(int d=0;d<hd;++d)dot+=double(__half2float(q[(b*heads+h)*hd+d]))*
                    __half2float(k[((size_t(b)*slots+p%slots)*kvh+kh)*hd+d]);
                scores[p-first]=dot*scale;maximum=std::max(maximum,dot*scale);
            }
            double denom=0;
            for(auto& x:scores){x=std::exp(x-maximum);denom+=x;}
            for(int d=0;d<hd;++d){
                double value=0;
                for(int p=first;p<n;++p)value+=scores[p-first]*
                    __half2float(v[((size_t(b)*slots+p%slots)*kvh+kh)*hd+d]);
                cpu_error=std::max(cpu_error,std::fabs(value/denom-__half2float(split[(b*heads+h)*hd+d])));
            }
        }
        require(cpu_error < 0.001,"CPU reference error exceeds 0.001 absolute");
        if(bench && (n==127||n==1024||n==4096||n==8192||n==12000)) {
            float ref=measure(stream.s,timings[0]),opt=measure(stream.s,timings[1]);
            std::printf("BENCH hd=%d kvh=%d heads=%d window=%d n=%d reference_us=%.3f split_us=%.3f speedup=%.3f\n",
                        hd,kvh,heads,window,n,ref,opt,ref/opt);
        }
        std::printf("PASS batch=%d heads=%d kvh=%d hd=%d window=%d n=%d peaked=%d bitwise=1 cpu_max=%.8g\n",
                    batch,heads,kvh,hd,window,n,peaked,cpu_error);std::fflush(stdout);
    }
}
int main(int argc,char** argv) {
    try {
        const bool bench=argc>1 && std::strcmp(argv[1],"--bench")==0;
        run(1,16,8,256,1024,1536,false,bench);
        run(1,16,1,512,0,16384,false,bench);
        run(2,8,8,128,512,1024,true,false);
        run(1,8,1,64,0,16384,true,false);
        run(1,16,1,512,0,16384,true,false);
        run(1,16,8,256,1024,1536,true,false);
        std::puts("PASS: 72 split-attention cases, CPU reference and graph replay across lengths");
    }catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
