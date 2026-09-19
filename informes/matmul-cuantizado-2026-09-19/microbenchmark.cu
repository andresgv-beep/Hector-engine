#include "../../kernels/matmul_cublas.cu"
#include <vector>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#define CHECK(x) do { auto e=(x); if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }while(0)
template<int B,int S> void run(int K,int N) {
 using namespace helios::kernels;
 std::vector<unsigned char> w(size_t(N)*((K+255)/256)*S);
 unsigned state=19; for(auto &v:w){state=state*1664525u+1013904223u;v=state>>24;}
 for(size_t i=0;i<w.size();i+=S){half h=__float2half(0.03125f);memcpy(w.data()+i,&h,2);}
 unsigned char* dw;half *a,*b;
 CHECK(cudaMalloc(&dw,w.size()));CHECK(cudaMalloc(&a,size_t(K)*N*2));CHECK(cudaMalloc(&b,size_t(K)*N*2));
 CHECK(cudaMemcpy(dw,w.data(),w.size(),cudaMemcpyHostToDevice));
 dim3 grid(N,((K+7)/8+255)/256);
 auto launch=[&](bool opt){if(opt) dequant_hq_symmetric_kernel<B,S,true><<<grid,256>>>(dw,b,K,N);else dequant_hq_symmetric_kernel<B,S><<<grid,256>>>(dw,a,K,N);};
 launch(0);launch(1);CHECK(cudaDeviceSynchronize());
 std::vector<half> ha(size_t(K)*N),hb(ha.size());CHECK(cudaMemcpy(ha.data(),a,ha.size()*2,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(hb.data(),b,hb.size()*2,cudaMemcpyDeviceToHost));
 if(memcmp(ha.data(),hb.data(),ha.size()*2))throw std::runtime_error("parity");
 cudaEvent_t st,en;CHECK(cudaEventCreate(&st));CHECK(cudaEventCreate(&en));
 for(int pair=0;pair<3;pair++){float ms[2];for(int j=0;j<2;j++){int mode=pair%2 ? 1-j:j;for(int i=0;i<5;i++)launch(mode);CHECK(cudaEventRecord(st));for(int i=0;i<50;i++)launch(mode);CHECK(cudaEventRecord(en));CHECK(cudaEventSynchronize(en));CHECK(cudaEventElapsedTime(&ms[mode],st,en));}printf("B=%d K=%d N=%d pair=%d before_us=%.3f after_us=%.3f exact=1\n",B,K,N,pair,ms[0]*20,ms[1]*20);fflush(stdout);}
 CHECK(cudaFree(dw));CHECK(cudaFree(a));CHECK(cudaFree(b));cudaEventDestroy(st);cudaEventDestroy(en);
}
int main(){try{for(int K:{256,2816,3840,7680,15360})for(int N:{128,4096}){run<4,152>(K,N);run<5,184>(K,N);}}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
