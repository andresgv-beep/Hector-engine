#include "engine.hpp"
#include "kernels.hpp"
#include "hnf_loader.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
void ck(cudaError_t x){if(x!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(x));}
int main(){
 cudaDeviceProp p;ck(cudaGetDeviceProperties(&p,0));printf("GPU %s SM=%d\n",p.name,p.multiProcessorCount);
 cudaStream_t stream;ck(cudaStreamCreate(&stream));
 for(bool dedicated:{false,true}){
  helios::EngineConfig cfg;cfg.stream=dedicated?stream:nullptr;helios::Engine e(cfg);helios::kernels::register_all_kernels(e);
  auto a=e.tensors().allocate_and_register("a",{32},helios::dtype::FP16());e.tensors().allocate_and_register("b",{32},helios::dtype::FP16());ck(cudaMemset(a,0,64));
  helios::CommandBuffer cb;cb.add_scale("b","a",2);e.execute(cb);e.sync();e.execute_graph_replay(cb);e.sync();printf("GRAPH dedicated=%d ready=%d\n",dedicated,e.graph_ready());
 }
 for(int hd:{256,512}){
  int kvh=hd==256?8:1,window=hd==256?1024:0,slots=hd==256?1536:16384;
  half *q,*k,*v,*out;int *len;
  ck(cudaMalloc(&q,16*hd*2));ck(cudaMalloc(&out,16*hd*2));ck(cudaMalloc(&k,size_t(slots)*kvh*hd*2));ck(cudaMalloc(&v,size_t(slots)*kvh*hd*2));ck(cudaMalloc(&len,4));
  ck(cudaMemset(q,0,16*hd*2));ck(cudaMemset(k,0,size_t(slots)*kvh*hd*2));ck(cudaMemset(v,0,size_t(slots)*kvh*hd*2));
  for(int n:{128,1024,4096,8192,12000}){
   ck(cudaMemcpy(len,&n,4,cudaMemcpyHostToDevice));
   auto launch=[&](){helios::kernels::launch_attention_cached_fp16_dp(q,k,v,out,1,len,16,kvh,hd,16384,1.f,window,stream,slots);};
   for(int i=0;i<10;i++)launch();ck(cudaStreamSynchronize(stream));
   cudaGraph_t g;cudaGraphExec_t x;ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));for(int i=0;i<50;i++)launch();ck(cudaStreamEndCapture(stream,&g));ck(cudaGraphInstantiate(&x,g,nullptr,nullptr,0));
   cudaEvent_t start,end;ck(cudaEventCreate(&start));ck(cudaEventCreate(&end));ck(cudaEventRecord(start,stream));for(int i=0;i<5;i++)ck(cudaGraphLaunch(x,stream));ck(cudaEventRecord(end,stream));ck(cudaEventSynchronize(end));float ms;ck(cudaEventElapsedTime(&ms,start,end));
   printf("ATTENTION hd=%d kvh=%d len=%d us=%.3f\n",hd,kvh,n,ms*1000/250);fflush(stdout);
   cudaEventDestroy(start);cudaEventDestroy(end);cudaGraphExecDestroy(x);cudaGraphDestroy(g);
  }
  cudaFree(q);cudaFree(k);cudaFree(v);cudaFree(out);cudaFree(len);
 }
 cudaStreamDestroy(stream);
}
