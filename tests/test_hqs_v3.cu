#include "kernels.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace helios;
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void need(bool v,const char* msg){if(!v)throw std::runtime_error(msg);}
void put(uint8_t* p,int i,int bits,unsigned v){int bit=i*bits,off=bit/8,shift=bit%8;p[off]|=v<<shift;if(shift+bits>8)p[off+1]|=v>>(8-shift);}
int get(const uint8_t* p,int i,int bits){int bit=i*bits,off=bit/8,shift=bit%8;unsigned v=p[off];if(shift+bits>8)v|=unsigned(p[off+1])<<8;return (v>>shift)&((1<<bits)-1);}
int main(){try{
 size_t comparisons=0;
 for(int bits:{3,4,5})for(int group:{16,32})for(int K:{8,248,256,264,3840,15360,16384,16640,33000}){
  const int N=7,Mmax=17,h=group==16?12:8,stride=h+bits*32,sb=(K+255)/256,rbits=std::max(bits,4),rstride=24+rbits*32;
  const std::string name="hq"+std::to_string(bits)+"3k_g"+std::to_string(group);
  const auto id=DTypeRegistry::instance().get_id(name);need(id!=DTYPE_INVALID,"dtype absent");need(dtype_size(id,256)==size_t(stride),"dtype size");
  std::vector<uint8_t>w(size_t(N)*sb*stride,0),ref(size_t(N)*sb*rstride,0);
  std::vector<half>expanded(size_t(N)*K),input(size_t(Mmax)*K);
  uint32_t seed=123;auto rnd=[&](){seed=seed*1664525u+1013904223u;return seed;};
  for(int row=0;row<N;++row)for(int b=0;b<sb;++b){
   auto* p=w.data()+(size_t(row)*sb+b)*stride;auto* r=ref.data()+(size_t(row)*sb+b)*rstride;
   half dh=__float2half(.001f*(1+(rnd()%20)));memcpy(p,&dh,2);memcpy(r,&dh,2);
   for(int g=0;g<256/group;++g)put(p+2,g,5,rnd()%32);
   for(int g=0;g<32;++g)put(r+4,g,5,get(p+2,g/(group/8),5));
   for(int i=0;i<256;++i){
    int code=rnd()>>24;code&=(1<<bits)-1;
    if(bits==4)p[h+i/2]|=code<<(i%2?0:4);else put(p+h,i,bits,code);
    int rc=bits==3?code+4:code;
    if(rbits==4)r[24+i/2]|=rc<<(i%2?0:4);else put(r+24,i,5,rc);
    float step=__half2float(dh)*float(get(p+2,i/group,5))*(1.f/31.f);
    if(b*256+i<K)expanded[size_t(row)*K+b*256+i]=__float2half(float(code-(1<<(bits-1)))*step);
   }
  }
  for(auto&v:input)v=__float2half((int(rnd()%201)-100)*.002f);
  uint8_t *dw,*dr;half *dx,*dy,*dyref,*de;
  ck(cudaMalloc(&dw,w.size()));ck(cudaMalloc(&dr,ref.size()));ck(cudaMalloc(&dx,input.size()*2));ck(cudaMalloc(&dy,size_t(Mmax)*N*2));ck(cudaMalloc(&dyref,size_t(Mmax)*N*2));ck(cudaMalloc(&de,expanded.size()*2));
  ck(cudaMemcpy(dw,w.data(),w.size(),cudaMemcpyHostToDevice));ck(cudaMemcpy(dr,ref.data(),ref.size(),cudaMemcpyHostToDevice));ck(cudaMemcpy(dx,input.data(),input.size()*2,cudaMemcpyHostToDevice));
  kernels::launch_dequant_hqs_v3(dw,de,K,N,bits,group);ck(cudaDeviceSynchronize());std::vector<half>got(expanded.size());ck(cudaMemcpy(got.data(),de,got.size()*2,cudaMemcpyDeviceToHost));need(!memcmp(got.data(),expanded.data(),got.size()*2),"dequant mismatch");comparisons+=got.size();
  for(int M:{1,8,9,17}){
   kernels::launch_matmul_hqs_v3(dx,dw,dy,M,K,N,bits,group);
   auto base=rbits==4?kernels::launch_matmul_hq42k:kernels::launch_matmul_hq52k;base(dx,dr,dyref,M,K,N,nullptr);ck(cudaDeviceSynchronize());
   std::vector<half>a(M*N),b(M*N);ck(cudaMemcpy(a.data(),dy,a.size()*2,cudaMemcpyDeviceToHost));ck(cudaMemcpy(b.data(),dyref,b.size()*2,cudaMemcpyDeviceToHost));need(!memcmp(a.data(),b.data(),a.size()*2),"matmul mismatch");comparisons+=a.size();
   // Warm allocations/tuning first, then check the real captured launcher.
   cudaStream_t stream;ck(cudaStreamCreate(&stream));cudaGraph_t graph;cudaGraphExec_t exec;
   ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));kernels::launch_matmul_hqs_v3(dx,dw,dy,M,K,N,bits,group,stream);ck(cudaStreamEndCapture(stream,&graph));ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
   ck(cudaGraphLaunch(exec,stream));ck(cudaStreamSynchronize(stream));ck(cudaMemcpy(a.data(),dy,a.size()*2,cudaMemcpyDeviceToHost));need(!memcmp(a.data(),b.data(),a.size()*2),"graph mismatch");comparisons+=a.size();cudaGraphExecDestroy(exec);cudaGraphDestroy(graph);cudaStreamDestroy(stream);
  }
  std::vector<int32_t>ids={0,6,2,-1,7};int32_t* did;half* emb;
  ck(cudaMalloc(&did,ids.size()*4));ck(cudaMalloc(&emb,ids.size()*K*2));ck(cudaMemcpy(did,ids.data(),ids.size()*4,cudaMemcpyHostToDevice));
  kernels::launch_embedding_hqs_v3(did,dw,emb,1,ids.size(),N,K,bits,group);ck(cudaDeviceSynchronize());std::vector<half>e(ids.size()*K);ck(cudaMemcpy(e.data(),emb,e.size()*2,cudaMemcpyDeviceToHost));
  for(size_t t=0;t<ids.size();++t)for(int i=0;i<K;++i){half expected=ids[t]>=0&&ids[t]<N?expanded[size_t(ids[t])*K+i]:__float2half(0);need(!memcmp(&expected,&e[t*K+i],2),"embedding mismatch");}comparisons+=e.size();
  cudaFree(dw);cudaFree(dr);cudaFree(dx);cudaFree(dy);cudaFree(dyref);cudaFree(de);cudaFree(did);cudaFree(emb);
 }
 // Scalar dequant and embedding tails, which do not have the GEMV K%8 constraint.
 std::cout<<"PASS x.3: "<<comparisons<<" exact half comparisons, 6 formats, GEMV/small batch/prefill/graphs/embedding"<<std::endl;
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<std::endl;return 1;}}
