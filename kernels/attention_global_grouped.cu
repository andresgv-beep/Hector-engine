// Gemma global decode: eight Q heads share one FP16 K/V head.
// QK and PV use mma.m16n8k16, with FP32 scores, softmax and accumulation.
#include "kernels.hpp"
#include <cuda_fp16.h>
#include <cstdint>

namespace helios { namespace kernels {
void launch_attention_flash_decode_merge_dp(const float*, half*, const int32_t*, int, int, cudaStream_t);
namespace {
constexpr int HD=512, HEADS=8, TILE=32, THREADS=128, STRIDE=520, PSTRIDE=40;

__device__ __forceinline__ void ld4(uint32_t (&r)[4], const half* p) {
    uint32_t a=static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]):"r"(a));
}
__device__ __forceinline__ void ld4t(uint32_t (&r)[4], const half* p) {
    uint32_t a=static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]):"r"(a));
}
__device__ __forceinline__ void ld2(uint32_t (&r)[2], const half* p) {
    uint32_t a=static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                 : "=r"(r[0]),"=r"(r[1]):"r"(a));
}
__device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
                 : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}
__device__ __forceinline__ void load_kv(half* dst,const half* src,int n) {
    for(int i=threadIdx.x;i<TILE*(HD/8);i+=THREADS) {
        int row=i/(HD/8),col=(i%(HD/8))*8;
        uint32_t a=static_cast<uint32_t>(__cvta_generic_to_shared(dst+row*STRIDE+col));
        const half* p=src+min(row,n-1)*HD+col;
        int bytes=row<n?16:0;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(a),"l"(p),"r"(bytes));
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
}

__global__ __launch_bounds__(THREADS,2) void attention_global_grouped_kernel(
    const half* __restrict__ q,const half* __restrict__ k,const half* __restrict__ v,
    float* __restrict__ partials,const int32_t* __restrict__ seq,float scale,int splits,int min_chunk) {
    const int len=*seq,chunk=max(min_chunk,(len+splits-1)/splits);
    const int begin=blockIdx.x*chunk,end=min(begin+chunk,len);
    if(begin>=len)return;
    const int lane=threadIdx.x%32,warp=threadIdx.x/32,hbase=blockIdx.y*HEADS;
    __shared__ __align__(16) half sq[HEADS*STRIDE];
    __shared__ __align__(16) half skv[TILE*STRIDE];
    __shared__ __align__(16) half sp[HEADS*PSTRIDE];
    __shared__ float scores[HEADS*TILE];
    __shared__ float maxima[HEADS],sums[HEADS],correction[HEADS];
    for(int i=threadIdx.x;i<HEADS*(HD/8);i+=THREADS) {
        int h=i/(HD/8),d=(i%(HD/8))*8;
        *reinterpret_cast<uint4*>(sq+h*STRIDE+d)=*reinterpret_cast<const uint4*>(q+(hbase+h)*HD+d);
    }
    if(threadIdx.x<HEADS){maxima[threadIdx.x]=-INFINITY;sums[threadIdx.x]=0;}
    float acc[8][4]={};
    __syncthreads();
    for(int pos=begin;pos<end;pos+=TILE) {
        int n=min(TILE,end-pos);
        load_kv(skv,k+size_t(pos)*HD,n);
        if(warp<2) {
            float dots[4]={};
            #pragma unroll
            for(int ds=0;ds<HD;ds+=16) {
                uint32_t a[4],b[2];
                ld4(a,skv+(warp*16+lane%16)*STRIDE+ds+(lane/16)*8);
                ld2(b,sq+(lane%8)*STRIDE+ds+((lane/8)%2)*8);
                mma(dots,a,b);
            }
            int r=warp*16+lane/4,h=(lane%4)*2;
            scores[h*TILE+r]=r<n?dots[0]*scale:-INFINITY;
            scores[(h+1)*TILE+r]=r<n?dots[1]*scale:-INFINITY;
            scores[h*TILE+r+8]=r+8<n?dots[2]*scale:-INFINITY;
            scores[(h+1)*TILE+r+8]=r+8<n?dots[3]*scale:-INFINITY;
        }
        __syncthreads();
        const int h=threadIdx.x/16,col=threadIdx.x%16;
        float mx=fmaxf(scores[h*TILE+col],scores[h*TILE+col+16]);
        #pragma unroll
        for(int o=8;o;o/=2)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,o,16));
        mx=fmaxf(mx,maxima[h]);
        float corr=expf(maxima[h]-mx);
        float p0=expf(scores[h*TILE+col]-mx),p1=expf(scores[h*TILE+col+16]-mx);
        sp[h*PSTRIDE+col]=__float2half_rn(p0);
        sp[h*PSTRIDE+col+16]=__float2half_rn(p1);
        float sum=p0+p1;
        #pragma unroll
        for(int o=8;o;o/=2)sum+=__shfl_xor_sync(0xffffffff,sum,o,16);
        // All lanes must finish reading the old shared maximum before lane 0
        // overwrites it; shuffle synchronization alone does not order memory.
        __syncwarp();
        if(col==0){sums[h]=sums[h]*corr+sum;maxima[h]=mx;correction[h]=corr;}
        __syncthreads();
        load_kv(skv,v+size_t(pos)*HD,n);
        int hc=(lane%4)*2;
        #pragma unroll
        for(int d=0;d<8;++d) {
            acc[d][0]*=correction[hc];acc[d][2]*=correction[hc];
            acc[d][1]*=correction[hc+1];acc[d][3]*=correction[hc+1];
            #pragma unroll
            for(int ks=0;ks<TILE;ks+=16) {
                uint32_t a[4],b[2];
                ld4t(a,skv+(ks+lane%8+(lane/16)*8)*STRIDE+(warp*8+d)*16+((lane/8)%2)*8);
                ld2(b,sp+(lane%8)*PSTRIDE+ks+((lane/8)%2)*8);
                mma(acc[d],a,b);
            }
        }
        __syncthreads();
    }
    if(threadIdx.x<HEADS) {
        float* dst=partials+(size_t(hbase+threadIdx.x)*splits+blockIdx.x)*(HD+2);
        dst[0]=maxima[threadIdx.x];dst[1]=sums[threadIdx.x];
    }
    const int hc=(lane%4)*2;
    #pragma unroll
    for(int d=0;d<8;++d) {
        int row=(warp*8+d)*16+lane/4;
        float* a=partials+(size_t(hbase+hc)*splits+blockIdx.x)*(HD+2)+2;
        float* b=a+size_t(splits)*(HD+2);
        a[row]=acc[d][0];b[row]=acc[d][1];a[row+8]=acc[d][2];b[row+8]=acc[d][3];
    }
}
} // namespace
void launch_attention_global_grouped_dp(const half* q,const half* k,const half* v,half* out,float* partials,
                                        const int32_t* seq,float scale,int splits,int min_chunk,cudaStream_t stream) {
    attention_global_grouped_kernel<<<dim3(splits,2),THREADS,0,stream>>>(q,k,v,partials,seq,scale,splits,min_chunk);
    launch_attention_flash_decode_merge_dp(partials,out,seq,splits,min_chunk,stream);
}
}} // namespace helios::kernels
