// Experimental x.3 embedding and prefill. GEMV lives beside the existing tuner.
#include "kernels.hpp"
#include "cublas_context.hpp"
#include <stdexcept>
namespace helios { namespace kernels {
namespace {
void checked(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template<int Bits,int Group>
__device__ float decode_v3(const uint8_t* block,int index) {
    constexpr int header=Group==16?12:8;
    const int bit=(index/Group)*5, byte=2+bit/8;
    const unsigned word=unsigned(block[byte]) | (unsigned(block[byte+1])<<8);
    const float d=__half2float(__ushort_as_half(unsigned(block[0])|(unsigned(block[1])<<8)));
    const float step=d*float((word>>(bit%8))&31)*(1.f/31.f);
    int code;
    if constexpr(Bits==4) code=(block[header+index/2]>>(index%2?0:4))&15;
    else {
        const int p=index*Bits, off=header+p/8;
        unsigned value=block[off];
        // The final code ends at the final payload byte; never overread.
        if ((p%8)+Bits>8) value|=unsigned(block[off+1])<<8;
        code=(value>>(p%8))&((1<<Bits)-1);
    }
    return float(code-(1<<(Bits-1)))*step;
}
template<int Bits,int Group,bool Packed>
__global__ void dequant_v3(const uint8_t* weights,half* output,int K,int N) {
    constexpr int bytes=(Group==16?12:8)+Bits*32;
    const int row=blockIdx.x;
    const int first=(blockIdx.y*blockDim.x+threadIdx.x)*8;
    if(row>=N || first>=K) return;
    const auto* block=weights+(size_t(row)*((K+255)/256)+first/256)*bytes;
    half values[8];
    #pragma unroll
    for(int j=0;j<8;++j) {
        if(first+j<K) {
            const half value=__float2half(decode_v3<Bits,Group>(block,first%256+j));
            if constexpr(Packed) values[j]=value;
            else output[size_t(row)*K+first+j]=value;
        }
    }
    if constexpr(Packed) {
        uint4 packed=make_uint4(
            __half_as_ushort(values[0])|(uint32_t(__half_as_ushort(values[1]))<<16),
            __half_as_ushort(values[2])|(uint32_t(__half_as_ushort(values[3]))<<16),
            __half_as_ushort(values[4])|(uint32_t(__half_as_ushort(values[5]))<<16),
            __half_as_ushort(values[6])|(uint32_t(__half_as_ushort(values[7]))<<16));
        *reinterpret_cast<uint4*>(output+size_t(row)*K+first)=packed;
    }
}
template<int Bits,int Group>
void dequant(const uint8_t* weights,half* output,int K,int N,cudaStream_t stream) {
    dim3 grid(N,((K+7)/8+255)/256);
    if(K%8==0) dequant_v3<Bits,Group,true><<<grid,256,0,stream>>>(weights,output,K,N);
    else dequant_v3<Bits,Group,false><<<grid,256,0,stream>>>(weights,output,K,N);
}
template<int Bits,int Group>
__global__ void embedding_v3(const int32_t* ids,const uint8_t* table,half* output,int vocab,int dim) {
    constexpr int bytes=(Group==16?12:8)+Bits*32;
    const int token=blockIdx.x, row=ids[token];
    for(int i=threadIdx.x;i<dim;i+=blockDim.x) {
        float value=0;
        if(row>=0 && row<vocab) {
            const auto* block=table+(size_t(row)*((dim+255)/256)+i/256)*bytes;
            value=decode_v3<Bits,Group>(block,i%256);
        }
        output[size_t(token)*dim+i]=__float2half(value);
    }
}
} // anonymous namespace
void launch_dequant_hqs_v3(const uint8_t* w,half* y,int K,int N,int bits,int group,cudaStream_t s) {
    if(K<=0 || N<=0) return;
#define D(B,G) if(bits==B && group==G) return dequant<B,G>(w,y,K,N,s)
    D(3,16);D(3,32);D(4,16);D(4,32);D(5,16);D(5,32);
#undef D
    throw std::runtime_error("Unsupported x.3 dequant format");
}
void launch_embedding_hqs_v3(const int32_t* ids,const uint8_t* w,half* y,int batch,int seq,int vocab,int dim,int bits,int group,cudaStream_t s) {
    if(batch<=0 || seq<=0 || vocab<=0 || dim<=0) return;
#define E(B,G) if(bits==B && group==G) {embedding_v3<B,G><<<batch*seq,256,0,s>>>(ids,w,y,vocab,dim);return;}
    E(3,16);E(3,32);E(4,16);E(4,32);E(5,16);E(5,32);
#undef E
    throw std::runtime_error("Unsupported x.3 embedding format");
}
void launch_matmul_hqs_v3_cublas(const half* x,const uint8_t* w,half* y,int M,int K,int N,int bits,int group,cudaStream_t s) {
    static half* scratch=nullptr;static size_t capacity=0;
    const size_t count=size_t(N)*K;
    if(count>capacity) {
        if(scratch) checked(cudaFree(scratch));
        scratch=nullptr;capacity=0;
        checked(cudaMalloc(&scratch,count*sizeof(half)));capacity=count;
    }
    const auto handle=cublas_handle_for_stream(s);
    if(!handle) throw std::runtime_error("x.3 cuBLAS handle unavailable");
    launch_dequant_hqs_v3(w,scratch,K,N,bits,group,s);
    const half alpha=__float2half(1.f),beta=__float2half(0.f);
    const auto result=cublasHgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,N,M,K,&alpha,scratch,K,x,K,&beta,y,N);
    if(result!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error("x.3 cublasHgemm failed");
}
} } // helios::kernels
