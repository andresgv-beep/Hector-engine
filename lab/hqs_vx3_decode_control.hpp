// Diagnostic CPU reconstruction only. Does not modify the deployed decoder.
#pragma once
#include <cuda_fp16.h>
#include <string>
#include <vector>
#include <cstring>
#include <stdexcept>
#include <cmath>
inline void decode_hqs_control(const std::string& dtype,const unsigned char* src,size_t count,half* dst) {
 if(dtype=="fp16") {std::memcpy(dst,src,count*2);return;}
 const int bits=dtype=="hq43k_g16"?4:dtype=="hq53k_g16"?5:0;
 if(!bits||count%256)throw std::runtime_error("unsupported HQS control geometry");
 const size_t stride=12+32*bits;
 auto unpack=[](const unsigned char* p,int i,int b){int bit=i*b,byte=bit/8,shift=bit%8;unsigned int n=p[byte];if(shift+b>8)n|=unsigned(p[byte+1])<<8;return (n>>shift)&((1u<<b)-1);};
 for(size_t off=0;off<count;off+=256){const auto* p=src+(off/256)*stride;half base;std::memcpy(&base,p,2);float d=__half2float(base);
  for(int j=0;j<256;++j){float scale=d*float(unpack(p+2,j/16,5))*(1.f/31.f);int code=bits==4?int((p[12+j/2]>>(j%2?0:4))&15):int(unpack(p+12,j,bits));float value=float(code-(1<<(bits-1)))*scale;
   if(!std::isfinite(value))throw std::runtime_error("nonfinite HQS control weight");dst[off+j]=__float2half(value);
  }
 }
}
