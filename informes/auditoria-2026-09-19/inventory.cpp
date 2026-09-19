#include "hnf_loader.hpp"
#include <cstdio>
#include <map>
#include <string>
int main(int argc,char**argv){helios::HnfLoader l;if(argc<2||!l.open(argv[1]))return 1;std::map<std::string,size_t> bytes;std::map<std::string,int> counts;int keq=0,global=0;for(auto&x:l.gemma4_config().layers){keq+=x.k_eq_v();global+=x.is_global_attention();}printf("layers=%zu global=%d k_eq_v=%d\n",l.gemma4_config().layers.size(),global,keq);for(auto&t:l.tensors()){bytes[t.dtype]+=t.size;counts[t.dtype]++;if(t.name.find("embedding")!=std::string::npos||t.name.find("lm_head")!=std::string::npos)printf("%s dtype=%s bytes=%llu offset=%llu\n",t.name.c_str(),t.dtype.c_str(),(unsigned long long)t.size,(unsigned long long)t.offset);}for(auto&x:bytes)printf("dtype=%s n=%d bytes=%zu\n",x.first.c_str(),counts[x.first],x.second);}
