// Q4_K_M vs HQS reconstruction attribution; identical FP16 arithmetic, no speed claims.
// Quantize original BF16 into physical Q8G32 blocks, reconstruct FP16, and
// execute the existing single-token Hector graphs, one resident layer at a time.
#include "hnf_loader.hpp"
#include "ggml.h"
#include "hqs_vx3_decode_control.hpp"
#include "graph_builder.hpp"
#include "gemma4_kv_cache.hpp"
#include "kernels.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <iomanip>
#include <unordered_map>
using namespace helios;
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Weight {std::string name; uint64_t offset; bool quant; std::vector<uint32_t> shape;std::string source="hnf";};
struct Position {size_t index; int doc,pos,input,target;};
#pragma pack(push,1)
struct Q8Block {half scale; int8_t code[32];};
#pragma pack(pop)
static_assert(sizeof(Q8Block)==34);
float bf16(uint16_t u){uint32_t bits=uint32_t(u)<<16;float v;std::memcpy(&v,&bits,4);return v;}
// HQS v4: superbloque de 256 = d, dmin (FP16) + 16 escalas (sb bits) + 16 mínimos (mb bits) + 128 B de códigos de 4 bits.
static size_t v4_block_bytes(int sb,int mb){return 4+(16*(sb+mb)+7)/8+128;}
static void decode_v4(const unsigned char* src,size_t blocks,int sb,int mb,half* out){
 const size_t hb=4+(16*(sb+mb)+7)/8,bb=hb+128;
 for(size_t i=0;i<blocks;++i){const unsigned char* b=src+i*bb;
  uint16_t dr,mr;std::memcpy(&dr,b,2);std::memcpy(&mr,b+2,2);__half_raw hd,hm;hd.x=dr;hm.x=mr;
  const float d=__half2float(half(hd)),dm=__half2float(half(hm));
  auto field=[&](size_t p,int bits){uint32_t v=0;for(int k=0;k<3;++k){size_t at=p/8+k;if(4+at<hb)v|=uint32_t(b[4+at])<<(8*k);}return (v>>(p%8))&((1u<<bits)-1);};
  for(int g=0;g<16;++g){const float a=d*float(field(size_t(g)*sb,sb)),m=dm*float(field(size_t(16)*sb+size_t(g)*mb,mb));
   for(int j=0;j<16;++j){const int k=g*16+j;const unsigned char c=b[hb+k/2];const float q=float((k&1)?(c>>4):(c&15));
    const float v=a*q-m;if(!std::isfinite(v))throw std::runtime_error("nonfinite v4 weight");out[i*256+k]=__float2half(v);}}}
}
int main(int argc,char**argv){try{
 if(argc==7 && std::string(argv[1])=="--decode") {
  std::string dtype=argv[3];size_t offset=std::stoull(argv[4]),count=std::stoull(argv[5]);
  size_t bytes=dtype=="fp16"?count*2:count/256*(dtype=="hq43k_g16"?140:172);
  std::ifstream in(argv[2],std::ios::binary);std::vector<unsigned char>b(bytes);in.seekg(offset);in.read(reinterpret_cast<char*>(b.data()),bytes);if(!in)throw std::runtime_error("probe read");
  std::vector<half>h(count);decode_hqs_control(dtype,b.data(),count,h.data());
  if(std::ifstream(argv[6]).good())throw std::runtime_error("probe refuses overwrite");std::ofstream out(argv[6],std::ios::binary);out.write(reinterpret_cast<const char*>(h.data()),count*2);return !out;
 }

 if(argc!=9)throw std::runtime_error("eval_ablation HNF RAW PLAN_TSV POSITIONS_TSV mix OUTPUT_JSON OUTPUT_F16_OR_DEV_NULL ORIGINAL_FULL_F32");
 const std::string mode=argv[5];if(mode!="mix")throw std::runtime_error("mode must be mix");
 if(std::ifstream(argv[6]).good()||(std::string(argv[7])!="/dev/null"&&std::ifstream(argv[7]).good()))throw std::runtime_error("refusing to overwrite output");
 std::vector<Weight> weights;std::ifstream wm(argv[3]);std::string line;
 while(std::getline(wm,line)){std::istringstream s(line);Weight w;int q,n;s>>w.name>>w.offset>>q>>n;w.quant=q;w.shape.resize(n);for(auto&d:w.shape)s>>d;s>>w.source;if(!s||(w.source!="hnf"&&w.source!="hqs16"&&w.source!="fp16"&&w.source!="q8"&&w.source!="q4_k"&&w.source!="q6_k"&&w.source!="f32"&&w.source!="v4"))throw std::runtime_error("weight plan TSV");weights.push_back(w);}
 std::vector<Position> positions;std::ifstream pm(argv[4]);Position p;while(pm>>p.index>>p.doc>>p.pos>>p.input>>p.target)positions.push_back(p);
 if(weights.empty()||positions.empty())throw std::runtime_error("empty manifest");
 for(size_t i=0;i<positions.size();++i)if(positions[i].pos!=(i&&positions[i-1].doc==positions[i].doc?positions[i-1].pos+1:0))throw std::runtime_error("non-contiguous prefix");
 HnfLoader loader;if(!loader.open(argv[1]))throw std::runtime_error("HNF open");
 auto config=loader.config();auto gemma=loader.gemma4_config();
 if(gemma.num_kv_shared_layers||gemma.has_flag(GEMMA4_EXT_FLAG_PLE))throw std::runtime_error("streamed control requires independent KV layers and no PLE");
 EngineConfig cfg;cfg.scratch_pool.auto_fraction=0;cfg.scratch_pool.min_size_bytes=0;Engine engine(cfg);kernels::register_all_kernels(engine);
 auto&reg=engine.tensors();
 std::unordered_map<std::string,const TensorEntry*> entries;for(auto*e:loader.tensors_for_block(BLOCK_TEXT_MODEL))entries.emplace(e->name,e);
 auto weight_dtype=[&](const Weight&w){if(w.source!="hnf")return dtype::FP16();auto id=DTypeRegistry::instance().get_id(entries.at(w.name)->dtype);if(id==DTYPE_INVALID)throw std::runtime_error("unknown HNF dtype");return id;};
 for(const auto&w:weights){if(entries.at(w.name)->shape!=w.shape)throw std::runtime_error("plan shape differs from HNF");TensorInfo t;t.shape=w.shape;t.dtype=weight_dtype(w);reg.register_tensor(w.name,t);}
 GraphBuilder builder;auto arch=builder.detect_architecture(engine,"text",config);
 builder.allocate_gemma4_scratch(engine,config,gemma,arch,1,1);
 Gemma4KVCache kv;kv.allocate(gemma,config.num_key_value_heads(),1,4096);kv.register_tensors(engine,"_evalkv");
 auto* token=reg.allocate_and_register("eval_token",{1,1},dtype::INT32());
 const size_t width=config.hidden_size(),vocab=config.vocab_size(),n=positions.size();
 auto* hidden=static_cast<half*>(reg.allocate_and_register("_control.hidden",{uint32_t(n),uint32_t(width)},dtype::FP16()));
 auto* scratch=reg.at("_s.hidden").ptr;std::ifstream raw(argv[2],std::ios::binary);if(!raw)throw std::runtime_error("raw open");
 std::ifstream hnfraw(argv[1],std::ios::binary);
 uint64_t quantized_values=0;double squared_error=0,squared_original=0;uint64_t clipped=0;bool codec_sample_saved=false;
 auto load=[&](const std::string& prefix){
  for(const auto&w:weights){if(w.name.compare(0,prefix.size(),prefix)!=0)continue;
   size_t count=1;for(auto d:w.shape)count*=d;
   reg.remove(w.name);void* dst=reg.allocate_and_register(w.name,w.shape,weight_dtype(w));
   if(w.source=="hnf"){auto*e=entries.at(w.name);std::vector<char>bytes(e->size);if(e->size!=reg.at(w.name).size_bytes||!loader.read_tensor_data(*e,bytes.data(),bytes.size()))throw std::runtime_error("HNF tensor read/size mismatch");ck(cudaMemcpy(dst,bytes.data(),bytes.size(),cudaMemcpyHostToDevice));continue;}
   if(w.source=="hqs16"){
    auto*e=entries.at(w.name);const auto dt=e->dtype;size_t bs=dt=="fp16"?1:256,bytes=dt=="fp16"?2:dt=="hq43k_g16"?140:dt=="hq53k_g16"?172:0;
    if(!bytes||count/bs*bytes!=e->size)throw std::runtime_error("HQS geometry mismatch");
    const size_t chunk=2097152;std::vector<unsigned char>packed(std::min(chunk,count)/bs*bytes);std::vector<half>halves(std::min(chunk,count));
    for(size_t start=0;start<count;start+=chunk){size_t take=std::min(chunk,count-start);hnfraw.seekg(e->offset+start/bs*bytes);hnfraw.read(reinterpret_cast<char*>(packed.data()),take/bs*bytes);if(!hnfraw)throw std::runtime_error("HQS raw read");decode_hqs_control(dt,packed.data(),take,halves.data());ck(cudaMemcpy(static_cast<char*>(dst)+start*2,halves.data(),take*2,cudaMemcpyHostToDevice));}
    continue;
   }
   if(w.source=="v4"){
    static std::ifstream v4bin(std::getenv("HQS_V4_BIN")?std::getenv("HQS_V4_BIN"):"",std::ios::binary);
    if(!v4bin)throw std::runtime_error("HQS_V4_BIN missing");
    const int sb=std::getenv("HQS_V4_SB")?std::atoi(std::getenv("HQS_V4_SB")):5,mb=std::getenv("HQS_V4_MB")?std::atoi(std::getenv("HQS_V4_MB")):5;
    if(count%256)throw std::runtime_error("v4 geometry");const size_t bb=v4_block_bytes(sb,mb),chunk=2097152;
    std::vector<unsigned char>packed(std::min(chunk,count)/256*bb);std::vector<half>halves(std::min(chunk,count));
    for(size_t start=0;start<count;start+=chunk){size_t take=std::min(chunk,count-start);v4bin.seekg(w.offset+start/256*bb);v4bin.read(reinterpret_cast<char*>(packed.data()),take/256*bb);if(!v4bin)throw std::runtime_error("v4 read");
     decode_v4(packed.data(),take/256,sb,mb,halves.data());ck(cudaMemcpy(static_cast<char*>(dst)+start*2,halves.data(),take*2,cudaMemcpyHostToDevice));}
    continue;
   }
   if(w.source=="q4_k"||w.source=="q6_k"||w.source=="f32"){
    auto type=w.source=="q4_k"?GGML_TYPE_Q4_K:w.source=="q6_k"?GGML_TYPE_Q6_K:GGML_TYPE_F32;
    const auto*traits=ggml_get_type_traits(type);const size_t bs=ggml_blck_size(type),bytes=ggml_type_size(type),chunk=2097152;
    std::vector<char> packed((std::min(chunk,count)/bs)*bytes);std::vector<float> floats(std::min(chunk,count));std::vector<half> halves(floats.size());
    for(size_t start=0;start<count;start+=chunk){size_t take=std::min(chunk,count-start);if(take%bs)throw std::runtime_error("GGUF block geometry");size_t size=take/bs*bytes;raw.seekg(w.offset+start/bs*bytes);raw.read(packed.data(),size);if(!raw)throw std::runtime_error("GGUF read");
     if(type==GGML_TYPE_F32)std::memcpy(floats.data(),packed.data(),size);else traits->to_float(packed.data(),floats.data(),take);
     for(size_t j=0;j<take;++j){if(!std::isfinite(floats[j]))throw std::runtime_error("nonfinite weight");halves[j]=__float2half(floats[j]);}
     ck(cudaMemcpy(static_cast<char*>(dst)+start*2,halves.data(),take*2,cudaMemcpyHostToDevice));
    }continue;
   }
   // Bound host memory, including the embedding table, to 2M weights.
   constexpr size_t chunk=2097152;std::vector<uint16_t> src(std::min(chunk,count));std::vector<half> decoded(src.size());std::vector<Q8Block> encoded((src.size()+31)/32);
   for(size_t start=0;start<count;start+=chunk){size_t take=std::min(chunk,count-start);raw.seekg(w.offset+start*2);raw.read(reinterpret_cast<char*>(src.data()),take*2);if(!raw)throw std::runtime_error("raw read");
    if(w.source=="q8"&&w.quant){if(take%32)throw std::runtime_error("Q8 geometry");
     for(size_t j=0;j<take;j+=32){auto& b=encoded[j/32];float maximum=0;for(int k=0;k<32;++k)maximum=std::max(maximum,std::abs(bf16(src[j+k])));b.scale=__float2half(maximum/127.f);float scale=__half2float(b.scale);
      for(int k=0;k<32;++k){float x=bf16(src[j+k]);float q=scale?std::nearbyint(x/scale):0;clipped+=q>127||q< -127;b.code[k]=static_cast<int8_t>(std::clamp(q,-127.f,127.f));}
     }
     // Decode only the stored scale and int8 payload, never original values.
     for(size_t j=0;j<take;j+=32){const auto&b=encoded[j/32];for(int k=0;k<32;++k){decoded[j+k]=__float2half(__half2float(b.scale)*float(b.code[k]));double x=bf16(src[j+k]),e=double(__half2float(decoded[j+k]))-x;squared_error+=e*e;squared_original+=x*x;}}
     quantized_values+=take;
     if(!codec_sample_saved){std::ofstream sample(std::string(argv[6])+".codec.bin",std::ios::binary);constexpr size_t sample_values=4096;sample.write(reinterpret_cast<const char*>(src.data()),sample_values*2);sample.write(reinterpret_cast<const char*>(encoded.data()),sample_values/32*sizeof(Q8Block));sample.write(reinterpret_cast<const char*>(decoded.data()),sample_values*2);if(!sample)throw std::runtime_error("codec sample write");codec_sample_saved=true;}
    }else for(size_t j=0;j<take;++j)decoded[j]=__float2half(bf16(src[j]));
    ck(cudaMemcpy(static_cast<char*>(dst)+start*2,decoded.data(),take*2,cudaMemcpyHostToDevice));
   }
  }
 };
 auto unload=[&](const std::string& prefix){engine.sync();for(const auto&w:weights)if(w.name.compare(0,prefix.size(),prefix)==0){reg.remove(w.name);TensorInfo t;t.shape=w.shape;t.dtype=weight_dtype(w);reg.register_tensor(w.name,t);}};
 const std::string embedding=arch.prefix+"."+arch.embedding_name;
 load(embedding);
 auto input=builder.build_gemma4_input(engine,config,gemma,arch,"eval_token",1,1);
 for(size_t i=0;i<n;++i){ck(cudaMemcpy(token,&positions[i].input,4,cudaMemcpyHostToDevice));engine.execute(input);engine.sync();ck(cudaMemcpy(hidden+i*width,scratch,width*2,cudaMemcpyDeviceToDevice));}
 unload(embedding);std::cout<<"EMBEDDED "<<n<<std::endl;
 for(uint32_t layer=0;layer<arch.num_layers;++layer){std::string prefix=arch.prefix+".layer"+std::to_string(layer)+".";load(prefix);
  for(size_t i=0;i<n;++i){ck(cudaMemcpy(scratch,hidden+i*width,width*2,cudaMemcpyDeviceToDevice));engine.execute(builder.build_gemma4_layer_cached(engine,config,gemma,arch,layer,1,1,{"_evalkv",uint32_t(positions[i].pos),4096}));engine.sync();ck(cudaMemcpy(hidden+i*width,scratch,width*2,cudaMemcpyDeviceToDevice));}
  unload(prefix);std::cout<<"LAYER "<<layer+1<<"/"<<arch.num_layers<<std::endl;
 }
 const std::string norm=arch.prefix+"."+arch.final_norm_name;load(norm);load(embedding);
 if(!arch.lm_head_name.empty()&&arch.lm_head_name!=arch.embedding_name)throw std::runtime_error("control assumes tied head");
 CommandBuffer head;head.add_rmsnorm("_s.normed","_s.hidden",norm,config.rms_norm_eps());head.add_matmul("_s.logits","_s.normed",embedding);head.commands().back().set("seq_len",uint32_t{1});
 if(gemma.has_flag(GEMMA4_EXT_FLAG_LOGIT_SOFTCAP))head.add_softcap("_s.logits","_s.logits",config.get<float>("final_logit_softcapping",30.f));
 std::ofstream jout(argv[6]),out(argv[7],std::ios::binary);if(!jout||!out)throw std::runtime_error("output open");
 std::ifstream reference(argv[8],std::ios::binary);if(!reference)throw std::runtime_error("original reference missing");std::vector<float>ref(vocab);
 jout<<std::setprecision(15)<<"{\"mode\":\""<<mode<<"\",\"vocab\":"<<vocab<<",\"positions\":[";std::vector<half>logits(vocab);double total=0,total_kl=0;
 for(size_t i=0;i<n;++i){ck(cudaMemcpy(scratch,hidden+i*width,width*2,cudaMemcpyDeviceToDevice));engine.execute(head);engine.sync();ck(cudaMemcpy(logits.data(),reg.at("_s.logits").ptr,vocab*2,cudaMemcpyDeviceToHost));if(positions[i].pos<16)out.write(reinterpret_cast<const char*>(logits.data()),vocab*2);if(!out)throw std::runtime_error("logit write");
  double peak=-1e30;int top=0;for(size_t j=0;j<vocab;++j){float x=__half2float(logits[j]);if(!std::isfinite(x))throw std::runtime_error("nonfinite logits");if(x>peak){peak=x;top=j;}}double sum=0;for(auto x:logits)sum+=std::exp(double(__half2float(x))-peak);double nll=peak+std::log(sum)-__half2float(logits[positions[i].target]);total+=nll;
  auto&p=positions[i];reference.seekg(p.index*vocab*4);reference.read(reinterpret_cast<char*>(ref.data()),vocab*4);if(!reference)throw std::runtime_error("reference read");double rp=-1e30;int rt=0;for(size_t j=0;j<vocab;++j){if(!std::isfinite(ref[j]))throw std::runtime_error("nonfinite original");if(ref[j]>rp){rp=ref[j];rt=j;}}double rs=0;for(float x:ref)rs+=std::exp(double(x)-rp);double rz=rp+std::log(rs),qz=peak+std::log(sum),kl=0;for(size_t j=0;j<vocab;++j){double lp=double(ref[j])-rz;kl+=std::exp(lp)*(lp-double(__half2float(logits[j]))+qz);}total_kl+=kl;
  if(i)jout<<",";jout<<"{\"index\":"<<p.index<<",\"doc\":"<<p.doc<<",\"pos\":"<<p.pos<<",\"input\":"<<p.input<<",\"target\":"<<p.target<<",\"argmax\":"<<top<<",\"nll\":"<<nll<<",\"kl\":"<<kl<<",\"original_argmax\":"<<rt<<",\"original_nll\":"<<rz-ref[p.target]<<"}";
  if(i%1024==0)std::cout<<"HEAD "<<i<<"/"<<n<<std::endl;
 }
 jout<<"],\"mean_nll\":"<<total/n<<",\"perplexity\":"<<std::exp(total/n)<<",\"kl_mean\":"<<total_kl/n<<",\"q8_values_including_embedding_reload\":"<<quantized_values<<",\"q8_relative_rmse\":"<<(squared_original?std::sqrt(squared_error/squared_original):0)<<",\"q8_clipped\":"<<clipped<<"}\n";
 if(!jout)throw std::runtime_error("json write");std::cout<<"DONE "<<mode<<" N="<<n<<" PPL="<<std::exp(total/n)<<std::endl;
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<std::endl;return 1;}}
