// Compare both attention implementations on the SAME activations/KV from a
// real model, layer by layer. No sampling or independent prefill can confound it.
#include "engine.hpp"
#include "gemma4_kv_cache.hpp"
#include "graph_builder.hpp"
#include "hnf_loader.hpp"
#include "htf_tokenizer.hpp"
#include "chat_template.hpp"
#include "kernels.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
int main(int argc,char** argv){
 if(argc!=2)return 2;
 try{
  struct Stream{cudaStream_t s;Stream(){ck(cudaStreamCreate(&s));}~Stream(){cudaStreamDestroy(s);}} stream;
  helios::EngineConfig ec;ec.stream=stream.s;
  helios::Engine engine(ec);helios::kernels::register_all_kernels(engine);
  helios::HnfLoader loader;
  if(!loader.open(argv[1]) || !loader.load_block(helios::BLOCK_TEXT_MODEL,engine))throw std::runtime_error("model load");
  const auto config=loader.config();const auto& gemma=loader.gemma4_config();
  if(!loader.has_gemma4_config() || config.num_attention_heads()!=16)
   throw std::runtime_error("this real-weight regression requires Gemma 4 with 16 query heads");
  for(const auto& layer:gemma.layers)if(layer.head_dim>512 || layer.head_dim%2)
   throw std::runtime_error("unsupported head dimension in real-weight regression");
  helios::GraphBuilder builder;const auto arch=builder.detect_architecture(engine,"text",config);
  builder.allocate_gemma4_scratch(engine,config,gemma,arch,1,512);
  helios::Gemma4KVCache kv;
  if(!kv.allocate(gemma,config.num_key_value_heads(),1,8192,512))throw std::runtime_error("KV allocation");
  kv.register_tensors(engine,"_verify_kv");
  void* input=engine.tensors().allocate_and_register("verify.tokens",{1,512},helios::dtype::INT32());
  auto candidate=static_cast<half*>(engine.tensors().allocate_and_register("verify.attention",{16*512},helios::dtype::FP16()));
  auto workspace=static_cast<float*>(engine.tensors().allocate_and_register("verify.partials",
    {uint32_t(helios::kernels::attention_cached_split_workspace_bytes(1,16)/4)},helios::dtype::FP32()));
  int comparisons=0;
  engine.register_kernel(helios::op::ATTENTION_CACHED(),[&](helios::ExecContext& ctx,const helios::Command& cmd){
   const int heads=cmd.get<uint32_t>("num_heads",0),kvh=cmd.get<uint32_t>("num_kv_heads",0),hd=cmd.get<uint32_t>("head_dim",0);
   const int maximum=cmd.get<uint32_t>("max_seq_len",0),window=cmd.get<uint32_t>("window_size",0),slots=cmd.get<uint32_t>("cache_slots",0);
   const float scale=cmd.get<float>("scale",1.f);
   helios::kernels::launch_attention_cached_fp16_dp(static_cast<const half*>(ctx.in(0)->ptr),static_cast<const half*>(ctx.in(1)->ptr),
    static_cast<const half*>(ctx.in(2)->ptr),static_cast<half*>(ctx.output->ptr),1,engine.device_total_seq(),heads,kvh,hd,maximum,scale,window,ctx.stream,slots);
   helios::kernels::launch_attention_cached_fp16_split_dp(static_cast<const half*>(ctx.in(0)->ptr),static_cast<const half*>(ctx.in(1)->ptr),
    static_cast<const half*>(ctx.in(2)->ptr),candidate,workspace,1,engine.device_total_seq(),heads,kvh,hd,maximum,scale,window,ctx.stream,slots);
   ck(cudaStreamSynchronize(ctx.stream));
   std::vector<half> a(heads*hd),b(heads*hd);
   ck(cudaMemcpy(a.data(),ctx.output->ptr,a.size()*2,cudaMemcpyDeviceToHost));
   ck(cudaMemcpy(b.data(),candidate,b.size()*2,cudaMemcpyDeviceToHost));
   if(std::memcmp(a.data(),b.data(),a.size()*2)){
    float worst=0;int count=0;
    for(size_t i=0;i<a.size();++i)if(std::memcmp(&a[i],&b[i],2)){++count;worst=std::max(worst,std::fabs(__half2float(a[i])-__half2float(b[i])));}
    std::fprintf(stderr,"DIFF %s count=%d max=%g\n",cmd.output.c_str(),count,worst);
    throw std::runtime_error("real attention not bitwise identical");
   }
   ++comparisons;
  });
  std::string prompt="Datos de referencia:\n";
  for(int i=0;i<384;++i)prompt+="El servidor registra eventos de red, memoria, disco y conexiones activas.\n";
  prompt+="Escribe una explicación técnica extensa de al menos mil palabras sobre cómo funciona un sistema operativo. Empieza directamente y desarrolla todos los detalles.";
  auto ids=loader.tokenizer()->encode(helios::format_gemma4_chat({{"user",prompt}}),false,false);
  uint32_t position=0;
  while(position<ids.size()){
   uint32_t n=std::min<size_t>(512,ids.size()-position);
   engine.tensors().at("verify.tokens").shape={1,n};
   ck(cudaMemcpy(input,ids.data()+position,n*4,cudaMemcpyHostToDevice));
   engine.execute(builder.build_gemma4_forward_cached(engine,config,gemma,arch,"verify.tokens",1,n,{"_verify_kv",position,8192}));
   engine.sync();position+=n;
  }
  for(int step=0;step<128;++step){
   std::vector<half> logits(config.vocab_size());
   ck(cudaMemcpy(logits.data(),builder.get_logits(engine)->ptr,logits.size()*2,cudaMemcpyDeviceToHost));
   int32_t token=0;for(size_t i=1;i<logits.size();++i)if(__half2float(logits[i])>__half2float(logits[token]))token=i;
   engine.tensors().at("verify.tokens").shape={1,1};ck(cudaMemcpy(input,&token,4,cudaMemcpyHostToDevice));
   engine.update_device_cache_pos(position,1);
   engine.execute(builder.build_gemma4_forward_cached(engine,config,gemma,arch,"verify.tokens",1,1,{"_verify_kv",position,8192}));
   engine.sync();++position;
  }
  std::printf("PASS: %d real layer/step attention outputs bitwise identical, end position=%u\n",comparisons,position);
 }catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
