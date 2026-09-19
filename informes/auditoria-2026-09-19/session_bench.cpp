#include "inference_session.hpp"
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>
#include <cstdlib>
#include <cstdint>
int main(int argc,char**argv) {
 if(argc<2) return 2;
 helios::Model::Config cfg; cfg.hnf_path=argv[1]; cfg.max_seq_len=16384;
 std::string err,code; auto model=helios::Model::load(cfg,&err);
 if(!model){fprintf(stderr,"%s\n",err.c_str());return 1;}
 helios::InferenceSession s; if(!s.attach(model,&err))return 1;
 std::atomic<bool> cancel{false};
 for(int rep: {0,128,384}) for(int trial=0;trial<2;trial++) {
  if(const char* only=getenv("AUDIT_REPEAT")){if(rep!=atoi(only))continue;}
  s.reset(); std::string p="Datos de referencia:\n";
  for(int i=0;i<rep;i++)p+="El servidor registra eventos de red, memoria, disco y conexiones activas.\n";
  p+="Escribe una explicación técnica extensa de al menos mil palabras sobre cómo funciona un sistema operativo. Empieza directamente y desarrolla todos los detalles.";
  helios::InferenceSession::GenConfig g;g.temperature=0;g.max_visible_tokens=128;g.max_thinking_tokens=0;g.close_turn=false;
  helios::InferenceSession::TurnStats st;helios::InferenceSession::FinishReason reason;std::string answer;
  if(argc>2 && getenv("AUDIT_PREFILL"))cudaProfilerStart();
  bool ok=s.run_turn({{"user",p}}, {},g,[&](const std::string&x){answer+=x;},{},[&](uint32_t,double){if(argc>2){if(getenv("AUDIT_PREFILL"))cudaProfilerStop();else cudaProfilerStart();}},cancel,&st,&reason,&code,&err);
  if(argc>2)cudaProfilerStop();
  size_t free,total;cudaMemGetInfo(&free,&total);
  printf("AUDIT repeat=%d trial=%d ok=%d prompt=%u output=%u prefill_ms=%.3f decode_ms=%.3f tok_s=%.3f used_MiB=%.1f reason=%s\n",rep,trial,ok,st.prefill_tokens,st.generated_tokens,st.prefill_ms,st.decode_ms,1000*st.generated_tokens/st.decode_ms,(total-free)/1048576.,helios::InferenceSession::finish_reason_name(reason));fflush(stdout);
  uint64_t hash=14695981039346656037ull;for(unsigned char ch:answer){hash^=ch;hash*=1099511628211ull;}printf("OUTPUT repeat=%d trial=%d hash=%llu bytes=%zu\n",rep,trial,(unsigned long long)hash,answer.size());fflush(stdout);
  if(!ok){fprintf(stderr,"%s %s\n",code.c_str(),err.c_str());return 1;}
  if(argc>2)return 0;
 }
}
