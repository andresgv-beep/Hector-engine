#include "src/hnf_loader.hpp"
#include "src/htf_tokenizer.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
int main(int argc, char** argv) {
    helios::HnfLoader L; if (!L.open(argv[1])) return 1;
    const auto* tok = L.tokenizer(); if (!tok) return 1;
    if (std::string(argv[2]) == "enc") {           // texto -> ids
        std::string s; int c; while ((c=getchar())!=EOF) s+=(char)c;
        auto ids = tok->encode(s, false, false);
        for (size_t i=0;i<ids.size();i++) printf("%d%s", ids[i], i+1<ids.size()?",":"");
        return 0;
    }
    std::vector<int32_t> ids; std::string cur;      // ids -> texto
    for (const char* q=argv[3]; ; ++q) { if(*q==','||!*q){ ids.push_back(atoi(cur.c_str())); cur.clear(); if(!*q)break;} else cur+=*q; }
    fputs(tok->decode(ids).c_str(), stdout);
    return 0;
}
