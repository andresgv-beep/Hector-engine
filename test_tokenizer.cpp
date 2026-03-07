// test_tokenizer.cpp
// ============================================================================
// Test HTF Tokenizer (carga desde HNF o HTF directo)
// ============================================================================

#include "src/hnf_loader.hpp"
#include "src/htf_tokenizer.hpp"
#include <iostream>
#include <fstream>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.hnf|tokenizer.htf> [text]" << std::endl;
        return 1;
    }
    
    std::string path = argv[1];
    std::string test_text = argc > 2 ? argv[2] : "Hello, world!";
    
    std::cout << "============================================" << std::endl;
    std::cout << "HELIOS HTF Tokenizer Test" << std::endl;
    std::cout << "============================================" << std::endl;
    
    const helios::HTFTokenizer* tokenizer = nullptr;
    helios::HTFTokenizer standalone_tokenizer;
    helios::HnfLoader loader;
    
    // Detectar si es HNF o HTF por magic bytes
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "Cannot open file: " << path << std::endl;
        return 1;
    }
    
    char magic[4];
    f.read(magic, 4);
    f.close();
    
    std::cout << ">>> Loading: " << path << std::endl;
    
    if (magic[0] == 'H' && magic[1] == 'N' && magic[2] == 'F') {
        // Es HNF - usar HnfLoader para extraer tokenizer
        std::cout << "  Detected: HNF format" << std::endl;
        
        // Crear engine dummy (no necesitamos GPU para tokenizer)
        helios::Engine engine;
        
        if (!loader.open(path)) {
            std::cerr << "Failed to open HNF" << std::endl;
            return 1;
        }
        
        if (!loader.has_tokenizer()) {
            std::cerr << "HNF does not contain tokenizer (block 0x9)" << std::endl;
            return 1;
        }
        
        tokenizer = loader.tokenizer();
        
    } else if (magic[0] == 'H' && magic[1] == 'T' && magic[2] == 'F') {
        // Es HTF directo
        std::cout << "  Detected: HTF format" << std::endl;
        
        if (!standalone_tokenizer.load_file(path, "text")) {
            std::cerr << "Failed to load HTF" << std::endl;
            return 1;
        }
        tokenizer = &standalone_tokenizer;
        
    } else {
        std::cerr << "Unknown format (magic: " << magic[0] << magic[1] << magic[2] << magic[3] << ")" << std::endl;
        return 1;
    }
    
    std::cout << "  Vocab size: " << tokenizer->vocab_size() << std::endl;
    std::cout << "  Encoding: ";
    switch (tokenizer->encoding_type()) {
        case helios::EncodingType::BPE: std::cout << "BPE"; break;
        case helios::EncodingType::SENTENCEPIECE: std::cout << "SentencePiece"; break;
        case helios::EncodingType::WORDPIECE: std::cout << "WordPiece"; break;
        default: std::cout << "Unknown";
    }
    if (tokenizer->byte_level()) std::cout << " (byte-level)";
    std::cout << std::endl;
    
    // Special tokens
    std::cout << "  Special tokens:" << std::endl;
    if (tokenizer->bos_token_id()) 
        std::cout << "    BOS: " << *tokenizer->bos_token_id() << std::endl;
    if (tokenizer->eos_token_id()) 
        std::cout << "    EOS: " << *tokenizer->eos_token_id() << std::endl;
    if (tokenizer->pad_token_id()) 
        std::cout << "    PAD: " << *tokenizer->pad_token_id() << std::endl;
    if (tokenizer->unk_token_id()) 
        std::cout << "    UNK: " << *tokenizer->unk_token_id() << std::endl;
    
    std::cout << std::endl;
    
    // Test encode
    std::cout << ">>> Encoding: \"" << test_text << "\"" << std::endl;
    
    auto ids = tokenizer->encode(test_text, false, false);
    
    std::cout << "  IDs (" << ids.size() << "): ";
    for (size_t i = 0; i < std::min(ids.size(), size_t(20)); i++) {
        std::cout << ids[i] << " ";
    }
    if (ids.size() > 20) std::cout << "...";
    std::cout << std::endl;
    
    // Show tokens
    std::cout << "  Tokens: ";
    for (size_t i = 0; i < std::min(ids.size(), size_t(20)); i++) {
        std::string tok = tokenizer->id_to_token(ids[i]);
        // Escape special chars for display
        std::string display;
        for (char c : tok) {
            if (c == '\n') display += "\\n";
            else if (c == '\t') display += "\\t";
            else if (c < 32) display += "?";
            else display += c;
        }
        std::cout << "[" << display << "] ";
    }
    std::cout << std::endl;
    
    // Test decode
    std::cout << std::endl;
    std::cout << ">>> Decoding back..." << std::endl;
    
    std::string decoded = tokenizer->decode(ids);
    std::cout << "  Result: \"" << decoded << "\"" << std::endl;
    
    // Verify roundtrip
    bool match = (decoded == test_text);
    std::cout << "  Roundtrip: " << (match ? "OK" : "MISMATCH") << std::endl;
    
    std::cout << std::endl;
    std::cout << "============================================" << std::endl;
    
    return match ? 0 : 1;
}
