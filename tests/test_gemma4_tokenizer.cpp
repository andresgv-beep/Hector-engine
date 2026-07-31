#include "hnf_loader.hpp"
#include "htf_tokenizer.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(1);
    }
}

void check_case(const helios::HTFTokenizer& tokenizer,
                const std::string& text,
                const std::vector<int32_t>& expected) {
    const auto actual = tokenizer.encode(text, false, false);
    require(actual == expected, "token IDs differ for: " + text);
    require(tokenizer.decode(actual) == text, "round-trip differs for: " + text);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "SKIP: pass gemma4-e2b-text-*.hnf to run real tokenizer checks\n";
        return 0;
    }

    helios::HnfLoader loader;
    require(loader.open(argv[1]), "cannot open HNF");
    require(loader.has_tokenizer(), "HNF has no HTF tokenizer block");
    const auto* tokenizer = loader.tokenizer();
    require(tokenizer != nullptr, "tokenizer was not loaded");

    require(tokenizer->is_htf_v13(), "expected HTF v1.3");
    require(tokenizer->encoding_type() == helios::EncodingType::BPE, "expected BPE");
    require(tokenizer->vocab_size() == 262144, "vocab size must be 262144");
    require(tokenizer->merge_count() == 514906, "merge count must be 514906");
    require(tokenizer->added_token_count() == 24, "added-token count must be 24");
    require(!tokenizer->byte_level(), "Gemma 4 is not GPT-2 ByteLevel");
    require(tokenizer->byte_fallback(), "Gemma 4 requires byte fallback");
    require(tokenizer->space_to_metaspace_normalizer(), "missing space-to-metaspace normalizer");
    require(tokenizer->metaspace_decoder(), "missing metaspace decoder");
    require(!tokenizer->add_bos_token_by_default(),
            "Gemma 4 IT chat template owns the explicit BOS token");

    require(tokenizer->bos_token_id() == 2, "BOS must be 2");
    require(tokenizer->eos_token_id() == 1, "EOS must be 1");
    require(tokenizer->pad_token_id() == 0, "PAD must be 0");
    require(tokenizer->unk_token_id() == 3, "UNK must be 3");
    require(tokenizer->token_to_id("<|turn>") == 105, "<|turn> must be 105");
    require(tokenizer->token_to_id("<turn|>") == 106, "<turn|> must be 106");
    require(tokenizer->token_to_id("<|channel>") == 100, "<|channel> must be 100");
    require(tokenizer->token_to_id("<channel|>") == 101, "<channel|> must be 101");
    require(tokenizer->token_to_id("<|think|>") == 98, "<|think|> must be 98");
    require(tokenizer->token_to_id("<|image|>") == 258880, "image token mismatch");
    require(tokenizer->token_to_id("<|audio|>") == 258881, "audio token mismatch");

    check_case(*tokenizer, "Hello world", {9259, 1902});
    check_case(*tokenizer, "Hola socio, ¿cómo estás?",
               {21529, 29944, 236764, 7196, 88565, 64135, 236881});
    check_case(*tokenizer, "línea 1\nlínea 2",
               {58740, 17802, 236743, 236770, 107, 58740, 17802, 236743, 236778});
    check_case(*tokenizer, "Unicode: café, 日本語, 😀",
               {123540, 236787, 33443, 236764, 33375, 238582, 236764, 163543});
    check_case(*tokenizer, "  espacios  dobles ",
               {138, 24132, 59615, 138, 46959, 1074, 236743});
    check_case(*tokenizer, "<|turn>user\nHola<turn|>\n<|turn>model\n",
               {105, 2364, 107, 21529, 106, 107, 105, 4368, 107});
    check_case(*tokenizer, "\xF4\x8F\xBF\xBF", {482, 381, 429, 429});

    require(tokenizer->encode("Hello world", true, false) ==
                std::vector<int32_t>({2, 9259, 1902}),
            "explicit BOS does not match upstream post-processor");
    require(tokenizer->encode("", false, true) == std::vector<int32_t>({1}),
            "EOS insertion failed");

    std::cout << "Gemma 4 HTF3 tokenizer: all reference vectors passed\n";
    return 0;
}
