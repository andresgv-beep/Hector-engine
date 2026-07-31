#include "chat_template.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(1);
    }
}

}  // namespace

int main() {
    using helios::ChatMessage;
    using helios::Gemma4ChatOptions;
    using helios::format_gemma4_chat;

    require(format_gemma4_chat({{"user", "  Hola  "}}) ==
                "<bos><|turn>user\nHola<turn|>\n<|turn>model\n",
            "single user turn");

    require(format_gemma4_chat({
                {"system", " Sé breve.\n"},
                {"user", "Hola"},
                {"assistant", "<|channel>thought\nsecreto<channel|>Buenas"},
                {"user", "Adiós"},
            }) ==
                "<bos><|turn>system\nSé breve.<turn|>\n"
                "<|turn>user\nHola<turn|>\n"
                "<|turn>model\nBuenas<turn|>\n"
                "<|turn>user\nAdiós<turn|>\n"
                "<|turn>model\n",
            "system and multi-turn history");

    Gemma4ChatOptions thinking;
    thinking.enable_thinking = true;
    require(format_gemma4_chat({{"user", "Piensa"}}, thinking) ==
                "<bos><|turn>system\n<|think|>\n<turn|>\n"
                "<|turn>user\nPiensa<turn|>\n<|turn>model\n",
            "thinking trigger");

    bool rejected = false;
    try {
        (void)format_gemma4_chat({{"user", "a"}, {"system", "late"}});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "late system message must be rejected");

    std::cout << "Gemma 4 canonical text chat template: passed\n";
    return 0;
}
