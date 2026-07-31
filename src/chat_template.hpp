#pragma once

#include <string>
#include <vector>

namespace helios {

struct ChatMessage {
    std::string role;
    std::string content;
};

struct Gemma4ChatOptions {
    bool add_generation_prompt = true;
    bool enable_thinking = false;
};

// Text-only subset of Google's canonical Gemma 4 template.  Media and tool
// calls belong to the later multimodal/tooling phases; normal system, user and
// assistant history is formatted byte-for-byte like chat_template.jinja.
std::string format_gemma4_chat(
    const std::vector<ChatMessage>& messages,
    const Gemma4ChatOptions& options = {});

}  // namespace helios
