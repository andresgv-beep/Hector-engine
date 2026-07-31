#include "chat_template.hpp"

#include <cctype>
#include <stdexcept>

namespace helios {
namespace {

std::string trim(const std::string& value) {
    size_t first = 0;
    while (first < value.size() &&
           std::isspace(static_cast<unsigned char>(value[first]))) {
        ++first;
    }
    size_t last = value.size();
    while (last > first &&
           std::isspace(static_cast<unsigned char>(value[last - 1]))) {
        --last;
    }
    return value.substr(first, last - first);
}

std::string strip_thinking(const std::string& value) {
    static const std::string channel_start = "<|channel>";
    static const std::string channel_end = "<channel|>";
    std::string result;
    size_t pos = 0;
    while (pos < value.size()) {
        const size_t channel = value.find(channel_start, pos);
        if (channel == std::string::npos) {
            result.append(value, pos, std::string::npos);
            break;
        }
        result.append(value, pos, channel - pos);
        const size_t end = value.find(channel_end, channel + channel_start.size());
        if (end == std::string::npos) break;
        pos = end + channel_end.size();
    }
    return trim(result);
}

}  // namespace

std::string format_gemma4_chat(const std::vector<ChatMessage>& messages,
                               const Gemma4ChatOptions& options) {
    std::string out = "<bos>";
    size_t first_message = 0;
    const bool has_system = !messages.empty() &&
        (messages.front().role == "system" || messages.front().role == "developer");

    if (options.enable_thinking || has_system) {
        out += "<|turn>system\n";
        if (options.enable_thinking) out += "<|think|>\n";
        if (has_system) {
            out += trim(messages.front().content);
            first_message = 1;
        }
        out += "<turn|>\n";
    }

    for (size_t i = first_message; i < messages.size(); ++i) {
        const auto& message = messages[i];
        if (message.role == "system" || message.role == "developer") {
            throw std::invalid_argument("Gemma 4 system/developer message must be first");
        }
        const std::string role = message.role == "assistant" ? "model" : message.role;
        if (role != "user" && role != "model") {
            throw std::invalid_argument("unsupported Gemma 4 text-chat role: " + message.role);
        }
        out += "<|turn>" + role + "\n";
        out += role == "model" ? strip_thinking(message.content) : trim(message.content);
        out += "<turn|>\n";
    }

    if (options.add_generation_prompt) out += "<|turn>model\n";
    return out;
}

}  // namespace helios
