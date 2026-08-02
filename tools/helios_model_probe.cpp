#include "model_capabilities.hpp"

#include <iostream>
#include <string>

namespace {

std::string json_escape(const std::string& input) {
    std::string output;
    output.reserve(input.size() + 8);
    for (const unsigned char c : input) {
        switch (c) {
            case '\\': output += "\\\\"; break;
            case '"': output += "\\\""; break;
            case '\n': output += "\\n"; break;
            case '\r': output += "\\r"; break;
            case '\t': output += "\\t"; break;
            default:
                if (c >= 0x20) output += static_cast<char>(c);
                break;
        }
    }
    return output;
}

void print_kv(const helios::ModelCapabilities& model) {
    std::cout << "schema=helios.model-capabilities.v1\n"
              << "schema_version=" << model.schema_version << '\n'
              << "text_architecture=" << model.text_architecture << '\n'
              << "multimodal=" << (model.multimodal ? 1 : 0) << '\n';
    for (const auto& capability : model.modalities) {
        const char* name = helios::model_modality_name(capability.modality);
        std::cout << "modality." << name << ".declared="
                  << (capability.declared ? 1 : 0) << '\n'
                  << "modality." << name << ".present="
                  << (capability.present ? 1 : 0) << '\n'
                  << "modality." << name << ".configured="
                  << (capability.configured ? 1 : 0) << '\n'
                  << "modality." << name << ".architecture="
                  << capability.architecture << '\n'
                  << "modality." << name << ".adapter="
                  << capability.adapter_id << '\n'
                  << "modality." << name << ".status="
                  << helios::adapter_status_name(capability.status) << '\n';
    }
    std::cout << "diagnostic_count=" << model.diagnostics.size() << '\n';
    for (size_t i = 0; i < model.diagnostics.size(); ++i) {
        std::string value = model.diagnostics[i];
        for (char& c : value) if (c == '\n' || c == '\r') c = ' ';
        std::cout << "diagnostic." << i << '=' << value << '\n';
    }
}

void print_json(const helios::ModelCapabilities& model) {
    std::cout << "{\"schema\":\"helios.model-capabilities.v1\","
              << "\"schema_version\":" << model.schema_version << ','
              << "\"text_architecture\":\""
              << json_escape(model.text_architecture) << "\","
              << "\"multimodal\":" << (model.multimodal ? "true" : "false")
              << ",\"modalities\":[";
    for (size_t i = 0; i < model.modalities.size(); ++i) {
        const auto& c = model.modalities[i];
        if (i) std::cout << ',';
        std::cout << "{\"name\":\"" << helios::model_modality_name(c.modality)
                  << "\",\"declared\":" << (c.declared ? "true" : "false")
                  << ",\"present\":" << (c.present ? "true" : "false")
                  << ",\"configured\":" << (c.configured ? "true" : "false")
                  << ",\"architecture\":\"" << json_escape(c.architecture)
                  << "\",\"adapter\":\"" << json_escape(c.adapter_id)
                  << "\",\"status\":\""
                  << helios::adapter_status_name(c.status) << "\"}";
    }
    std::cout << "],\"diagnostics\":[";
    for (size_t i = 0; i < model.diagnostics.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << '"' << json_escape(model.diagnostics[i]) << '"';
    }
    std::cout << "]}\n";
}

void print_human(const helios::ModelCapabilities& model) {
    std::cout << "HNF capabilities v" << model.schema_version
              << " · text=" << model.text_architecture
              << " · multimodal=" << (model.multimodal ? "yes" : "no")
              << '\n';
    for (const auto& capability : model.modalities) {
        if (!capability.declared && !capability.present) continue;
        std::cout << "  " << helios::model_modality_name(capability.modality)
                  << ": " << helios::adapter_status_name(capability.status);
        if (!capability.architecture.empty()) {
            std::cout << " · arch=" << capability.architecture;
        }
        if (!capability.adapter_id.empty()) {
            std::cout << " · adapter=" << capability.adapter_id;
        }
        std::cout << '\n';
    }
    for (const auto& diagnostic : model.diagnostics) {
        std::cout << "  warning: " << diagnostic << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    enum class Format { Human, Kv, Json } format = Format::Human;
    const char* path = nullptr;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--kv") format = Format::Kv;
        else if (arg == "--json") format = Format::Json;
        else if (!path) path = argv[i];
        else {
            std::cerr << "usage: helios_model_probe [--kv|--json] model.hnf\n";
            return 2;
        }
    }
    if (!path) {
        std::cerr << "usage: helios_model_probe [--kv|--json] model.hnf\n";
        return 2;
    }

    helios::HnfLoader loader;
    if (!loader.load_metadata(path)) {
        std::cerr << "helios_model_probe: " << loader.last_error() << '\n';
        return 1;
    }
    const auto model = helios::inspect_model_capabilities(loader);
    if (format == Format::Kv) print_kv(model);
    else if (format == Format::Json) print_json(model);
    else print_human(model);
    return 0;
}
