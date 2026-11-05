// Configuración efectiva que deja el cargador, clave a clave y ordenada: sirve para
// comparar dos HNF del mismo modelo (por ejemplo antes y después de una edición de pesos).
// uso: hnf_config_dump modelo.hnf
#include "hnf_loader.hpp"

#include <iostream>
#include <map>
#include <sstream>
#include <variant>

int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "uso: hnf_config_dump modelo.hnf\n"; return 2; }
    helios::HnfLoader loader;
    if (!loader.load_metadata(argv[1])) { std::cerr << "hnf_config_dump: " << loader.last_error() << '\n'; return 1; }
    std::map<std::string, std::string> sorted;
    for (const auto& [key, value] : loader.config().params) {
        std::ostringstream text;
        std::visit([&](const auto& v) { text << v; }, value);
        sorted[key] = text.str();
    }
    std::cout << "format " << loader.format_major() << "\ntensors " << loader.tensors().size() << '\n';
    for (const auto& [key, value] : sorted) std::cout << key << " = " << value << '\n';
    return 0;
}
