#pragma once

// Parser JSON mínimo pero CORRECTO, para el protocolo de helios_runtime.
//
// La primera versión buscaba claves con `find()` y contaba llaves ignorando
// las que iban dentro de cadenas. Un mensaje con código —
//   {"content":"if (x) { return {\"ok\": true}; }"}
// — la rompía, y `"temperature"` se encontraba aunque estuviera dentro del
// texto del usuario. Un cliente hostil no hace falta: basta con pegarle código
// a Héctor.
//
// Esto tokeniza de verdad, respeta escapes y pares surrogate, y busca claves
// por ÁMBITO en vez de por posición en la línea.

#include <cctype>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace helios::mini_json {

struct Value;
using Object = std::map<std::string, Value>;
using Array = std::vector<Value>;

struct Value {
    enum class Type { Null, Bool, Number, String, Array, Object } type = Type::Null;
    bool boolean = false;
    double number = 0.0;
    std::string text;
    std::shared_ptr<Array> array;
    std::shared_ptr<Object> object;

    bool is_object() const { return type == Type::Object; }
    bool is_array() const { return type == Type::Array; }

    // Acceso por ámbito: `get("generation").get("temperature")`. Nunca busca
    // una clave "en cualquier parte de la línea".
    const Value* get(const std::string& key) const {
        if (type != Type::Object || !object) return nullptr;
        auto it = object->find(key);
        return it == object->end() ? nullptr : &it->second;
    }
    std::string str(const std::string& def = {}) const {
        return type == Type::String ? text : def;
    }
    double num(double def) const { return type == Type::Number ? number : def; }
    bool boo(bool def) const { return type == Type::Bool ? boolean : def; }
};

class Parser {
public:
    // Devuelve false y llena `error` si el JSON está mal formado. Un cliente
    // que manda basura recibe un error explícito, no un objeto a medias.
    static bool parse(const std::string& src, Value* out, std::string* error);

private:
    explicit Parser(const std::string& s) : s_(s) {}
    bool value(Value* v);
    bool object(Value* v);
    bool array(Value* v);
    bool string(std::string* out);
    bool number(Value* v);
    bool literal(const char* lit, Value* v);
    void skip_ws();
    bool fail(const char* why);

    const std::string& s_;
    size_t i_ = 0;
    std::string err_;
};

inline void Parser::skip_ws() {
    while (i_ < s_.size() &&
           (s_[i_] == ' ' || s_[i_] == '\t' || s_[i_] == '\n' || s_[i_] == '\r')) {
        i_++;
    }
}

inline bool Parser::fail(const char* why) {
    if (err_.empty()) {
        err_ = std::string(why) + " en la posición " + std::to_string(i_);
    }
    return false;
}

inline bool Parser::string(std::string* out) {
    if (i_ >= s_.size() || s_[i_] != '"') return fail("se esperaba una cadena");
    i_++;
    out->clear();
    while (i_ < s_.size()) {
        unsigned char c = static_cast<unsigned char>(s_[i_]);
        if (c == '"') { i_++; return true; }
        if (c != '\\') {
            if (c < 0x20) return fail("carácter de control sin escapar");
            *out += static_cast<char>(c);
            i_++;
            continue;
        }
        i_++;
        if (i_ >= s_.size()) return fail("escape truncado");
        char e = s_[i_++];
        switch (e) {
            case '"':  *out += '"';  break;
            case '\\': *out += '\\'; break;
            case '/':  *out += '/';  break;
            case 'b':  *out += '\b'; break;
            case 'f':  *out += '\f'; break;
            case 'n':  *out += '\n'; break;
            case 'r':  *out += '\r'; break;
            case 't':  *out += '\t'; break;
            case 'u': {
                if (i_ + 4 > s_.size()) return fail("\\u truncado");
                uint32_t cp = 0;
                for (int k = 0; k < 4; k++) {
                    char h = s_[i_ + k];
                    cp <<= 4;
                    if (h >= '0' && h <= '9') cp |= (h - '0');
                    else if (h >= 'a' && h <= 'f') cp |= (h - 'a' + 10);
                    else if (h >= 'A' && h <= 'F') cp |= (h - 'A' + 10);
                    else return fail("\\u con dígito no hexadecimal");
                }
                i_ += 4;
                // Par surrogate: un emoji escapado viaja como 😀 y
                // tratarlos por separado produce UTF-8 inválido.
                if (cp >= 0xD800 && cp <= 0xDBFF) {
                    if (i_ + 6 > s_.size() || s_[i_] != '\\' || s_[i_ + 1] != 'u') {
                        return fail("surrogate alto sin su pareja");
                    }
                    uint32_t lo = 0;
                    for (int k = 0; k < 4; k++) {
                        char h = s_[i_ + 2 + k];
                        lo <<= 4;
                        if (h >= '0' && h <= '9') lo |= (h - '0');
                        else if (h >= 'a' && h <= 'f') lo |= (h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') lo |= (h - 'A' + 10);
                        else return fail("surrogate bajo inválido");
                    }
                    if (lo < 0xDC00 || lo > 0xDFFF) return fail("surrogate bajo fuera de rango");
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    i_ += 6;
                } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                    return fail("surrogate bajo huérfano");
                }
                if (cp < 0x80) {
                    *out += static_cast<char>(cp);
                } else if (cp < 0x800) {
                    *out += static_cast<char>(0xC0 | (cp >> 6));
                    *out += static_cast<char>(0x80 | (cp & 0x3F));
                } else if (cp < 0x10000) {
                    *out += static_cast<char>(0xE0 | (cp >> 12));
                    *out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                    *out += static_cast<char>(0x80 | (cp & 0x3F));
                } else {
                    *out += static_cast<char>(0xF0 | (cp >> 18));
                    *out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
                    *out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                    *out += static_cast<char>(0x80 | (cp & 0x3F));
                }
                break;
            }
            default: return fail("escape desconocido");
        }
    }
    return fail("cadena sin cerrar");
}

inline bool Parser::number(Value* v) {
    size_t start = i_;
    if (i_ < s_.size() && (s_[i_] == '-' || s_[i_] == '+')) i_++;
    bool any = false;
    while (i_ < s_.size() && (isdigit((unsigned char)s_[i_]) || s_[i_] == '.' ||
                              s_[i_] == 'e' || s_[i_] == 'E' ||
                              ((s_[i_] == '-' || s_[i_] == '+') &&
                               (s_[i_ - 1] == 'e' || s_[i_ - 1] == 'E')))) {
        if (isdigit((unsigned char)s_[i_])) any = true;
        i_++;
    }
    if (!any) return fail("número inválido");
    try {
        v->number = std::stod(s_.substr(start, i_ - start));
    } catch (...) {
        return fail("número fuera de rango");
    }
    v->type = Value::Type::Number;
    return true;
}

inline bool Parser::literal(const char* lit, Value* v) {
    size_t n = strlen(lit);
    if (s_.compare(i_, n, lit) != 0) return fail("literal desconocido");
    i_ += n;
    if (lit[0] == 'n') { v->type = Value::Type::Null; }
    else { v->type = Value::Type::Bool; v->boolean = (lit[0] == 't'); }
    return true;
}

inline bool Parser::array(Value* v) {
    i_++;  // [
    v->type = Value::Type::Array;
    v->array = std::make_shared<Array>();
    skip_ws();
    if (i_ < s_.size() && s_[i_] == ']') { i_++; return true; }
    while (true) {
        Value item;
        if (!value(&item)) return false;
        v->array->push_back(std::move(item));
        skip_ws();
        if (i_ >= s_.size()) return fail("array sin cerrar");
        if (s_[i_] == ',') { i_++; skip_ws(); continue; }
        if (s_[i_] == ']') { i_++; return true; }
        return fail("se esperaba ',' o ']'");
    }
}

inline bool Parser::object(Value* v) {
    i_++;  // {
    v->type = Value::Type::Object;
    v->object = std::make_shared<Object>();
    skip_ws();
    if (i_ < s_.size() && s_[i_] == '}') { i_++; return true; }
    while (true) {
        skip_ws();
        std::string key;
        if (!string(&key)) return false;
        skip_ws();
        if (i_ >= s_.size() || s_[i_] != ':') return fail("se esperaba ':'");
        i_++;
        Value item;
        if (!value(&item)) return false;
        (*v->object)[key] = std::move(item);
        skip_ws();
        if (i_ >= s_.size()) return fail("objeto sin cerrar");
        if (s_[i_] == ',') { i_++; continue; }
        if (s_[i_] == '}') { i_++; return true; }
        return fail("se esperaba ',' o '}'");
    }
}

inline bool Parser::value(Value* v) {
    skip_ws();
    if (i_ >= s_.size()) return fail("fin inesperado");
    char c = s_[i_];
    if (c == '{') return object(v);
    if (c == '[') return array(v);
    if (c == '"') {
        v->type = Value::Type::String;
        return string(&v->text);
    }
    if (c == 't') return literal("true", v);
    if (c == 'f') return literal("false", v);
    if (c == 'n') return literal("null", v);
    return number(v);
}

inline bool Parser::parse(const std::string& src, Value* out, std::string* error) {
    Parser p(src);
    if (!p.value(out)) { *error = p.err_; return false; }
    p.skip_ws();
    if (p.i_ != src.size()) {
        *error = "sobra contenido tras el valor JSON";
        return false;
    }
    return true;
}

}  // namespace helios::mini_json
