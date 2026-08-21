#pragma once
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace mmb::json {

struct value;
using object = std::map<std::string, value>;
using array = std::vector<value>;

struct value {
    using storage = std::variant<std::nullptr_t, bool, double, std::string, array, object>;
    storage data;

    bool is_object() const { return std::holds_alternative<object>(data); }
    bool is_array() const { return std::holds_alternative<array>(data); }
    bool is_string() const { return std::holds_alternative<std::string>(data); }
    bool is_number() const { return std::holds_alternative<double>(data); }
    bool is_bool() const { return std::holds_alternative<bool>(data); }

    const object & as_object() const {
        if (!is_object()) throw std::runtime_error("JSON value is not an object");
        return std::get<object>(data);
    }
    const array & as_array() const {
        if (!is_array()) throw std::runtime_error("JSON value is not an array");
        return std::get<array>(data);
    }
    const std::string & as_string() const {
        if (!is_string()) throw std::runtime_error("JSON value is not a string");
        return std::get<std::string>(data);
    }
    double as_number() const {
        if (!is_number()) throw std::runtime_error("JSON value is not a number");
        return std::get<double>(data);
    }
    bool as_bool() const {
        if (!is_bool()) throw std::runtime_error("JSON value is not a boolean");
        return std::get<bool>(data);
    }
};

class parser {
public:
    explicit parser(std::string_view input) : input_(input) {}

    value parse() {
        skip_ws();
        value v = parse_value();
        skip_ws();
        if (pos_ != input_.size()) fail("trailing data");
        return v;
    }

private:
    std::string_view input_;
    size_t pos_ = 0;

    [[noreturn]] void fail(const char * message) const {
        throw std::runtime_error(std::string("JSON parse error at byte ") +
                                 std::to_string(pos_) + ": " + message);
    }

    void skip_ws() {
        while (pos_ < input_.size() &&
               (input_[pos_] == ' ' || input_[pos_] == '\n' ||
                input_[pos_] == '\r' || input_[pos_] == '\t')) {
            ++pos_;
        }
    }

    bool take(char c) {
        skip_ws();
        if (pos_ < input_.size() && input_[pos_] == c) {
            ++pos_;
            return true;
        }
        return false;
    }

    void expect(char c) {
        if (!take(c)) fail("unexpected token");
    }

    value parse_value() {
        skip_ws();
        if (pos_ >= input_.size()) fail("unexpected end of input");
        switch (input_[pos_]) {
            case '{': return value{parse_object()};
            case '[': return value{parse_array()};
            case '"': return value{parse_string()};
            case 't': consume_literal("true"); return value{true};
            case 'f': consume_literal("false"); return value{false};
            case 'n': consume_literal("null"); return value{nullptr};
            default:
                if (input_[pos_] == '-' || std::isdigit(static_cast<unsigned char>(input_[pos_]))) {
                    return value{parse_number()};
                }
                fail("invalid value");
        }
    }

    object parse_object() {
        expect('{');
        object out;
        skip_ws();
        if (take('}')) return out;
        while (true) {
            skip_ws();
            if (pos_ >= input_.size() || input_[pos_] != '"') fail("object key must be a string");
            std::string key = parse_string();
            expect(':');
            auto [it, inserted] = out.emplace(std::move(key), parse_value());
            if (!inserted) fail("duplicate object key");
            if (take('}')) return out;
            expect(',');
        }
    }

    array parse_array() {
        expect('[');
        array out;
        skip_ws();
        if (take(']')) return out;
        while (true) {
            out.push_back(parse_value());
            if (take(']')) return out;
            expect(',');
        }
    }

    static void append_utf8(std::string & out, uint32_t cp) {
        if (cp <= 0x7f) {
            out.push_back(static_cast<char>(cp));
        } else if (cp <= 0x7ff) {
            out.push_back(static_cast<char>(0xc0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
        } else if (cp <= 0xffff) {
            out.push_back(static_cast<char>(0xe0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
        } else {
            out.push_back(static_cast<char>(0xf0 | (cp >> 18)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3f)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
        }
    }

    uint32_t parse_hex4() {
        if (pos_ + 4 > input_.size()) fail("short unicode escape");
        uint32_t cp = 0;
        for (int i = 0; i < 4; ++i) {
            char c = input_[pos_++];
            cp <<= 4;
            if (c >= '0' && c <= '9') cp |= static_cast<uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f') cp |= static_cast<uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') cp |= static_cast<uint32_t>(c - 'A' + 10);
            else fail("invalid unicode escape");
        }
        return cp;
    }

    std::string parse_string() {
        if (pos_ >= input_.size() || input_[pos_] != '"') fail("expected string");
        ++pos_;
        std::string out;
        while (pos_ < input_.size()) {
            char c = input_[pos_++];
            if (c == '"') return out;
            if (static_cast<unsigned char>(c) < 0x20) fail("control byte in string");
            if (c != '\\') {
                out.push_back(c);
                continue;
            }
            if (pos_ >= input_.size()) fail("short escape");
            char esc = input_[pos_++];
            switch (esc) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': {
                    uint32_t cp = parse_hex4();
                    if (cp >= 0xd800 && cp <= 0xdbff) {
                        if (pos_ + 2 > input_.size() || input_[pos_] != '\\' || input_[pos_ + 1] != 'u') {
                            fail("unpaired high surrogate");
                        }
                        pos_ += 2;
                        uint32_t low = parse_hex4();
                        if (low < 0xdc00 || low > 0xdfff) fail("invalid low surrogate");
                        cp = 0x10000 + ((cp - 0xd800) << 10) + (low - 0xdc00);
                    } else if (cp >= 0xdc00 && cp <= 0xdfff) {
                        fail("unpaired low surrogate");
                    }
                    append_utf8(out, cp);
                    break;
                }
                default: fail("invalid escape");
            }
        }
        fail("unterminated string");
    }

    double parse_number() {
        skip_ws();
        size_t start = pos_;
        if (input_[pos_] == '-') ++pos_;
        if (pos_ >= input_.size()) fail("short number");
        if (input_[pos_] == '0') {
            ++pos_;
        } else {
            if (!std::isdigit(static_cast<unsigned char>(input_[pos_]))) fail("invalid number");
            while (pos_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[pos_]))) ++pos_;
        }
        if (pos_ < input_.size() && input_[pos_] == '.') {
            ++pos_;
            if (pos_ >= input_.size() || !std::isdigit(static_cast<unsigned char>(input_[pos_]))) fail("invalid fraction");
            while (pos_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[pos_]))) ++pos_;
        }
        if (pos_ < input_.size() && (input_[pos_] == 'e' || input_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < input_.size() && (input_[pos_] == '+' || input_[pos_] == '-')) ++pos_;
            if (pos_ >= input_.size() || !std::isdigit(static_cast<unsigned char>(input_[pos_]))) fail("invalid exponent");
            while (pos_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[pos_]))) ++pos_;
        }
        std::string tmp(input_.substr(start, pos_ - start));
        char * end = nullptr;
        double result = std::strtod(tmp.c_str(), &end);
        if (end == nullptr || *end != '\0') fail("invalid number");
        return result;
    }

    void consume_literal(std::string_view literal) {
        if (input_.substr(pos_, literal.size()) != literal) fail("invalid literal");
        pos_ += literal.size();
    }
};

inline value parse(std::string_view input) {
    return parser(input).parse();
}

inline value parse_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open JSON file: " + path);
    std::string data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return parse(data);
}

inline const value & require(const object & obj, const char * key) {
    auto it = obj.find(key);
    if (it == obj.end()) throw std::runtime_error(std::string("missing JSON field: ") + key);
    return it->second;
}

inline uint64_t require_u64(const object & obj, const char * key) {
    double d = require(obj, key).as_number();
    if (d < 0 || d > 9007199254740991.0 || d != static_cast<double>(static_cast<uint64_t>(d))) {
        throw std::runtime_error(std::string("invalid non-negative integer field: ") + key);
    }
    return static_cast<uint64_t>(d);
}

inline uint32_t require_u32(const object & obj, const char * key) {
    uint64_t v = require_u64(obj, key);
    if (v > 0xffffffffULL) throw std::runtime_error(std::string("integer too large: ") + key);
    return static_cast<uint32_t>(v);
}

} // namespace mmb::json
