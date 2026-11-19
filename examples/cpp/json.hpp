// Minimal RFC 8259 JSON parser and DOM value in pure C++20.
#pragma once

#include <charconv>
#include <cstdint>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace kern::json {

struct Value;
using Array = std::vector<Value>;
using Object = std::map<std::string, Value, std::less<>>;

struct Value {
    using Storage = std::variant<std::nullptr_t, bool, int64_t, double, std::string, Array, Object>;
    Storage data{nullptr};

    Value() = default;
    Value(std::nullptr_t) : data(nullptr) {}
    Value(bool v) : data(v) {}
    Value(int64_t v) : data(v) {}
    Value(int v) : data(static_cast<int64_t>(v)) {}
    Value(double v) : data(v) {}
    Value(std::string v) : data(std::move(v)) {}
    Value(const char* v) : data(std::string(v)) {}
    Value(Array v) : data(std::move(v)) {}
    Value(Object v) : data(std::move(v)) {}

    bool is_null() const { return std::holds_alternative<std::nullptr_t>(data); }
    bool is_bool() const { return std::holds_alternative<bool>(data); }
    bool is_int() const { return std::holds_alternative<int64_t>(data); }
    bool is_double() const { return std::holds_alternative<double>(data); }
    bool is_number() const { return is_int() || is_double(); }
    bool is_string() const { return std::holds_alternative<std::string>(data); }
    bool is_array() const { return std::holds_alternative<Array>(data); }
    bool is_object() const { return std::holds_alternative<Object>(data); }

    bool as_bool() const { return std::get<bool>(data); }
    int64_t as_int() const { return std::get<int64_t>(data); }
    double as_double() const { return is_int() ? static_cast<double>(as_int()) : std::get<double>(data); }
    const std::string& as_string() const { return std::get<std::string>(data); }
    const Array& as_array() const { return std::get<Array>(data); }
    const Object& as_object() const { return std::get<Object>(data); }
    Object& as_object() { return std::get<Object>(data); }

    bool contains(std::string_view key) const {
        if (!is_object()) return false;
        const auto& obj = as_object();
        return obj.find(key) != obj.end();
    }

    const Value& at(std::string_view key) const {
        const auto& obj = as_object();
        auto it = obj.find(key);
        if (it == obj.end()) {
            throw std::runtime_error("missing JSON key: " + std::string(key));
        }
        return it->second;
    }

    const Value* get(std::string_view key) const {
        if (!is_object()) return nullptr;
        const auto& obj = as_object();
        auto it = obj.find(key);
        return it == obj.end() ? nullptr : &it->second;
    }
};

namespace detail {

class Parser {
public:
    explicit Parser(std::string_view src) : src_(src) {}

    Value parse_root() {
        skip_ws();
        Value v = parse_value();
        skip_ws();
        if (pos_ != src_.size()) {
            fail("trailing characters after JSON value");
        }
        return v;
    }

private:
    std::string_view src_;
    size_t pos_ = 0;

    [[noreturn]] void fail(const std::string& msg) const {
        throw std::runtime_error("JSON parse error at offset " + std::to_string(pos_) + ": " + msg);
    }

    void skip_ws() {
        while (pos_ < src_.size()) {
            char c = src_[pos_];
            if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
                ++pos_;
            } else {
                break;
            }
        }
    }

    char peek() const {
        if (pos_ >= src_.size()) fail("unexpected end of input");
        return src_[pos_];
    }

    char next() {
        char c = peek();
        ++pos_;
        return c;
    }

    void expect(std::string_view lit) {
        if (src_.substr(pos_, lit.size()) != lit) {
            fail("expected '" + std::string(lit) + "'");
        }
        pos_ += lit.size();
    }

    static void append_utf8(std::string& out, uint32_t cp) {
        if (cp <= 0x7F) {
            out.push_back(static_cast<char>(cp));
        } else if (cp <= 0x7FF) {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp <= 0xFFFF) {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }

    uint32_t parse_hex4() {
        if (pos_ + 4 > src_.size()) fail("incomplete \\uXXXX escape");
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) {
            char c = src_[pos_++];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= static_cast<uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f') v |= static_cast<uint32_t>(10 + c - 'a');
            else if (c >= 'A' && c <= 'F') v |= static_cast<uint32_t>(10 + c - 'A');
            else fail("invalid hex digit in \\uXXXX escape");
        }
        return v;
    }

    std::string parse_string() {
        if (next() != '"') fail("expected '\"'");
        std::string out;
        while (pos_ < src_.size()) {
            char c = next();
            if (c == '"') return out;
            if (c == '\\') {
                char e = next();
                switch (e) {
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
                        if (cp >= 0xD800 && cp <= 0xDBFF && src_.substr(pos_, 2) == "\\u") {
                            pos_ += 2;
                            uint32_t low = parse_hex4();
                            cp = 0x10000 + (((cp - 0xD800) << 10) | (low - 0xDC00));
                        }
                        append_utf8(out, cp);
                        break;
                    }
                    default: fail("invalid escape sequence");
                }
            } else {
                out.push_back(c);
            }
        }
        fail("unterminated string");
    }

    Value parse_number() {
        size_t start = pos_;
        if (src_[pos_] == '-') ++pos_;
        bool is_float = false;
        while (pos_ < src_.size() && src_[pos_] >= '0' && src_[pos_] <= '9') ++pos_;
        if (pos_ < src_.size() && src_[pos_] == '.') {
            is_float = true;
            ++pos_;
            while (pos_ < src_.size() && src_[pos_] >= '0' && src_[pos_] <= '9') ++pos_;
        }
        if (pos_ < src_.size() && (src_[pos_] == 'e' || src_[pos_] == 'E')) {
            is_float = true;
            ++pos_;
            if (pos_ < src_.size() && (src_[pos_] == '+' || src_[pos_] == '-')) ++pos_;
            while (pos_ < src_.size() && src_[pos_] >= '0' && src_[pos_] <= '9') ++pos_;
        }
        std::string_view tok = src_.substr(start, pos_ - start);
        if (!is_float) {
            int64_t v = 0;
            auto [ptr, ec] = std::from_chars(tok.data(), tok.data() + tok.size(), v);
            if (ec == std::errc{} && ptr == tok.data() + tok.size()) {
                return Value(v);
            }
        }
        double d = std::stod(std::string(tok));
        return Value(d);
    }

    Value parse_array() {
        next();  // '['
        Array arr;
        skip_ws();
        if (peek() == ']') {
            next();
            return Value(std::move(arr));
        }
        while (true) {
            skip_ws();
            arr.push_back(parse_value());
            skip_ws();
            char c = next();
            if (c == ']') break;
            if (c != ',') fail("expected ',' or ']' in array");
        }
        return Value(std::move(arr));
    }

    Value parse_object() {
        next();  // '{'
        Object obj;
        skip_ws();
        if (peek() == '}') {
            next();
            return Value(std::move(obj));
        }
        while (true) {
            skip_ws();
            std::string key = parse_string();
            skip_ws();
            if (next() != ':') fail("expected ':' in object");
            skip_ws();
            obj.emplace(std::move(key), parse_value());
            skip_ws();
            char c = next();
            if (c == '}') break;
            if (c != ',') fail("expected ',' or '}' in object");
        }
        return Value(std::move(obj));
    }

    Value parse_value() {
        char c = peek();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') return Value(parse_string());
        if (c == 't') { expect("true"); return Value(true); }
        if (c == 'f') { expect("false"); return Value(false); }
        if (c == 'n') { expect("null"); return Value(nullptr); }
        if (c == '-' || (c >= '0' && c <= '9')) return parse_number();
        fail("unexpected character");
    }
};

}  // namespace detail

inline Value parse(std::string_view src) { return detail::Parser(src).parse_root(); }

inline Value parse_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open file: " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return parse(ss.str());
}

}  // namespace kern::json
