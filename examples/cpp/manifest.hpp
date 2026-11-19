// Pure-C++20 reader for a kern v7 manifest and its serving protocol.
//
// Parses a manifest JSON, expands any `constants` table (without rewriting
// identifier/reference fields), evaluates the closed scalar expression set
// (`ceil_div`, `mul`, `add`, `clamp`), and projects `Protocol` so a driver
// never hardcodes buffer, var, or program names. Zero CUDA dependencies.
#pragma once

#include "json.hpp"

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

namespace kern {

using VarMap = std::unordered_map<std::string, int64_t>;
using Dim = std::variant<int64_t, std::string>;

inline size_t dtype_bytes(std::string_view dt) {
    if (dt == "u8" || dt == "i8" || dt == "fp8e4m3") return 1;
    if (dt == "bf16" || dt == "f16" || dt == "u16" || dt == "i16") return 2;
    if (dt == "f32" || dt == "i32" || dt == "u32") return 4;
    if (dt == "f64" || dt == "i64" || dt == "u64") return 8;
    throw std::runtime_error("unsupported dtype: " + std::string(dt));
}

inline int64_t eval_expr(const json::Value& expr, const VarMap& vals) {
    if (expr.is_int()) return expr.as_int();
    if (expr.is_string()) {
        auto it = vals.find(expr.as_string());
        if (it == vals.end()) {
            throw std::runtime_error("unknown var in expression: " + expr.as_string());
        }
        return it->second;
    }
    if (expr.is_object()) {
        if (const auto* a = expr.get("ceil_div")) {
            const auto& xs = a->as_array();
            int64_t e = eval_expr(xs[0], vals);
            int64_t c = eval_expr(xs[1], vals);
            return (e + c - 1) / c;
        }
        if (const auto* a = expr.get("mul")) {
            const auto& xs = a->as_array();
            return eval_expr(xs[0], vals) * eval_expr(xs[1], vals);
        }
        if (const auto* a = expr.get("add")) {
            const auto& xs = a->as_array();
            return eval_expr(xs[0], vals) + eval_expr(xs[1], vals);
        }
        if (const auto* a = expr.get("clamp")) {
            const auto& xs = a->as_array();
            int64_t e = eval_expr(xs[0], vals);
            int64_t start = eval_expr(xs[1], vals);
            int64_t len = eval_expr(xs[2], vals);
            return std::min(std::max(e - start, int64_t{0}), len);
        }
    }
    throw std::runtime_error("unsupported manifest expression");
}

namespace detail {

inline bool is_name_key(std::string_view k) {
    static const std::unordered_set<std::string_view> kNames = {
        "model", "dtype", "kind", "fill", "axis", "index_into", "tensor",
        "source", "sha256", "module", "entry", "params", "op", "buf",
        "state", "var", "scratch", "context", "label", "lane", "after",
    };
    return kNames.contains(k);
}

inline json::Value expand_constants(const json::Value& v, const json::Object& consts) {
    if (consts.empty()) return v;
    if (v.is_string()) {
        auto it = consts.find(v.as_string());
        return it != consts.end() ? it->second : v;
    }
    if (v.is_array()) {
        json::Array out;
        out.reserve(v.as_array().size());
        for (const auto& item : v.as_array()) {
            out.push_back(expand_constants(item, consts));
        }
        return json::Value(std::move(out));
    }
    if (v.is_object()) {
        json::Object out;
        for (const auto& [k, val] : v.as_object()) {
            out.emplace(k, is_name_key(k) ? val : expand_constants(val, consts));
        }
        return json::Value(std::move(out));
    }
    return v;
}

inline std::vector<Dim> parse_shape(const json::Value& arr) {
    std::vector<Dim> out;
    for (const auto& d : arr.as_array()) {
        if (d.is_int()) out.emplace_back(d.as_int());
        else out.emplace_back(d.as_string());
    }
    return out;
}

}  // namespace detail

struct Filled {
    std::string name;
    std::string fill;
    std::string dtype;
    std::vector<Dim> shape;
    std::string axis;  // "rows" | "groups" | "ranks" | "fixed"
};

struct Table {
    std::string name;
    std::string state;
    std::string dtype;
    int64_t stride = 1;
    std::vector<Dim> shape;
    int64_t lines_per_slot = 0;  // 0 for page tables, >0 for per-seq line tables
};

struct Forward {
    std::string name;
    int64_t groups = 1;
    Dim rows;  // int64_t for fixed-row decode/round, string var name for chunk prefill
    std::optional<std::string> context_var;
    bool graph = false;
    std::optional<std::string> emits;
    std::optional<std::string> count;
};

struct Sampling {
    float temperature = 0.0f;
    int32_t top_k = 0;
    float top_p = 1.0f;
    float min_p = 0.0f;
    int64_t seed = 0;
};

struct Protocol;

struct Manifest {
    json::Value raw;
    std::string model;
    VarMap vars_max;

    static Manifest from_json(const json::Value& input) {
        json::Value expanded;
        if (const auto* c = input.get("constants"); c && c->is_object() && !c->as_object().empty()) {
            json::Object obj;
            for (const auto& [k, v] : input.as_object()) {
                if (k == "constants") continue;
                obj.emplace(k, detail::expand_constants(v, c->as_object()));
            }
            expanded = json::Value(std::move(obj));
        } else {
            expanded = input;
        }

        const auto* sv = expanded.get("schema_version");
        if (!sv || !sv->is_int() || sv->as_int() != 7) {
            throw std::runtime_error("unsupported schema_version; expected 7");
        }

        Manifest m;
        m.raw = std::move(expanded);
        m.model = m.raw.at("model").as_string();
        if (const auto* vars = m.raw.get("vars")) {
            for (const auto& [k, v] : vars->as_object()) {
                m.vars_max[k] = v.at("max").as_int();
            }
        }
        return m;
    }

    static Manifest load(const std::string& path) { return from_json(json::parse_file(path)); }

    const json::Object& vars() const {
        static const json::Object kEmpty;
        const auto* v = raw.get("vars");
        return v ? v->as_object() : kEmpty;
    }

    const json::Object& states() const {
        static const json::Object kEmpty;
        const auto* v = raw.get("states");
        return v ? v->as_object() : kEmpty;
    }

    const json::Object& buffers() const { return raw.at("buffers").as_object(); }

    const json::Object& modules() const {
        static const json::Object kEmpty;
        const auto* v = raw.get("modules");
        return v ? v->as_object() : kEmpty;
    }

    const json::Object& ops() const { return raw.at("ops").as_object(); }
    const json::Object& programs() const { return raw.at("programs").as_object(); }

    std::vector<int64_t> shape_at(const json::Value& shape_arr, const VarMap* vals = nullptr) const {
        const VarMap& v = vals ? *vals : vars_max;
        std::vector<int64_t> out;
        for (const auto& d : shape_arr.as_array()) {
            out.push_back(d.is_string() ? v.at(d.as_string()) : d.as_int());
        }
        return out;
    }

    std::vector<int64_t> shape_at(const std::vector<Dim>& dims, const VarMap* vals = nullptr) const {
        const VarMap& v = vals ? *vals : vars_max;
        std::vector<int64_t> out;
        for (const auto& d : dims) {
            if (std::holds_alternative<std::string>(d)) {
                out.push_back(v.at(std::get<std::string>(d)));
            } else {
                out.push_back(std::get<int64_t>(d));
            }
        }
        return out;
    }

    size_t buffer_bytes(std::string_view name) const {
        const auto& buf = raw.at("buffers").at(name);
        size_t numel = 1;
        for (int64_t d : shape_at(buf.at("shape"))) numel *= static_cast<size_t>(d);
        return numel * dtype_bytes(buf.at("dtype").as_string());
    }

    size_t state_bytes(std::string_view name, int64_t capacity_tokens, int64_t seq_slots) const {
        const auto& st = raw.at("states").at(name);
        if (const auto* p = st.get("bytes_per_token")) {
            return static_cast<size_t>(p->as_int() * capacity_tokens);
        }
        if (const auto* p = st.get("bytes_per_seq")) {
            return static_cast<size_t>(p->as_int() * seq_slots);
        }
        return static_cast<size_t>(st.at("bytes").as_int());
    }

    std::vector<std::string> once_programs() const {
        std::vector<std::string> out;
        for (const auto& [name, prog] : programs()) {
            if (const auto* o = prog.get("once"); o && o->is_bool() && o->as_bool()) {
                out.push_back(name);
            }
        }
        std::sort(out.begin(), out.end());
        return out;
    }

    Protocol protocol() const;
};

struct Protocol {
    std::optional<std::string> rows_var;
    std::optional<std::string> groups_var;
    int64_t rows_max = 1;
    int64_t groups_max = 1;
    std::map<std::string, Filled> inputs;
    std::map<std::string, std::vector<Filled>> outputs;
    std::vector<Table> page_tables;
    std::vector<Table> line_tables;
    std::map<std::string, Forward> forwards;

    static Protocol from_manifest(const Manifest& m) {
        Protocol p;
        std::optional<std::string> ranks_var;
        for (const auto& [k, v] : m.vars()) {
            if (const auto* ax = v.get("axis"); ax && ax->is_string()) {
                if (ax->as_string() == "rows") p.rows_var = k;
                else if (ax->as_string() == "groups") p.groups_var = k;
                else if (ax->as_string() == "ranks") ranks_var = k;
            }
        }
        if (p.rows_var) p.rows_max = m.vars_max.at(*p.rows_var);
        if (p.groups_var) p.groups_max = m.vars_max.at(*p.groups_var);

        auto axis_of = [&](const std::vector<Dim>& shape) -> std::string {
            if (shape.empty() || !std::holds_alternative<std::string>(shape[0])) return "fixed";
            const auto& d0 = std::get<std::string>(shape[0]);
            if (p.rows_var && d0 == *p.rows_var) return "rows";
            if (p.groups_var && d0 == *p.groups_var) return "groups";
            if (ranks_var && d0 == *ranks_var) return "ranks";
            return "fixed";
        };

        for (const auto& [name, buf] : m.buffers()) {
            const auto& kind = buf.at("kind").as_string();
            auto shape = detail::parse_shape(buf.at("shape"));
            if (const auto* fill = buf.get("fill"); fill && fill->is_string()) {
                Filled f{name, fill->as_string(), buf.at("dtype").as_string(), shape, axis_of(shape)};
                if (kind == "input" || kind == "inout") {
                    p.inputs[f.fill] = std::move(f);
                } else if (kind == "output") {
                    p.outputs[f.fill].push_back(std::move(f));
                }
                continue;
            }

            if (const auto* dom = buf.get("domain");
                (kind == "input" || kind == "inout") && dom && dom->is_object() && dom->contains("index_into")) {
                const auto& target = dom->at("index_into").as_string();
                const auto& states = m.states();
                if (auto it = states.find(target); it != states.end()) {
                    const auto& st = it->second;
                    int64_t stride = dom->contains("stride") ? dom->at("stride").as_int() : 1;
                    if (st.contains("bytes_per_token")) {
                        p.page_tables.push_back(Table{name, target, buf.at("dtype").as_string(), stride, shape, 0});
                    } else if (st.contains("bytes_per_seq")) {
                        int64_t lps = st.at("bytes_per_seq").as_int() / stride;
                        p.line_tables.push_back(Table{name, target, buf.at("dtype").as_string(), stride, shape, lps});
                    }
                }
            }
        }

        std::set<std::string> tok_outs;
        if (auto it = p.outputs.find("tokens"); it != p.outputs.end()) {
            for (const auto& f : it->second) tok_outs.insert(f.name);
        }
        std::set<std::string> cnt_outs;
        if (auto it = p.outputs.find("count"); it != p.outputs.end()) {
            for (const auto& f : it->second) cnt_outs.insert(f.name);
        }

        for (const auto& [pname, prog] : m.programs()) {
            const auto* batch = prog.get("batch");
            if (!batch || !batch->is_object()) continue;

            std::set<std::string> written;
            for (const auto& call : prog.at("calls").as_array()) {
                const auto& op = m.raw.at("ops").at(call.at("op").as_string());
                const auto& params = op.at("params").as_array();
                const auto& args = call.at("args").as_array();
                for (size_t i = 0; i < std::min(params.size(), args.size()); ++i) {
                    const auto& ps = params[i].as_string();
                    if ((ps.starts_with("out ") || ps.starts_with("inout ")) &&
                        args[i].is_object() && args[i].contains("buf")) {
                        written.insert(args[i].at("buf").as_string());
                    }
                }
            }

            Forward fwd;
            fwd.name = pname;
            fwd.groups = batch->at("groups").as_int();
            const auto& r = batch->at("rows");
            fwd.rows = r.is_int() ? Dim(r.as_int()) : Dim(r.as_string());
            if (const auto* ctx = batch->get("context"); ctx && ctx->is_string()) {
                fwd.context_var = ctx->as_string();
            }
            if (const auto* g = prog.get("graph"); g && g->is_bool()) {
                fwd.graph = g->as_bool();
            }
            for (const auto& w : written) {
                if (!fwd.emits && tok_outs.contains(w)) fwd.emits = w;
                if (!fwd.count && cnt_outs.contains(w)) fwd.count = w;
            }
            p.forwards.emplace(pname, std::move(fwd));
        }

        return p;
    }

    int64_t page_unit() const {
        int64_t u = 1;
        for (const auto& t : page_tables) u = std::max(u, t.stride);
        return u;
    }

    const Forward& chunk_forward() const {
        for (const auto& [_, f] : forwards) {
            if (std::holds_alternative<std::string>(f.rows)) return f;
        }
        throw std::runtime_error("manifest declares no variable-rows prefill program");
    }

    const Forward& step_forward(std::optional<int64_t> rows = std::nullopt, int64_t groups = 1) const {
        std::vector<const Forward*> cands;
        for (const auto& [_, f] : forwards) {
            if (!std::holds_alternative<int64_t>(f.rows) || f.groups < groups) continue;
            if (rows && std::get<int64_t>(f.rows) != *rows) continue;
            cands.push_back(&f);
        }
        if (cands.empty()) {
            throw std::runtime_error("no forward matching requested rows/groups");
        }
        std::sort(cands.begin(), cands.end(), [](const Forward* a, const Forward* b) {
            int64_t ra = std::get<int64_t>(a->rows);
            int64_t rb = std::get<int64_t>(b->rows);
            if (ra != rb) return ra > rb;
            return a->groups < b->groups;
        });
        return *cands[0];
    }

    VarMap vars_for(const Forward& fwd, int64_t groups, int64_t rows,
                    std::optional<int64_t> context = std::nullopt) const {
        VarMap v;
        if (groups_var) v[*groups_var] = groups;
        if (rows_var) v[*rows_var] = rows;
        if (fwd.context_var && context) v[*fwd.context_var] = *context;
        return v;
    }
};

inline Protocol Manifest::protocol() const { return Protocol::from_manifest(*this); }

}  // namespace kern
