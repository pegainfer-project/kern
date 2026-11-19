// Pure-CPU tests for `manifest.hpp` and `json.hpp` (no CUDA required).

#include "executor.hpp"

#include <cassert>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

template <typename Fn>
void expect_throw(Fn&& fn, const char* label) {
    bool threw = false;
    try {
        fn();
    } catch (const std::exception&) {
        threw = true;
    }
    if (!threw) {
        std::cerr << "FAIL: expected exception for " << label << "\n";
        std::exit(1);
    }
}

void test_minimal_example_parses(const std::filesystem::path& root) {
    auto m = kern::Manifest::load((root / "examples" / "minimal.json").string());
    assert(m.model == "toy");
    assert(m.vars_max.size() == 1 && m.vars_max.at("tokens") == 1024);
    assert(m.buffer_bytes("x") == 1024 * 64 * 2);
    assert(m.state_bytes("kv", 512, 2) == 512 * 256);
}

void test_constants_expansion_and_expressions() {
    auto v = kern::json::parse(R"({
        "schema_version": 7,
        "model": "const-test",
        "constants": {"H": 64, "MAX_T": 128},
        "vars": {"tokens": {"max": "MAX_T", "axis": "rows"}},
        "buffers": {"x": {"dtype": "bf16", "shape": ["tokens", "H"], "kind": "input"}},
        "ops": {},
        "programs": {}
    })");
    auto m = kern::Manifest::from_json(v);
    assert(m.vars_max.at("tokens") == 128);
    assert(m.buffer_bytes("x") == 128 * 64 * 2);

    kern::VarMap vals = {{"tokens", 17}};
    assert(kern::eval_expr(kern::json::parse(R"({"ceil_div": ["tokens", 16]})"), vals) == 2);
    assert(kern::eval_expr(kern::json::parse(R"({"mul": [{"add": ["tokens", 1]}, 4]})"), {{"tokens", 3}}) == 16);
    assert(kern::eval_expr(kern::json::parse(R"({"clamp": ["tokens", 10, 8]})"), {{"tokens", 15}}) == 5);
    expect_throw([&] { kern::eval_expr(kern::json::parse(R"({"sub": ["tokens", 1]})"), vals); }, "invalid expr");
}

void test_constants_do_not_substitute_reference_names(const std::filesystem::path& root) {
    auto raw = kern::json::parse_file((root / "examples" / "minimal.json").string());
    raw.as_object()["constants"] = kern::json::parse(R"({"x": 64, "tokens": 32})");
    raw.as_object()["buffers"].as_object()["w"].as_object()["shape"] = kern::json::parse(R"(["x"])");

    auto m = kern::Manifest::from_json(raw);
    const auto& w_shape = m.raw.at("buffers").at("w").at("shape").as_array();
    assert(w_shape.size() == 1 && w_shape[0].as_int() == 64);
    const auto& args = m.raw.at("programs").at("step").at("calls").as_array()[0].at("args").as_array();
    assert(args.front().at("buf").as_string() == "x");
    assert(args.back().at("var").as_string() == "tokens");
}

void test_unsupported_schema_version_raises() {
    auto v = kern::json::parse(R"({"schema_version": 6, "model": "old", "buffers": {}, "ops": {}, "programs": {}})");
    expect_throw([&] { kern::Manifest::from_json(v); }, "schema_version 6");
}

void test_protocol_projection() {
    auto v = kern::json::parse(R"({
        "schema_version": 7,
        "model": "spec-test",
        "vars": {
            "tokens": {"max": 1024, "axis": "rows"},
            "seqs": {"max": 16, "axis": "groups"},
            "ctx": {"max": 8192}
        },
        "states": {
            "kv": {"bytes_per_token": 32768},
            "gdn": {"bytes_per_seq": 49152}
        },
        "buffers": {
            "token_ids": {"dtype": "i64", "shape": ["tokens"], "kind": "input", "fill": "token"},
            "positions": {"dtype": "i64", "shape": ["tokens"], "kind": "input", "fill": "position"},
            "pages": {"dtype": "i32", "shape": ["seqs", 64], "kind": "input", "domain": {"index_into": "kv", "stride": 64}},
            "gdn_lines": {"dtype": "i32", "shape": [48, "seqs"], "kind": "input", "domain": {"index_into": "gdn", "stride": 1024}},
            "next_token": {"dtype": "i64", "shape": ["seqs"], "kind": "output", "fill": "tokens"},
            "verify_tokens": {"dtype": "i64", "shape": ["seqs", 16], "kind": "output", "fill": "tokens"},
            "accepted": {"dtype": "i32", "shape": ["seqs"], "kind": "output", "fill": "count"}
        },
        "ops": {
            "emit_one": {"params": ["out buffer<i64>"], "impl": {"launches": []}},
            "emit_round": {"params": ["out buffer<i64>", "out buffer<i32>"], "impl": {"launches": []}}
        },
        "programs": {
            "load_b": {"once": true, "calls": []},
            "load_a": {"once": true, "calls": []},
            "prefill": {"batch": {"groups": 1, "rows": "tokens", "context": "ctx"}, "calls": [{"op": "emit_one", "args": [{"buf": "next_token"}]}]},
            "decode": {"batch": {"groups": 1, "rows": 1}, "graph": true, "calls": [{"op": "emit_one", "args": [{"buf": "next_token"}]}]},
            "round": {"batch": {"groups": 16, "rows": 16}, "graph": true, "calls": [{"op": "emit_round", "args": [{"buf": "verify_tokens"}, {"buf": "accepted"}]}]}
        }
    })");
    auto m = kern::Manifest::from_json(v);
    auto once = m.once_programs();
    assert((once == std::vector<std::string>{"load_a", "load_b"}));

    auto p = m.protocol();
    assert(p.rows_var == "tokens" && p.groups_var == "seqs" && p.page_unit() == 64);
    assert(p.line_tables.size() == 1 && p.line_tables[0].lines_per_slot == 48);
    assert(p.chunk_forward().name == "prefill");
    assert(p.chunk_forward().emits == "next_token");
    auto pv = p.vars_for(p.chunk_forward(), 1, 128, 512);
    assert(pv.at("seqs") == 1 && pv.at("tokens") == 128 && pv.at("ctx") == 512);

    assert(p.step_forward().name == "round");
    assert(p.step_forward().emits == "verify_tokens" && p.step_forward().count == "accepted");
    assert(p.step_forward(1).name == "decode");
    auto sv = p.vars_for(p.step_forward(), 1, 16);
    assert(sv.at("seqs") == 1 && sv.at("tokens") == 16);
    expect_throw([&] { p.step_forward(1, 2); }, "groups exceeds bound");
}

void test_sha256() {
    const char* s = "abc";
    auto h = kern::detail::sha256_hex(reinterpret_cast<const uint8_t*>(s), 3);
    assert(h == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

}  // namespace

int main() {
    auto root = std::filesystem::weakly_canonical(std::filesystem::absolute(__FILE__))
                    .parent_path().parent_path().parent_path();
    test_minimal_example_parses(root);
    test_constants_expansion_and_expressions();
    test_constants_do_not_substitute_reference_names(root);
    test_unsupported_schema_version_raises();
    test_protocol_projection();
    test_sha256();
    std::cout << "All 6 C++ manifest/protocol tests passed.\n";
    return 0;
}
