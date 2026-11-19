// Minimal model-agnostic C++20 CLI runner over `manifest.hpp` and `executor.hpp`.
//
// Takes comma-separated prompt token IDs, drives chunked prefill and fixed-row
// decode or speculative rounds solely through `Protocol`, and prints emitted
// token IDs. Zero Rust, PyTorch, or tokenizer dependencies.
//
//   ./kern-run --manifest model.json --kernels cubins/ --weights ckpt/ --tokens 791,6864,315 --steps 32

#include "executor.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<int64_t> parse_tokens(const std::string& s) {
    std::vector<int64_t> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(std::stoll(item));
    }
    return out;
}

std::vector<int64_t> generate(kern::Executor& rt, const std::vector<int64_t>& prompt_ids,
                              int64_t steps, std::optional<int64_t> rows, bool use_graph) {
    const auto& p = rt.protocol;
    const auto& chunk_fwd = p.chunk_forward();
    const auto& step_fwd = p.step_forward(rows, 1);
    int64_t step_rows = std::get<int64_t>(step_fwd.rows);

    int64_t pos = 0;
    std::vector<int64_t> generated;

    // 1. Chunked prefill
    while (pos < static_cast<int64_t>(prompt_ids.size())) {
        int64_t n = std::min<int64_t>(p.rows_max, static_cast<int64_t>(prompt_ids.size()) - pos);
        std::vector<int64_t> chunk(prompt_ids.begin() + pos, prompt_ids.begin() + pos + n);
        std::vector<int64_t> slots(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i) slots[static_cast<size_t>(i)] = pos + i;
        auto vals = rt.stage(chunk_fwd, {pos}, {n}, chunk, slots);
        rt.run(chunk_fwd.name, vals, use_graph);
        rt.sync();
        pos += n;
        if (pos == static_cast<int64_t>(prompt_ids.size())) {
            auto out = rt.emitted(chunk_fwd);
            generated.insert(generated.end(), out.begin(), out.end());
        }
    }

    // 2. Decode or speculative rounds (CUDA Graph)
    while (static_cast<int64_t>(generated.size()) < steps) {
        int64_t anchor = generated.empty() ? prompt_ids.back() : generated.back();
        std::vector<int64_t> ids(static_cast<size_t>(step_rows), 0);
        ids[0] = anchor;
        std::vector<int64_t> slots(static_cast<size_t>(step_rows));
        for (int64_t i = 0; i < step_rows; ++i) slots[static_cast<size_t>(i)] = pos + i;
        auto vals = rt.stage(step_fwd, {pos}, {step_rows}, ids, slots);
        rt.run(step_fwd.name, vals, use_graph);
        rt.sync();
        auto out = rt.emitted(step_fwd);
        pos += static_cast<int64_t>(out.size());
        generated.insert(generated.end(), out.begin(), out.end());
    }

    if (static_cast<int64_t>(generated.size()) > steps) {
        generated.resize(static_cast<size_t>(steps));
    }
    return generated;
}

void usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " --manifest <file> --kernels <dir> --weights <path>..."
              << " --tokens <id,id,...> [--steps 32] [--rows N] [--capacity 4096]"
              << " [--eager] [--gpu 0]\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string manifest_path;
    std::string kernels_dir;
    std::vector<std::filesystem::path> weights;
    std::vector<int64_t> prompt_ids;
    int64_t steps = 32;
    std::optional<int64_t> rows;
    int64_t capacity = 4096;
    bool eager = false;
    int gpu = 0;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const std::string& flag) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + flag);
            return argv[++i];
        };
        if (a == "--manifest") manifest_path = need(a);
        else if (a == "--kernels") kernels_dir = need(a);
        else if (a == "--weights") {
            weights.emplace_back(need(a));
            while (i + 1 < argc && argv[i + 1][0] != '-') weights.emplace_back(argv[++i]);
        } else if (a == "--tokens") prompt_ids = parse_tokens(need(a));
        else if (a == "--steps") steps = std::stoll(need(a));
        else if (a == "--rows") rows = std::stoll(need(a));
        else if (a == "--capacity") capacity = std::stoll(need(a));
        else if (a == "--eager") eager = true;
        else if (a == "--gpu") gpu = std::stoi(need(a));
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else { std::cerr << "unknown arg: " << a << "\n"; usage(argv[0]); return 2; }
    }

    if (manifest_path.empty() || kernels_dir.empty() || weights.empty() || prompt_ids.empty()) {
        usage(argv[0]);
        return 2;
    }

    try {
        auto t0 = std::chrono::steady_clock::now();
        kern::Executor rt(kern::Manifest::load(manifest_path), kernels_dir, weights, capacity, 2, gpu);
        rt.init_single_sequence(1);
        auto t1 = std::chrono::steady_clock::now();
        auto out = generate(rt, prompt_ids, steps, rows, !eager);
        auto t2 = std::chrono::steady_clock::now();

        for (size_t i = 0; i < out.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << out[i];
        }
        std::cout << '\n';

        double load_s = std::chrono::duration<double>(t1 - t0).count();
        double gen_s = std::chrono::duration<double>(t2 - t1).count();
        std::cerr << "loaded in " << load_s << "s, generated " << out.size()
                  << " tokens in " << gen_s << "s (" << (gen_s > 0 ? out.size() / gen_s : 0.0)
                  << " tok/s)\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
