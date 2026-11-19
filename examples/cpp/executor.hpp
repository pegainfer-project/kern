// Pure-C++20 single-GPU executor for a kern v7 manifest (cubin + JSON + safetensors).
//
// Runs a manifest directly through the CUDA Driver API and cuBLASLt (loaded
// via dlopen) and mmap'd safetensors, with zero Rust, PyTorch, or compile-time
// CUDA toolkit dependencies.
#pragma once

#include "cuda_api.hpp"
#include "manifest.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace kern {

namespace detail {

// Standard FIPS 180-4 SHA-256 returning lowercase 64-char hex.
inline std::string sha256_hex(const uint8_t* data, size_t len) {
    static constexpr uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    };
    auto rotr = [](uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); };
    uint32_t h[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    };

    auto compress = [&](const uint8_t blk[64]) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (uint32_t(blk[i * 4]) << 24) | (uint32_t(blk[i * 4 + 1]) << 16) |
                   (uint32_t(blk[i * 4 + 2]) << 8) | uint32_t(blk[i * 4 + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            uint32_t ch = (e & f) ^ ((~e) & g);
            uint32_t t1 = hh + S1 + ch + K[i] + w[i];
            uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2 = S0 + maj;
            hh = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    };

    size_t i = 0;
    for (; i + 64 <= len; i += 64) compress(data + i);
    uint8_t tail[128] = {};
    size_t rem = len - i;
    if (rem) std::memcpy(tail, data + i, rem);
    tail[rem] = 0x80;
    size_t total_blocks = (rem + 9 <= 64) ? 1 : 2;
    uint64_t bit_len = uint64_t(len) * 8;
    for (int j = 0; j < 8; ++j) {
        tail[total_blocks * 64 - 1 - j] = static_cast<uint8_t>((bit_len >> (j * 8)) & 0xFF);
    }
    compress(tail);
    if (total_blocks == 2) compress(tail + 64);

    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (uint32_t x : h) oss << std::setw(8) << x;
    return oss.str();
}

struct MmapFile {
    const uint8_t* ptr = nullptr;
    size_t size = 0;

    explicit MmapFile(const std::filesystem::path& p) {
        int fd = ::open(p.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open: " + p.string());
        struct stat st{};
        if (::fstat(fd, &st) != 0) {
            ::close(fd);
            throw std::runtime_error("fstat failed: " + p.string());
        }
        size = static_cast<size_t>(st.st_size);
        void* m = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (m == MAP_FAILED) throw std::runtime_error("mmap failed: " + p.string());
        ptr = static_cast<const uint8_t*>(m);
    }

    ~MmapFile() {
        if (ptr) ::munmap(const_cast<uint8_t*>(ptr), size);
    }

    MmapFile(const MmapFile&) = delete;
    MmapFile& operator=(const MmapFile&) = delete;
    MmapFile(MmapFile&& o) noexcept : ptr(o.ptr), size(o.size) { o.ptr = nullptr; o.size = 0; }
};

inline size_t safetensors_dtype_bytes(std::string_view dt) {
    if (dt == "U8" || dt == "I8" || dt == "F8_E4M3" || dt == "F8_E5M2" || dt == "BOOL") return 1;
    if (dt == "BF16" || dt == "F16" || dt == "U16" || dt == "I16") return 2;
    if (dt == "F32" || dt == "I32" || dt == "U32") return 4;
    if (dt == "F64" || dt == "I64" || dt == "U64") return 8;
    throw std::runtime_error("unsupported safetensors dtype: " + std::string(dt));
}

inline int tma_dtype(std::string_view dt) {
    if (dt == "u8") return cuda::CU_TENSOR_MAP_DATA_TYPE_UINT8;
    if (dt == "u16") return cuda::CU_TENSOR_MAP_DATA_TYPE_UINT16;
    if (dt == "u32") return cuda::CU_TENSOR_MAP_DATA_TYPE_UINT32;
    if (dt == "i32") return cuda::CU_TENSOR_MAP_DATA_TYPE_INT32;
    if (dt == "u64") return cuda::CU_TENSOR_MAP_DATA_TYPE_UINT64;
    if (dt == "i64") return cuda::CU_TENSOR_MAP_DATA_TYPE_INT64;
    if (dt == "f16") return cuda::CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    if (dt == "bf16") return cuda::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    if (dt == "f32") return cuda::CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    if (dt == "tf32") return cuda::CU_TENSOR_MAP_DATA_TYPE_TFLOAT32;
    throw std::runtime_error("unsupported TMA dtype: " + std::string(dt));
}

inline int tma_swizzle(int64_t s) {
    if (s == 0) return cuda::CU_TENSOR_MAP_SWIZZLE_NONE;
    if (s == 32) return cuda::CU_TENSOR_MAP_SWIZZLE_32B;
    if (s == 64) return cuda::CU_TENSOR_MAP_SWIZZLE_64B;
    if (s == 128) return cuda::CU_TENSOR_MAP_SWIZZLE_128B;
    throw std::runtime_error("unsupported TMA swizzle: " + std::to_string(s));
}

inline int tma_l2(int64_t s) {
    if (s == 0) return cuda::CU_TENSOR_MAP_L2_PROMOTION_NONE;
    if (s == 64) return cuda::CU_TENSOR_MAP_L2_PROMOTION_L2_64B;
    if (s == 128) return cuda::CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
    if (s == 256) return cuda::CU_TENSOR_MAP_L2_PROMOTION_L2_256B;
    throw std::runtime_error("unsupported TMA l2_promotion: " + std::to_string(s));
}

}  // namespace detail

class Executor {
public:
    Manifest manifest;
    Protocol protocol;
    int64_t capacity_tokens;
    int64_t seq_slots;
    int gpu;

    Executor(Manifest m, const std::filesystem::path& kernels_dir,
             const std::vector<std::filesystem::path>& weights,
             int64_t capacity_tokens = 4096, int64_t seq_slots = 2, int gpu = 0)
        : manifest(std::move(m)),
          protocol(manifest.protocol()),
          capacity_tokens(capacity_tokens),
          seq_slots(seq_slots),
          gpu(gpu),
          cu_(cuda::Api::load()) {
        cu_.check(cu_.cuInit(0), "cuInit");
        cuda::CUdevice dev = 0;
        cu_.check(cu_.cuDeviceGet(&dev, gpu), "cuDeviceGet");
        cu_.check(cu_.cuDevicePrimaryCtxRetain(&ctx_, dev), "cuDevicePrimaryCtxRetain");
        cu_.check(cu_.cuCtxSetCurrent(ctx_), "cuCtxSetCurrent");
        cu_.check(cu_.cuStreamCreate(&stream_, 0), "cuStreamCreate");

        if (cu_.cublasLtCreate) {
            cuda::Api::check_blas(cu_.cublasLtCreate(&blt_), "cublasLtCreate");
            workspace_bytes_ = 32 * 1024 * 1024;
            workspace_ = alloc_zero(workspace_bytes_);
        }

        // 1. Load sha256-pinned cubin modules.
        std::unordered_map<std::string, std::vector<uint8_t>> by_sha;
        for (const auto& entry : std::filesystem::directory_iterator(kernels_dir)) {
            if (!entry.is_regular_file() || entry.path().extension() != ".cubin") continue;
            std::ifstream f(entry.path(), std::ios::binary);
            std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            by_sha[detail::sha256_hex(bytes.data(), bytes.size())] = std::move(bytes);
        }
        for (const auto& [name, mod] : manifest.modules()) {
            const auto& sha = mod.at("sha256").as_string();
            auto it = by_sha.find(sha);
            if (it == by_sha.end()) {
                throw std::runtime_error("module `" + name + "` sha256 " + sha + " not found in " + kernels_dir.string());
            }
            cuda::CUmodule cmod = nullptr;
            cu_.check(cu_.cuModuleLoadData(&cmod, it->second.data()), "cuModuleLoadData");
            modules_[name] = cmod;
        }

        // 2. Allocate zero-initialized states, buffers, and op-private scratch.
        for (const auto& [name, _] : manifest.states()) {
            size_t n = manifest.state_bytes(name, capacity_tokens, seq_slots);
            states_[name] = {alloc_zero(n), n};
        }
        for (const auto& [name, _] : manifest.buffers()) {
            size_t n = manifest.buffer_bytes(name);
            buffers_[name] = {alloc_zero(n), n};
        }
        for (const auto& [op_name, op] : manifest.ops()) {
            const auto& impl = op.at("impl");
            if (const auto* sc = impl.get("scratch"); sc && sc->is_object()) {
                for (const auto& [sname, sbuf] : sc->as_object()) {
                    size_t numel = 1;
                    for (int64_t d : manifest.shape_at(sbuf.at("shape"))) numel *= static_cast<size_t>(d);
                    size_t n = numel * dtype_bytes(sbuf.at("dtype").as_string());
                    scratch_[op_name + "::" + sname] = {alloc_zero(n), n};
                }
            }
        }

        // 3. Load checkpoint weights via `bind` (mmap'd safetensors).
        load_weights(weights);

        // 4. Lower programs and run `once` setup programs.
        for (const auto& [pname, prog] : manifest.programs()) {
            compiled_[pname] = compile_program(prog);
        }
        for (const auto& pname : manifest.once_programs()) {
            run(pname, {}, false);
        }
        sync();
    }

    ~Executor() {
        for (auto& [_, g] : graphs_) cu_.cuGraphExecDestroy(g);
        for (auto& [_, m] : modules_) cu_.cuModuleUnload(m);
        for (auto& [_, b] : scratch_) cu_.cuMemFree(b.ptr);
        for (auto& [_, b] : buffers_) cu_.cuMemFree(b.ptr);
        for (auto& [_, b] : states_) cu_.cuMemFree(b.ptr);
        if (workspace_) cu_.cuMemFree(workspace_);
        if (blt_ && cu_.cublasLtDestroy) cu_.cublasLtDestroy(blt_);
        if (stream_) cu_.cuStreamDestroy(stream_);
    }

    Executor(const Executor&) = delete;
    Executor& operator=(const Executor&) = delete;

    void sync() const { cu_.check(cu_.cuStreamSynchronize(stream_), "cuStreamSynchronize"); }

    void run(const std::string& program, const VarMap& vals, bool graph = true) {
        const auto& prog = manifest.raw.at("programs").at(program);
        bool use_graph = graph && prog.contains("graph") && prog.at("graph").as_bool();
        if (!use_graph) {
            issue(program, vals);
            return;
        }
        std::string key = graph_key(program, vals);
        auto it = graphs_.find(key);
        if (it == graphs_.end()) {
            cu_.check(cu_.cuStreamBeginCapture(stream_, cuda::CU_STREAM_CAPTURE_MODE_GLOBAL), "cuStreamBeginCapture");
            issue(program, vals);
            cuda::CUgraph g = nullptr;
            cu_.check(cu_.cuStreamEndCapture(stream_, &g), "cuStreamEndCapture");
            cuda::CUgraphExec exec = nullptr;
            cu_.check(cu_.cuGraphInstantiateWithFlags(&exec, g, 0), "cuGraphInstantiateWithFlags");
            cu_.cuGraphDestroy(g);
            it = graphs_.emplace(std::move(key), exec).first;
        }
        cu_.check(cu_.cuGraphLaunch(it->second, stream_), "cuGraphLaunch");
    }

    void write_bytes(const std::string& name, const void* src, size_t nbytes) {
        const auto& b = buffers_.at(name);
        if (nbytes > b.bytes) throw std::runtime_error("write exceeds buffer `" + name + "`");
        cu_.check(cu_.cuMemcpyHtoDAsync(b.ptr, src, nbytes, stream_), "cuMemcpyHtoDAsync");
    }

    void read_bytes(const std::string& name, void* dst, size_t nbytes) const {
        const auto& b = buffers_.at(name);
        if (nbytes > b.bytes) throw std::runtime_error("read exceeds buffer `" + name + "`");
        cu_.check(cu_.cuMemcpyDtoH(dst, b.ptr, nbytes), "cuMemcpyDtoH");
    }

    std::vector<int64_t> read_ints(const std::string& name, size_t count) const {
        const auto& dt = manifest.raw.at("buffers").at(name).at("dtype").as_string();
        std::vector<int64_t> out(count);
        if (dt == "i64") {
            read_bytes(name, out.data(), count * sizeof(int64_t));
        } else if (dt == "i32") {
            std::vector<int32_t> tmp(count);
            read_bytes(name, tmp.data(), count * sizeof(int32_t));
            for (size_t i = 0; i < count; ++i) out[i] = tmp[i];
        } else {
            throw std::runtime_error("read_ints: unsupported dtype `" + dt + "` for `" + name + "`");
        }
        return out;
    }

    void write_ints(const std::string& name, const std::string& dt, const std::vector<int64_t>& vals) {
        if (dt == "i64") {
            write_bytes(name, vals.data(), vals.size() * sizeof(int64_t));
        } else if (dt == "i32") {
            std::vector<int32_t> tmp(vals.size());
            for (size_t i = 0; i < vals.size(); ++i) tmp[i] = static_cast<int32_t>(vals[i]);
            write_bytes(name, tmp.data(), tmp.size() * sizeof(int32_t));
        } else if (dt == "u32") {
            std::vector<uint32_t> tmp(vals.size());
            for (size_t i = 0; i < vals.size(); ++i) tmp[i] = static_cast<uint32_t>(vals[i]);
            write_bytes(name, tmp.data(), tmp.size() * sizeof(uint32_t));
        } else if (dt == "u8") {
            std::vector<uint8_t> tmp(vals.size());
            for (size_t i = 0; i < vals.size(); ++i) tmp[i] = static_cast<uint8_t>(vals[i]);
            write_bytes(name, tmp.data(), tmp.size());
        } else {
            throw std::runtime_error("write_ints: unsupported dtype `" + dt + "`");
        }
    }

    void write_floats(const std::string& name, const std::vector<float>& vals) {
        write_bytes(name, vals.data(), vals.size() * sizeof(float));
    }

    void write_tables(const std::vector<std::vector<int64_t>>& pages_per_seq,
                      const std::vector<int64_t>& seq_slot_ids) {
        for (const auto& tbl : protocol.page_tables) {
            auto shape = manifest.shape_at(tbl.shape);
            int64_t cols = shape.size() > 1 ? shape[1] : shape[0];
            if (shape.size() == 2) {
                std::vector<int64_t> flat(pages_per_seq.size() * static_cast<size_t>(cols), 0);
                for (size_t i = 0; i < pages_per_seq.size(); ++i) {
                    size_t n = std::min<size_t>(pages_per_seq[i].size(), static_cast<size_t>(cols));
                    for (size_t j = 0; j < n; ++j) flat[i * cols + j] = pages_per_seq[i][j];
                }
                write_ints(tbl.name, tbl.dtype, flat);
            } else {
                std::vector<int64_t> flat(static_cast<size_t>(cols), 0);
                if (!pages_per_seq.empty()) {
                    size_t n = std::min<size_t>(pages_per_seq[0].size(), static_cast<size_t>(cols));
                    for (size_t j = 0; j < n; ++j) flat[j] = pages_per_seq[0][j];
                }
                write_ints(tbl.name, tbl.dtype, flat);
            }
        }
        for (const auto& tbl : protocol.line_tables) {
            auto shape = manifest.shape_at(tbl.shape);
            size_t total = 1;
            for (int64_t d : shape) total *= static_cast<size_t>(d);
            std::vector<int64_t> lines(total, 0);
            int64_t d0 = shape[0], d1 = shape.size() > 1 ? shape[1] : 1, d2 = shape.size() > 2 ? shape[2] : 1;
            for (size_t i = 0; i < seq_slot_ids.size(); ++i) {
                int64_t base = seq_slot_ids[i] * tbl.lines_per_slot;
                for (int64_t r = 0; r < d0; ++r) {
                    lines[static_cast<size_t>((r * d1 + static_cast<int64_t>(i)) * d2)] = base + r;
                }
            }
            write_ints(tbl.name, tbl.dtype, lines);
        }
    }

    void init_single_sequence(int64_t seq_slot = 1) {
        int64_t n_pages = capacity_tokens / protocol.page_unit();
        std::vector<int64_t> pages(static_cast<size_t>(n_pages));
        for (int64_t i = 0; i < n_pages; ++i) pages[static_cast<size_t>(i)] = i;
        write_tables({pages}, {seq_slot});
    }

    VarMap stage(const Forward& fwd, const std::vector<int64_t>& positions,
                 const std::vector<int64_t>& lens, const std::vector<int64_t>& token_ids,
                 const std::vector<int64_t>& slots, const Sampling& sampling = Sampling{}) {
        int64_t groups = static_cast<int64_t>(lens.size());
        int64_t rows = 0;
        int64_t context = 0;
        std::vector<int64_t> starts;
        for (size_t i = 0; i < lens.size(); ++i) {
            starts.push_back(rows);
            rows += lens[i];
            context += positions[i] + lens[i];
        }

        for (const auto& [role, filled] : protocol.inputs) {
            if (role == "temperature") {
                write_floats(filled.name, std::vector<float>(groups, sampling.temperature));
            } else if (role == "top_p") {
                write_floats(filled.name, std::vector<float>(groups, sampling.top_p));
            } else if (role == "min_p") {
                write_floats(filled.name, std::vector<float>(groups, sampling.min_p));
            } else if (role == "top_k") {
                write_ints(filled.name, filled.dtype, std::vector<int64_t>(groups, sampling.top_k));
            } else if (role == "seed") {
                write_ints(filled.name, filled.dtype, std::vector<int64_t>(groups, sampling.seed));
            } else if (role == "token") {
                if (filled.axis == "groups") {
                    std::vector<int64_t> vals;
                    for (int64_t s : starts) vals.push_back(token_ids[static_cast<size_t>(s)]);
                    write_ints(filled.name, filled.dtype, vals);
                } else {
                    write_ints(filled.name, filled.dtype, token_ids);
                }
            } else if (role == "valid") {
                write_ints(filled.name, filled.dtype, std::vector<int64_t>(static_cast<size_t>(rows), 1));
            } else if (role == "position") {
                std::vector<int64_t> vals;
                vals.reserve(static_cast<size_t>(rows));
                for (size_t i = 0; i < lens.size(); ++i) {
                    for (int64_t j = 0; j < lens[i]; ++j) vals.push_back(positions[i] + j);
                }
                write_ints(filled.name, filled.dtype, vals);
            } else if (role == "slot") {
                write_ints(filled.name, filled.dtype, slots);
            } else if (role == "seq_len") {
                std::vector<int64_t> vals;
                for (size_t i = 0; i < lens.size(); ++i) vals.push_back(positions[i] + lens[i]);
                write_ints(filled.name, filled.dtype, vals);
            } else if (role == "cu_seqlens") {
                auto vals = starts;
                vals.push_back(rows);
                write_ints(filled.name, filled.dtype, vals);
            }
        }
        return protocol.vars_for(fwd, groups, rows, context);
    }

    std::vector<std::vector<int64_t>> emitted_batch(const Forward& fwd, int64_t groups) const {
        if (!fwd.emits) return std::vector<std::vector<int64_t>>(static_cast<size_t>(groups));
        std::vector<int64_t> counts = fwd.count ? read_ints(*fwd.count, static_cast<size_t>(groups))
                                                : std::vector<int64_t>(static_cast<size_t>(groups), 1);
        const auto& shape = manifest.raw.at("buffers").at(*fwd.emits).at("shape").as_array();
        int64_t width = shape.size() > 1 ? shape[1].as_int() : 1;
        auto raw = read_ints(*fwd.emits, static_cast<size_t>(groups * width));
        std::vector<std::vector<int64_t>> out(static_cast<size_t>(groups));
        for (int64_t i = 0; i < groups; ++i) {
            int64_t c = counts[static_cast<size_t>(i)];
            out[static_cast<size_t>(i)].assign(raw.begin() + i * width, raw.begin() + i * width + c);
        }
        return out;
    }

    std::vector<int64_t> emitted(const Forward& fwd) const { return emitted_batch(fwd, 1)[0]; }

private:
    struct DevBuf {
        cuda::CUdeviceptr ptr = 0;
        size_t bytes = 0;
    };

    struct CallArg {
        std::string kind;  // "ptr", "var", "expr", "i32", "i64", "u32", "u64", "f32", "u8"
        uint64_t u64 = 0;
        double f64 = 0.0;
        size_t bytes = 0;
        json::Value expr;
    };

    struct DynField {
        size_t at = 0;
        int width = 4;
        json::Value expr;
    };

    struct Slot {
        enum class Kind { ConstU64, ConstScalar, Dyn, Pack } kind = Kind::ConstU64;
        std::string ptype;
        uint64_t u64 = 0;
        double f64 = 0.0;
        json::Value expr;
        std::vector<uint8_t> pack_base;
        std::vector<DynField> pack_dyn;
    };

    struct When {
        std::string var;
        std::optional<int64_t> min_val;
        std::optional<int64_t> max_val;
    };

    struct Step {
        bool is_extern = false;
        std::string entry;
        cuda::CUfunction fn = nullptr;
        std::optional<When> when;
        std::array<unsigned int, 3> block = {1, 1, 1};
        std::array<json::Value, 3> grid;
        json::Value shmem;
        std::vector<Slot> args;
    };

    const cuda::Api& cu_;
    cuda::CUcontext ctx_ = nullptr;
    cuda::CUstream stream_ = nullptr;
    cuda::cublasLtHandle_t blt_ = nullptr;
    cuda::CUdeviceptr workspace_ = 0;
    size_t workspace_bytes_ = 0;

    std::unordered_map<std::string, cuda::CUmodule> modules_;
    std::unordered_map<std::string, cuda::CUfunction> funcs_;
    std::unordered_map<std::string, DevBuf> states_;
    std::unordered_map<std::string, DevBuf> buffers_;
    std::unordered_map<std::string, DevBuf> scratch_;
    std::unordered_map<std::string, std::vector<Step>> compiled_;
    std::unordered_map<std::string, cuda::CUgraphExec> graphs_;

    cuda::CUdeviceptr alloc_zero(size_t bytes) const {
        if (bytes == 0) bytes = 1;
        cuda::CUdeviceptr ptr = 0;
        cu_.check(cu_.cuMemAlloc(&ptr, bytes), "cuMemAlloc");
        cu_.check(cu_.cuMemsetD8(ptr, 0, bytes), "cuMemsetD8");
        return ptr;
    }

    static std::string graph_key(const std::string& prog, const VarMap& vals) {
        std::map<std::string, int64_t> sorted(vals.begin(), vals.end());
        std::ostringstream oss;
        oss << prog;
        for (const auto& [k, v] : sorted) oss << ';' << k << '=' << v;
        return oss.str();
    }

    cuda::CUfunction get_func(const std::string& mod_name, const std::string& entry, int64_t max_shmem) {
        std::string key = mod_name + "::" + entry;
        auto it = funcs_.find(key);
        cuda::CUfunction fn = nullptr;
        if (it == funcs_.end()) {
            cu_.check(cu_.cuModuleGetFunction(&fn, modules_.at(mod_name), entry.c_str()), "cuModuleGetFunction");
            funcs_[key] = fn;
        } else {
            fn = it->second;
        }
        if (max_shmem > 48 * 1024) {
            cu_.check(cu_.cuFuncSetAttribute(fn, cuda::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                             static_cast<int>(max_shmem)),
                      "cuFuncSetAttribute");
        }
        return fn;
    }

    void load_weights(const std::vector<std::filesystem::path>& weight_paths) {
        std::vector<std::filesystem::path> files;
        for (const auto& p : weight_paths) {
            if (std::filesystem::is_directory(p)) {
                std::vector<std::filesystem::path> dir_files;
                for (const auto& e : std::filesystem::directory_iterator(p)) {
                    if (e.is_regular_file() && e.path().extension() == ".safetensors") {
                        dir_files.push_back(e.path());
                    }
                }
                std::sort(dir_files.begin(), dir_files.end());
                files.insert(files.end(), dir_files.begin(), dir_files.end());
            } else {
                files.push_back(p);
            }
        }

        struct TensorRef {
            size_t file_idx = 0;
            size_t data_start = 0;
            size_t elem_bytes = 0;
            std::vector<int64_t> shape;
        };

        std::vector<detail::MmapFile> mapped;
        std::vector<size_t> data_bases;
        std::unordered_map<std::string, TensorRef> index;

        for (size_t fi = 0; fi < files.size(); ++fi) {
            mapped.emplace_back(files[fi]);
            const auto& mf = mapped.back();
            if (mf.size < 8) throw std::runtime_error("truncated safetensors: " + files[fi].string());
            uint64_t hdr_len = 0;
            std::memcpy(&hdr_len, mf.ptr, 8);
            size_t base = 8 + static_cast<size_t>(hdr_len);
            data_bases.push_back(base);
            auto hdr = json::parse(std::string_view(reinterpret_cast<const char*>(mf.ptr + 8), hdr_len));
            for (const auto& [k, v] : hdr.as_object()) {
                if (k == "__metadata__") continue;
                TensorRef ref;
                ref.file_idx = fi;
                ref.elem_bytes = detail::safetensors_dtype_bytes(v.at("dtype").as_string());
                const auto& offs = v.at("data_offsets").as_array();
                ref.data_start = base + static_cast<size_t>(offs[0].as_int());
                for (const auto& d : v.at("shape").as_array()) ref.shape.push_back(d.as_int());
                index.emplace(k, std::move(ref));
            }
        }

        for (const auto& [name, buf] : manifest.buffers()) {
            if (buf.at("kind").as_string() != "weight") continue;
            const auto& dst = buffers_.at(name);
            size_t off = 0;
            for (const auto& seg : buf.at("bind").as_array()) {
                const auto& tname = seg.at("tensor").as_string();
                const auto& ref = index.at(tname);
                const uint8_t* src = mapped[ref.file_idx].ptr + ref.data_start;
                int64_t rows_total = ref.shape.empty() ? 1 : ref.shape[0];
                int64_t r0 = 0, r1 = rows_total;
                if (const auto* r = seg.get("rows")) {
                    r0 = r->as_array()[0].as_int();
                    r1 = r->as_array()[1].as_int();
                }
                int64_t row_elems = 1;
                for (size_t d = 1; d < ref.shape.size(); ++d) row_elems *= ref.shape[d];

                if (const auto* c = seg.get("cols")) {
                    int64_t c0 = c->as_array()[0].as_int();
                    int64_t c1 = c->as_array()[1].as_int();
                    size_t slice_bytes = static_cast<size_t>(c1 - c0) * ref.elem_bytes;
                    size_t row_bytes = static_cast<size_t>(row_elems) * ref.elem_bytes;
                    std::vector<uint8_t> packed(static_cast<size_t>(r1 - r0) * slice_bytes);
                    for (int64_t r = r0; r < r1; ++r) {
                        std::memcpy(packed.data() + static_cast<size_t>(r - r0) * slice_bytes,
                                    src + static_cast<size_t>(r) * row_bytes + static_cast<size_t>(c0) * ref.elem_bytes,
                                    slice_bytes);
                    }
                    cu_.check(cu_.cuMemcpyHtoDAsync(dst.ptr + off, packed.data(), packed.size(), stream_), "weight copy");
                    sync();
                    off += packed.size();
                } else {
                    size_t row_bytes = static_cast<size_t>(row_elems) * ref.elem_bytes;
                    size_t n = static_cast<size_t>(r1 - r0) * row_bytes;
                    cu_.check(cu_.cuMemcpyHtoDAsync(dst.ptr + off, src + static_cast<size_t>(r0) * row_bytes, n, stream_),
                              "weight copy");
                    off += n;
                }
            }
            if (off != dst.bytes) {
                throw std::runtime_error("weight `" + name + "`: bound " + std::to_string(off) +
                                         " bytes, expected " + std::to_string(dst.bytes));
            }
        }
        sync();
    }

    CallArg resolve_call_arg(const json::Value& arg) const {
        if (const auto* b = arg.get("buf")) {
            const auto& db = buffers_.at(b->as_string());
            size_t off = arg.contains("offset") ? static_cast<size_t>(arg.at("offset").as_int()) : 0;
            return {"ptr", db.ptr + off, 0.0, db.bytes - off, {}};
        }
        if (const auto* s = arg.get("state")) {
            const auto& db = states_.at(s->as_string());
            size_t off = arg.contains("offset") ? static_cast<size_t>(arg.at("offset").as_int()) : 0;
            return {"ptr", db.ptr + off, 0.0, db.bytes - off, {}};
        }
        if (const auto* v = arg.get("var")) return {"var", 0, 0.0, 0, *v};
        if (const auto* e = arg.get("expr")) return {"expr", 0, 0.0, 0, *e};
        for (const char* k : {"i32", "i64", "u32", "u64", "u8"}) {
            if (const auto* v = arg.get(k)) return {k, static_cast<uint64_t>(v->as_int()), 0.0, 0, {}};
        }
        if (const auto* v = arg.get("f32")) return {"f32", 0, v->as_double(), 0, {}};
        throw std::runtime_error("unsupported call arg");
    }

    std::array<uint8_t, 128> encode_tensormap(const json::Value& tm, uint64_t ptr, size_t total_bytes) const {
        if (!cu_.cuTensorMapEncodeTiled) {
            throw std::runtime_error("cuTensorMapEncodeTiled unavailable in this CUDA driver");
        }
        std::vector<uint64_t> dims, strides;
        std::vector<uint32_t> box, elem_strides;
        for (const auto& d : tm.at("dims").as_array()) dims.push_back(static_cast<uint64_t>(d.as_int()));
        for (const auto& s : tm.at("strides").as_array()) strides.push_back(static_cast<uint64_t>(s.as_int()));
        for (const auto& b : tm.at("box").as_array()) box.push_back(static_cast<uint32_t>(b.as_int()));
        elem_strides.assign(dims.size(), 1);
        if (!dims.empty() && dims.back() == 0) {
            dims.back() = total_bytes / strides.back();
        }
        int oob = (tm.contains("oob_nan") && tm.at("oob_nan").as_bool())
                      ? cuda::CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
                      : cuda::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
        cuda::CUtensorMap tmap{};
        cu_.check(cu_.cuTensorMapEncodeTiled(
                      &tmap, detail::tma_dtype(tm.at("dtype").as_string()), static_cast<uint32_t>(dims.size()),
                      reinterpret_cast<void*>(ptr), dims.data(), strides.data(), box.data(), elem_strides.data(),
                      cuda::CU_TENSOR_MAP_INTERLEAVE_NONE,
                      detail::tma_swizzle(tm.contains("swizzle") ? tm.at("swizzle").as_int() : 0),
                      detail::tma_l2(tm.contains("l2_promotion") ? tm.at("l2_promotion").as_int() : 0), oob),
                  "cuTensorMapEncodeTiled");
        std::array<uint8_t, 128> out{};
        std::memcpy(out.data(), &tmap, 128);
        return out;
    }

    static void pack_int(std::vector<uint8_t>& buf, size_t at, int w, int64_t v) {
        if (w == 1) { uint8_t x = static_cast<uint8_t>(v); std::memcpy(buf.data() + at, &x, 1); }
        else if (w == 4) { int32_t x = static_cast<int32_t>(v); std::memcpy(buf.data() + at, &x, 4); }
        else if (w == 8) { int64_t x = v; std::memcpy(buf.data() + at, &x, 8); }
        else throw std::runtime_error("unsupported pack width: " + std::to_string(w));
    }

    Slot compile_slot(const json::Value& larg, const std::string& ptype, const std::string& op_name,
                      const std::vector<CallArg>& call_args) const {
        if (const auto* pi = larg.get("param")) {
            const auto& ca = call_args[static_cast<size_t>(pi->as_int())];
            if (ca.kind == "ptr") return {Slot::Kind::ConstU64, ptype, ca.u64, 0.0, {}, {}, {}};
            if (ca.kind == "var" || ca.kind == "expr") return {Slot::Kind::Dyn, ptype, 0, 0.0, ca.expr, {}, {}};
            return {Slot::Kind::ConstScalar, ptype, ca.u64, ca.f64, {}, {}, {}};
        }
        if (const auto* sc = larg.get("scratch")) {
            return {Slot::Kind::ConstU64, ptype, scratch_.at(op_name + "::" + sc->as_string()).ptr, 0.0, {}, {}, {}};
        }
        if (const auto* pk = larg.get("pack")) {
            Slot s;
            s.kind = Slot::Kind::Pack;
            s.pack_base.assign(static_cast<size_t>(pk->at("size").as_int()), 0);
            for (const auto& fld : pk->at("fields").as_array()) {
                size_t at = static_cast<size_t>(fld.at("at").as_int());
                if (const auto* tm = fld.get("tensormap")) {
                    const auto& ca = call_args[static_cast<size_t>(tm->at("param").as_int())];
                    auto bytes = encode_tensormap(*tm, ca.u64, ca.bytes);
                    std::memcpy(s.pack_base.data() + at, bytes.data(), 128);
                } else if (const auto* fp = fld.get("param")) {
                    const auto& ca = call_args[static_cast<size_t>(fp->as_int())];
                    int w = fld.contains("width") ? static_cast<int>(fld.at("width").as_int()) : 0;
                    if (ca.kind == "ptr") {
                        std::memcpy(s.pack_base.data() + at, &ca.u64, 8);
                    } else if (ca.kind == "var" || ca.kind == "expr") {
                        s.pack_dyn.push_back({at, w ? w : 4, ca.expr});
                    } else if (ca.kind == "f32") {
                        float fv = static_cast<float>(ca.f64);
                        std::memcpy(s.pack_base.data() + at, &fv, 4);
                    } else {
                        int eff_w = w ? w : ((ca.kind == "i64" || ca.kind == "u64") ? 8 : 4);
                        pack_int(s.pack_base, at, eff_w, static_cast<int64_t>(ca.u64));
                    }
                } else if (const auto* fs = fld.get("scratch")) {
                    uint64_t ptr = scratch_.at(op_name + "::" + fs->as_string()).ptr;
                    std::memcpy(s.pack_base.data() + at, &ptr, 8);
                } else if (const auto* fv = fld.get("var")) {
                    int w = fld.contains("width") ? static_cast<int>(fld.at("width").as_int()) : 4;
                    s.pack_dyn.push_back({at, w, *fv});
                } else if (const auto* fe = fld.get("expr")) {
                    int w = fld.contains("width") ? static_cast<int>(fld.at("width").as_int()) : 4;
                    s.pack_dyn.push_back({at, w, *fe});
                } else if (const auto* v = fld.get("i32")) {
                    pack_int(s.pack_base, at, 4, v->as_int());
                } else if (const auto* v = fld.get("i64")) {
                    pack_int(s.pack_base, at, 8, v->as_int());
                } else if (const auto* v = fld.get("u8")) {
                    pack_int(s.pack_base, at, 1, v->as_int());
                } else if (const auto* v = fld.get("f32")) {
                    float fv = static_cast<float>(v->as_double());
                    std::memcpy(s.pack_base.data() + at, &fv, 4);
                } else {
                    throw std::runtime_error("unsupported pack field");
                }
            }
            return s;
        }
        for (const char* k : {"i32", "i64", "u32", "u64", "u8"}) {
            if (const auto* v = larg.get(k)) {
                return {Slot::Kind::ConstScalar, ptype, static_cast<uint64_t>(v->as_int()), 0.0, {}, {}, {}};
            }
        }
        if (const auto* v = larg.get("f32")) {
            return {Slot::Kind::ConstScalar, ptype, 0, v->as_double(), {}, {}, {}};
        }
        throw std::runtime_error("unsupported launch arg");
    }

    std::vector<Step> compile_program(const json::Value& prog) {
        std::vector<Step> steps;
        for (const auto& call : prog.at("calls").as_array()) {
            const auto& op_name = call.at("op").as_string();
            const auto& op = manifest.raw.at("ops").at(op_name);
            std::vector<CallArg> call_args;
            for (const auto& a : call.at("args").as_array()) call_args.push_back(resolve_call_arg(a));

            for (const auto& launch : op.at("impl").at("launches").as_array()) {
                const auto& lparams = (launch.contains("params") ? launch.at("params") : op.at("params")).as_array();
                json::Array default_args;
                const json::Array* largs = nullptr;
                if (const auto* la = launch.get("args"); la && la->is_array()) {
                    largs = &la->as_array();
                } else {
                    for (size_t i = 0; i < lparams.size(); ++i) {
                        json::Object o;
                        o.emplace("param", json::Value(static_cast<int64_t>(i)));
                        default_args.emplace_back(std::move(o));
                    }
                    largs = &default_args;
                }

                Step st;
                st.entry = launch.at("entry").as_string();
                st.is_extern = st.entry.starts_with("extern:");
                if (const auto* w = launch.get("when"); w && w->is_object()) {
                    When wh;
                    wh.var = w->at("var").as_string();
                    if (const auto* mn = w->get("min")) wh.min_val = mn->as_int();
                    if (const auto* mx = w->get("max")) wh.max_val = mx->as_int();
                    st.when = std::move(wh);
                }
                for (size_t i = 0; i < std::min(largs->size(), lparams.size()); ++i) {
                    st.args.push_back(compile_slot((*largs)[i], lparams[i].as_string(), op_name, call_args));
                }

                if (!st.is_extern) {
                    const auto& blk = launch.at("block").as_array();
                    const auto& grd = launch.at("grid").as_array();
                    for (int i = 0; i < 3; ++i) {
                        st.block[i] = static_cast<unsigned int>(blk[i].as_int());
                        st.grid[i] = grd[i];
                    }
                    if (const auto* sh = launch.get("shared_mem")) st.shmem = *sh;
                    int64_t max_sh = st.shmem.is_null() ? 0 : eval_expr(st.shmem, manifest.vars_max);
                    st.fn = get_func(launch.at("module").as_string(), st.entry, max_sh);
                }
                steps.push_back(std::move(st));
            }
        }
        return steps;
    }

    static uint64_t slot_u64(const Slot& s, const VarMap& vals) {
        if (s.kind == Slot::Kind::ConstU64 || s.kind == Slot::Kind::ConstScalar) return s.u64;
        if (s.kind == Slot::Kind::Dyn) return static_cast<uint64_t>(eval_expr(s.expr, vals));
        throw std::runtime_error("expected scalar/ptr slot");
    }

    void gemm_bf16_tn(uint64_t a_ptr, uint64_t w_ptr, uint64_t c_ptr,
                      uint64_t m, uint64_t n, uint64_t k, uint64_t ldc, uint64_t lda, float beta) const {
        if (m == 0 || n == 0 || k == 0) return;
        if (!blt_) throw std::runtime_error("libcublasLt.so unavailable for extern:cublaslt_bf16_tn");
        cuda::cublasLtMatmulDesc_t desc = nullptr;
        cuda::cublasLtMatrixLayout_t w_lay = nullptr, a_lay = nullptr, c_lay = nullptr;
        cuda::cublasLtMatmulPreference_t pref = nullptr;

        cuda::Api::check_blas(cu_.cublasLtMatmulDescCreate(&desc, cuda::CUBLAS_COMPUTE_32F, cuda::CUDA_R_32F),
                              "cublasLtMatmulDescCreate");
        int transa = cuda::CUBLAS_OP_T;
        cuda::Api::check_blas(cu_.cublasLtMatmulDescSetAttribute(desc, cuda::CUBLASLT_MATMUL_DESC_TRANSA, &transa, 4),
                              "cublasLtMatmulDescSetAttribute");
        cuda::Api::check_blas(cu_.cublasLtMatrixLayoutCreate(&w_lay, cuda::CUDA_R_16BF, k, n, static_cast<int64_t>(k)),
                              "layout W");
        cuda::Api::check_blas(cu_.cublasLtMatrixLayoutCreate(&a_lay, cuda::CUDA_R_16BF, k, m, static_cast<int64_t>(lda)),
                              "layout A");
        cuda::Api::check_blas(cu_.cublasLtMatrixLayoutCreate(&c_lay, cuda::CUDA_R_16BF, n, m, static_cast<int64_t>(ldc)),
                              "layout C");
        cuda::Api::check_blas(cu_.cublasLtMatmulPreferenceCreate(&pref), "prefCreate");
        cuda::Api::check_blas(cu_.cublasLtMatmulPreferenceSetAttribute(
                                  pref, cuda::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes_, sizeof(size_t)),
                              "prefSet");

        cuda::cublasLtMatmulHeuristicResult_t heur{};
        int returned = 0;
        cuda::Api::check_blas(cu_.cublasLtMatmulAlgoGetHeuristic(blt_, desc, w_lay, a_lay, c_lay, c_lay, pref, 1,
                                                                 &heur, &returned),
                              "cublasLtMatmulAlgoGetHeuristic");
        if (returned == 0) throw std::runtime_error("cublasLt found no heuristic algorithm");

        float alpha = 1.0f;
        cuda::Api::check_blas(cu_.cublasLtMatmul(blt_, desc, &alpha,
                                                 reinterpret_cast<const void*>(w_ptr), w_lay,
                                                 reinterpret_cast<const void*>(a_ptr), a_lay,
                                                 &beta, reinterpret_cast<const void*>(c_ptr), c_lay,
                                                 reinterpret_cast<void*>(c_ptr), c_lay, &heur.algo,
                                                 reinterpret_cast<void*>(workspace_), workspace_bytes_, stream_),
                              "cublasLtMatmul");

        cu_.cublasLtMatmulPreferenceDestroy(pref);
        cu_.cublasLtMatrixLayoutDestroy(c_lay);
        cu_.cublasLtMatrixLayoutDestroy(a_lay);
        cu_.cublasLtMatrixLayoutDestroy(w_lay);
        cu_.cublasLtMatmulDescDestroy(desc);
    }

    void issue(const std::string& pname, const VarMap& vals) {
        for (const auto& step : compiled_.at(pname)) {
            if (step.when) {
                int64_t v = vals.at(step.when->var);
                if ((step.when->min_val && v < *step.when->min_val) ||
                    (step.when->max_val && v > *step.when->max_val)) {
                    continue;
                }
            }
            if (step.is_extern) {
                if (step.entry == "extern:cublaslt_bf16_tn" || step.entry == "extern:cublaslt_bf16_tn_acc") {
                    uint64_t a = slot_u64(step.args[0], vals);
                    uint64_t w = slot_u64(step.args[1], vals);
                    uint64_t c = slot_u64(step.args[2], vals);
                    uint64_t m = slot_u64(step.args[3], vals);
                    uint64_t n = slot_u64(step.args[4], vals);
                    uint64_t k = slot_u64(step.args[5], vals);
                    uint64_t ldc = step.args.size() > 6 ? slot_u64(step.args[6], vals) : n;
                    uint64_t lda = step.args.size() > 7 ? slot_u64(step.args[7], vals) : k;
                    float beta = step.entry.ends_with("_acc") ? 1.0f : 0.0f;
                    gemm_bf16_tn(a, w, c, m, n, k, ldc, lda, beta);
                    continue;
                }
                throw std::runtime_error("unsupported extern: " + step.entry);
            }

            unsigned int gx = static_cast<unsigned int>(eval_expr(step.grid[0], vals));
            unsigned int gy = static_cast<unsigned int>(eval_expr(step.grid[1], vals));
            unsigned int gz = static_cast<unsigned int>(eval_expr(step.grid[2], vals));
            if (gx == 0 || gy == 0 || gz == 0) continue;
            unsigned int shmem = step.shmem.is_null() ? 0 : static_cast<unsigned int>(eval_expr(step.shmem, vals));

            std::vector<std::vector<uint8_t>> storage(step.args.size());
            std::vector<void*> ptrs(step.args.size());
            for (size_t i = 0; i < step.args.size(); ++i) {
                const auto& s = step.args[i];
                auto& buf = storage[i];
                if (s.kind == Slot::Kind::ConstU64) {
                    buf.resize(8);
                    std::memcpy(buf.data(), &s.u64, 8);
                } else if (s.kind == Slot::Kind::ConstScalar) {
                    if (s.ptype == "f32") {
                        float fv = static_cast<float>(s.f64);
                        buf.resize(4);
                        std::memcpy(buf.data(), &fv, 4);
                    } else if (s.ptype == "i64" || s.ptype == "u64") {
                        buf.resize(8);
                        std::memcpy(buf.data(), &s.u64, 8);
                    } else if (s.ptype == "u8") {
                        buf.resize(1);
                        buf[0] = static_cast<uint8_t>(s.u64);
                    } else {
                        int32_t iv = static_cast<int32_t>(s.u64);
                        buf.resize(4);
                        std::memcpy(buf.data(), &iv, 4);
                    }
                } else if (s.kind == Slot::Kind::Dyn) {
                    int64_t v = eval_expr(s.expr, vals);
                    if (s.ptype == "i64" || s.ptype == "u64") {
                        buf.resize(8);
                        std::memcpy(buf.data(), &v, 8);
                    } else if (s.ptype == "u8") {
                        buf.resize(1);
                        buf[0] = static_cast<uint8_t>(v);
                    } else {
                        int32_t iv = static_cast<int32_t>(v);
                        buf.resize(4);
                        std::memcpy(buf.data(), &iv, 4);
                    }
                } else {
                    buf = s.pack_base;
                    for (const auto& df : s.pack_dyn) {
                        pack_int(buf, df.at, df.width, eval_expr(df.expr, vals));
                    }
                }
                ptrs[i] = buf.data();
            }
            cu_.check(cu_.cuLaunchKernel(step.fn, gx, gy, gz, step.block[0], step.block[1], step.block[2],
                                         shmem, stream_, ptrs.data(), nullptr),
                      "cuLaunchKernel");
        }
    }
};

}  // namespace kern
