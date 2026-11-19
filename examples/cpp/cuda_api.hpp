// Minimal dlopen bindings for the CUDA Driver API and cuBLASLt.
//
// Zero compile-time dependency on CUDA headers or toolkit: builds on any
// Linux host and loads libcuda.so.1 / libcublasLt.so at runtime.
#pragma once

#include <cstddef>
#include <cstdint>
#include <dlfcn.h>
#include <stdexcept>
#include <string>
#include <string_view>

namespace kern::cuda {

using CUresult = int;
using CUdevice = int;
using CUcontext = void*;
using CUmodule = void*;
using CUfunction = void*;
using CUstream = void*;
using CUgraph = void*;
using CUgraphExec = void*;
using CUdeviceptr = uint64_t;

struct alignas(64) CUtensorMap {
    uint64_t opaque[16];
};

constexpr CUresult CUDA_SUCCESS = 0;
constexpr int CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8;
constexpr int CU_STREAM_CAPTURE_MODE_GLOBAL = 0;

// CUtensorMapDataType
constexpr int CU_TENSOR_MAP_DATA_TYPE_UINT8 = 0;
constexpr int CU_TENSOR_MAP_DATA_TYPE_UINT16 = 1;
constexpr int CU_TENSOR_MAP_DATA_TYPE_UINT32 = 2;
constexpr int CU_TENSOR_MAP_DATA_TYPE_INT32 = 3;
constexpr int CU_TENSOR_MAP_DATA_TYPE_UINT64 = 4;
constexpr int CU_TENSOR_MAP_DATA_TYPE_INT64 = 5;
constexpr int CU_TENSOR_MAP_DATA_TYPE_FLOAT16 = 6;
constexpr int CU_TENSOR_MAP_DATA_TYPE_FLOAT32 = 7;
constexpr int CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 = 9;
constexpr int CU_TENSOR_MAP_DATA_TYPE_TFLOAT32 = 11;

// CUtensorMapInterleave / Swizzle / L2promotion / FloatOOBfill
constexpr int CU_TENSOR_MAP_INTERLEAVE_NONE = 0;
constexpr int CU_TENSOR_MAP_SWIZZLE_NONE = 0;
constexpr int CU_TENSOR_MAP_SWIZZLE_32B = 1;
constexpr int CU_TENSOR_MAP_SWIZZLE_64B = 2;
constexpr int CU_TENSOR_MAP_SWIZZLE_128B = 3;
constexpr int CU_TENSOR_MAP_L2_PROMOTION_NONE = 0;
constexpr int CU_TENSOR_MAP_L2_PROMOTION_L2_64B = 1;
constexpr int CU_TENSOR_MAP_L2_PROMOTION_L2_128B = 2;
constexpr int CU_TENSOR_MAP_L2_PROMOTION_L2_256B = 3;
constexpr int CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE = 0;
constexpr int CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA = 1;

// cuBLASLt ABI types and constants (CUDA 12/13).
using cublasStatus_t = int;
using cublasLtHandle_t = void*;
using cublasLtMatmulDesc_t = void*;
using cublasLtMatrixLayout_t = void*;
using cublasLtMatmulPreference_t = void*;

struct cublasLtMatmulAlgo_t {
    uint64_t data[8];
};

struct cublasLtMatmulHeuristicResult_t {
    cublasLtMatmulAlgo_t algo;
    size_t workspaceSize;
    cublasStatus_t state;
    float wavesCount;
    int reserved[4];
};

constexpr cublasStatus_t CUBLAS_STATUS_SUCCESS = 0;
constexpr int CUDA_R_32F = 0;
constexpr int CUDA_R_16BF = 14;
constexpr int CUBLAS_OP_N = 0;
constexpr int CUBLAS_OP_T = 1;
constexpr int CUBLAS_COMPUTE_32F = 68;
constexpr int CUBLASLT_MATMUL_DESC_TRANSA = 3;
constexpr int CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1;

struct Api {
    void* libcuda = nullptr;
    void* libcublaslt = nullptr;

    CUresult (*cuInit)(unsigned int) = nullptr;
    CUresult (*cuGetErrorString)(CUresult, const char**) = nullptr;
    CUresult (*cuDeviceGet)(CUdevice*, int) = nullptr;
    CUresult (*cuDevicePrimaryCtxRetain)(CUcontext*, CUdevice) = nullptr;
    CUresult (*cuCtxSetCurrent)(CUcontext) = nullptr;
    CUresult (*cuMemAlloc)(CUdeviceptr*, size_t) = nullptr;
    CUresult (*cuMemFree)(CUdeviceptr) = nullptr;
    CUresult (*cuMemsetD8)(CUdeviceptr, unsigned char, size_t) = nullptr;
    CUresult (*cuMemcpyHtoDAsync)(CUdeviceptr, const void*, size_t, CUstream) = nullptr;
    CUresult (*cuMemcpyDtoH)(void*, CUdeviceptr, size_t) = nullptr;
    CUresult (*cuStreamCreate)(CUstream*, unsigned int) = nullptr;
    CUresult (*cuStreamDestroy)(CUstream) = nullptr;
    CUresult (*cuStreamSynchronize)(CUstream) = nullptr;
    CUresult (*cuModuleLoadData)(CUmodule*, const void*) = nullptr;
    CUresult (*cuModuleUnload)(CUmodule) = nullptr;
    CUresult (*cuModuleGetFunction)(CUfunction*, CUmodule, const char*) = nullptr;
    CUresult (*cuFuncSetAttribute)(CUfunction, int, int) = nullptr;
    CUresult (*cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int,
                               unsigned int, unsigned int, unsigned int,
                               unsigned int, CUstream, void**, void**) = nullptr;
    CUresult (*cuTensorMapEncodeTiled)(CUtensorMap*, int, uint32_t, void*,
                                       const uint64_t*, const uint64_t*, const uint32_t*,
                                       const uint32_t*, int, int, int, int) = nullptr;
    CUresult (*cuStreamBeginCapture)(CUstream, int) = nullptr;
    CUresult (*cuStreamEndCapture)(CUstream, CUgraph*) = nullptr;
    CUresult (*cuGraphInstantiateWithFlags)(CUgraphExec*, CUgraph, uint64_t) = nullptr;
    CUresult (*cuGraphLaunch)(CUgraphExec, CUstream) = nullptr;
    CUresult (*cuGraphExecDestroy)(CUgraphExec) = nullptr;
    CUresult (*cuGraphDestroy)(CUgraph) = nullptr;

    cublasStatus_t (*cublasLtCreate)(cublasLtHandle_t*) = nullptr;
    cublasStatus_t (*cublasLtDestroy)(cublasLtHandle_t) = nullptr;
    cublasStatus_t (*cublasLtMatmulDescCreate)(cublasLtMatmulDesc_t*, int, int) = nullptr;
    cublasStatus_t (*cublasLtMatmulDescDestroy)(cublasLtMatmulDesc_t) = nullptr;
    cublasStatus_t (*cublasLtMatmulDescSetAttribute)(cublasLtMatmulDesc_t, int, const void*, size_t) = nullptr;
    cublasStatus_t (*cublasLtMatrixLayoutCreate)(cublasLtMatrixLayout_t*, int, uint64_t, uint64_t, int64_t) = nullptr;
    cublasStatus_t (*cublasLtMatrixLayoutDestroy)(cublasLtMatrixLayout_t) = nullptr;
    cublasStatus_t (*cublasLtMatmulPreferenceCreate)(cublasLtMatmulPreference_t*) = nullptr;
    cublasStatus_t (*cublasLtMatmulPreferenceDestroy)(cublasLtMatmulPreference_t) = nullptr;
    cublasStatus_t (*cublasLtMatmulPreferenceSetAttribute)(cublasLtMatmulPreference_t, int, const void*, size_t) = nullptr;
    cublasStatus_t (*cublasLtMatmulAlgoGetHeuristic)(cublasLtHandle_t, cublasLtMatmulDesc_t,
                                                     cublasLtMatrixLayout_t, cublasLtMatrixLayout_t,
                                                     cublasLtMatrixLayout_t, cublasLtMatrixLayout_t,
                                                     cublasLtMatmulPreference_t, int,
                                                     cublasLtMatmulHeuristicResult_t*, int*) = nullptr;
    cublasStatus_t (*cublasLtMatmul)(cublasLtHandle_t, cublasLtMatmulDesc_t, const void*,
                                     const void*, cublasLtMatrixLayout_t, const void*, cublasLtMatrixLayout_t,
                                     const void*, const void*, cublasLtMatrixLayout_t,
                                     void*, cublasLtMatrixLayout_t, const cublasLtMatmulAlgo_t*,
                                     void*, size_t, CUstream) = nullptr;

    static const Api& load() {
        static Api api = load_impl();
        return api;
    }

    void check(CUresult err, const char* what) const {
        if (err != CUDA_SUCCESS) {
            const char* msg = "unknown";
            if (cuGetErrorString) cuGetErrorString(err, &msg);
            throw std::runtime_error(std::string(what) + " failed: CUDA error " + std::to_string(err) +
                                     " (" + (msg ? msg : "unknown") + ")");
        }
    }

    static void check_blas(cublasStatus_t err, const char* what) {
        if (err != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error(std::string(what) + " failed: cublasStatus " + std::to_string(err));
        }
    }

private:
    template <typename Fn>
    static void sym(void* lib, const char* name, Fn& out, bool required = true) {
        void* p = dlsym(lib, name);
        if (!p && required) {
            throw std::runtime_error(std::string("missing symbol ") + name);
        }
        out = reinterpret_cast<Fn>(p);
    }

    static Api load_impl() {
        Api a;
        a.libcuda = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
        if (!a.libcuda) a.libcuda = dlopen("libcuda.so", RTLD_LAZY | RTLD_LOCAL);
        if (!a.libcuda) throw std::runtime_error(std::string("cannot load libcuda.so.1: ") + dlerror());

        sym(a.libcuda, "cuInit", a.cuInit);
        sym(a.libcuda, "cuGetErrorString", a.cuGetErrorString);
        sym(a.libcuda, "cuDeviceGet", a.cuDeviceGet);
        sym(a.libcuda, "cuDevicePrimaryCtxRetain", a.cuDevicePrimaryCtxRetain);
        sym(a.libcuda, "cuCtxSetCurrent", a.cuCtxSetCurrent);
        sym(a.libcuda, "cuMemAlloc_v2", a.cuMemAlloc);
        sym(a.libcuda, "cuMemFree_v2", a.cuMemFree);
        sym(a.libcuda, "cuMemsetD8_v2", a.cuMemsetD8);
        sym(a.libcuda, "cuMemcpyHtoDAsync_v2", a.cuMemcpyHtoDAsync);
        sym(a.libcuda, "cuMemcpyDtoH_v2", a.cuMemcpyDtoH);
        sym(a.libcuda, "cuStreamCreate", a.cuStreamCreate);
        sym(a.libcuda, "cuStreamDestroy_v2", a.cuStreamDestroy);
        sym(a.libcuda, "cuStreamSynchronize", a.cuStreamSynchronize);
        sym(a.libcuda, "cuModuleLoadData", a.cuModuleLoadData);
        sym(a.libcuda, "cuModuleUnload", a.cuModuleUnload);
        sym(a.libcuda, "cuModuleGetFunction", a.cuModuleGetFunction);
        sym(a.libcuda, "cuFuncSetAttribute", a.cuFuncSetAttribute);
        sym(a.libcuda, "cuLaunchKernel", a.cuLaunchKernel);
        sym(a.libcuda, "cuTensorMapEncodeTiled", a.cuTensorMapEncodeTiled, false);
        sym(a.libcuda, "cuStreamBeginCapture_v2", a.cuStreamBeginCapture);
        sym(a.libcuda, "cuStreamEndCapture", a.cuStreamEndCapture);
        sym(a.libcuda, "cuGraphInstantiateWithFlags", a.cuGraphInstantiateWithFlags);
        sym(a.libcuda, "cuGraphLaunch", a.cuGraphLaunch);
        sym(a.libcuda, "cuGraphExecDestroy", a.cuGraphExecDestroy);
        sym(a.libcuda, "cuGraphDestroy", a.cuGraphDestroy);

        for (const char* name : {"libcublasLt.so.13", "libcublasLt.so.12", "libcublasLt.so"}) {
            a.libcublaslt = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
            if (a.libcublaslt) break;
        }
        if (a.libcublaslt) {
            sym(a.libcublaslt, "cublasLtCreate", a.cublasLtCreate);
            sym(a.libcublaslt, "cublasLtDestroy", a.cublasLtDestroy);
            sym(a.libcublaslt, "cublasLtMatmulDescCreate", a.cublasLtMatmulDescCreate);
            sym(a.libcublaslt, "cublasLtMatmulDescDestroy", a.cublasLtMatmulDescDestroy);
            sym(a.libcublaslt, "cublasLtMatmulDescSetAttribute", a.cublasLtMatmulDescSetAttribute);
            sym(a.libcublaslt, "cublasLtMatrixLayoutCreate", a.cublasLtMatrixLayoutCreate);
            sym(a.libcublaslt, "cublasLtMatrixLayoutDestroy", a.cublasLtMatrixLayoutDestroy);
            sym(a.libcublaslt, "cublasLtMatmulPreferenceCreate", a.cublasLtMatmulPreferenceCreate);
            sym(a.libcublaslt, "cublasLtMatmulPreferenceDestroy", a.cublasLtMatmulPreferenceDestroy);
            sym(a.libcublaslt, "cublasLtMatmulPreferenceSetAttribute", a.cublasLtMatmulPreferenceSetAttribute);
            sym(a.libcublaslt, "cublasLtMatmulAlgoGetHeuristic", a.cublasLtMatmulAlgoGetHeuristic);
            sym(a.libcublaslt, "cublasLtMatmul", a.cublasLtMatmul);
        }
        return a;
    }
};

}  // namespace kern::cuda
