//! The extern ops the runtime provides itself, all cuBLAS: the kernels a
//! manifest names as `extern:` instead of shipping. `Blas` is the handle
//! and workspace the f32-result GEMM needs beside the cublasLt one.

use std::os::raw::c_void;
use std::sync::Arc;

use cudarc::cublas;
use cudarc::cublaslt::{self, CudaBlasLT, Matmul, MatmulConfig, MatmulShared};
use cudarc::driver::{sys, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceSlice, SyncOnDrop};
use half::bf16;

use crate::compile::RVal;
use crate::error::{bail, Error, Result};

/// Raw device pointer presented as a `DevicePtr<bf16>`/`DevicePtrMut<bf16>`
/// for the cublasLt extern op.
struct RawBf16 {
    ptr: sys::CUdeviceptr,
    len: usize,
    stream: Arc<CudaStream>,
}

impl DeviceSlice<bf16> for RawBf16 {
    fn len(&self) -> usize {
        self.len
    }
    fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

impl DevicePtr<bf16> for RawBf16 {
    fn device_ptr<'a>(&'a self, _: &'a CudaStream) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        (self.ptr, SyncOnDrop::Record(None))
    }
}

impl DevicePtrMut<bf16> for RawBf16 {
    fn device_ptr_mut<'a>(&'a mut self, _: &'a CudaStream) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        (self.ptr, SyncOnDrop::Record(None))
    }
}

/// `extern:cublaslt_bf16_tn`: row-major `C[m,n] = A[m,k] @ W[n,k]^T`,
/// resolved args `[a, w, c, m, n, k]`. Column-major mapping: compute
/// `C_cm[n,m] = W_cm^T[n,k] x A_cm[k,m]` -> transa=T on W (lda=k),
/// transb=N on A (ldb=k), m'=n, n'=m, ldc=n.
/// `extern:cublaslt_bf16_tn_acc` is the same with beta=1: `C += A @ W^T`.
pub(crate) fn gemm_bf16_tn(blt: &CudaBlasLT, stream: &Arc<CudaStream>, args: &[RVal], beta: f32) -> Result<()> {
    // `c[m, n] (+)= a[m, k] @ w[n, k]^T`; an optional 7th arg is C's row
    // stride in elements (default n), and an optional 8th arg is A's row
    // stride (default k). This permits group views without transposing A.
    let (a, w, c, m, n, k, ldc, a_stride) = match args {
        [a, w, c, m, n, k] => (a, w, c, m.val, n.val, k.val, n.val, k.val),
        [a, w, c, m, n, k, ldc] => (a, w, c, m.val, n.val, k.val, ldc.val, k.val),
        [a, w, c, m, n, k, ldc, a_stride] => (a, w, c, m.val, n.val, k.val, ldc.val, a_stride.val),
        _ => bail!(Manifest, "gemm expects 6, 7 or 8 args, got {}", args.len()),
    };
    if ldc < n {
        bail!(Manifest, "gemm: ldc {ldc} < n {n}");
    }
    if a_stride < k {
        bail!(Manifest, "gemm: A row stride {a_stride} < k {k}");
    }
    if m == 0 || n == 0 || k == 0 {
        return Ok(());
    }
    if a.bytes < ((m - 1) * a_stride + k) * 2 || w.bytes < n * k * 2 || c.bytes < ((m - 1) * ldc + n) * 2 {
        bail!(Manifest, "gemm: operands too small for m={m} n={n} k={k} ldc={ldc} A row stride={a_stride}");
    }
    let view = |rv: &RVal| RawBf16 { ptr: rv.val, len: (rv.bytes / 2) as usize, stream: stream.clone() };
    let cfg = MatmulConfig {
        transa: true,
        transb: false,
        transc: false,
        m: n,
        n: m,
        k,
        alpha: 1.0,
        beta,
        lda: k as i64,
        ldb: a_stride as i64,
        ldc: ldc as i64,
        stride_a: None,
        stride_b: None,
        stride_c: None,
        stride_bias: None,
        batch_size: None,
    };
    let mut out = view(c);
    unsafe {
        blt.matmul(cfg, &view(w), &view(a), &mut out, None, None)
            .map_err(|e| Error::Cuda(format!("cublasLt matmul (m={m} n={n} k={k}): {e:?}")))?;
    }
    Ok(())
}

/// A cuBLAS handle bound to the runtime's stream with its own workspace, for
/// the f32-result GEMM built-in (`cublasGemmEx`; cublasLt's typed `Matmul`
/// only lands in the operand type). Kept separate from the Lt handle so the
/// two never share a workspace.
pub(crate) struct Blas {
    handle: cublas::sys::cublasHandle_t,
    _workspace: CudaSlice<u8>,
    /// The workspace's address, taken once: reading it through cudarc's
    /// `device_ptr` records a cross-stream wait that invalidates a capture.
    ws: sys::CUdeviceptr,
}

// The handle is only ever used from the runtime's own thread, on its stream.
unsafe impl Send for Blas {}
unsafe impl Sync for Blas {}

impl Blas {
    const WORKSPACE: usize = 32 << 20;

    pub(crate) fn new(stream: &Arc<CudaStream>) -> Result<Blas> {
        let handle = cublas::result::create_handle().map_err(|e| Error::Cuda(format!("cublasCreate: {e:?}")))?;
        let workspace: CudaSlice<u8> = stream.alloc_zeros(Self::WORKSPACE)?;
        let ws = workspace.device_ptr(stream).0;
        unsafe {
            cublas::result::set_stream(handle, stream.cu_stream() as *mut _)
                .map_err(|e| Error::Cuda(format!("cublasSetStream: {e:?}")))?;
            cublas::sys::cublasSetWorkspace_v2(handle, ws as *mut c_void, Self::WORKSPACE)
                .result()
                .map_err(|e| Error::Cuda(format!("cublasSetWorkspace: {e:?}")))?;
            cublas::sys::cublasSetMathMode(handle, cublas::sys::cublasMath_t::CUBLAS_TENSOR_OP_MATH)
                .result()
                .map_err(|e| Error::Cuda(format!("cublasSetMathMode: {e:?}")))?;
        }
        Ok(Blas { handle, _workspace: workspace, ws })
    }
}

impl Drop for Blas {
    fn drop(&mut self) {
        unsafe {
            let _ = cublas::sys::cublasDestroy_v2(self.handle);
        }
    }
}

/// `extern:cublas_bf16_tn_f32`: row-major `C[m,n] = A[m,k] @ W[n,k]^T` with
/// bf16 operands, f32 accumulation and an **f32** result — cublasGemmEx with
/// `CUBLAS_COMPUTE_32F` / `CUBLAS_GEMM_DEFAULT_TENSOR_OP`, the call an
/// engine that lands its own bf16 partials makes. Args as
/// [`gemm_bf16_tn`]: `[a, w, c, m, n, k]` or 7 with C's row stride.
pub(crate) fn gemm_bf16_tn_f32(blas: &Blas, args: &[RVal]) -> Result<()> {
    let (a, w, c, m, n, k, ldc) = match args {
        [a, w, c, m, n, k] => (a, w, c, m.val, n.val, k.val, n.val),
        [a, w, c, m, n, k, ldc] => (a, w, c, m.val, n.val, k.val, ldc.val),
        _ => bail!(Manifest, "gemm expects 6 or 7 args, got {}", args.len()),
    };
    if ldc < n {
        bail!(Manifest, "gemm: ldc {ldc} < n {n}");
    }
    if m == 0 || n == 0 || k == 0 {
        return Ok(());
    }
    if a.bytes < m * k * 2 || w.bytes < n * k * 2 || c.bytes < ((m - 1) * ldc + n) * 4 {
        bail!(
            Manifest,
            "gemm f32: operands too small for m={m} n={n} k={k} ldc={ldc}: a {} B, w {} B, c {} B",
            a.bytes,
            w.bytes,
            c.bytes
        );
    }
    let dim = |v: u64| i32::try_from(v).map_err(|_| Error::Manifest(format!("gemm f32: dimension {v} exceeds i32")));
    let (alpha, beta) = (1.0f32, 0.0f32);
    use cublas::sys::{cublasComputeType_t, cublasGemmAlgo_t, cublasOperation_t, cudaDataType};
    unsafe {
        cublas::result::gemm_ex(
            blas.handle,
            cublasOperation_t::CUBLAS_OP_T,
            cublasOperation_t::CUBLAS_OP_N,
            dim(n)?,
            dim(m)?,
            dim(k)?,
            &alpha as *const f32 as *const c_void,
            w.val as *const c_void,
            cudaDataType::CUDA_R_16BF,
            dim(k)?,
            a.val as *const c_void,
            cudaDataType::CUDA_R_16BF,
            dim(k)?,
            &beta as *const f32 as *const c_void,
            c.val as *mut c_void,
            cudaDataType::CUDA_R_32F,
            dim(ldc)?,
            cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        )
        .map_err(|e| Error::Cuda(format!("cublasGemmEx f32 (m={m} n={n} k={k}): {e:?}")))?;
    }
    Ok(())
}

/// `extern:cublaslt_fp8_tn` / `extern:cublaslt_fp8_tn_f32`: row-major
/// `C[m,n] = (a_scale * w_scale) * A[m,k] @ W[n,k]^T` with e4m3 operands,
/// f32 accumulation and a bf16 or f32 result; args
/// `[a, w, c, a_scale, w_scale, m, n, k]`, optionally a 9th C row stride.
/// The scales are one f32 each on the device (cublasLt's per-tensor
/// `A_SCALE_POINTER` / `B_SCALE_POINTER`), so a manifest can quantize the
/// activation in-graph and the GEMM never sees a host value. Same
/// column-major mapping as the bf16 path; cublasLt's fp8 rule that A is
/// transposed and B is not is exactly it. The workspace is `Blas`'s: both
/// GEMMs run on the one stream, so the buffer is never shared in flight.
pub(crate) fn gemm_fp8_tn(
    blt: &CudaBlasLT,
    blas: &Blas,
    stream: &Arc<CudaStream>,
    args: &[RVal],
    f32_out: bool,
) -> Result<()> {
    let (a, w, c, a_scale, w_scale, m, n, k, ldc) = match args {
        [a, w, c, sa, sw, m, n, k] => (a, w, c, sa, sw, m.val, n.val, k.val, n.val),
        [a, w, c, sa, sw, m, n, k, ldc] => (a, w, c, sa, sw, m.val, n.val, k.val, ldc.val),
        _ => bail!(Manifest, "fp8 gemm expects 8 or 9 args, got {}", args.len()),
    };
    if ldc < n {
        bail!(Manifest, "fp8 gemm: ldc {ldc} < n {n}");
    }
    if k % 16 != 0 {
        bail!(Manifest, "fp8 gemm: k {k} is not a multiple of 16 (cublasLt fp8 operand rows are 16-byte aligned)");
    }
    if m == 0 || n == 0 || k == 0 {
        return Ok(());
    }
    let out = if f32_out { 4 } else { 2 };
    if a.bytes < m * k
        || w.bytes < n * k
        || c.bytes < ((m - 1) * ldc + n) * out
        || a_scale.bytes < 4
        || w_scale.bytes < 4
    {
        bail!(Manifest, "fp8 gemm: operands too small for m={m} n={n} k={k} ldc={ldc}");
    }
    use cublaslt::result;
    use cublaslt::sys::{cublasComputeType_t, cublasLtMatmulDescAttributes_t as Attr, cudaDataType};
    let e4m3 = cudaDataType::CUDA_R_8F_E4M3;
    let cd = if f32_out { cudaDataType::CUDA_R_32F } else { cudaDataType::CUDA_R_16BF };
    let lt = |e: cublaslt::result::CublasError, what: &str| {
        Error::Cuda(format!("cublasLt fp8 {what} (m={m} n={n} k={k}): {e:?}"))
    };
    unsafe {
        let a_layout = result::create_matrix_layout(e4m3, k, n, k as i64).map_err(|e| lt(e, "layout"))?;
        let b_layout = result::create_matrix_layout(e4m3, k, m, k as i64).map_err(|e| lt(e, "layout"))?;
        let c_layout = result::create_matrix_layout(cd, n, m, ldc as i64).map_err(|e| lt(e, "layout"))?;
        let desc = result::create_matmul_desc(cublasComputeType_t::CUBLAS_COMPUTE_32F, cudaDataType::CUDA_R_32F)
            .map_err(|e| lt(e, "desc"))?;
        let set = |attr: Attr, v: *const c_void, size: usize| {
            result::set_matmul_desc_attribute(desc, attr, v, size).map_err(|e| lt(e, "attribute"))
        };
        let (t, nt) = (cublas::sys::cublasOperation_t::CUBLAS_OP_T, cublas::sys::cublasOperation_t::CUBLAS_OP_N);
        set(Attr::CUBLASLT_MATMUL_DESC_TRANSA, &t as *const _ as *const c_void, 4)?;
        set(Attr::CUBLASLT_MATMUL_DESC_TRANSB, &nt as *const _ as *const c_void, 4)?;
        let (sa, sw) = (a_scale.val as *const c_void, w_scale.val as *const c_void);
        set(Attr::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sw as *const _ as *const c_void, 8)?;
        set(Attr::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sa as *const _ as *const c_void, 8)?;
        let pref = result::create_matmul_pref().map_err(|e| lt(e, "preference"))?;
        let ws_size = Blas::WORKSPACE;
        result::set_matmul_pref_attribute(
            pref,
            cublaslt::sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &ws_size as *const _ as *const c_void,
            std::mem::size_of::<usize>(),
        )
        .map_err(|e| lt(e, "preference"))?;
        let heuristic =
            result::get_matmul_algo_heuristic(*blt.handle(), desc, a_layout, b_layout, c_layout, c_layout, pref)
                .map_err(|e| lt(e, "heuristic"))?;
        let (alpha, beta) = (1.0f32, 0.0f32);
        let r = result::matmul(
            *blt.handle(),
            desc,
            &alpha as *const _ as *const c_void,
            &beta as *const _ as *const c_void,
            w.val as *const c_void,
            a_layout,
            a.val as *const c_void,
            b_layout,
            c.val as *const c_void,
            c_layout,
            c.val as *mut c_void,
            c_layout,
            &heuristic.algo,
            blas.ws as *mut c_void,
            ws_size,
            stream.cu_stream() as *mut _,
        )
        .map_err(|e| lt(e, "matmul"));
        let _ = result::destroy_matmul_pref(pref);
        let _ = result::destroy_matmul_desc(desc);
        for l in [a_layout, b_layout, c_layout] {
            let _ = result::destroy_matrix_layout(l);
        }
        r
    }
}
