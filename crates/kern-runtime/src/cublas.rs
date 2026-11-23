//! The extern ops the runtime provides itself, all cuBLAS: the kernels a
//! manifest names as `extern:` instead of shipping. Each stream launches them
//! through its cuBLASLt handle.

use std::os::raw::c_void;
use std::sync::Arc;

use cudarc::cublas;
use cudarc::cublaslt::{self, CudaBlasLT, Matmul, MatmulConfig, MatmulShared};
use cudarc::driver::{sys, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceSlice, SyncOnDrop};
use half::bf16;
use kern_manifest::types::GemmAlgo;

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

/// The operands of a `cublaslt_bf16_tn[_acc]` call, `[a, w, c, m, n, k]` plus
/// an optional C row stride (default n) and A row stride (default k), in
/// elements, so a group view needs no transposed copy of A; checked against
/// their buffers, `None` for an empty product.
pub(crate) fn bf16_operands(args: &[RVal]) -> Result<Option<(&RVal, &RVal, &RVal, GemmShape)>> {
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
        return Ok(None);
    }
    if a.bytes < ((m - 1) * a_stride + k) * 2 || w.bytes < n * k * 2 || c.bytes < ((m - 1) * ldc + n) * 2 {
        bail!(Manifest, "gemm: operands too small for m={m} n={n} k={k} ldc={ldc} A row stride={a_stride}");
    }
    Ok(Some((a, w, c, GemmShape { m, n, k, ldc, a_stride })))
}

/// `extern:cublaslt_bf16_tn`: row-major `C[m,n] = A[m,k] @ W[n,k]^T`,
/// resolved args `[a, w, c, m, n, k]`. Column-major mapping: compute
/// `C_cm[n,m] = W_cm^T[n,k] x A_cm[k,m]` -> transa=T on W (lda=k),
/// transb=N on A (ldb=k), m'=n, n'=m, ldc=n.
/// `extern:cublaslt_bf16_tn_acc` is the same with beta=1: `C += A @ W^T`.
/// Optional row strides as [`bf16_operands`] reads them.
pub(crate) fn gemm_bf16_tn(blt: &CudaBlasLT, stream: &Arc<CudaStream>, args: &[RVal], beta: f32) -> Result<()> {
    let Some((a, w, c, GemmShape { m, n, k, ldc, a_stride })) = bf16_operands(args)? else {
        return Ok(());
    };
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

/// A bf16 GEMM's shape: `C[m, n] = A[m, k] · W[n, k]ᵀ` with C's and A's row strides.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GemmShape {
    m: u64,
    n: u64,
    k: u64,
    ldc: u64,
    a_stride: u64,
}

/// A matmul descriptor and the `w`, `a`, `c` layouts of a bf16 `C = A · Wᵀ`,
/// column-major as in [`gemm_bf16_tn`]; destroyed on drop.
struct Described {
    desc: cublaslt::sys::cublasLtMatmulDesc_t,
    layouts: [cublaslt::sys::cublasLtMatrixLayout_t; 3],
}

impl Described {
    /// The descriptors of a bf16 `C = A · Wᵀ` whose result lands as `out`:
    /// bf16 here, or f32 for `extern:cublas_bf16_tn_f32` — cuBLASLt takes C's
    /// and D's type from their layout, independent of the operands'.
    fn new(s: GemmShape, out: cublaslt::sys::cudaDataType) -> Result<Self> {
        use cublaslt::result;
        use cublaslt::sys::{cublasComputeType_t, cublasLtMatmulDescAttributes_t as Attr, cudaDataType};
        let err =
            |e: cublaslt::result::CublasError| Error::Cuda(format!("cublasLt descriptor ({s:?}, {out:?}): {e:?}"));
        let bf = cudaDataType::CUDA_R_16BF;
        unsafe {
            let desc = result::create_matmul_desc(cublasComputeType_t::CUBLAS_COMPUTE_32F, cudaDataType::CUDA_R_32F)
                .map_err(err)?;
            let mut d = Described { desc, layouts: [std::ptr::null_mut(); 3] };
            let t = cublas::sys::cublasOperation_t::CUBLAS_OP_T;
            result::set_matmul_desc_attribute(
                desc,
                Attr::CUBLASLT_MATMUL_DESC_TRANSA,
                &t as *const _ as *const c_void,
                4,
            )
            .map_err(err)?;
            d.layouts[0] = result::create_matrix_layout(bf, s.k, s.n, s.k as i64).map_err(err)?;
            d.layouts[1] = result::create_matrix_layout(bf, s.k, s.m, s.a_stride as i64).map_err(err)?;
            d.layouts[2] = result::create_matrix_layout(out, s.n, s.m, s.ldc as i64).map_err(err)?;
            Ok(d)
        }
    }
}

impl Drop for Described {
    fn drop(&mut self) {
        use cublaslt::result;
        unsafe {
            for l in self.layouts {
                if !l.is_null() {
                    let _ = result::destroy_matrix_layout(l);
                }
            }
            let _ = result::destroy_matmul_desc(self.desc);
        }
    }
}

/// A launch's pinned cuBLASLt algorithm, built from the manifest's config.
#[derive(Clone, Copy)]
pub(crate) struct Pinned(cublaslt::sys::cublasLtMatmulAlgo_t);

impl Pinned {
    /// `cublasLtMatmulAlgoInit` for the bf16 GEMM's types, then every config
    /// attribute set as `algo` says.
    pub(crate) fn new(blt: &CudaBlasLT, algo: &GemmAlgo) -> Result<Pinned> {
        use cublaslt::sys::{
            self as lt, cublasComputeType_t, cublasLtMatmulAlgoConfigAttributes_t as Cfg, cudaDataType,
        };
        let bf = cudaDataType::CUDA_R_16BF;
        let mut raw = std::mem::MaybeUninit::<lt::cublasLtMatmulAlgo_t>::zeroed();
        let status = unsafe {
            lt::cublasLtMatmulAlgoInit(
                *blt.handle(),
                cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cudaDataType::CUDA_R_32F,
                bf,
                bf,
                bf,
                bf,
                algo.id,
                raw.as_mut_ptr(),
            )
        };
        if status != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            bail!(KernelArtifact, "cublasLt has no bf16 algorithm {} here ({status:?})", algo.id);
        }
        let mut raw = unsafe { raw.assume_init() };
        let mut set = |attr: Cfg, (v, size): (*const c_void, usize), what: &str| {
            let status = unsafe { lt::cublasLtMatmulAlgoConfigSetAttribute(&mut raw, attr, v, size) };
            if status == lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                Ok(())
            } else {
                Err(Error::KernelArtifact(format!("cublasLt algorithm {}: {what} rejected ({status:?})", algo.id)))
            }
        };
        fn raw_of<T>(v: &T) -> (*const c_void, usize) {
            (v as *const T as *const c_void, std::mem::size_of::<T>())
        }
        set(Cfg::CUBLASLT_ALGO_CONFIG_TILE_ID, raw_of(&algo.tile), "tile")?;
        set(Cfg::CUBLASLT_ALGO_CONFIG_STAGES_ID, raw_of(&algo.stages), "stages")?;
        set(Cfg::CUBLASLT_ALGO_CONFIG_SPLITK_NUM, raw_of(&algo.split_k), "split_k")?;
        set(Cfg::CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, raw_of(&algo.reduction), "reduction")?;
        set(Cfg::CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, raw_of(&algo.swizzle), "swizzle")?;
        set(Cfg::CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, raw_of(&algo.custom), "custom")?;
        set(Cfg::CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, raw_of(&algo.inner_shape), "inner_shape")?;
        set(Cfg::CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, raw_of(&algo.cluster_shape), "cluster_shape")?;
        Ok(Pinned(raw))
    }

    /// `cublasLtMatmulAlgoCheck` at `shape`, the workspace within [`Workspace`]'s.
    pub(crate) fn check(&self, blt: &CudaBlasLT, shape: GemmShape) -> Result<()> {
        use cublaslt::sys as lt;
        let d = Described::new(shape, cublaslt::sys::cudaDataType::CUDA_R_16BF)?;
        let mut result = std::mem::MaybeUninit::<lt::cublasLtMatmulHeuristicResult_t>::zeroed();
        let [w, a, c] = d.layouts;
        let status =
            unsafe { lt::cublasLtMatmulAlgoCheck(*blt.handle(), d.desc, w, a, c, c, &self.0, result.as_mut_ptr()) };
        if status != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            bail!(KernelArtifact, "the pinned cublasLt algorithm does not run at {shape:?} ({status:?})");
        }
        let need = unsafe { result.assume_init() }.workspaceSize;
        if need > Workspace::BYTES {
            bail!(
                KernelArtifact,
                "the pinned cublasLt algorithm needs {need} bytes of workspace at {shape:?}, over {}",
                Workspace::BYTES
            );
        }
        Ok(())
    }
}

/// [`gemm_bf16_tn`] on a pinned algorithm: no heuristic, the workspace [`Workspace`]'s.
pub(crate) fn gemm_bf16_tn_pinned(
    blt: &CudaBlasLT,
    ws: &Workspace,
    stream: &Arc<CudaStream>,
    args: &[RVal],
    beta: f32,
    algo: &Pinned,
) -> Result<()> {
    let Some((a, w, c, shape)) = bf16_operands(args)? else {
        return Ok(());
    };
    let d = Described::new(shape, cublaslt::sys::cudaDataType::CUDA_R_16BF)?;
    let [wl, al, cl] = d.layouts;
    let alpha = 1.0f32;
    unsafe {
        cublaslt::result::matmul(
            *blt.handle(),
            d.desc,
            &alpha as *const _ as *const c_void,
            &beta as *const _ as *const c_void,
            w.val as *const c_void,
            wl,
            a.val as *const c_void,
            al,
            c.val as *const c_void,
            cl,
            c.val as *mut c_void,
            cl,
            &algo.0,
            ws.ws as *mut c_void,
            Workspace::BYTES,
            stream.cu_stream() as *mut _,
        )
    }
    .map_err(|e| Error::Cuda(format!("cublasLt matmul, pinned algorithm ({shape:?}): {e:?}")))
}

/// The explicit workspace owned by one stream. Launches on that stream
/// can reuse the buffer in stream order.
///
/// The 32 MiB budget preserves the f32 GEMM results measured against
/// `cublasGemmEx` on L20. A zero budget changed the heuristic's selection.
pub(crate) struct Workspace {
    _buffer: CudaSlice<u8>,
    /// The workspace's address, taken once: reading it through cudarc's
    /// `device_ptr` records a cross-stream wait that invalidates a capture.
    ws: sys::CUdeviceptr,
}

impl Workspace {
    const BYTES: usize = 32 << 20;

    pub(crate) fn new(stream: &Arc<CudaStream>) -> Result<Self> {
        let buffer: CudaSlice<u8> = stream.alloc_zeros(Self::BYTES)?;
        let ws = buffer.device_ptr(stream).0;
        Ok(Self { _buffer: buffer, ws })
    }
}

/// `extern:cublas_bf16_tn_f32`: row-major `C[m,n] = A[m,k] @ W[n,k]^T` with
/// bf16 operands, f32 accumulation and an f32 result. Takes
/// `[a, w, c, m, n, k]` and an optional output row stride.
///
/// An f32 output layout lets this launch use the stream's Lt handle and
/// avoid the classic handle that could fail during destruction in a host
/// process with mismatched cuBLAS and cuBLASLt builds.
pub(crate) fn gemm_bf16_tn_f32(
    blt: &CudaBlasLT,
    ws: &Workspace,
    stream: &Arc<CudaStream>,
    args: &[RVal],
) -> Result<()> {
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
    for dim in [m, n, k, ldc] {
        if dim > i32::MAX as u64 {
            bail!(Manifest, "gemm f32: dimension {dim} exceeds i32");
        }
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
    let shape = GemmShape { m, n, k, ldc, a_stride: k };
    let d = Described::new(shape, cublaslt::sys::cudaDataType::CUDA_R_32F)?;
    let [wl, al, cl] = d.layouts;
    let (alpha, beta) = (1.0f32, 0.0f32);
    unsafe {
        cublaslt::result::matmul(
            *blt.handle(),
            d.desc,
            &alpha as *const _ as *const c_void,
            &beta as *const _ as *const c_void,
            w.val as *const c_void,
            wl,
            a.val as *const c_void,
            al,
            c.val as *const c_void,
            cl,
            c.val as *mut c_void,
            cl,
            std::ptr::null(),
            ws.ws as *mut c_void,
            Workspace::BYTES,
            stream.cu_stream() as *mut _,
        )
    }
    .map_err(|e| Error::Cuda(format!("cublasLt matmul f32 (m={m} n={n} k={k}): {e:?}")))
}

/// How an fp8 GEMM's operands are scaled: one f32 a tensor, or an e8m0 a 32-element block of K (MXFP8).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Fp8Scale {
    Tensor,
    Block32,
}

/// `extern:cublaslt_fp8_tn` / `extern:cublaslt_fp8_tn_f32`: row-major
/// `C[m,n] = (a_scale * w_scale) * A[m,k] @ W[n,k]^T`, e4m3 operands, f32
/// accumulation, bf16 or f32 result; args `[a, w, c, a_scale, w_scale, m,
/// n, k]` plus an optional C row stride. Each scale is one f32 on the
/// device (cublasLt's per-tensor scale pointers), so a manifest quantizes
/// the activation in-graph. Same column-major mapping as the bf16 path,
/// which is exactly cublasLt's fp8 rule (A transposed, B not). The
/// workspace belongs to the stream that launches the GEMM.
///
/// `extern:cublaslt_mxfp8_tn` ([`Fp8Scale::Block32`]): the same product over
/// MXFP8 operands, bf16 result: `a_scale` / `w_scale` are e8m0 bytes, one per
/// 32 elements of a row's K, in cuBLAS's 128x4 tiled layout (FlashInfer's
/// `F8_128x4`): rows padded to 128 and K blocks to 4, tile (r / 128, b / 4) at
/// `512 (r / 128 · ⌈K/128⌉ + b / 4)`, entry (r, b) at `(r % 32) 16 + (r % 128 / 32) 4 + b % 4` in it.
pub(crate) fn gemm_fp8_tn(
    blt: &CudaBlasLT,
    ws: &Workspace,
    stream: &Arc<CudaStream>,
    args: &[RVal],
    f32_out: bool,
    scale: Fp8Scale,
) -> Result<()> {
    let (a, w, c, a_scale, w_scale, m, n, k, ldc) = match args {
        [a, w, c, sa, sw, m, n, k] => (a, w, c, sa, sw, m.val, n.val, k.val, n.val),
        [a, w, c, sa, sw, m, n, k, ldc] => (a, w, c, sa, sw, m.val, n.val, k.val, ldc.val),
        _ => bail!(Manifest, "fp8 gemm expects 8 or 9 args, got {}", args.len()),
    };
    if ldc < n {
        bail!(Manifest, "fp8 gemm: ldc {ldc} < n {n}");
    }
    let k_align = if scale == Fp8Scale::Block32 { 32 } else { 16 };
    if k % k_align != 0 {
        bail!(Manifest, "fp8 gemm: k {k} is not a multiple of {k_align}");
    }
    if m == 0 || n == 0 || k == 0 {
        return Ok(());
    }
    let sf_bytes = |rows: u64| match scale {
        Fp8Scale::Tensor => 4,
        Fp8Scale::Block32 => rows.div_ceil(128) * 128 * (k / 32).div_ceil(4) * 4,
    };
    let out = if f32_out { 4 } else { 2 };
    if a.bytes < m * k
        || w.bytes < n * k
        || c.bytes < ((m - 1) * ldc + n) * out
        || a_scale.bytes < sf_bytes(m)
        || w_scale.bytes < sf_bytes(n)
    {
        bail!(Manifest, "fp8 gemm: operands or scales too small for m={m} n={n} k={k} ldc={ldc}");
    }
    use cublaslt::result;
    use cublaslt::sys::{
        cublasComputeType_t, cublasLtMatmulDescAttributes_t as Attr, cublasLtMatmulMatrixScale_t as Mode, cudaDataType,
    };
    let e4m3 = cudaDataType::CUDA_R_8F_E4M3;
    let cd = if f32_out { cudaDataType::CUDA_R_32F } else { cudaDataType::CUDA_R_16BF };
    let lt = |e: cublaslt::result::CublasError, what: &str| {
        Error::Cuda(format!("cublasLt fp8 {what} (m={m} n={n} k={k}, {scale:?} scales): {e:?}"))
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
        if scale == Fp8Scale::Block32 {
            let mode = Mode::CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
            set(Attr::CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode as *const _ as *const c_void, 4)?;
            set(Attr::CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode as *const _ as *const c_void, 4)?;
        }
        let (sa, sw) = (a_scale.val as *const c_void, w_scale.val as *const c_void);
        set(Attr::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sw as *const _ as *const c_void, 8)?;
        set(Attr::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sa as *const _ as *const c_void, 8)?;
        let pref = result::create_matmul_pref().map_err(|e| lt(e, "preference"))?;
        let ws_size = Workspace::BYTES;
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
            ws.ws as *mut c_void,
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
