//! Load-time lowering. Everything name-shaped in a program dies here:
//! kernel steps resolve to CUDA functions, buffer/state/scratch references
//! to device addresses (static once allocated), symbol names to indices
//! into the dense env. What execution replays is a flat launch list whose
//! slots are either finished values or symbol-indexed expressions — no name
//! lookups, no wiring, and no panics left for the hot path.

use std::collections::BTreeMap;
use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use cudarc::driver::{result as cu, sys, CudaStream};
use kern_manifest::types::{Arg, Dim, Dispatch, Expr, Kernel, Manifest, ParamType, StepArg};
use sha2::Digest;

use crate::cubin::param_sizes;
use crate::device::{alloc, DeviceBuf};
use crate::error::{bail, cuda_check, Error, Result};

/// A scalar expression with symbol names resolved to indices into the dense
/// env (manifest symbol order). Division by zero is rejected at compile
/// time; overflow stays a runtime error (value-dependent).
#[derive(Clone)]
pub(crate) enum CExpr {
    Const(u64),
    Sym(usize),
    CeilDiv(Box<CExpr>, u64),
    Mul(Box<CExpr>, u64),
}

impl CExpr {
    pub(crate) fn eval(&self, env: &[u64]) -> Result<u64> {
        match self {
            CExpr::Const(c) => Ok(*c),
            CExpr::Sym(i) => Ok(env[*i]),
            CExpr::CeilDiv(e, c) => Ok(e.eval(env)?.checked_add(c - 1).ok_or_else(overflow)? / c),
            CExpr::Mul(e, c) => e.eval(env)?.checked_mul(*c).ok_or_else(overflow),
        }
    }
}

fn overflow() -> Error {
    Error::Manifest("expression eval: arithmetic overflow".into())
}

/// A launch value: the low bytes of `val` are what the param slot receives;
/// `bytes` is the remaining buffer size for pointer args (0 for scalars),
/// used by the extern gemm.
#[derive(Clone, Copy)]
pub(crate) struct RVal {
    pub(crate) val: u64,
    pub(crate) bytes: u64,
}

/// One launch parameter, lowered.
#[derive(Clone)]
pub(crate) enum Slot {
    /// Known at load time: a device pointer (base + offset) or a literal.
    Const(RVal),
    /// Symbol-dependent scalar, evaluated against the dense env per run.
    Expr(CExpr),
}

pub(crate) enum LaunchKind {
    Cubin {
        func: sys::CUfunction,
        block: [u32; 3],
        grid: [CExpr; 3],
        shared_mem: Option<CExpr>,
        /// Launch with programmatic stream serialization (see `Step::pdl`).
        pdl: bool,
    },
    /// `extern:cublaslt_bf16_tn` / `..._acc` (beta 0.0 / 1.0).
    Gemm { beta: f32 },
}

pub(crate) struct Launch {
    pub(crate) kind: LaunchKind,
    pub(crate) slots: Vec<Slot>,
    /// Error context: which dispatch and impl step this launch came from.
    pub(crate) ctx: String,
}

pub(crate) struct CompiledProgram {
    pub(crate) launches: Vec<Launch>,
    /// Launch index range `[lo, hi)` of every dispatch, in dispatch order
    /// (a multi-step impl contributes several launches).
    pub(crate) dispatch_ranges: Vec<(usize, usize)>,
}

/// One kernel implementation step, resolved against the loaded modules.
enum StepImpl {
    Cubin { func: sys::CUfunction, module: String },
    GemmBf16Tn { beta: f32 },
}

/// A kernel implementation, resolved: one entry per step, plus the private
/// scratch buffers the impl declared (allocated once at symbol max, reused
/// every dispatch — contents are dead outside a single dispatch).
pub(crate) struct ResolvedKernel {
    steps: Vec<StepImpl>,
    pub(crate) scratch: BTreeMap<String, DeviceBuf>,
}

impl ResolvedKernel {
    /// The module each step resolved to, in step order (introspection).
    pub(crate) fn step_modules(&self) -> Vec<String> {
        self.steps
            .iter()
            .map(|s| match s {
                StepImpl::Cubin { module, .. } => module.clone(),
                StepImpl::GemmBf16Tn { .. } => "runtime built-in (cublasLt)".into(),
            })
            .collect()
    }
}

/// Byte size of a shaped declaration at symbol upper bounds.
pub(crate) fn shaped_bytes(
    what: &str,
    shape: &[Dim],
    dtype_bytes: u64,
    max_env: &BTreeMap<String, u64>,
) -> Result<u64> {
    let mut elems = 1u64;
    for d in shape {
        let n = match d {
            Dim::Const(c) => *c,
            Dim::Sym(s) => *max_env
                .get(s)
                .ok_or_else(|| Error::Manifest(format!("{what}: unknown symbol `{s}` in shape")))?,
        };
        elems = elems
            .checked_mul(n)
            .ok_or_else(|| Error::Manifest(format!("{what}: size overflow")))?;
    }
    elems
        .checked_mul(dtype_bytes)
        .ok_or_else(|| Error::Manifest(format!("{what}: size overflow")))
}

/// Resolve every kernel: match each step's symbol + declared param layout
/// against the loaded modules (or an extern built-in), verify pinned cubin
/// hashes, opt into >48KB dynamic shared memory, allocate scratch.
pub(crate) fn resolve_kernels(
    manifest: &Manifest,
    modules: &[(String, sys::CUmodule)],
    remote: &BTreeMap<String, PathBuf>,
    kernels_dir: &Path,
    stream: &Arc<CudaStream>,
    max_env: &BTreeMap<String, u64>,
) -> Result<BTreeMap<String, ResolvedKernel>> {
    let mut kernels = BTreeMap::new();
    for (name, k) in &manifest.kernels {
        let mut steps = Vec::new();
        for (si, st) in k.imp.steps.iter().enumerate() {
            if let Some(ext) = st.symbol.strip_prefix("extern:") {
                match ext {
                    "cublaslt_bf16_tn" => steps.push(StepImpl::GemmBf16Tn { beta: 0.0 }),
                    "cublaslt_bf16_tn_acc" => steps.push(StepImpl::GemmBf16Tn { beta: 1.0 }),
                    _ => bail!(Manifest, "kernel `{name}` step #{si}: unsupported extern op `{ext}`"),
                }
                continue;
            }
            // Pinned-artifact integrity: when the step names its cubin
            // (the pluggable path), verify the file hash if declared.
            if let (Some(cb), Some(sha)) = (&st.cubin, &st.sha256) {
                let path = match remote.get(cb.as_str()) {
                    Some(p) => p.clone(),
                    None => kernels_dir.join(cb),
                };
                let data = std::fs::read(&path).map_err(|e| {
                    Error::KernelArtifact(format!(
                        "kernel `{name}` step #{si}: reading {}: {e}",
                        path.display()
                    ))
                })?;
                let got = format!("{:x}", sha2::Sha256::digest(&data));
                if got != sha.to_lowercase() {
                    bail!(
                        KernelArtifact,
                        "kernel `{name}` step #{si}: cubin `{cb}` sha256 mismatch: \
                         manifest declares {sha}, file is {got}"
                    );
                }
            }
            let want: Vec<usize> = st.params.iter().map(|p| p.size_bytes() as usize).collect();
            let sym = CString::new(st.symbol.as_str())
                .map_err(|e| Error::Manifest(format!("kernel `{name}` symbol: {e}")))?;
            let mut resolved = None;
            let mut seen = Vec::new();
            for (file, cmod) in modules {
                if let Some(cb) = &st.cubin {
                    if file != cb {
                        continue;
                    }
                }
                let Ok(func) = (unsafe { cu::module::get_function(*cmod, sym.clone()) }) else {
                    continue;
                };
                let got = param_sizes(func)?;
                if got == want {
                    resolved = Some(StepImpl::Cubin { func, module: file.clone() });
                    break;
                }
                seen.push(format!("{file}: {got:?}"));
            }
            let Some(r) = resolved else {
                bail!(
                    KernelArtifact,
                    "kernel `{name}` step #{si} ({}): no loaded instance matches declared \
                     param layout {want:?} (cubin filter {:?}); saw {seen:?}",
                    st.symbol,
                    st.cubin
                );
            };
            // Opt in to >48KB dynamic shared memory where the step needs it.
            if let (StepImpl::Cubin { func, .. }, Some(sm)) = (&r, &st.shared_mem) {
                let bytes = sm
                    .eval(max_env)
                    .map_err(|e| Error::Manifest(format!("kernel `{name}`: {e}")))?;
                if bytes > 48 * 1024 {
                    cuda_check(
                        unsafe {
                            sys::cuFuncSetAttribute(
                                *func,
                                sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                bytes as i32,
                            )
                        },
                        "cuFuncSetAttribute",
                    )?;
                }
            }
            steps.push(r);
        }
        let mut scratch = BTreeMap::new();
        for (sname, sd) in &k.imp.scratch {
            let bytes = shaped_bytes(
                &format!("kernel `{name}` scratch `{sname}`"),
                &sd.shape,
                sd.dtype.bytes(),
                max_env,
            )?;
            scratch.insert(sname.clone(), alloc(stream, bytes)?);
        }
        kernels.insert(name.clone(), ResolvedKernel { steps, scratch });
    }
    Ok(kernels)
}

/// Lower every program's dispatch list into a flat launch list.
pub(crate) fn compile_programs(
    manifest: &Manifest,
    kernels: &BTreeMap<String, ResolvedKernel>,
    buffers: &BTreeMap<String, DeviceBuf>,
    states: &BTreeMap<String, DeviceBuf>,
) -> Result<BTreeMap<String, CompiledProgram>> {
    let syms: BTreeMap<&str, usize> =
        manifest.symbols.keys().enumerate().map(|(i, s)| (s.as_str(), i)).collect();
    let mut programs = BTreeMap::new();
    for (pname, prog) in &manifest.programs {
        let mut launches = Vec::new();
        let mut dispatch_ranges = Vec::with_capacity(prog.dispatches.len());
        for (di, d) in prog.dispatches.iter().enumerate() {
            let dctx = dispatch_ctx(di, d);
            let (Some(k), Some(rk)) = (manifest.kernels.get(&d.kernel), kernels.get(&d.kernel))
            else {
                bail!(Manifest, "program `{pname}` {dctx}: unknown kernel");
            };
            let lo = launches.len();
            compile_dispatch(d, k, rk, &dctx, buffers, states, &syms, &mut launches).map_err(
                |e| Error::Dispatch {
                    context: format!("program `{pname}` {dctx}"),
                    source: Box::new(e),
                },
            )?;
            dispatch_ranges.push((lo, launches.len()));
        }
        programs.insert(pname.clone(), CompiledProgram { launches, dispatch_ranges });
    }
    Ok(programs)
}

/// Error context locating one entry of a program's dispatch list.
fn dispatch_ctx(i: usize, d: &Dispatch) -> String {
    match &d.label {
        Some(l) => format!("dispatch #{i} `{l}` (kernel `{}`)", d.kernel),
        None => format!("dispatch #{i} (kernel `{}`)", d.kernel),
    }
}

#[allow(clippy::too_many_arguments)]
fn compile_dispatch(
    d: &Dispatch,
    k: &Kernel,
    rk: &ResolvedKernel,
    dctx: &str,
    buffers: &BTreeMap<String, DeviceBuf>,
    states: &BTreeMap<String, DeviceBuf>,
    syms: &BTreeMap<&str, usize>,
    launches: &mut Vec<Launch>,
) -> Result<()> {
    if d.args.len() != k.params.len() {
        bail!(Manifest, "kernel takes {} args, dispatch passes {}", k.params.len(), d.args.len());
    }
    // Lower the interface args once; each step then wires its own launch
    // params from these, its scratch, and its private literals.
    let mut vals = Vec::with_capacity(d.args.len());
    for (arg, pty) in d.args.iter().zip(&k.params) {
        vals.push(match pty {
            ParamType::Buf { .. } | ParamType::Ptr { .. } => {
                Slot::Const(pointer_arg(arg, buffers, states)?)
            }
            ParamType::Scalar(_) => scalar_arg(arg, syms)?,
        });
    }
    for (si, (st, imp)) in k.imp.steps.iter().zip(&rk.steps).enumerate() {
        let mut slots = Vec::with_capacity(st.args.len());
        for sa in &st.args {
            slots.push(match sa {
                StepArg::Arg { arg } => vals.get(*arg).cloned().ok_or_else(|| {
                    Error::Manifest(format!("step #{si}: forwarded arg #{arg} out of range"))
                })?,
                StepArg::Scratch { scratch, offset } => {
                    let Some(b) = rk.scratch.get(scratch) else {
                        bail!(Manifest, "step #{si}: unknown scratch `{scratch}`");
                    };
                    Slot::Const(offset_into(b, *offset, || format!("scratch `{scratch}`"))?)
                }
                StepArg::I32 { i32: v } => lit(*v as u32 as u64),
                StepArg::U32 { u32: v } => lit(*v as u64),
                StepArg::I64 { i64: v } => lit(*v as u64),
                StepArg::U8 { u8: v } => lit(*v as u64),
                StepArg::F32 { f32: v } => lit(v.to_bits() as u64),
            });
        }
        let kind = match imp {
            StepImpl::GemmBf16Tn { beta } => {
                if slots.len() != 6 {
                    bail!(Manifest, "step #{si}: extern gemm takes 6 args, got {}", slots.len());
                }
                LaunchKind::Gemm { beta: *beta }
            }
            StepImpl::Cubin { func, .. } => LaunchKind::Cubin {
                func: *func,
                block: st.block,
                grid: [
                    compile_expr(&st.grid[0], syms)?,
                    compile_expr(&st.grid[1], syms)?,
                    compile_expr(&st.grid[2], syms)?,
                ],
                shared_mem: st.shared_mem.as_ref().map(|e| compile_expr(e, syms)).transpose()?,
                pdl: st.pdl,
            },
        };
        launches.push(Launch {
            kind,
            slots,
            ctx: format!("{dctx} step #{si} (`{}`)", st.symbol),
        });
    }
    Ok(())
}

/// Lower a buffer/state arg to its finished pointer value.
fn pointer_arg(
    arg: &Arg,
    buffers: &BTreeMap<String, DeviceBuf>,
    states: &BTreeMap<String, DeviceBuf>,
) -> Result<RVal> {
    let (map, name, offset, what) = match arg {
        Arg::Buf { buf, offset } => (buffers, buf, *offset, "buffer"),
        Arg::State { state, offset } => (states, state, *offset, "state"),
        _ => bail!(Manifest, "expected buffer/state arg, got {arg}"),
    };
    let Some(b) = map.get(name) else {
        bail!(Manifest, "unknown {what} `{name}`");
    };
    offset_into(b, offset, || format!("{what} `{name}`"))
}

fn offset_into(b: &DeviceBuf, offset: u64, what: impl Fn() -> String) -> Result<RVal> {
    let Some(bytes) = b.bytes.checked_sub(offset) else {
        bail!(Manifest, "offset {offset} outside {} ({} bytes)", what(), b.bytes);
    };
    Ok(RVal { val: b.ptr + offset, bytes })
}

/// Lower a scalar arg: literals finish now, symbols and expressions become
/// dense-indexed expressions.
fn scalar_arg(arg: &Arg, syms: &BTreeMap<&str, usize>) -> Result<Slot> {
    Ok(match arg {
        Arg::Sym { sym } => Slot::Expr(CExpr::Sym(sym_index(syms, sym)?)),
        Arg::Expr { expr } => Slot::Expr(compile_expr(expr, syms)?),
        Arg::I32 { i32: v } => lit(*v as u32 as u64),
        Arg::U32 { u32: v } => lit(*v as u64),
        Arg::I64 { i64: v } => lit(*v as u64),
        Arg::U8 { u8: v } => lit(*v as u64),
        Arg::F32 { f32: v } => lit(v.to_bits() as u64),
        Arg::Buf { .. } | Arg::State { .. } => bail!(Manifest, "expected scalar arg, got {arg}"),
    })
}

fn lit(val: u64) -> Slot {
    Slot::Const(RVal { val, bytes: 0 })
}

fn compile_expr(e: &Expr, syms: &BTreeMap<&str, usize>) -> Result<CExpr> {
    Ok(match e {
        Expr::Const(c) => CExpr::Const(*c),
        Expr::Sym { sym } => CExpr::Sym(sym_index(syms, sym)?),
        Expr::CeilDiv { ceil_div: (inner, c) } => {
            if *c == 0 {
                bail!(Manifest, "expression: division by zero");
            }
            CExpr::CeilDiv(Box::new(compile_expr(inner, syms)?), *c)
        }
        Expr::Mul { mul: (inner, c) } => CExpr::Mul(Box::new(compile_expr(inner, syms)?), *c),
    })
}

fn sym_index(syms: &BTreeMap<&str, usize>, sym: &str) -> Result<usize> {
    match syms.get(sym) {
        Some(&i) => Ok(i),
        None => bail!(Manifest, "unknown symbol `{sym}`"),
    }
}
