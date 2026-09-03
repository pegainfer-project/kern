//! Load-time verification: refuse any manifest that is not provably
//! self-consistent, before anything touches the GPU. All errors are
//! collected and reported together, rustc-style.
//!
//! Checks:
//!   1. schema_version
//!   2. vars: max > 0
//!   3. states: exactly one of bytes_per_token / bytes / bytes_per_seq is non-zero
//!   4. buffers: shapes resolve, byte sizes don't overflow at var upper
//!      bounds; a declared domain is well-formed (bound kinds vs dtype,
//!      `index_into` resolves, min <= max at the var corners)
//!   5. modules: sha256 shape, source well-formed (registry refs parse)
//!   6. ops (interface + implementation):
//!      - scratch shapes resolve, every scratch is used
//!      - per launch: module ref resolves; geometry present iff not
//!        `extern:`; CUDA block/grid limits at var upper bounds (and grid
//!        non-zero at lower bounds), shared-mem opt-in cap, arg/param arity,
//!        per-position wiring types (a forwarded interface param must match
//!        kind/dtype, a launch may not write through an interface `in`
//!        param, literal types)
//!      - impl dataflow: scratch never read before a launch wrote it; every
//!        interface `out` param written by some launch
//!   7. calls: op refs resolve, arg/param arity and per-position type match
//!      against the interface, var ranges fit scalar params
//!   8. dataflow per program: no read-before-write, no writes to input,
//!      weight or peer buffers, every output / carry buffer written by some
//!      program; a program is `once` or has a `batch`, not both; a batch
//!      has `groups >= 1` and constant `rows >= 1` or a declared var
//!   9. no unused declarations; a `fill` sits on an input or output of an
//!      integer dtype, input roles on inputs and output roles on outputs
//!      (whether the fills add up to a serving contract is
//!      [`crate::protocol`]'s question, not this one's)
//!  10. topology: group sizes > 0, every group used; a `peer` buffer is
//!      `u64[group size]` `of` an exported buffer or a state; `export`,
//!      `of` and `group` only where they mean something; `{"rank": g}`
//!      binds only to i32/i64 params; an op with an extern launch never
//!      receives a peer buffer (runtime built-ins may not touch peer memory)
//!
//! What this deliberately cannot check: kernel *behavior*, and the
//! *semantics* of interface params (that a replacement implementation
//! interprets position #3 as the same row stride). A cubin that lies about
//! what it touches is inside the trust boundary; the manifest only makes
//! the lie explicit and diffable. Cross-checking launch param layouts against
//! `cuFuncGetParamInfo` is a load-time (phase 2) concern in the runtime
//! crate, since it needs the CUDA driver.

use crate::types::*;
use std::collections::{BTreeMap, BTreeSet};

const MAX_GRID_X: u64 = (1 << 31) - 1;
const MAX_GRID_YZ: u64 = 65_535;
const MAX_BLOCK_THREADS: u64 = 1024;
const MAX_BLOCK_Z: u32 = 64;
/// Per-block dynamic shared memory after the `cuFuncSetAttribute` opt-in the
/// runtime performs for any launch declaring `shared_mem`: 227 KiB on
/// sm90/sm100/sm103 datacenter parts.
const MAX_DYN_SHARED_MEM: u64 = 232_448;
/// Blocks per thread-block cluster (non-portable maximum on sm_90+).
const MAX_CLUSTER_BLOCKS: u64 = 16;

/// Sized like buffers: dtype bytes x dims at var upper bounds.
fn shaped_size(
    what: &str,
    dtype: DType,
    shape: &[Dim],
    env_max: &BTreeMap<String, u64>,
    used_vars: &mut BTreeSet<String>,
    errs: &mut Vec<String>,
) -> Option<u64> {
    if shape.is_empty() {
        errs.push(format!("{what}: shape must not be empty"));
        return None;
    }
    let mut dim_err = false;
    let mut size: Option<u64> = Some(dtype.bytes());
    for dim in shape {
        let extent = match dim {
            Dim::Const(0) => {
                errs.push(format!("{what}: zero-sized dimension"));
                dim_err = true;
                None
            }
            Dim::Const(c) => Some(*c),
            Dim::Var(s) => match env_max.get(s) {
                Some(mx) => {
                    used_vars.insert(s.clone());
                    Some(*mx)
                }
                None => {
                    errs.push(format!("{what}: unknown var `{s}` in shape"));
                    dim_err = true;
                    None
                }
            },
        };
        size = match (size, extent) {
            (Some(a), Some(e)) => a.checked_mul(e),
            _ => None,
        };
    }
    if size.is_none() && !dim_err {
        errs.push(format!("{what}: byte size overflows u64 at var upper bounds"));
    }
    size
}

/// A manifest that passed [`verify`]: every reference resolves, every
/// launch fits the device limits at the var bounds, every program's
/// dataflow is sound. Constructed by `verify` alone; a runtime loads only
/// one of these, so nothing downstream checks again. Derefs to the
/// [`Manifest`].
#[derive(Debug, Clone)]
pub struct Verified(Manifest);

impl Verified {
    /// Parse and verify; a parse error is reported as the one diagnostic.
    pub fn from_json(s: &str) -> Result<Verified, VerifyErrors> {
        let m = Manifest::from_json(s).map_err(|e| VerifyErrors(vec![e.to_string()]))?;
        verify(m)
    }

    pub fn into_inner(self) -> Manifest {
        self.0
    }
}

impl std::ops::Deref for Verified {
    type Target = Manifest;
    fn deref(&self) -> &Manifest {
        &self.0
    }
}

/// Every diagnostic [`verify`] collected, reported together rustc-style.
/// Derefs to the individual messages.
#[derive(Debug)]
pub struct VerifyErrors(pub Vec<String>);

impl std::fmt::Display for VerifyErrors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("manifest failed verification:")?;
        for e in &self.0 {
            write!(f, "\n  - {e}")?;
        }
        Ok(())
    }
}

impl std::error::Error for VerifyErrors {}

impl std::ops::Deref for VerifyErrors {
    type Target = [String];
    fn deref(&self) -> &[String] {
        &self.0
    }
}

/// Verify `m`; the manifest comes back as a [`Verified`] or not at all.
pub fn verify(m: Manifest) -> Result<Verified, VerifyErrors> {
    let errs = diagnostics(&m);
    if errs.is_empty() {
        Ok(Verified(m))
    } else {
        Err(VerifyErrors(errs))
    }
}

/// Every rule, every violation.
fn diagnostics(m: &Manifest) -> Vec<String> {
    let mut errs: Vec<String> = Vec::new();
    let mut used_vars: BTreeSet<String> = BTreeSet::new();
    let mut used_buffers: BTreeSet<String> = BTreeSet::new();
    let mut used_states: BTreeSet<String> = BTreeSet::new();
    let mut used_modules: BTreeSet<String> = BTreeSet::new();
    let mut used_ops: BTreeSet<String> = BTreeSet::new();
    let mut used_groups: BTreeSet<String> = BTreeSet::new();

    // 1. format
    if m.schema_version != SCHEMA_VERSION {
        errs.push(format!("unsupported schema_version {} (this runtime reads {SCHEMA_VERSION})", m.schema_version));
    }

    // 2. vars
    for (name, v) in &m.vars {
        if v.max < Var::MIN {
            errs.push(format!("var `{name}`: max must be >= {}", Var::MIN));
        }
    }
    let env_max: BTreeMap<String, u64> = m.vars.iter().map(|(k, v)| (k.clone(), v.max)).collect();
    let env_min: BTreeMap<String, u64> = m.vars.keys().map(|k| (k.clone(), Var::MIN)).collect();

    // 10a. topology
    if let Some(t) = &m.topology {
        for (name, size) in &t.groups {
            if *size == 0 {
                errs.push(format!("topology group `{name}`: size must be > 0"));
            }
        }
    }
    let group_ctx = |g: &str, errs: &mut Vec<String>, used_groups: &mut BTreeSet<String>, ctx: &str| -> Option<u64> {
        match m.group_size(g) {
            Some(sz) => {
                used_groups.insert(g.to_string());
                Some(sz)
            }
            None => {
                errs.push(match &m.topology {
                    Some(_) => format!("{ctx}: unknown topology group `{g}`"),
                    None => format!("{ctx}: group `{g}` but the manifest declares no topology"),
                });
                None
            }
        }
    };

    // 3. states
    for (name, st) in &m.states {
        let set = [st.bytes_per_token, st.bytes, st.bytes_per_seq].iter().filter(|&&b| b > 0).count();
        match set {
            0 => errs.push(format!("state `{name}`: one of bytes_per_token / bytes / bytes_per_seq must be > 0")),
            1 => {}
            _ => errs.push(format!("state `{name}`: bytes_per_token, bytes and bytes_per_seq are exclusive")),
        }
    }

    // 4. buffers
    let mut buf_sizes: BTreeMap<&str, u64> = BTreeMap::new();
    for (name, b) in &m.buffers {
        if let Some(sz) =
            shaped_size(&format!("buffer `{name}`"), b.dtype, &b.shape, &env_max, &mut used_vars, &mut errs)
        {
            buf_sizes.insert(name, sz);
        }
        if let Some(d) = &b.domain {
            check_domain(name, b, d, m, &env_max, &env_min, &mut used_vars, &mut errs);
        }
        let ctx = format!("buffer `{name}`");
        // 9b. fill
        if let Some(fill) = b.fill {
            let output = matches!(fill, Fill::Tokens | Fill::Count | Fill::Error);
            match (b.kind, output) {
                (BufferKind::Input, false) | (BufferKind::Output, true) => {}
                (BufferKind::Input | BufferKind::Output, _) => errs.push(format!(
                    "{ctx}: fill `{fill}` is {} but the buffer is {}",
                    if output { "read from an output" } else { "written into an input" },
                    b.kind
                )),
                (kind, _) => errs.push(format!("{ctx}: a fill sits on an input or output, not a {kind} buffer")),
            }
            if !matches!(b.dtype, DType::I32 | DType::I64) {
                errs.push(format!("{ctx}: fill `{fill}` needs an i32 or i64 buffer, not {}", b.dtype));
            }
        }
        // 10b. export / peer
        if b.kind == BufferKind::Peer {
            if b.export {
                errs.push(format!("{ctx}: a peer buffer cannot itself be exported"));
            }
            if b.dtype != DType::U64 {
                errs.push(format!("{ctx}: a peer buffer holds device addresses, dtype must be u64, not {}", b.dtype));
            }
            if b.domain.is_some() {
                errs.push(format!("{ctx}: a peer buffer's contents are runtime-filled addresses; it takes no domain"));
            }
            match &b.of {
                None => errs.push(format!(
                    "{ctx}: a peer buffer must name the exported buffer or state it holds addresses `of`"
                )),
                Some(of) if of == name => errs.push(format!("{ctx}: a peer buffer cannot be `of` itself")),
                Some(of) => match (m.buffers.get(of), m.states.get(of)) {
                    (Some(_), Some(_)) => errs.push(format!("{ctx}: `of` `{of}` is both a buffer and a state")),
                    (Some(target), None) => {
                        used_buffers.insert(of.clone());
                        if target.kind == BufferKind::Peer {
                            errs.push(format!("{ctx}: `of` `{of}` is itself a peer buffer"));
                        } else if !target.export {
                            errs.push(format!("{ctx}: `of` buffer `{of}` is not exported"));
                        }
                    }
                    (None, Some(_)) => {
                        used_states.insert(of.clone());
                    }
                    (None, None) => errs.push(format!("{ctx}: `of` unknown buffer/state `{of}`")),
                },
            }
            match &b.group {
                None => errs
                    .push(format!("{ctx}: a peer buffer must name the topology `group` its addresses are indexed by")),
                Some(g) => {
                    if let Some(sz) = group_ctx(g, &mut errs, &mut used_groups, &ctx) {
                        match b.shape.as_slice() {
                            [Dim::Const(n)] if *n == sz => {}
                            _ => errs.push(format!(
                                "{ctx}: a peer buffer over group `{g}` has shape [{sz}], one address per member"
                            )),
                        }
                    }
                }
            }
        } else {
            if b.of.is_some() {
                errs.push(format!("{ctx}: `of` only applies to peer buffers"));
            }
            if b.group.is_some() {
                errs.push(format!("{ctx}: `group` only applies to peer buffers"));
            }
        }
    }

    // 5. modules
    for (name, md) in &m.modules {
        let ctx = format!("module `{name}`");
        if md.sha256.len() != 64 || !md.sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
            errs.push(format!("{ctx}: sha256 `{}` is not 64 hex chars", md.sha256));
        }
        if md.source.is_empty() {
            errs.push(format!("{ctx}: empty source"));
        }
        if let Some(Err(e)) = RegistryRef::parse(&md.source) {
            errs.push(format!("{ctx}: {e}"));
        }
    }

    // 6. ops: interface + implementation
    // Per op: (interface param, footprint bytes, context) of every tensormap
    // launch arg, checked against the bound buffer at each call (rule 7).
    let mut op_tensormaps: BTreeMap<&str, Vec<(usize, u64, String)>> = BTreeMap::new();
    for (oname, op) in &m.ops {
        let imp = &op.imp;

        for (i, p) in op.params.iter().enumerate() {
            if *p == ParamType::TensorMap {
                errs.push(format!(
                    "op `{oname}`: interface param #{i} is a tensormap; tensormaps are launch-private, \
                     the interface takes the buffer they describe"
                ));
            }
            if matches!(p, ParamType::Bytes(_)) {
                errs.push(format!(
                    "op `{oname}`: interface param #{i} is a byte aggregate; packs are launch-private, \
                     the interface takes the buffers and scalars they are assembled from"
                ));
            }
        }

        for (sname, s) in &imp.scratch {
            shaped_size(
                &format!("op `{oname}` scratch `{sname}`"),
                s.dtype,
                &s.shape,
                &env_max,
                &mut used_vars,
                &mut errs,
            );
        }

        if imp.launches.is_empty() {
            errs.push(format!("op `{oname}`: implementation has no launches"));
        }

        // Impl-level dataflow over slots: interface `out` params and scratch
        // start unwritten; `in`/`inout` interface params are caller-provided.
        let mut iface_written: Vec<bool> = op.params.iter().map(|p| p.dir() != Some(Dir::Out)).collect();
        let mut scratch_written: BTreeSet<&str> = BTreeSet::new();
        let mut scratch_used: BTreeSet<&str> = BTreeSet::new();

        for (li, launch) in imp.launches.iter().enumerate() {
            let ctx = format!("op `{oname}` launch #{li} ({})", launch.entry());
            match launch {
                Launch::Extern(e) => {
                    if !e.entry.starts_with("extern:") {
                        errs.push(format!(
                            "{ctx}: a launch without a module must be a runtime built-in (`extern:<name>`)"
                        ));
                    }
                }
                Launch::Kernel(k) => {
                    if k.entry.is_empty() {
                        errs.push(format!("{ctx}: empty entry"));
                    }
                    if k.entry.starts_with("extern:") {
                        errs.push(format!("{ctx}: an extern entry has no module or launch geometry"));
                    }
                    if m.modules.contains_key(&k.module) {
                        used_modules.insert(k.module.clone());
                    } else {
                        errs.push(format!("{ctx}: unknown module `{}`", k.module));
                    }
                    let block = &k.block;
                    let threads: u64 = block.iter().map(|&x| x as u64).product();
                    if block.contains(&0) || threads > MAX_BLOCK_THREADS {
                        errs.push(format!(
                            "{ctx}: block {block:?} exceeds {MAX_BLOCK_THREADS} threads or has a zero dim"
                        ));
                    }
                    if block[2] > MAX_BLOCK_Z {
                        errs.push(format!("{ctx}: block.z {} > {MAX_BLOCK_Z}", block[2]));
                    }
                    for (axis, e) in ["x", "y", "z"].iter().zip(&k.grid) {
                        let ectx = format!("{ctx}: grid.{axis}");
                        check_expr(e, m, &mut used_vars, &mut errs, &ectx);
                        match e.eval(&env_max) {
                            Ok(v) => {
                                let limit = if *axis == "x" { MAX_GRID_X } else { MAX_GRID_YZ };
                                if v > limit {
                                    errs.push(format!("{ectx}: {v} exceeds CUDA limit {limit} at var upper bounds"));
                                }
                            }
                            Err(err) => errs.push(format!("{ectx}: {err}")),
                        }
                        if let Ok(0) = e.eval(&env_min) {
                            errs.push(format!("{ectx}: evaluates to 0 at var lower bounds"));
                        }
                    }
                    if let Some(e) = &k.shared_mem {
                        let ectx = format!("{ctx}: shared_mem");
                        check_expr(e, m, &mut used_vars, &mut errs, &ectx);
                        if let Ok(v) = e.eval(&env_max) {
                            if v > MAX_DYN_SHARED_MEM {
                                errs.push(format!(
                                    "{ectx}: {v} bytes exceeds opt-in limit {MAX_DYN_SHARED_MEM} at var upper bounds"
                                ));
                            }
                        }
                    }
                    if let Some(cl) = &k.cluster {
                        let blocks: u64 = cl.iter().map(|&x| x as u64).product();
                        if cl.contains(&0) || blocks > MAX_CLUSTER_BLOCKS {
                            errs.push(format!(
                                "{ctx}: cluster {cl:?} has a zero dim or more than {MAX_CLUSTER_BLOCKS} blocks"
                            ));
                        } else {
                            for (axis, (e, &c)) in ["x", "y", "z"].iter().zip(k.grid.iter().zip(cl)) {
                                for (env, at) in [(&env_max, "upper"), (&env_min, "lower")] {
                                    if let Ok(v) = e.eval(env) {
                                        if v % c as u64 != 0 {
                                            errs.push(format!(
                                                "{ctx}: grid.{axis} = {v} at var {at} bounds is not a multiple of cluster.{axis} = {c}"
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if launch.is_extern()
                && launch.params_of(op).iter().any(|p| matches!(p, ParamType::TensorMap | ParamType::Bytes(_)))
            {
                errs.push(format!("{ctx}: an extern launch takes pointers and scalars, not a tensormap or a pack"));
            }

            let params = launch.params_of(op);
            let args = launch.args_of(op);
            if args.len() != params.len() {
                errs.push(format!("{ctx}: takes {} params, got {} args", params.len(), args.len()));
                continue;
            }
            for (j, (arg, param)) in args.iter().zip(params).enumerate() {
                let actx = format!("{ctx}: arg #{j}");
                match arg {
                    LaunchArg::Param { param: i } => {
                        let Some(iface) = op.params.get(*i) else {
                            errs.push(format!(
                                "{actx}: interface param #{i} out of range ({} interface params)",
                                op.params.len()
                            ));
                            continue;
                        };
                        // Kind and dtype must match; a launch may not write
                        // through an interface param declared `in`.
                        let compatible = match (iface, param) {
                            (ParamType::Buf { dtype: a, .. }, ParamType::Buf { dtype: b, .. }) => a == b,
                            (ParamType::State { .. }, ParamType::State { .. }) => true,
                            (ParamType::Scalar(a), ParamType::Scalar(b)) => a == b,
                            _ => false,
                        };
                        if !compatible {
                            errs.push(format!(
                                "{actx}: interface param `{iface}` does not match launch param `{param}`"
                            ));
                            continue;
                        }
                        if let (Some(idir), Some(ldir)) = (iface.dir(), param.dir()) {
                            if matches!(ldir, Dir::Out | Dir::InOut) && idir == Dir::In {
                                errs.push(format!("{actx}: launch writes through interface `in` param #{i}"));
                            }
                            if matches!(ldir, Dir::In | Dir::InOut) && !iface_written[*i] {
                                errs.push(format!(
                                    "{actx}: interface `out` param #{i} read before any launch wrote it"
                                ));
                            }
                            if matches!(ldir, Dir::Out | Dir::InOut) {
                                iface_written[*i] = true;
                            }
                        }
                    }
                    LaunchArg::Scratch { scratch } => {
                        // Borrow the declaration's key: the wiring may be a
                        // defaulted (owned) list that dies with this launch.
                        let Some((scratch, sdecl)) = imp.scratch.get_key_value(scratch) else {
                            errs.push(format!("{actx}: unknown scratch `{scratch}`"));
                            continue;
                        };
                        scratch_used.insert(scratch.as_str());
                        let ParamType::Buf { dtype, dir } = param else {
                            errs.push(format!("{actx}: scratch `{scratch}` bound to non-buffer param `{param}`"));
                            continue;
                        };
                        if sdecl.dtype != *dtype {
                            errs.push(format!(
                                "{actx}: scratch `{scratch}` has dtype {} but param expects {}",
                                sdecl.dtype, dtype
                            ));
                        }
                        if matches!(dir, Dir::In | Dir::InOut) && !scratch_written.contains(scratch.as_str()) {
                            errs.push(format!("{actx}: scratch `{scratch}` is read before any launch wrote it"));
                        }
                        if matches!(dir, Dir::Out | Dir::InOut) {
                            scratch_written.insert(scratch.as_str());
                        }
                    }
                    LaunchArg::I32 { .. } if matches!(param, ParamType::Scalar(ScalarType::I32)) => {}
                    LaunchArg::I64 { .. } if matches!(param, ParamType::Scalar(ScalarType::I64)) => {}
                    LaunchArg::F32 { .. } if matches!(param, ParamType::Scalar(ScalarType::F32)) => {}
                    LaunchArg::U8 { .. } if matches!(param, ParamType::Scalar(ScalarType::U8)) => {}
                    LaunchArg::Rank { rank } => {
                        if !matches!(param, ParamType::Scalar(ScalarType::I32 | ScalarType::I64)) {
                            errs.push(format!("{actx}: a rank binds only to an i32 or i64 param, not `{param}`"));
                        }
                        group_ctx(rank, &mut errs, &mut used_groups, &actx);
                    }
                    LaunchArg::TensorMap { tensormap: t } => {
                        if *param != ParamType::TensorMap {
                            errs.push(format!("{actx}: a tensormap binds only to a `tensormap` param, not `{param}`"));
                        }
                        let i = t.param;
                        let Some(iface) = op.params.get(i) else {
                            errs.push(format!(
                                "{actx}: interface param #{i} out of range ({} interface params)",
                                op.params.len()
                            ));
                            continue;
                        };
                        let (ParamType::Buf { dir, .. } | ParamType::State { dir }) = iface else {
                            errs.push(format!(
                                "{actx}: tensormap over interface param #{i} (`{iface}`), which is not a buffer or state"
                            ));
                            continue;
                        };
                        for e in t.check() {
                            errs.push(format!("{actx}: {e}"));
                        }
                        // The descriptor reads or writes per the interface
                        // direction; the kernel is trusted not to store
                        // through an `in` buffer's descriptor.
                        if matches!(dir, Dir::Out | Dir::InOut) {
                            iface_written[i] = true;
                        }
                        if let Some(fp) = t.footprint() {
                            op_tensormaps.entry(oname.as_str()).or_default().push((i, fp, actx.clone()));
                        }
                    }
                    LaunchArg::Pack { pack } => {
                        match param {
                            ParamType::Bytes(n) if *n == pack.size => {}
                            ParamType::Bytes(n) => {
                                errs.push(format!("{actx}: pack of {} bytes bound to a `bytes<{n}>` param", pack.size))
                            }
                            _ => errs.push(format!("{actx}: a pack binds only to a `bytes<n>` param, not `{param}`")),
                        }
                        for e in pack.check(|i| op.params.get(i).map(|p| p.size_bytes() as u32)) {
                            errs.push(format!("{actx}: {e}"));
                        }
                        for (k, f) in pack.fields.iter().enumerate() {
                            let fctx = format!("{actx}: field #{k}");
                            match &f.src {
                                FieldSrc::Param { param: i } => {
                                    let Some(iface) = op.params.get(*i) else {
                                        errs.push(format!(
                                            "{fctx}: interface param #{i} out of range ({} interface params)",
                                            op.params.len()
                                        ));
                                        continue;
                                    };
                                    // A pointer field is read or written per
                                    // the interface direction; the kernel is
                                    // trusted like a tensormap's.
                                    if matches!(iface.dir(), Some(Dir::Out | Dir::InOut)) {
                                        iface_written[*i] = true;
                                    }
                                }
                                FieldSrc::Scratch { scratch } => {
                                    let Some((scratch, _)) = imp.scratch.get_key_value(scratch) else {
                                        errs.push(format!("{fctx}: unknown scratch `{scratch}`"));
                                        continue;
                                    };
                                    scratch_used.insert(scratch.as_str());
                                    scratch_written.insert(scratch.as_str());
                                }
                                FieldSrc::Var { var } => {
                                    if m.vars.contains_key(var) {
                                        used_vars.insert(var.clone());
                                    } else {
                                        errs.push(format!("{fctx}: unknown var `{var}`"));
                                    }
                                }
                                FieldSrc::Expr { expr } => check_expr(expr, m, &mut used_vars, &mut errs, &fctx),
                                FieldSrc::Rank { rank } => {
                                    group_ctx(rank, &mut errs, &mut used_groups, &fctx);
                                }
                                FieldSrc::I32 { .. }
                                | FieldSrc::I64 { .. }
                                | FieldSrc::F32 { .. }
                                | FieldSrc::U8 { .. } => {}
                            }
                        }
                    }
                    arg => {
                        errs.push(format!("{actx}: {arg} does not match launch param `{param}`"));
                    }
                }
            }
        }

        for (i, (p, written)) in op.params.iter().zip(&iface_written).enumerate() {
            if !written {
                errs.push(format!("op `{oname}`: interface `{p}` param #{i} is never written by any launch"));
            }
        }
        for sname in imp.scratch.keys() {
            if !scratch_used.contains(sname.as_str()) {
                errs.push(format!("op `{oname}`: scratch `{sname}` is never used"));
            }
        }
    }

    // 7 + 8. programs
    if m.programs.is_empty() {
        errs.push("no programs declared".to_string());
    }
    let initially_written: BTreeSet<String> = m
        .buffers
        .iter()
        .filter(|(_, b)| {
            // Carry buffers hold another program's output; whether that
            // program ran first is the caller's sequencing contract, so
            // per-program dataflow treats them as initially written.
            // Peer buffers are filled by the runtime when the group's
            // handles are imported, before any program runs.
            matches!(b.kind, BufferKind::Input | BufferKind::Weight | BufferKind::Carry | BufferKind::Peer)
        })
        .map(|(n, _)| n.clone())
        .collect();
    let mut actually_written: BTreeSet<String> = BTreeSet::new();

    for (pname, p) in &m.programs {
        if p.once && p.batch.is_some() {
            errs.push(format!(
                "program `{pname}`: `once` (run after load) and `batch` (driven per step) are exclusive"
            ));
        }
        if let Some(batch) = &p.batch {
            if batch.groups == 0 {
                errs.push(format!("program `{pname}`: batch.groups must be >= 1"));
            }
            match &batch.rows {
                Dim::Const(0) => errs.push(format!("program `{pname}`: batch.rows must be >= 1")),
                Dim::Const(_) => {}
                Dim::Var(v) => {
                    if m.vars.contains_key(v) {
                        used_vars.insert(v.clone());
                    } else {
                        errs.push(format!("program `{pname}`: batch.rows names unknown var `{v}`"));
                    }
                }
            }
        }
        let mut written = initially_written.clone();
        for (i, c) in p.calls.iter().enumerate() {
            let ctx = match &c.label {
                Some(l) => format!("program `{pname}` call #{i} ({l})"),
                None => format!("program `{pname}` call #{i}"),
            };
            let Some(op) = m.ops.get(&c.op) else {
                errs.push(format!("{ctx}: unknown op `{}`", c.op));
                continue;
            };
            used_ops.insert(c.op.clone());
            let has_extern = op.imp.launches.iter().any(|l| l.is_extern());

            if c.args.len() != op.params.len() {
                errs.push(format!("{ctx}: op `{}` takes {} params, got {} args", c.op, op.params.len(), c.args.len()));
                continue;
            }
            for (j, (arg, param)) in c.args.iter().zip(&op.params).enumerate() {
                let actx = format!("{ctx}: arg #{j}");
                match (arg, param) {
                    (Arg::Buf { buf, offset }, ParamType::Buf { dtype, dir }) => {
                        used_buffers.insert(buf.clone());
                        let Some(b) = m.buffers.get(buf) else {
                            errs.push(format!("{actx}: unknown buffer `{buf}`"));
                            continue;
                        };
                        if b.dtype != *dtype {
                            errs.push(format!(
                                "{actx}: buffer `{buf}` has dtype {} but param expects {}",
                                b.dtype, dtype
                            ));
                        }
                        if *offset > 0 {
                            if offset % b.dtype.bytes() != 0 {
                                errs.push(format!(
                                    "{actx}: offset {offset} into buffer `{buf}` is not {}-aligned for {}",
                                    b.dtype.bytes(),
                                    b.dtype
                                ));
                            }
                            if let Some(&sz) = buf_sizes.get(buf.as_str()) {
                                if *offset >= sz {
                                    errs.push(format!(
                                        "{actx}: offset {offset} is outside buffer `{buf}` ({sz} bytes at var upper bounds)"
                                    ));
                                }
                            }
                        }
                        if matches!(dir, Dir::In | Dir::InOut) && !written.contains(buf) {
                            errs.push(format!("{actx}: buffer `{buf}` is read before ever being written"));
                        }
                        if let (Some(tms), Some(&sz)) = (op_tensormaps.get(c.op.as_str()), buf_sizes.get(buf.as_str()))
                        {
                            for (_, fp, tctx) in tms.iter().filter(|(p, _, _)| *p == j) {
                                let avail = sz.saturating_sub(*offset);
                                if *fp > avail {
                                    errs.push(format!(
                                        "{actx}: {tctx} addresses {fp} bytes but buffer `{buf}` has {avail} bytes past offset {offset} at var upper bounds"
                                    ));
                                }
                            }
                        }
                        if b.kind == BufferKind::Peer && has_extern {
                            errs.push(format!(
                                "{actx}: peer buffer `{buf}` passed to op `{}`, which has an extern launch; runtime built-ins never receive peer memory",
                                c.op
                            ));
                        }
                        if matches!(dir, Dir::Out | Dir::InOut) {
                            if matches!(b.kind, BufferKind::Input | BufferKind::Weight | BufferKind::Peer) {
                                errs.push(format!("{actx}: op writes to read-only {} buffer `{buf}`", b.kind));
                            }
                            written.insert(buf.clone());
                            actually_written.insert(buf.clone());
                        }
                    }
                    (Arg::State { state, .. }, ParamType::State { .. }) => {
                        // state offsets are provider layout arithmetic over a
                        // runtime-scaled pool; there is no static bound to
                        // check them against.
                        used_states.insert(state.clone());
                        if !m.states.contains_key(state) {
                            errs.push(format!("{actx}: unknown state `{state}`"));
                        }
                    }
                    (Arg::Var { var }, ParamType::Scalar(st)) => {
                        used_vars.insert(var.clone());
                        match m.vars.get(var) {
                            None => errs.push(format!("{actx}: unknown var `{var}`")),
                            Some(v) => {
                                if *st == ScalarType::F32 {
                                    errs.push(format!("{actx}: var `{var}` cannot bind to an f32 param"));
                                } else if !scalar_fits(*st, v.max) {
                                    errs.push(format!("{actx}: var `{var}` max {} exceeds {st} range", v.max));
                                }
                            }
                        }
                    }
                    (Arg::Expr { expr }, ParamType::Scalar(st)) => {
                        check_expr(expr, m, &mut used_vars, &mut errs, &actx);
                        if *st == ScalarType::F32 {
                            errs.push(format!("{actx}: an expression cannot bind to an f32 param"));
                        } else if let Ok(v) = expr.eval(&env_max) {
                            if !scalar_fits(*st, v) {
                                errs.push(format!(
                                    "{actx}: expression reaches {v} at var upper bounds, exceeding {st} range"
                                ));
                            }
                        }
                    }
                    (Arg::I32 { .. }, ParamType::Scalar(ScalarType::I32))
                    | (Arg::I64 { .. }, ParamType::Scalar(ScalarType::I64))
                    | (Arg::F32 { .. }, ParamType::Scalar(ScalarType::F32))
                    | (Arg::U8 { .. }, ParamType::Scalar(ScalarType::U8)) => {}
                    (Arg::Rank { rank }, ParamType::Scalar(ScalarType::I32 | ScalarType::I64)) => {
                        group_ctx(rank, &mut errs, &mut used_groups, &actx);
                    }
                    (arg, param) => {
                        errs.push(format!("{actx}: {arg} does not match param `{param}`"));
                    }
                }
            }
        }
    }
    // Outputs and carries must be produced by *some* program — a
    // prefill-style program whose only effect is state mutation legitimately
    // writes none itself.
    for (bname, b) in &m.buffers {
        if matches!(b.kind, BufferKind::Output | BufferKind::Carry) && !actually_written.contains(bname) {
            errs.push(format!("{} buffer `{bname}` is never written by any program", b.kind));
        }
    }

    // 9. unused declarations
    for name in m.buffers.keys() {
        if !used_buffers.contains(name) {
            errs.push(format!("buffer `{name}` is never used by any program"));
        }
    }
    for name in m.ops.keys() {
        if !used_ops.contains(name) {
            errs.push(format!("op `{name}` is never called"));
        }
    }
    for name in m.modules.keys() {
        if !used_modules.contains(name) {
            errs.push(format!("module `{name}` is never launched"));
        }
    }
    for name in m.states.keys() {
        if !used_states.contains(name) {
            errs.push(format!("state `{name}` is never used by any program"));
        }
    }
    for name in m.vars.keys() {
        if !used_vars.contains(name) {
            errs.push(format!("var `{name}` is never used"));
        }
    }
    if let Some(t) = &m.topology {
        for name in t.groups.keys() {
            if !used_groups.contains(name) {
                errs.push(format!("topology group `{name}` is never used (no peer buffer or rank arg names it)"));
            }
        }
    }

    errs
}

fn scalar_fits(st: ScalarType, v: u64) -> bool {
    match st {
        ScalarType::I32 => v <= i32::MAX as u64,
        ScalarType::I64 => v <= i64::MAX as u64,
        ScalarType::U8 => v <= u8::MAX as u64,
        ScalarType::F32 => false,
    }
}

/// A domain is a prior on contents; the verifier only proves it is
/// well-formed against the declaration it decorates (never that any kernel
/// honours it).
#[allow(clippy::too_many_arguments)]
fn check_domain(
    name: &str,
    b: &Buffer,
    d: &Domain,
    m: &Manifest,
    env_max: &BTreeMap<String, u64>,
    env_min: &BTreeMap<String, u64>,
    used_vars: &mut BTreeSet<String>,
    errs: &mut Vec<String>,
) {
    let ctx = format!("buffer `{name}` domain");
    let is_float = matches!(b.dtype, DType::Bf16 | DType::F16 | DType::F32 | DType::Fp8E4m3);
    if d.index_into.is_some() && (d.min.is_some() || d.max.is_some()) {
        errs.push(format!("{ctx}: `index_into` and `min`/`max` are mutually exclusive"));
    }
    if d.index_into.is_none() && d.min.is_none() && d.max.is_none() && !d.monotone {
        errs.push(format!("{ctx}: empty (declare bounds, `index_into`, or `monotone`)"));
    }
    if d.stride == 0 {
        errs.push(format!("{ctx}: `stride` must be > 0"));
    }
    if d.stride > 1 && d.index_into.is_none() {
        errs.push(format!("{ctx}: `stride` only applies with `index_into`"));
    }
    if let Some(t) = &d.index_into {
        if is_float {
            errs.push(format!("{ctx}: a {} buffer cannot index anything", b.dtype));
        }
        match (m.buffers.contains_key(t), m.states.contains_key(t)) {
            (false, false) => errs.push(format!("{ctx}: `index_into` unknown buffer/state `{t}`")),
            (true, true) => errs.push(format!("{ctx}: `index_into` `{t}` is both a buffer and a state")),
            (false, true) if m.states[t].is_per_seq() && !m.states[t].bytes_per_seq.is_multiple_of(d.stride.max(1)) => {
                errs.push(format!(
                    "{ctx}: `index_into` per-sequence state `{t}` in lines of {} bytes, which do not divide its {} bytes per sequence",
                    d.stride, m.states[t].bytes_per_seq
                ))
            }
            (true, false) if t == name => errs.push(format!("{ctx}: a buffer cannot index itself")),
            _ => {}
        }
    }
    if d.monotone && b.shape.len() != 1 {
        errs.push(format!("{ctx}: `monotone` requires a one-dimensional buffer"));
    }
    for (which, bound) in [("min", &d.min), ("max", &d.max)] {
        let Some(bound) = bound else { continue };
        match bound {
            Bound::Float(_) if !is_float => {
                errs.push(format!("{ctx}: float `{which}` on a {} buffer", b.dtype));
            }
            Bound::Expr(e) => check_expr(e, m, used_vars, errs, &format!("{ctx}: `{which}`")),
            _ => {}
        }
    }
    if let (Some(lo), Some(hi)) = (&d.min, &d.max) {
        // Must hold at every var value the bounds can take; both corners
        // suffice for the monotone expression set.
        for env in [env_min, env_max] {
            if let (Ok(lo), Ok(hi)) = (lo.eval(env), hi.eval(env)) {
                if lo > hi {
                    errs.push(format!("{ctx}: min {lo} > max {hi}"));
                    break;
                }
            }
        }
    }
}

fn check_expr(e: &Expr, m: &Manifest, used_vars: &mut BTreeSet<String>, errs: &mut Vec<String>, ctx: &str) {
    match e {
        Expr::Const(_) => {}
        Expr::Var(var) => {
            if m.vars.contains_key(var) {
                used_vars.insert(var.clone());
            } else {
                errs.push(format!("{ctx}: unknown var `{var}`"));
            }
        }
        Expr::CeilDiv { ceil_div: (inner, c) } => {
            if *c == 0 {
                errs.push(format!("{ctx}: division by zero"));
            }
            check_expr(inner, m, used_vars, errs, ctx);
        }
        Expr::Mul { mul: (inner, c) } => {
            if *c == 0 {
                errs.push(format!("{ctx}: multiplication by constant zero"));
            }
            check_expr(inner, m, used_vars, errs, ctx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Manifest;

    /// The base fixture deliberately exercises the impl machinery: `embed`
    /// is the minimal single-launch op (ABI = interface, wiring defaulted);
    /// `attn` is a two-launch implementation with a private scratch buffer.
    const BASE: &str = r#"{
      "schema_version": 4, "model": "toy",
      "vars": { "tokens": { "max": 128 } },
      "states": { "kv": { "bytes_per_token": 4096 } },
      "buffers": {
        "x": { "dtype": "i32", "shape": ["tokens"], "kind": "input" },
        "w": { "dtype": "bf16", "shape": [64, 64], "kind": "weight" },
        "h": { "dtype": "bf16", "shape": ["tokens", 64], "kind": "workspace" },
        "y": { "dtype": "bf16", "shape": ["tokens", 64], "kind": "output" }
      },
      "modules": {
        "toy": { "source": "toy.cubin", "sha256": "abababababababababababababababababababababababababababababababab" }
      },
      "ops": {
        "embed": {
          "params": ["in buffer<i32>", "in buffer<bf16>", "out buffer<bf16>", "i32"],
          "impl": {
            "launches": [
              { "module": "toy", "entry": "embed_k",
                "block": [128, 1, 1], "grid": [{ "ceil_div": ["tokens", 128] }, 1, 1] }
            ]
          }
        },
        "attn": {
          "params": ["in buffer<bf16>", "inout state", "out buffer<bf16>", "i32", "i64"],
          "impl": {
            "scratch": {
              "part": { "dtype": "f32", "shape": ["tokens", 8] }
            },
            "launches": [
              {
                "module": "toy", "entry": "attn_part_k",
                "params": ["in buffer<bf16>", "inout state", "out buffer<f32>", "i32", "i64"],
                "block": [128, 1, 1],
                "grid": ["tokens", 8, 1],
                "args": [{ "param": 0 }, { "param": 1 }, { "scratch": "part" }, { "param": 3 }, { "param": 4 }]
              },
              {
                "module": "toy", "entry": "attn_reduce_k",
                "params": ["in buffer<f32>", "out buffer<bf16>", "i32"],
                "block": [128, 1, 1],
                "grid": ["tokens", 1, 1],
                "args": [{ "scratch": "part" }, { "param": 2 }, { "i32": 8 }]
              }
            ]
          }
        }
      },
      "programs": {
        "decode": { "calls": [
          { "label": "embed", "op": "embed",
            "args": [{ "buf": "x" }, { "buf": "w" }, { "buf": "h" }, { "var": "tokens" }] },
          { "label": "attn", "op": "attn",
            "args": [{ "buf": "h" }, { "state": "kv" }, { "buf": "y" }, { "var": "tokens" }, { "i64": 0 }] }
        ] }
      }
    }"#;

    fn base() -> serde_json::Value {
        serde_json::from_str(BASE).unwrap()
    }

    fn check(v: serde_json::Value) -> Result<Verified, VerifyErrors> {
        let m: Manifest = serde_json::from_value(v).map_err(|e| VerifyErrors(vec![e.to_string()]))?;
        verify(m)
    }

    fn assert_err(v: serde_json::Value, needle: &str) {
        let errs = check(v).expect_err("expected verification failure");
        assert!(errs.iter().any(|e| e.contains(needle)), "no error containing `{needle}` in {errs:#?}");
    }

    #[test]
    fn base_manifest_verifies() {
        check(base()).unwrap();
    }

    #[test]
    fn roundtrip_keeps_defaults_implicit() {
        let m = Manifest::from_json(BASE).unwrap();
        let j = m.to_json();
        let v: serde_json::Value = serde_json::from_str(&j).unwrap();
        let embed = &v["ops"]["embed"]["impl"]["launches"][0];
        assert!(
            embed.get("args").is_none() && embed.get("params").is_none(),
            "defaulted wiring must not be materialized: {embed}"
        );
        let again = Manifest::from_json(&j).unwrap();
        assert_eq!(again.to_json(), j);
    }

    #[test]
    fn launch_defaults_resolve_to_interface() {
        let m = Manifest::from_json(BASE).unwrap();
        let op = &m.ops["embed"];
        let l = &op.imp.launches[0];
        assert_eq!(l.params_of(op), &op.params[..]);
        assert_eq!(
            l.args_of(op).as_ref(),
            &[
                LaunchArg::Param { param: 0 },
                LaunchArg::Param { param: 1 },
                LaunchArg::Param { param: 2 },
                LaunchArg::Param { param: 3 }
            ]
        );
    }

    #[test]
    fn unknown_module() {
        let mut v = base();
        v["ops"]["embed"]["impl"]["launches"][0]["module"] = "ghost".into();
        assert_err(v, "unknown module `ghost`");
    }

    #[test]
    fn unused_module() {
        let mut v = base();
        v["modules"]["spare"] = serde_json::json!({ "source": "spare.cubin", "sha256": "cd".repeat(32) });
        assert_err(v, "module `spare` is never launched");
    }

    #[test]
    fn module_sha256_shape() {
        let mut v = base();
        v["modules"]["toy"]["sha256"] = "abc".into();
        assert_err(v, "is not 64 hex chars");
    }

    #[test]
    fn registry_module_malformed_ref() {
        let mut v = base();
        v["modules"]["toy"]["source"] = "hf:org/repo".into();
        assert_err(v, "invalid registry ref `hf:org/repo`");
    }

    #[test]
    fn registry_module_verifies() {
        let mut v = base();
        v["modules"]["toy"]["source"] = "hf:org/repo/pkg/embed.cubin@v1".into();
        check(v).unwrap();
    }

    #[test]
    fn registry_ref_parsing() {
        use crate::types::RegistryRef;
        assert!(RegistryRef::parse("embed.cubin").is_none());
        let r = RegistryRef::parse("hf:org/repo/a/b.cubin").unwrap().unwrap();
        assert_eq!((r.org.as_str(), r.repo.as_str()), ("org", "repo"));
        assert_eq!((r.path.as_str(), r.revision.as_str()), ("a/b.cubin", "main"));
        let r = RegistryRef::parse("hf:org/repo/a.cubin@abc123").unwrap().unwrap();
        assert_eq!(r.revision, "abc123");
        for bad in ["hf:org", "hf:org/repo", "hf:org/repo/", "hf:org//x", "hf:o/r/x@", "hf:o/r/../x", "hf:o/r/a//b"] {
            assert!(RegistryRef::parse(bad).unwrap().is_err(), "{bad}");
        }
    }

    #[test]
    fn extern_launch_has_no_geometry() {
        let mut v = base();
        v["ops"]["embed"]["impl"]["launches"][0] = serde_json::json!({ "entry": "extern:cublaslt_bf16_tn" });
        // arity: extern gemm inherits the 4-param interface here; only the
        // geometry rule is under test
        check(v).unwrap();
        // geometry on an extern: matches neither launch shape
        let mut v = base();
        v["ops"]["embed"]["impl"]["launches"][0] =
            serde_json::json!({ "entry": "extern:cublaslt_bf16_tn", "block": [1, 1, 1], "grid": [1, 1, 1] });
        assert!(check(v).is_err());
        // a module with an extern entry
        let mut v = base();
        v["ops"]["embed"]["impl"]["launches"][0]["entry"] = "extern:x".into();
        assert_err(v, "an extern entry has no module or launch geometry");
    }

    #[test]
    fn kernel_launch_needs_module_and_geometry() {
        // no module, no geometry, not extern: parses as an extern launch and
        // the verifier names the rule
        let mut v = base();
        v["ops"]["embed"]["impl"]["launches"][0] = serde_json::json!({ "entry": "embed_k" });
        assert_err(v, "a launch without a module must be a runtime built-in");
        // geometry without a module: no shape accepts it
        let mut v = base();
        v["ops"]["embed"]["impl"]["launches"][0].as_object_mut().unwrap().remove("module");
        assert!(check(v).is_err());
        let mut v = base();
        v["ops"]["embed"]["impl"]["launches"][0].as_object_mut().unwrap().remove("grid");
        assert!(check(v).is_err());
    }

    #[test]
    fn wrong_version() {
        let mut v = base();
        v["schema_version"] = 3.into();
        assert_err(v, "unsupported schema_version 3");
    }

    #[test]
    fn dtype_mismatch() {
        let mut v = base();
        v["buffers"]["h"]["dtype"] = "f32".into();
        assert_err(v, "has dtype f32 but param expects bf16");
    }

    #[test]
    fn unknown_op() {
        let mut v = base();
        v["programs"]["decode"]["calls"][0]["op"] = "nope".into();
        assert_err(v, "unknown op `nope`");
    }

    #[test]
    fn arg_count_mismatch() {
        let mut v = base();
        v["programs"]["decode"]["calls"][1]["args"].as_array_mut().unwrap().pop();
        assert_err(v, "takes 5 params, got 4 args");
    }

    #[test]
    fn read_before_write() {
        let mut v = base();
        let calls = v["programs"]["decode"]["calls"].as_array_mut().unwrap();
        calls.swap(0, 1);
        assert_err(v, "read before ever being written");
    }

    #[test]
    fn write_to_weight() {
        let mut v = base();
        v["programs"]["decode"]["calls"][0]["args"][2] = serde_json::json!({ "buf": "w" });
        assert_err(v, "writes to read-only weight buffer `w`");
    }

    #[test]
    fn var_exceeds_i32() {
        let mut v = base();
        v["vars"]["tokens"]["max"] = 3_000_000_000u64.into();
        assert_err(v, "exceeds i32 range");
    }

    #[test]
    fn output_never_written() {
        let mut v = base();
        v["programs"]["decode"]["calls"].as_array_mut().unwrap().pop();
        assert_err(v, "output buffer `y` is never written");
    }

    #[test]
    fn duplicate_name_rejected() {
        let dup = BASE.replace(
            r#""x": { "dtype": "i32", "shape": ["tokens"], "kind": "input" },"#,
            r#""x": { "dtype": "i32", "shape": ["tokens"], "kind": "input" },
               "x": { "dtype": "i32", "shape": ["tokens"], "kind": "input" },"#,
        );
        let err = Manifest::from_json(&dup).expect_err("duplicate must fail");
        assert!(err.to_string().contains("duplicate name `x`"), "{err}");
    }

    #[test]
    fn unknown_field_rejected() {
        let mut v = base();
        v["surprise"] = 1.into();
        let errs = check(v).expect_err("unknown field must fail");
        assert!(errs[0].contains("unknown field"), "{errs:?}");
    }

    #[test]
    fn grid_division_by_zero() {
        let mut v = base();
        v["ops"]["embed"]["impl"]["launches"][0]["grid"][0] = serde_json::json!({ "ceil_div": ["tokens", 0] });
        assert_err(v, "division by zero");
    }

    #[test]
    fn unused_buffer() {
        let mut v = base();
        v["buffers"]["dead"] = serde_json::json!({ "dtype": "bf16", "shape": [8], "kind": "workspace" });
        assert_err(v, "buffer `dead` is never used");
    }

    #[test]
    fn block_too_large() {
        let mut v = base();
        v["ops"]["attn"]["impl"]["launches"][0]["block"] = serde_json::json!([1024, 2, 1]);
        assert_err(v, "exceeds 1024 threads");
    }

    #[test]
    fn buf_offset_ok() {
        let mut v = base();
        // h is bf16 [tokens=128, 64] -> 16384 bytes max
        v["programs"]["decode"]["calls"][1]["args"][0] = serde_json::json!({ "buf": "h", "offset": 128 });
        check(v).unwrap();
    }

    #[test]
    fn buf_offset_misaligned() {
        let mut v = base();
        v["programs"]["decode"]["calls"][1]["args"][0] = serde_json::json!({ "buf": "h", "offset": 3 });
        assert_err(v, "not 2-aligned");
    }

    #[test]
    fn buf_offset_out_of_range() {
        let mut v = base();
        v["programs"]["decode"]["calls"][1]["args"][0] = serde_json::json!({ "buf": "h", "offset": 16384 });
        assert_err(v, "outside buffer `h`");
    }

    #[test]
    fn u8_param_and_var_range() {
        let mut v = base();
        v["ops"]["attn"]["params"][3] = "u8".into();
        v["ops"]["attn"]["impl"]["launches"][0]["params"][3] = "u8".into();
        v["programs"]["decode"]["calls"][1]["args"][3] = serde_json::json!({ "u8": 1 });
        check(v).unwrap();
        // binding a var with max 300 to a u8 param is rejected
        let mut v = base();
        v["ops"]["attn"]["params"][3] = "u8".into();
        v["ops"]["attn"]["impl"]["launches"][0]["params"][3] = "u8".into();
        v["vars"]["tokens"]["max"] = 300.into();
        assert_err(v, "exceeds u8 range");
    }

    #[test]
    fn u32_scalar_is_gone() {
        let mut v = base();
        v["ops"]["attn"]["params"][3] = "u32".into();
        let errs = check(v).expect_err("u32 scalars are not a thing");
        assert!(errs[0].contains("invalid param type"), "{errs:?}");
    }

    #[test]
    fn shared_mem_within_limit_ok() {
        let mut v = base();
        v["ops"]["attn"]["impl"]["launches"][0]["shared_mem"] = 167_184u64.into();
        check(v).unwrap();
    }

    #[test]
    fn shared_mem_exceeds_limit() {
        let mut v = base();
        v["ops"]["attn"]["impl"]["launches"][0]["shared_mem"] = 300_000u64.into();
        assert_err(v, "exceeds opt-in limit 232448");
    }

    #[test]
    fn buffer_arg_to_state_param() {
        let mut v = base();
        v["programs"]["decode"]["calls"][1]["args"][1] = serde_json::json!({ "buf": "h" });
        assert_err(v, "does not match param `inout state`");
    }

    #[test]
    fn grid_exceeds_cuda_limit_y() {
        let mut v = base();
        v["ops"]["attn"]["impl"]["launches"][0]["grid"][1] = 100_000u64.into();
        assert_err(v, "exceeds CUDA limit 65535");
    }

    #[test]
    fn bad_param_string_rejected() {
        let mut v = base();
        v["ops"]["attn"]["params"][0] = "buffer<bf16>".into();
        let errs = check(v).expect_err("param without direction must fail");
        assert!(errs[0].contains("invalid param type"), "{errs:?}");
    }

    // --- impl-layer checks ---

    #[test]
    fn launch_param_index_out_of_range() {
        let mut v = base();
        v["ops"]["embed"]["impl"]["launches"][0]["args"] =
            serde_json::json!([{ "param": 0 }, { "param": 1 }, { "param": 2 }, { "param": 9 }]);
        assert_err(v, "interface param #9 out of range");
    }

    #[test]
    fn launch_writes_interface_in_param() {
        let mut v = base();
        v["ops"]["embed"]["impl"]["launches"][0]["params"] =
            serde_json::json!(["out buffer<i32>", "in buffer<bf16>", "out buffer<bf16>", "i32"]);
        assert_err(v, "writes through interface `in` param #0");
    }

    #[test]
    fn launch_iface_kind_mismatch() {
        let mut v = base();
        // launch param says bf16 buffer where the interface forwards i32
        v["ops"]["embed"]["impl"]["launches"][0]["params"] =
            serde_json::json!(["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32"]);
        assert_err(v, "does not match launch param");
    }

    #[test]
    fn defaulted_args_with_explicit_params_must_agree_in_arity() {
        let mut v = base();
        v["ops"]["embed"]["impl"]["launches"][0]["params"] =
            serde_json::json!(["in buffer<i32>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32"]);
        assert_err(v, "takes 5 params, got 4 args");
    }

    #[test]
    fn scratch_read_before_write() {
        let mut v = base();
        let launches = v["ops"]["attn"]["impl"]["launches"].as_array_mut().unwrap();
        launches.swap(0, 1);
        assert_err(v, "scratch `part` is read before any launch wrote it");
    }

    #[test]
    fn scratch_unknown() {
        let mut v = base();
        v["ops"]["attn"]["impl"]["launches"][0]["args"][2] = serde_json::json!({ "scratch": "nope" });
        assert_err(v, "unknown scratch `nope`");
    }

    #[test]
    fn scratch_unused() {
        let mut v = base();
        v["ops"]["attn"]["impl"]["scratch"]["dead"] = serde_json::json!({ "dtype": "f32", "shape": [4] });
        assert_err(v, "scratch `dead` is never used");
    }

    #[test]
    fn scratch_dtype_mismatch() {
        let mut v = base();
        v["ops"]["attn"]["impl"]["launches"][0]["params"][2] = "out buffer<bf16>".into();
        assert_err(v, "scratch `part` has dtype f32 but param expects bf16");
    }

    #[test]
    fn scratch_offset_is_gone() {
        let mut v = base();
        v["ops"]["attn"]["impl"]["launches"][1]["args"][0] = serde_json::json!({ "scratch": "part", "offset": 4096 });
        let errs = check(v).expect_err("scratch offsets are not a thing");
        assert!(errs[0].contains("did not match any variant"), "{errs:?}");
        // and a typo in a call arg is a parse error, not a silently ignored key
        let mut v = base();
        v["programs"]["decode"]["calls"][1]["args"][0] = serde_json::json!({ "buf": "h", "offest": 128 });
        check(v).expect_err("unknown keys in args must fail");
    }

    #[test]
    fn interface_out_never_written() {
        let mut v = base();
        // reduce launch now writes scratch instead of the interface out param
        v["ops"]["attn"]["impl"]["launches"][1]["params"][1] = "out buffer<f32>".into();
        v["ops"]["attn"]["impl"]["launches"][1]["args"][1] = serde_json::json!({ "scratch": "part" });
        assert_err(v, "param #2 is never written by any launch");
    }

    #[test]
    fn empty_impl_rejected() {
        let mut v = base();
        v["ops"]["embed"]["impl"]["launches"] = serde_json::json!([]);
        assert_err(v, "implementation has no launches");
    }

    #[test]
    fn state_bytes_forms() {
        let mut v = base();
        v["states"]["kv"] = serde_json::json!({ "bytes": 4096 });
        check(v).unwrap();
        let mut v = base();
        v["states"]["kv"] = serde_json::json!({ "bytes": 4096, "bytes_per_token": 1 });
        assert_err(v, "are exclusive");
        let mut v = base();
        v["states"]["kv"] = serde_json::json!({});
        assert_err(v, "must be > 0");
        let mut v = base();
        v["states"]["kv"] = serde_json::json!({ "bytes": 4096, "align": 256 });
        let errs = check(v).expect_err("align is gone");
        assert!(errs[0].contains("unknown field"), "{errs:?}");
    }

    // --- topology / export / peer / rank ---

    /// The base fixture plus an `ep` group of 4, an exported flag buffer,
    /// its peer address array and a barrier op taking both plus the rank.
    fn peer_base() -> serde_json::Value {
        let mut v = base();
        v["topology"] = serde_json::json!({ "groups": { "ep": 4 } });
        v["buffers"]["flags"] = serde_json::json!({ "dtype": "u32", "shape": [64], "kind": "carry", "export": true });
        v["buffers"]["flags_peers"] =
            serde_json::json!({ "dtype": "u64", "shape": [4], "kind": "peer", "of": "flags", "group": "ep" });
        v["ops"]["barrier"] = serde_json::json!({
            "params": ["inout buffer<u32>", "in buffer<u64>", "i32", "i32"],
            "impl": { "launches": [
                { "module": "toy", "entry": "barrier_k", "block": [32, 1, 1], "grid": [1, 1, 1],
                  "args": [{ "param": 0 }, { "param": 1 }, { "param": 2 }, { "rank": "ep" }] }
            ] }
        });
        v["programs"]["decode"]["calls"].as_array_mut().unwrap().push(serde_json::json!({
            "op": "barrier",
            "args": [{ "buf": "flags" }, { "buf": "flags_peers" }, { "rank": "ep" }, { "i32": 0 }]
        }));
        v
    }

    #[test]
    fn peer_manifest_verifies() {
        check(peer_base()).unwrap();
        // a peer array of a state
        let mut v = peer_base();
        v["buffers"]["flags_peers"]["of"] = "kv".into();
        check(v).unwrap();
        // rank into an i64 param
        let mut v = peer_base();
        v["ops"]["barrier"]["params"][2] = "i64".into();
        check(v).unwrap();
    }

    #[test]
    fn peer_shape_and_dtype() {
        let mut v = peer_base();
        v["buffers"]["flags_peers"]["shape"] = serde_json::json!([8]);
        assert_err(v, "has shape [4], one address per member");
        let mut v = peer_base();
        v["buffers"]["flags_peers"]["shape"] = serde_json::json!([4, 1]);
        assert_err(v, "has shape [4], one address per member");
        let mut v = peer_base();
        v["buffers"]["flags_peers"]["dtype"] = "i64".into();
        v["ops"]["barrier"]["params"][1] = "in buffer<i64>".into();
        assert_err(v, "dtype must be u64");
    }

    #[test]
    fn peer_of_must_be_exported() {
        let mut v = peer_base();
        v["buffers"]["flags"]["export"] = false.into();
        assert_err(v, "`of` buffer `flags` is not exported");
        let mut v = peer_base();
        v["buffers"]["flags_peers"]["of"] = "ghost".into();
        assert_err(v, "`of` unknown buffer/state `ghost`");
        let mut v = peer_base();
        v["buffers"]["flags_peers"]["of"] = "flags_peers".into();
        assert_err(v, "cannot be `of` itself");
        let mut v = peer_base();
        v["buffers"]["flags_peers"].as_object_mut().unwrap().remove("of");
        assert_err(v, "must name the exported buffer or state");
    }

    #[test]
    fn peer_group_rules() {
        let mut v = peer_base();
        v["buffers"]["flags_peers"]["group"] = "tp".into();
        assert_err(v, "unknown topology group `tp`");
        let mut v = peer_base();
        v.as_object_mut().unwrap().remove("topology");
        assert_err(v, "declares no topology");
        let mut v = peer_base();
        v["topology"]["groups"]["cp"] = 2.into();
        assert_err(v, "topology group `cp` is never used");
        let mut v = peer_base();
        v["topology"]["groups"]["ep"] = 0.into();
        assert_err(v, "size must be > 0");
        let mut v = peer_base();
        v["buffers"]["flags"]["group"] = "ep".into();
        assert_err(v, "`group` only applies to peer buffers");
        let mut v = peer_base();
        v["buffers"]["flags"]["of"] = "flags".into();
        assert_err(v, "`of` only applies to peer buffers");
        let mut v = peer_base();
        v["buffers"]["flags_peers"]["export"] = true.into();
        assert_err(v, "cannot itself be exported");
    }

    #[test]
    fn peer_is_read_only_and_never_extern() {
        let mut v = peer_base();
        v["ops"]["barrier"]["params"][1] = "inout buffer<u64>".into();
        assert_err(v, "writes to read-only peer buffer `flags_peers`");
        // a peer buffer reaching an op with an extern launch
        let mut v = peer_base();
        v["ops"]["barrier"]["impl"]["launches"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({ "entry": "extern:cublaslt_bf16_tn", "params": [], "args": [] }));
        assert_err(v, "runtime built-ins never receive peer memory");
    }

    #[test]
    fn rank_binds_integers_only() {
        let mut v = peer_base();
        v["ops"]["barrier"]["params"][2] = "f32".into();
        v["ops"]["barrier"]["impl"]["launches"][0]["args"][2] = serde_json::json!({ "f32": 0.0 });
        assert_err(v, "rank in group `ep` does not match param `f32`");
        let mut v = peer_base();
        v["ops"]["barrier"]["params"][3] = "u8".into();
        v["programs"]["decode"]["calls"][2]["args"][3] = serde_json::json!({ "u8": 0 });
        assert_err(v, "a rank binds only to an i32 or i64 param, not `u8`");
        let mut v = peer_base();
        v["ops"]["barrier"]["impl"]["launches"][0]["args"][3] = serde_json::json!({ "rank": "nope" });
        assert_err(v, "unknown topology group `nope`");
    }

    /// A launch taking a tensor map over interface buffer #1 (u8 raw bytes),
    /// launched as clusters of 2.
    fn pack_base() -> serde_json::Value {
        let mut v = base();
        v["buffers"]["pk_out"] = serde_json::json!({ "dtype": "f32", "shape": [64], "kind": "output" });
        v["ops"]["pk"] = serde_json::json!({
            "params": ["out buffer<f32>", "in buffer<bf16>", "i32"],
            "impl": { "launches": [
                { "module": "toy", "entry": "pk_k", "block": [128, 1, 1], "grid": [1, 1, 1],
                  "params": ["bytes<24>", "in buffer<bf16>"],
                  "args": [{ "pack": { "size": 24, "fields": [
                                { "at": 0, "param": 0 }, { "at": 8, "param": 2 }, { "at": 12, "var": "tokens" },
                                { "at": 16, "i64": 4608 } ] } },
                           { "param": 1 }] }
            ] }
        });
        v["programs"]["decode"]["calls"].as_array_mut().unwrap().push(serde_json::json!({
            "op": "pk",
            "args": [{ "buf": "pk_out" }, { "buf": "w" }, { "var": "tokens" }]
        }));
        v
    }

    #[test]
    fn pack_manifest_verifies() {
        check(pack_base()).unwrap();
        assert_eq!("bytes<48>".parse::<ParamType>(), Ok(ParamType::Bytes(48)));
        assert_eq!(ParamType::Bytes(48).size_bytes(), 48);
        assert!("bytes<0>".parse::<ParamType>().is_err());
    }

    #[test]
    fn pack_rules() {
        let mut v = pack_base();
        v["ops"]["pk"]["impl"]["launches"][0]["params"][0] = "bytes<32>".into();
        assert_err(v, "pack of 24 bytes bound to a `bytes<32>` param");
        let mut v = pack_base();
        v["ops"]["pk"]["impl"]["launches"][0]["args"][0]["pack"]["fields"][1]["at"] = 4.into();
        assert_err(v, "field #1 overlaps field #0");
        let mut v = pack_base();
        v["ops"]["pk"]["impl"]["launches"][0]["args"][0]["pack"]["fields"][3]["at"] = 20.into();
        assert_err(v, "field #3 at 20 spans 8 bytes, past the 24 byte image");
        let mut v = pack_base();
        v["ops"]["pk"]["impl"]["launches"][0]["args"][0]["pack"]["fields"][2]["var"] = "ghost".into();
        assert_err(v, "unknown var `ghost`");
        let mut v = pack_base();
        v["ops"]["pk"]["params"][0] = "bytes<8>".into();
        assert_err(v, "interface param #0 is a byte aggregate");
        let mut v = pack_base();
        v["ops"]["pk"]["impl"]["launches"][0]["args"][1] = serde_json::json!({ "pack": { "size": 8, "fields": [] } });
        assert_err(v, "a pack binds only to a `bytes<n>` param");
    }

    #[test]
    fn tensormap_spans_the_buffer() {
        // outermost 0 = as many slices as the buffer holds: no footprint to check against the call
        let mut v = tensormap_base();
        v["ops"]["tm"]["impl"]["launches"][0]["args"][1]["tensormap"]["dims"] = serde_json::json!([128, 0]);
        check(v).unwrap();
        let mut v = tensormap_base();
        v["ops"]["tm"]["impl"]["launches"][0]["args"][1]["tensormap"]["dims"] = serde_json::json!([0, 4]);
        assert_err(v, "dims[0] is 0; only the outermost dim may span the buffer");
    }

    fn tensormap_base() -> serde_json::Value {
        let mut v = base();
        v["buffers"]["tm_out"] = serde_json::json!({ "dtype": "bf16", "shape": [64], "kind": "output" });
        v["buffers"]["raw"] = serde_json::json!({ "dtype": "u8", "shape": [512], "kind": "input" });
        v["ops"]["tm"] = serde_json::json!({
            "params": ["out buffer<bf16>", "in buffer<u8>"],
            "impl": { "launches": [
                { "module": "toy", "entry": "tm_k", "block": [128, 1, 1], "grid": [2, 1, 1], "cluster": [2, 1, 1],
                  "params": ["out buffer<bf16>", "tensormap"],
                  "args": [{ "param": 0 }, { "tensormap": { "param": 1, "dtype": "u8", "dims": [128, 4],
                                                            "strides": [128], "box": [128, 4], "swizzle": 128 } }] }
            ] }
        });
        v["programs"]["decode"]["calls"].as_array_mut().unwrap().push(serde_json::json!({
            "op": "tm",
            "args": [{ "buf": "tm_out" }, { "buf": "raw" }]
        }));
        v
    }

    #[test]
    fn tensormap_manifest_verifies() {
        check(tensormap_base()).unwrap();
        // the call's offset shrinks what the descriptor may address: 512 - 0 ok, exactly fits
        let mut v = tensormap_base();
        v["programs"]["decode"]["calls"][2]["args"][1]["offset"] = 0.into();
        check(v).unwrap();
    }

    #[test]
    fn tensormap_shape_rules() {
        let mut v = tensormap_base();
        v["ops"]["tm"]["impl"]["launches"][0]["args"][1]["tensormap"]["box"] = serde_json::json!([512, 4]);
        assert_err(v, "box[0] = 512, must be 1..=256");
        let mut v = tensormap_base();
        v["ops"]["tm"]["impl"]["launches"][0]["args"][1]["tensormap"]["strides"] = serde_json::json!([120]);
        assert_err(v, "strides[0] = 120 is not a positive multiple of 16");
        let mut v = tensormap_base();
        v["ops"]["tm"]["impl"]["launches"][0]["args"][1]["tensormap"]["swizzle"] = 64.into();
        assert_err(v, "box[0] spans 128 bytes, more than the 64 byte swizzle span");
        let mut v = tensormap_base();
        v["ops"]["tm"]["impl"]["launches"][0]["args"][1]["tensormap"]["dims"] = serde_json::json!([128, 5]);
        assert_err(v, "addresses 640 bytes but buffer `raw` has 512 bytes past offset 0");
        let mut v = tensormap_base();
        v["programs"]["decode"]["calls"][2]["args"][1]["offset"] = 16.into();
        assert_err(v, "has 496 bytes past offset 16");
    }

    #[test]
    fn tensormap_binding_rules() {
        // only over a buffer param
        let mut v = tensormap_base();
        v["ops"]["tm"]["impl"]["launches"][0]["args"][1]["tensormap"]["param"] = 0.into();
        v["ops"]["tm"]["impl"]["launches"][0]["args"][0] =
            serde_json::json!({ "tensormap": { "param": 1, "dtype": "u8", "dims": [128], "box": [128] } });
        assert_err(v, "arg #0: a tensormap binds only to a `tensormap` param, not `out buffer<bf16>`");
        // never on the interface
        let mut v = tensormap_base();
        v["ops"]["tm"]["params"][1] = "tensormap".into();
        assert_err(v, "interface param #1 is a tensormap; tensormaps are launch-private");
        // never into an extern
        let mut v = tensormap_base();
        v["ops"]["tm"]["impl"]["launches"][0] = serde_json::json!({
            "entry": "extern:cublaslt_bf16_tn", "params": ["out buffer<bf16>", "tensormap"],
            "args": [{ "param": 0 }, { "tensormap": { "param": 1, "dtype": "u8", "dims": [128], "box": [128] } }]
        });
        assert_err(v, "an extern launch takes pointers and scalars, not a tensormap");
    }

    #[test]
    fn cluster_divides_grid() {
        let mut v = tensormap_base();
        v["ops"]["tm"]["impl"]["launches"][0]["grid"] = serde_json::json!([3, 1, 1]);
        assert_err(v, "grid.x = 3 at var upper bounds is not a multiple of cluster.x = 2");
        let mut v = tensormap_base();
        v["ops"]["tm"]["impl"]["launches"][0]["cluster"] = serde_json::json!([4, 4, 2]);
        assert_err(v, "has a zero dim or more than 16 blocks");
        let mut v = tensormap_base();
        v["ops"]["tm"]["impl"]["launches"][0]["cluster"] = serde_json::json!([2, 0, 1]);
        assert_err(v, "has a zero dim or more than 16 blocks");
    }

    #[test]
    fn exported_buffer_without_peer_array_verifies() {
        // export alone is legal (a rank may hand its handle to something
        // outside the manifest); the group must still be used somewhere.
        let mut v = peer_base();
        v["buffers"].as_object_mut().unwrap().remove("flags_peers");
        v["ops"]["barrier"]["params"] = serde_json::json!(["inout buffer<u32>", "i32", "i32"]);
        v["ops"]["barrier"]["impl"]["launches"][0]["args"] =
            serde_json::json!([{ "param": 0 }, { "param": 1 }, { "rank": "ep" }]);
        v["programs"]["decode"]["calls"][2]["args"] =
            serde_json::json!([{ "buf": "flags" }, { "rank": "ep" }, { "i32": 0 }]);
        check(v).unwrap();
    }

    // --- domains ---

    #[test]
    fn domain_index_into_and_bounds_verify() {
        let mut v = base();
        v["buffers"]["x"]["domain"] = serde_json::json!({ "index_into": "w" });
        v["buffers"]["y"]["domain"] = serde_json::json!({ "min": -1.5, "max": 1.5 });
        check(v).unwrap();
        let mut v = base();
        v["buffers"]["x"]["domain"] = serde_json::json!({ "min": 0, "max": "tokens", "monotone": true });
        check(v).unwrap();
        let mut v = base();
        v["buffers"]["x"]["domain"] = serde_json::json!({ "index_into": "kv", "stride": 16 });
        check(v).unwrap();
    }

    #[test]
    fn domain_resolves() {
        use crate::types::ResolvedDomain;
        let mut v = base();
        v["buffers"]["x"]["domain"] = serde_json::json!({ "index_into": "kv", "stride": 16 });
        let m: Manifest = serde_json::from_value(v).unwrap();
        let env = BTreeMap::from([("tokens".to_string(), 4u64)]);
        let r = m.buffers["x"]
            .domain
            .as_ref()
            .unwrap()
            .resolve(&m, &env, &Provision { tokens: 4096, seq_slots: m.seq_slots() })
            .unwrap();
        assert_eq!(r, ResolvedDomain { lo: Some(0.0), hi: Some(255.0), monotone: false });
        assert!(r.contains(255.0) && !r.contains(256.0) && !r.contains(-1.0));

        let mut v = base();
        v["buffers"]["x"]["domain"] = serde_json::json!({ "index_into": "w" });
        let m: Manifest = serde_json::from_value(v).unwrap();
        let r = m.buffers["x"]
            .domain
            .as_ref()
            .unwrap()
            .resolve(&m, &env, &Provision { tokens: 0, seq_slots: m.seq_slots() })
            .unwrap();
        assert_eq!(r.hi, Some(63.0));

        let mut v = base();
        v["buffers"]["x"]["domain"] = serde_json::json!({ "min": 1, "max": "tokens" });
        let m: Manifest = serde_json::from_value(v).unwrap();
        let r = m.buffers["x"]
            .domain
            .as_ref()
            .unwrap()
            .resolve(&m, &env, &Provision { tokens: 0, seq_slots: m.seq_slots() })
            .unwrap();
        assert_eq!((r.lo, r.hi), (Some(1.0), Some(4.0)));
    }

    #[test]
    fn domain_rejects_malformed() {
        let mut v = base();
        v["buffers"]["x"]["domain"] = serde_json::json!({ "index_into": "nope" });
        assert_err(v, "unknown buffer/state `nope`");
        let mut v = base();
        v["buffers"]["x"]["domain"] = serde_json::json!({ "index_into": "w", "max": 3 });
        assert_err(v, "mutually exclusive");
        let mut v = base();
        v["buffers"]["x"]["domain"] = serde_json::json!({ "min": 0.5 });
        assert_err(v, "float `min` on a i32 buffer");
        let mut v = base();
        v["buffers"]["h"]["domain"] = serde_json::json!({ "index_into": "w" });
        assert_err(v, "a bf16 buffer cannot index anything");
        let mut v = base();
        v["buffers"]["h"]["domain"] = serde_json::json!({ "min": 0, "monotone": true });
        assert_err(v, "`monotone` requires a one-dimensional buffer");
        let mut v = base();
        v["buffers"]["x"]["domain"] = serde_json::json!({ "min": 5, "max": 2 });
        assert_err(v, "min 5 > max 2");
        let mut v = base();
        v["buffers"]["x"]["domain"] = serde_json::json!({});
        assert_err(v, "empty");
        let mut v = base();
        v["buffers"]["x"]["domain"] = serde_json::json!({ "max": "ghost" });
        assert_err(v, "unknown var `ghost`");
        let mut v = base();
        v["buffers"]["x"]["domain"] = serde_json::json!({ "min": 0, "stride": 4 });
        assert_err(v, "`stride` only applies with `index_into`");
    }
}
