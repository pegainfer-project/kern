//! Static device memory and KV page capacity accounting.
//!
//! Mirrors `kern_runtime::Runtime::load` without touching CUDA:
//! every non-`source` device buffer and every op's `impl.scratch` is sized at
//! `vars` upper bounds, `source` buffers share one 256-byte-aligned first-fit
//! staging window across `derive` calls, `cublas::Blas` reserves 32 MiB, and
//! remaining device memory less 1 GiB [`HEADROOM_BYTES`], driver context
//! overhead, fixed states, and per-sequence slots is divided into whole pages
//! of [`kern_pool::page_unit`] tokens.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{bail, Result};
use kern_manifest::types::{Arg, BufferKind, Dim, Manifest, Placement};
use kern_pool::{page_unit, row_tokens};
use serde::Serialize;

/// Safety margin left free on the device (`kern_runtime::HEADROOM`: 1 GiB).
pub const HEADROOM_BYTES: u64 = 1 << 30;

/// Fixed stream workspace allocated by `cublas::Blas::new` (32 MiB).
pub const CUBLAS_WORKSPACE_BYTES: u64 = 32 << 20;

/// Default CUDA context + cuBLASLt handle estimate when uncalibrated (512 MiB).
pub const DEFAULT_DRIVER_OVERHEAD_BYTES: u64 = 512 << 20;

const DERIVE_ALIGN: u64 = 256;

/// Device specification and capacity overrides for [`MemoryBudget::compute`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct DeviceMemorySpec {
    /// Total physical VRAM on one rank, in bytes.
    pub total_vram_bytes: u64,
    /// CUDA driver context, module image, and allocator reserve, in bytes.
    pub driver_overhead_bytes: u64,
    /// Active sequence slots requested (`None` uses `manifest.seq_slots() - 1`,
    /// matching `Runtime::load` with `capacity: None`).
    pub active_slots: Option<u64>,
    /// Explicit token cap (`None` takes every whole page that fits).
    pub explicit_tokens: Option<u64>,
}

impl DeviceMemorySpec {
    pub fn with_gib(total_gib: f64) -> Self {
        Self {
            total_vram_bytes: (total_gib * (1u64 << 30) as f64).round() as u64,
            driver_overhead_bytes: DEFAULT_DRIVER_OVERHEAD_BYTES,
            active_slots: None,
            explicit_tokens: None,
        }
    }
}

/// Complete static VRAM breakdown and resulting KV page budget for one rank.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MemoryBudget {
    pub model: String,
    pub total_vram_bytes: u64,
    pub headroom_bytes: u64,
    pub driver_overhead_bytes: u64,
    pub weights_bytes: u64,
    pub host_weights_bytes: u64,
    pub activations_bytes: u64,
    pub op_scratch_bytes: u64,
    pub derive_staging_bytes: u64,
    pub cublas_workspace_bytes: u64,
    pub fixed_states_bytes: u64,
    pub page_unit: u64,
    pub token_unit_bytes: u64,
    pub page_bytes: u64,
    pub slot_unit_bytes: u64,
    /// Total provisioned slots including reserved null slot 0 (`0` when the
    /// manifest has no per-sequence states).
    pub provisioned_slots: u64,
    pub seq_states_bytes: u64,
    /// Raw pages allocated by `Runtime::load` (`Pool::pages()`).
    pub raw_pages: usize,
    /// Pages available to `kern-serve`'s scheduler after reserving 1 pad page.
    pub scheduler_pages: usize,
    pub kv_states_bytes: u64,
    /// Maximum tokens one sequence may hold (`row_tokens` capped by pool pages).
    pub max_seq_tokens: usize,
}

impl MemoryBudget {
    /// Compute the static VRAM breakdown and page capacity for `manifest`.
    pub fn compute(manifest: &Manifest, spec: DeviceMemorySpec) -> Result<Self> {
        let vars_max: BTreeMap<String, u64> =
            manifest.vars.iter().map(|(k, v)| (k.clone(), v.max)).collect();

        let derived_carries = derived_carry_names(manifest);
        let mut weights_bytes = 0u64;
        let mut host_weights_bytes = 0u64;
        let mut activations_bytes = 0u64;
        let mut sources = BTreeMap::new();

        for (name, b) in &manifest.buffers {
            let bytes = shaped_bytes(&format!("buffer `{name}`"), &b.shape, b.dtype.bytes(), &vars_max)?;
            if b.kind == BufferKind::Source {
                sources.insert(name.clone(), bytes);
                continue;
            }
            if b.placement == Placement::Host {
                host_weights_bytes = host_weights_bytes.saturating_add(bytes);
                continue;
            }
            if b.kind == BufferKind::Weight
                || (b.kind == BufferKind::Carry && derived_carries.contains(name.as_str()))
            {
                weights_bytes = weights_bytes.saturating_add(bytes);
            } else {
                activations_bytes = activations_bytes.saturating_add(bytes);
            }
        }

        let mut op_scratch_bytes = 0u64;
        for (op_name, op) in &manifest.ops {
            for (sc_name, sc) in &op.imp.scratch {
                let bytes = shaped_bytes(
                    &format!("op `{op_name}` scratch `{sc_name}`"),
                    &sc.shape,
                    sc.dtype.bytes(),
                    &vars_max,
                )?;
                op_scratch_bytes = op_scratch_bytes.saturating_add(bytes);
            }
        }

        let derive_staging_bytes = derive_staging_bytes(manifest, &sources);
        let cublas_workspace_bytes = CUBLAS_WORKSPACE_BYTES;

        let unit = page_unit(manifest);
        let token_unit_bytes: u64 = manifest
            .states
            .values()
            .filter(|s| s.is_owned())
            .map(|s| s.bytes_per_token)
            .sum();
        let page_bytes = token_unit_bytes.saturating_mul(unit);
        let slot_unit_bytes: u64 = manifest
            .states
            .values()
            .filter(|s| s.is_owned())
            .map(|s| s.bytes_per_seq)
            .sum();
        let fixed_states_bytes: u64 = manifest
            .states
            .values()
            .filter(|s| s.is_owned() && s.bytes_per_token == 0 && s.bytes_per_seq == 0)
            .map(|s| s.bytes)
            .sum();

        let provisioned_slots = if slot_unit_bytes == 0 {
            0
        } else {
            match spec.active_slots {
                Some(active) => active.saturating_add(1),
                None => manifest.seq_slots(),
            }
        };
        let seq_states_bytes = provisioned_slots.saturating_mul(slot_unit_bytes);

        let non_state_used = spec
            .driver_overhead_bytes
            .saturating_add(weights_bytes)
            .saturating_add(activations_bytes)
            .saturating_add(op_scratch_bytes)
            .saturating_add(derive_staging_bytes)
            .saturating_add(cublas_workspace_bytes);

        let free_before_states = spec.total_vram_bytes.saturating_sub(non_state_used);
        let state_fixed = fixed_states_bytes.saturating_add(seq_states_bytes);

        let raw_pages = match spec.explicit_tokens.filter(|_| page_bytes > 0) {
            Some(asked) => {
                let p = asked / unit;
                if p == 0 {
                    bail!("requested state capacity {asked} tokens is smaller than one page ({unit} tokens)");
                }
                p as usize
            }
            None if page_bytes == 0 => 0,
            None => {
                let p = free_before_states
                    .checked_sub(HEADROOM_BYTES)
                    .and_then(|b| b.checked_sub(state_fixed))
                    .map_or(0, |b| b / page_bytes);
                if p == 0 {
                    bail!(
                        "insufficient VRAM ({:.2} GiB total): 0 KV pages fit after {:.2} GiB non-state, {:.2} GiB fixed/seq states, and 1.00 GiB headroom",
                        gib(spec.total_vram_bytes),
                        gib(non_state_used),
                        gib(state_fixed),
                    );
                }
                p as usize
            }
        };

        let scheduler_pages = if page_bytes > 0 { raw_pages.saturating_sub(1) } else { 0 };
        let kv_states_bytes = (raw_pages as u64).saturating_mul(page_bytes);
        let row_cap = row_tokens(manifest, unit).map_or(raw_pages, |t| (t / unit) as usize);
        let max_seq_tokens = row_cap.min(scheduler_pages.max(raw_pages.min(1))) * unit as usize;

        Ok(Self {
            model: manifest.model.clone(),
            total_vram_bytes: spec.total_vram_bytes,
            headroom_bytes: HEADROOM_BYTES,
            driver_overhead_bytes: spec.driver_overhead_bytes,
            weights_bytes,
            host_weights_bytes,
            activations_bytes,
            op_scratch_bytes,
            derive_staging_bytes,
            cublas_workspace_bytes,
            fixed_states_bytes,
            page_unit: unit,
            token_unit_bytes,
            page_bytes,
            slot_unit_bytes,
            provisioned_slots,
            seq_states_bytes,
            raw_pages,
            scheduler_pages,
            kv_states_bytes,
            max_seq_tokens,
        })
    }

    /// Infer `driver_overhead_bytes` from one observed `Runtime::load` page
    /// count (`paged states: <raw_pages> pages` in `kern-serve` logs) so
    /// subsequent manifests and slot counts on the same GPU are calibrated.
    pub fn calibrate_driver_overhead(
        manifest: &Manifest,
        total_vram_bytes: u64,
        active_slots: Option<u64>,
        observed_raw_pages: usize,
    ) -> Result<u64> {
        let zero_spec = DeviceMemorySpec {
            total_vram_bytes,
            driver_overhead_bytes: 0,
            active_slots,
            explicit_tokens: None,
        };
        let base = Self::compute(manifest, zero_spec)?;
        if base.page_bytes == 0 {
            bail!("cannot calibrate driver overhead from a manifest with no paged states");
        }
        let non_driver = base
            .weights_bytes
            .saturating_add(base.activations_bytes)
            .saturating_add(base.op_scratch_bytes)
            .saturating_add(base.derive_staging_bytes)
            .saturating_add(base.cublas_workspace_bytes)
            .saturating_add(base.fixed_states_bytes)
            .saturating_add(base.seq_states_bytes)
            .saturating_add(HEADROOM_BYTES);
        let kv_used = (observed_raw_pages as u64).saturating_mul(base.page_bytes);
        // Place calibrated overhead at the midpoint of the half-page rounding interval
        // `[total - non_driver - (pages + 1)*page_bytes + 1, total - non_driver - pages*page_bytes]`.
        let max_overhead = total_vram_bytes
            .checked_sub(non_driver.saturating_add(kv_used))
            .ok_or_else(|| anyhow::anyhow!("observed {observed_raw_pages} pages exceed total VRAM {total_vram_bytes}"))?;
        let slack = (base.page_bytes / 2).min(max_overhead);
        Ok(max_overhead - slack)
    }

    /// Total KV token capacity available to the scheduler.
    pub fn scheduler_tokens(&self) -> u64 {
        (self.scheduler_pages as u64).saturating_mul(self.page_unit)
    }
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

fn shaped_bytes(what: &str, shape: &[Dim], elem_bytes: u64, vars_max: &BTreeMap<String, u64>) -> Result<u64> {
    let mut total = elem_bytes;
    for dim in shape {
        let ext = match dim {
            Dim::Const(c) => *c,
            Dim::Var(v) => *vars_max
                .get(v)
                .ok_or_else(|| anyhow::anyhow!("{what}: unknown var `{v}`"))?,
        };
        total = total
            .checked_mul(ext)
            .ok_or_else(|| anyhow::anyhow!("{what}: byte size overflows u64"))?;
    }
    Ok(total)
}

fn derived_carry_names(manifest: &Manifest) -> BTreeSet<&str> {
    let mut names = BTreeSet::new();
    for prog in manifest.programs.values().filter(|p| p.once || p.derive) {
        for call in &prog.calls {
            let Some(op) = manifest.ops.get(&call.op) else { continue };
            for (idx, arg) in call.args.iter().enumerate() {
                if let Arg::Buf { buf, .. } = arg {
                    if op.params.get(idx).and_then(|p| p.dir()).is_some_and(|d| d != kern_manifest::types::Dir::In) {
                        names.insert(buf.as_str());
                    }
                }
            }
        }
    }
    names
}

#[derive(Clone, Copy)]
struct Window {
    first: usize,
    last: usize,
    bytes: u64,
}

fn derive_staging_bytes(manifest: &Manifest, sources: &BTreeMap<String, u64>) -> u64 {
    if sources.is_empty() {
        return 0;
    }
    let calls: Vec<_> = manifest
        .programs
        .iter()
        .filter(|(_, p)| p.derive)
        .flat_map(|(_, p)| &p.calls)
        .collect();
    let mut live: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    for (k, c) in calls.iter().enumerate() {
        for a in &c.args {
            let Arg::Buf { buf, .. } = a else { continue };
            if sources.contains_key(buf) {
                live.entry(buf.as_str())
                    .and_modify(|(_, last)| *last = k)
                    .or_insert((k, k));
            }
        }
    }
    let windows: Vec<Window> = live
        .iter()
        .map(|(n, &(first, last))| Window { first, last, bytes: sources[*n] })
        .collect();
    let mut order: Vec<usize> = (0..windows.len()).collect();
    order.sort_by_key(|&i| (windows[i].first, i));
    let mut at = vec![0u64; windows.len()];
    let mut placed: Vec<usize> = Vec::new();
    for i in order {
        let w = windows[i];
        let mut taken: Vec<(u64, u64)> = placed
            .iter()
            .filter(|&&j| windows[j].last >= w.first)
            .map(|&j| (at[j], at[j] + windows[j].bytes))
            .collect();
        taken.sort_unstable();
        let mut lo = 0;
        for (a, b) in taken {
            if lo + w.bytes <= a {
                break;
            }
            lo = lo.max(b.next_multiple_of(DERIVE_ALIGN));
        }
        at[i] = lo;
        placed.push(i);
    }
    windows
        .iter()
        .zip(&at)
        .map(|(w, a)| a + w.bytes)
        .max()
        .unwrap_or(0)
}
