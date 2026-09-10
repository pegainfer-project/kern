//! Load: everything that turns a verified manifest, a kernel directory
//! and a checkpoint into a [`Runtime`] that can run. Names stop here:
//! ops resolve to functions, buffers and states to device addresses,
//! programs to flat launch lists (`compile`), and after `load` returns
//! the execution path performs no name lookups.
//!
//! The states' budget is decided here too. Buffers, scratch and fixed
//! states are allocated as declared, at var max; what the paged and
//! per-sequence states get is the caller's [`Capacity`] or, without one,
//! whatever the device has left less [`HEADROOM`], carved into physical
//! chunks that the pool hands out as pages and slots.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use cudarc::cublaslt::CudaBlasLT;
use cudarc::driver::CudaContext;
use kern_manifest::types::{BufferKind, Manifest, Placement, Provision};
use kern_manifest::Verified;

use crate::chunks::Kind;
use crate::device::{
    alloc, alloc_host, alloc_vmm, chunk_granularity, copy_2d, Arena, Blas, DeviceBuf, Mapper, Physical, Share,
};
use crate::error::bail;
use crate::lease::Remaps;
use crate::pages::{chunks_for, page_unit, Pool};
use crate::peers::PeerSlot;
use crate::{compile, cubin, weights, Capacity, Error, Result, Runtime, Topology, HEADROOM};

impl Runtime {
    /// Load every `*.cubin` under `kernels_dir`, resolve
    /// ops, allocate all buffers and states, and lower every program.
    /// `capacity` sizes the pooled states ([`Capacity`]; a fixed-`bytes`
    /// state is allocated as declared); `None` fits them to the device:
    /// whatever memory is free once everything else is allocated, less
    /// [`HEADROOM`], but never more than every sequence the manifest can
    /// run at once could reference, with a slot per row the manifest
    /// bounds.
    /// A manifest with a `topology` needs this rank's [`Topology`]: one
    /// entry per declared group, sizes matching; without one the argument
    /// is ignored.
    pub fn load(
        manifest: &Verified,
        kernels_dir: &std::path::Path,
        gpu: usize,
        capacity: Option<Capacity>,
        topology: Option<&Topology>,
    ) -> Result<Runtime> {
        Self::load_with_host_weights(manifest, kernels_dir, gpu, capacity, topology, &crate::HostWeights::new())
    }

    /// Load a rank using host weights shared within one checkpoint snapshot.
    /// All participating ranks must bind their weights before serving begins.
    pub fn load_with_host_weights(
        manifest: &Verified,
        kernels_dir: &std::path::Path,
        gpu: usize,
        capacity: Option<Capacity>,
        topology: Option<&Topology>,
        host_weights: &crate::HostWeights,
    ) -> Result<Runtime> {
        let manifest = manifest.clone();
        let mut ranks = BTreeMap::new();
        if let Some(t) = &manifest.topology {
            let Some(mine) = topology else {
                bail!(Api, "the manifest declares a topology ({}); load needs this rank's place in it", fmt_groups(t));
            };
            for (g, &size) in &t.groups {
                let Some(r) = mine.groups.get(g) else {
                    bail!(Api, "topology group `{g}`: no rank given");
                };
                if r.size != size {
                    bail!(Api, "topology group `{g}`: manifest declares {size} members, rank given for {}", r.size);
                }
                if r.index >= size {
                    bail!(Api, "topology group `{g}`: rank {} outside 0..{size}", r.index);
                }
                ranks.insert(g.clone(), r.index);
            }
        }
        let dev = gpu as i32;

        let ctx = CudaContext::new(gpu)?;
        // A created (non-legacy) stream: the NULL stream cannot be captured
        // into a CUDA graph.
        let stream = ctx.new_stream()?;
        let xfer = ctx.new_stream()?;
        let blt = CudaBlasLT::new(stream.clone())?;
        let blas = Blas::new(&stream)?;
        ctx.bind_to_thread()?;

        let remote = cubin::fetch_registry_cubins(&manifest)?;
        let wanted: BTreeSet<String> = manifest.modules.values().map(|md| md.sha256.to_lowercase()).collect();
        let modules = cubin::load_pinned_modules(kernels_dir, &remote, &wanted)?;

        // Paged states are paged: every `index_into` one comes in `stride`
        // tokens per index (the KV block table's page). A capacity that is
        // not a whole number of pages would provision a torn last page —
        // slots the domain says are valid but a page-major kernel writes
        // past the pool for. Round down; the caller asked for "about this
        // many tokens", not for that page. (A per-sequence state's stride
        // is bytes per line, not tokens: not a page.)
        let page = page_unit(&manifest);
        let vars_max: BTreeMap<_, _> = manifest.vars.iter().map(|(s, v)| (s.clone(), v.max)).collect();

        // Buffer sizes are static: shapes only reference vars, sized at max.
        // Exported buffers come from the virtual-memory API with a fabric
        // handle; a peer array is an ordinary local buffer the runtime
        // fills; everything else is pool memory.
        let mut buffers = BTreeMap::new();
        let mut peers = BTreeMap::new();
        for (name, b) in &manifest.buffers {
            let bytes = compile::shaped_bytes(&format!("buffer `{name}`"), &b.shape, b.dtype.bytes(), &vars_max)?;
            let buf = if b.placement == Placement::Host {
                alloc_host(&stream, host_weights.acquire(name, b, bytes, &ctx)?)?
            } else if b.export {
                alloc_vmm(&stream, dev, bytes, Share::Required, &format!("buffer `{name}`"))?
            } else {
                alloc(&stream, bytes)?
            };
            if b.kind == BufferKind::Peer {
                // The verifier guarantees `of` and `group` are set.
                peers.insert(
                    name.clone(),
                    PeerSlot {
                        of: b.of.clone().unwrap_or_default(),
                        group: b.group.clone().unwrap_or_default(),
                        filled: false,
                    },
                );
            }
            buffers.insert(name.clone(), buf);
        }
        let mut staging = BTreeMap::new();
        for (name, b) in &manifest.buffers {
            if b.kind == BufferKind::Input {
                let mut pinned = unsafe { ctx.alloc_pinned::<u8>(buffers[name].bytes.max(1) as usize)? };
                pinned.as_mut_slice()?.fill(0);
                staging.insert(name.clone(), pinned);
            }
        }

        // Op scratch is allocated here, at var max, like the buffers.
        let resolved = compile::resolve_ops(&manifest, &modules, kernels_dir, &stream, &vars_max)?;

        // Everything but the states is on the device now: what is left is
        // the states' to take (weights are bound into buffers already
        // sized, so binding later costs nothing more). A fixed state is
        // allocated as declared; the paged and per-sequence states share
        // one budget of physical chunks, pages and sequence slots made out
        // of it as the pool decides.
        let token_bytes: u64 = manifest.states.values().map(|s| s.bytes_per_token).sum();
        let paged_bytes = token_bytes * page;
        let slot_bytes: u64 = manifest.states.values().map(|s| s.bytes_per_seq).sum();
        let fixed_bytes: u64 =
            manifest.states.values().filter(|s| s.bytes_per_token == 0 && s.bytes_per_seq == 0).map(|s| s.bytes).sum();
        let first_slots = match (slot_bytes > 0, capacity) {
            (false, _) => 0,
            (true, Some(c)) => c.seqs + 2,
            (true, None) => manifest.seq_slots(),
        };
        let chunk = chunk_size(&manifest, page, chunk_granularity(dev)? as u64);
        let chunks = match capacity.and_then(|c| c.tokens) {
            Some(asked) => {
                let aligned = asked / page * page;
                if aligned == 0 {
                    return Err(Error::Manifest(format!(
                        "state capacity {asked} tokens is smaller than one page ({page} tokens)"
                    )));
                }
                if aligned != asked {
                    tracing::warn!("state capacity {asked} is not a multiple of the page unit {page}; using {aligned}");
                }
                chunks_for(&manifest, aligned, first_slots, chunk)
            }
            None => fit_budget(&ctx, fixed_bytes, paged_bytes, slot_bytes * first_slots)? / chunk,
        };
        let chunks =
            u32::try_from(chunks).map_err(|_| Error::Manifest(format!("{chunks} chunks of state: too many")))?;

        let mut states = BTreeMap::new();
        for (name, s) in &manifest.states {
            if s.bytes_per_token == 0 && s.bytes_per_seq == 0 {
                let buf = alloc_vmm(&stream, dev, s.bytes, Share::IfSupported, &format!("state `{name}`"))?;
                states.insert(name.clone(), buf);
            }
        }
        let (pool, initial) = Pool::new(&manifest, chunk, chunks, first_slots as usize)?;
        let physical = Physical::create(dev, chunk as usize, chunks as usize)?;
        let mut arenas = Vec::with_capacity(pool.pooled().len());
        for a in pool.pooled() {
            let arena = Arena::reserve(dev, chunk as usize, a.positions)?;
            let objects = match a.kind {
                Kind::Page => pool.total(),
                Kind::Slot => pool.slots(),
            } as u64;
            let buf = DeviceBuf::reserved(&stream, arena.ptr(), objects * a.object, a.positions as u64 * chunk);
            states.insert(a.state.clone(), buf);
            arenas.push(arena);
        }
        let mut mapper = Mapper::new(arenas, physical);
        mapper.run(&initial)?;
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        tracing::debug!(
            "state budget {:.1} GiB in {chunks} chunks of {} MiB: {} pages of {page} tokens, {} sequence slots",
            gib(chunks as u64 * chunk),
            chunk >> 20,
            pool.total(),
            pool.slots(),
        );
        for p in peers.values() {
            if let Some(st) = states.get(&p.of) {
                if !st.is_shareable() {
                    bail!(Cuda, "state `{}` has no fabric handle on device {dev}, but a peer buffer is `of` it", p.of);
                }
            }
        }
        let resolution = resolved.iter().map(|(n, rk)| (n.clone(), rk.launch_modules())).collect();
        let peer_names: BTreeSet<String> = peers.keys().cloned().collect();
        let place = compile::Ranks { ranks: &ranks, peer_buffers: &peer_names };
        let programs = compile::compile_programs(&manifest, &resolved, &buffers, &states, &place)?;
        let scratch = resolved.into_values().flat_map(|rk| rk.scratch.into_values()).collect();

        let provision = Provision { tokens: pool.pages_max() as u64 * page, seq_slots: pool.slots_max() as u64 };
        let remaps = Remaps::spawn(Arc::clone(&ctx), mapper)?;
        let mut rt = Runtime {
            host_weights_ready: !manifest.buffers.values().any(|b| b.placement == Placement::Host),
            manifest,
            ctx,
            stream,
            xfer,
            host: None,
            parking: Vec::new(),
            blt,
            blas,
            provision,
            pool: Arc::new(pool),
            remaps,
            remap_count: 0,
            buffers,
            states,
            staging,
            programs,
            scratch,
            resolution,
            n_modules: modules.len(),
            graphs: BTreeMap::new(),
            eager: false,
            gpu,
            ranks,
            peers,
            imports: Vec::new(),
        };
        rt.zero_fresh(&initial)?;
        rt.stream.synchronize()?;
        Ok(rt)
    }

    /// Assemble every `weight` buffer from the checkpoint tensors its
    /// `bind` names, out of one or more safetensors blobs (the model's
    /// shards, a draft's next to them). Only headers are parsed; each
    /// segment is one copy straight out of the blob. A tensor name that
    /// appears in more than one blob is ambiguous and refused.
    pub fn load_weights(&mut self, blobs: &[&[u8]]) -> Result<()> {
        if self.host_weights_ready && self.buffers.values().any(|b| b.host_weight().is_some()) {
            bail!(Api, "host weights are an immutable snapshot; use a new runtime and scope to reload");
        }
        self.ctx.bind_to_thread()?;
        let sts = blobs
            .iter()
            .map(|b| safetensors::SafeTensors::deserialize(b))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| Error::WeightArtifact(format!("unparseable safetensors: {e}")))?;
        let lookup = |tensor: &str| -> Result<weights::TensorInfo> {
            let found: Vec<_> =
                sts.iter().enumerate().filter_map(|(i, st)| st.tensor(tensor).ok().map(|t| (i, t))).collect();
            let (blob, t) = match found.as_slice() {
                [one] => one.clone(),
                [] => bail!(WeightArtifact, "tensor `{tensor}` is in none of the {} artifact(s)", sts.len()),
                many => bail!(WeightArtifact, "tensor `{tensor}` is in {} artifacts", many.len()),
            };
            let Some(dtype) = weights::dtype_of(t.dtype()) else {
                bail!(WeightArtifact, "tensor `{tensor}`: dtype {:?} has no manifest dtype", t.dtype());
            };
            let base = blobs[blob].as_ptr() as usize;
            let offset = t.data().as_ptr() as usize - base;
            Ok(weights::TensorInfo { blob, offset, dtype, shape: t.shape().iter().map(|&d| d as u64).collect() })
        };
        for (name, b) in &self.manifest.buffers {
            if b.kind != BufferKind::Weight {
                continue;
            }
            let dst = &self.buffers[name];
            let copies = weights::plan(name, b, dst.bytes, lookup, |group| self.ranks.get(group).copied())?;
            if let Some(host) = dst.host_weight() {
                host.initialize(|bytes| {
                    copy_host_weights(bytes, blobs, &copies);
                    Ok(())
                })?;
                continue;
            }
            for c in copies {
                let src = &blobs[c.blob][c.src..c.src + (c.pitch * (c.rows - 1) + c.width) as usize];
                if c.pitch == c.width {
                    let mut view = dst.view(c.dst as usize..(c.dst + c.width * c.rows) as usize)?;
                    self.stream.memcpy_htod(src, &mut view)?;
                } else {
                    let stream = self.stream.cu_stream();
                    copy_2d(
                        stream,
                        (dst.ptr + c.dst, c.width),
                        (src.as_ptr() as u64, c.pitch),
                        c.width,
                        c.rows,
                        false,
                    )?;
                }
            }
        }
        self.stream.synchronize()?;
        self.host_weights_ready = true;
        Ok(())
    }
}

/// Execute copies already bounded and laid out by `weights::plan`.
/// Full-width tensor segments use one memcpy, even for hundreds of millions
/// of short rows. Rectangular slices retain their source row pitch.
fn copy_host_weights(dst: &mut [u8], blobs: &[&[u8]], copies: &[weights::Copy]) {
    for c in copies {
        if c.pitch == c.width {
            let len = (c.width * c.rows) as usize;
            let dest = c.dst as usize;
            dst[dest..dest + len].copy_from_slice(&blobs[c.blob][c.src..c.src + len]);
        } else {
            for row in 0..c.rows {
                let src = (c.src as u64 + row * c.pitch) as usize;
                let dest = (c.dst + row * c.width) as usize;
                dst[dest..dest + c.width as usize].copy_from_slice(&blobs[c.blob][src..src + c.width as usize]);
            }
        }
    }
}

fn fmt_groups(t: &kern_manifest::types::Topology) -> String {
    t.groups.iter().map(|(g, n)| format!("{g}={n}")).collect::<Vec<_>>().join(", ")
}

/// The chunk the pooled states are backed in: a multiple of the
/// allocation granularity `g`, at most half the smallest page or slot so
/// an object spans at least two (a chunk shared at a boundary is one of
/// many), at most 64 MiB. Mapping costs per chunk, so bigger is cheaper.
fn chunk_size(m: &Manifest, page: u64, g: u64) -> u64 {
    let smallest = m
        .states
        .values()
        .filter_map(|s| match (s.bytes_per_token, s.bytes_per_seq) {
            (t, 0) if t > 0 => Some(t * page),
            (0, q) if q > 0 => Some(q),
            _ => None,
        })
        .min()
        .unwrap_or(g);
    (smallest / 2 / g).clamp(1, (64 << 20) / g.max(1)).max(1) * g
}

/// State budget in bytes that fits the device: free memory (after every
/// buffer, scratch and fixed state) less [`HEADROOM`]; it must hold the
/// first sequence slots and a page.
fn fit_budget(ctx: &CudaContext, fixed: u64, page_bytes: u64, first_slots_bytes: u64) -> Result<u64> {
    let (free, total) = ctx.mem_get_info()?;
    let (free, total) = (free as u64, total as u64);
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    let Some(budget) = free.checked_sub(HEADROOM).and_then(|b| b.checked_sub(fixed)) else {
        return Err(Error::Cuda(format!(
            "{:.2} GiB free of {:.1} on the device: nothing left for the states after {:.1} GiB headroom",
            gib(free),
            gib(total),
            gib(HEADROOM)
        )));
    };
    if budget < first_slots_bytes + page_bytes {
        return Err(Error::Cuda(format!(
            "{:.2} GiB free of {:.1} on the device: the first sequence slots are {:.2} GiB and a page {:.2} GiB, headroom {:.1} GiB",
            gib(free),
            gib(total),
            gib(first_slots_bytes),
            gib(page_bytes),
            gib(HEADROOM)
        )));
    }
    tracing::info!(
        "state budget {:.1} GiB; device {:.1} GiB free of {:.1}, {:.1} GiB headroom",
        gib(budget),
        gib(free),
        gib(total),
        gib(HEADROOM)
    );
    Ok(budget)
}

#[cfg(test)]
mod host_copy_tests {
    use super::*;

    #[test]
    fn contiguous_and_strided_rectangles_preserve_destination_neighbors() {
        let contiguous = [90, 91, 1, 2, 3, 4, 5, 6, 92];
        let strided = [90, 1, 2, 80, 81, 3, 4, 82, 83, 5, 6, 84];
        let blobs: &[&[u8]] = &[&contiguous, &strided];
        for (blob, src, pitch) in [(0, 2, 2), (1, 1, 4)] {
            let mut dst = [77; 12];
            copy_host_weights(&mut dst, blobs, &[weights::Copy { dst: 3, blob, src, width: 2, rows: 3, pitch }]);
            assert_eq!(dst, [77, 77, 77, 1, 2, 3, 4, 5, 6, 77, 77, 77]);
        }
    }

    #[test]
    fn mixed_segments_from_multiple_blobs_assemble_in_order() {
        let blobs: &[&[u8]] = &[&[90, 1, 2, 3, 4, 91], &[80, 5, 6, 81, 82, 7, 8, 83]];
        let copies = [
            weights::Copy { dst: 0, blob: 0, src: 1, width: 2, rows: 2, pitch: 2 },
            weights::Copy { dst: 4, blob: 1, src: 1, width: 2, rows: 2, pitch: 4 },
        ];
        let mut dst = [0; 8];
        copy_host_weights(&mut dst, blobs, &copies);
        assert_eq!(dst, [1, 2, 3, 4, 5, 6, 7, 8]);
    }
}
