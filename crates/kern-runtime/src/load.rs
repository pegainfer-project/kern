//! Load: everything that turns a verified manifest, a kernel directory
//! and a checkpoint into a [`Runtime`] that can run. Names stop here:
//! ops resolve to functions, buffers and states to device addresses,
//! programs to flat launch lists (`compile`), weights to copies out of a
//! checkpoint (`load_weights`), and after `load` returns the execution
//! path performs no name lookups.
//!
//! The states' budget is decided here too. Buffers, scratch and fixed
//! states are allocated as declared, at var max; what the paged and
//! per-sequence states get is the caller's [`Capacity`] or, without one,
//! whatever the device has left less [`HEADROOM`], carved into physical
//! chunks that the pool hands out as pages and slots.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use cudarc::cublaslt::CudaBlasLT;
use cudarc::driver::{CudaContext, CudaStream};
use kern_manifest::types::{Buffer, BufferKind, DType, Dim, Manifest, Placement, Provision, Segment};
use kern_manifest::Verified;

use crate::cublas::Blas;
use crate::device::{
    alloc, alloc_host, alloc_vmm, chunk_granularity, copy_1d, copy_2d, Arena, DeviceBuf, Mapper, Physical, Space,
};
use crate::error::bail;
use crate::lease::Remaps;
use crate::peers::PeerSlot;
use crate::weights::{Blob, Tensors};
use crate::{compile, cubin, weights, Capacity, Error, Result, Runtime, Topology, HEADROOM};
use kern_pool::{chunks_for, page_unit, Kind, Pool};

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
        Self::load_from(manifest, kernels_dir, gpu, capacity, topology, host_weights, None)
    }

    /// Load a rank over the weights a runtime on the same device left
    /// behind ([`Runtime::into_resident`]): every weight buffer declared
    /// as one of them is taken filled instead of allocated, and
    /// `load_weights` copies only what was not. The caller binds the same
    /// checkpoint to both runtimes, as the same rank.
    pub fn load_over(
        manifest: &Verified,
        kernels_dir: &std::path::Path,
        gpu: usize,
        capacity: Option<Capacity>,
        topology: Option<&Topology>,
        host_weights: &crate::HostWeights,
        resident: Resident,
    ) -> Result<Runtime> {
        Self::load_from(manifest, kernels_dir, gpu, capacity, topology, host_weights, Some(resident))
    }

    fn load_from(
        manifest: &Verified,
        kernels_dir: &std::path::Path,
        gpu: usize,
        capacity: Option<Capacity>,
        topology: Option<&Topology>,
        host_weights: &crate::HostWeights,
        mut resident: Option<Resident>,
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
        if let Some(r) = &resident {
            if r.gpu != gpu {
                bail!(Api, "resident weights are on gpu {}; this runtime loads on gpu {gpu}", r.gpu);
            }
            if r.ranks != ranks {
                bail!(Api, "resident weights are rank {:?}'s; this runtime is rank {ranks:?}", r.ranks);
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
        // Exported buffers come from the virtual-memory API so peers can
        // map them; a peer array is an ordinary local buffer the runtime
        // fills; everything else is pool memory.
        let mut buffers = BTreeMap::new();
        let mut peers = BTreeMap::new();
        let (mut filled, mut kept) = (BTreeSet::new(), 0u64);
        for (name, b) in &manifest.buffers {
            let bytes = compile::shaped_bytes(&format!("buffer `{name}`"), &b.shape, b.dtype.bytes(), &vars_max)?;
            let buf = if let Some(r) = resident.as_mut().and_then(|r| r.take(name, b, bytes)) {
                filled.insert(name.clone());
                kept += bytes;
                r.on(&stream)
            } else if b.placement == Placement::Host {
                alloc_host(&stream, host_weights.acquire(name, b, bytes, &ctx)?)?
            } else if b.export {
                alloc_vmm(&stream, dev, bytes, &format!("buffer `{name}`"))?
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
        let tokens = match capacity.and_then(|c| c.tokens) {
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
                Some(aligned)
            }
            None => None,
        };
        let chunks = match tokens {
            Some(t) => chunks_for(&manifest, t, first_slots, chunk),
            None => fit_budget(&ctx, fixed_bytes, paged_bytes, slot_bytes * first_slots)? / chunk,
        };
        let chunks =
            u32::try_from(chunks).map_err(|_| Error::Manifest(format!("{chunks} chunks of state: too many")))?;

        let mut states = BTreeMap::new();
        for (name, s) in &manifest.states {
            if s.bytes_per_token == 0 && s.bytes_per_seq == 0 {
                let buf = alloc_vmm(&stream, dev, s.bytes, &format!("state `{name}`"))?;
                states.insert(name.clone(), buf);
            }
        }
        let t0 = std::time::Instant::now();
        let (pool, initial) = Pool::new(&manifest, chunk, chunks, first_slots as usize, tokens)?;
        let planned = t0.elapsed();
        let physical = Physical::create(dev, chunk as usize, chunks as usize)?;
        let created = t0.elapsed() - planned;
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
        let t1 = std::time::Instant::now();
        mapper.run(&initial)?;
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        tracing::debug!(
            "state budget {:.1} GiB in {chunks} chunks of {} MiB: {} pages of {page} tokens, {} sequence slots; planned in {planned:?}, created in {created:?}, mapped in {:?}",
            gib(chunks as u64 * chunk),
            chunk >> 20,
            pool.total(),
            pool.slots(),
            t1.elapsed(),
        );
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
            filled,
            kept,
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
            compare: Default::default(),
        };
        rt.zero_fresh(&initial)?;
        rt.stream.synchronize()?;
        Ok(rt)
    }

    /// Assemble every `weight` buffer from the checkpoint tensors its
    /// `bind` names, each segment one copy straight out of the tensor's
    /// bytes, wherever [`Tensors`] says they are: this process's memory
    /// (safetensors blobs) or memory this device reads from another
    /// process (a weight cache's buckets, mapped with [`Runtime::map`]).
    pub fn load_weights(&mut self, tensors: &dyn Tensors) -> Result<()> {
        if self.host_weights_ready && self.buffers.values().any(|b| b.host_weight().is_some()) {
            bail!(Api, "host weights are an immutable snapshot; use a new runtime and scope to reload");
        }
        self.ctx.bind_to_thread()?;
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        // Copies of a few MiB each are launch-latency bound on one stream.
        let lanes = (0..LOAD_LANES).map(|_| self.ctx.new_stream()).collect::<std::result::Result<Vec<_>, _>>()?;
        let mut planned = Vec::new();
        for (name, b) in &self.manifest.buffers {
            if b.kind != BufferKind::Weight || self.filled.contains(name) {
                continue;
            }
            let dst = &self.buffers[name];
            let copies =
                weights::plan(name, b, dst.bytes, |t| tensors.find(t), |group| self.ranks.get(group).copied())?;
            planned.push((dst, copies));
        }
        // Device copies go out first and run while a host weight is filled.
        let t0 = std::time::Instant::now();
        let (mut device, mut host, mut filled) = ((0u64, 0usize), (0u64, 0usize, 0f64), (0u64, 0f64));
        for (dst, copies) in planned.iter().filter(|(d, _)| d.host_weight().is_none()) {
            for c in copies {
                copy_to_device(&lanes[device.1 % LOAD_LANES], dst, c)?;
                device.1 += 1;
            }
            device.0 += dst.bytes;
        }
        for (dst, copies) in planned.iter() {
            let Some(h) = dst.host_weight() else { continue };
            let t = std::time::Instant::now();
            let mine = h.initialize(|bytes| copy_to_host(&self.stream, bytes, copies))?;
            let secs = t.elapsed().as_secs_f64();
            host = (host.0 + dst.bytes, host.1 + copies.len(), host.2 + secs);
            if mine {
                filled = (filled.0 + dst.bytes, filled.1 + secs);
            }
        }
        lanes.iter().try_for_each(|s| s.synchronize())?;
        self.stream.synchronize()?;
        tracing::info!(
            "gpu {} weights: device {:.1} GiB in {} copies ({:.1} GiB resident already), host {:.1} GiB in {} copies of which this rank filled {:.1} GiB in {:.1}s ({:.0} GiB/s); {:.1}s in all",
            self.gpu,
            gib(device.0),
            device.1,
            gib(self.kept),
            gib(host.0),
            host.1,
            gib(filled.0),
            filled.1,
            gib(filled.0) / filled.1.max(1e-9),
            t0.elapsed().as_secs_f64(),
        );
        self.host_weights_ready = true;
        self.filled
            .extend(self.manifest.buffers.iter().filter(|(_, b)| b.kind == BufferKind::Weight).map(|(n, _)| n.clone()));
        Ok(())
    }

    /// Bytes of weights this runtime took from a [`Resident`] instead of copying.
    pub fn kept_bytes(&self) -> u64 {
        self.kept
    }

    /// The device weight buffers, filled, out of a runtime that is otherwise
    /// dropped here: what [`Runtime::load_over`] on the same device takes
    /// instead of loading again. Buffers whose weights were never bound,
    /// and host-placed ones (shared through [`crate::HostWeights`]), stay
    /// behind.
    pub fn into_resident(mut self) -> Resident {
        let mut all = std::mem::take(&mut self.buffers);
        let buffers = self
            .filled
            .iter()
            .filter_map(|n| {
                let buf = all.remove(n)?;
                buf.host_weight().is_none().then(|| (n.clone(), (Identity::of(&self.manifest.buffers[n]), buf)))
            })
            .collect();
        Resident { gpu: self.gpu, ranks: self.ranks.clone(), buffers }
    }
}

/// Weight buffers that outlived their runtime, waiting for the next one
/// on the same device. A buffer is handed over when the next manifest
/// declares it the same way — type, shape, placement, export and the
/// tensors bound into it — under the same name; a kernel swap changes
/// none of that. What the next runtime does not take is freed with this.
pub struct Resident {
    gpu: usize,
    ranks: BTreeMap<String, u64>,
    buffers: BTreeMap<String, (Identity, DeviceBuf)>,
}

impl Resident {
    pub fn bytes(&self) -> u64 {
        self.buffers.values().map(|(_, b)| b.bytes).sum()
    }

    fn take(&mut self, name: &str, b: &Buffer, bytes: u64) -> Option<DeviceBuf> {
        let same = self.buffers.get(name).is_some_and(|(id, buf)| *id == Identity::of(b) && buf.bytes == bytes);
        same.then(|| self.buffers.remove(name)).flatten().map(|(_, buf)| buf)
    }
}

/// What decides that two manifests mean the same bytes by a buffer name.
#[derive(PartialEq, Eq)]
struct Identity {
    dtype: DType,
    shape: Vec<Dim>,
    kind: BufferKind,
    placement: Placement,
    export: bool,
    bind: Vec<Segment>,
}

impl Identity {
    fn of(b: &Buffer) -> Identity {
        Identity {
            dtype: b.dtype,
            shape: b.shape.clone(),
            kind: b.kind,
            placement: b.placement,
            export: b.export,
            bind: b.bind.clone(),
        }
    }
}

const LOAD_LANES: usize = 8;

/// One planned copy into a device buffer. Host bytes go up in one memcpy
/// when contiguous, else as one 2D copy; device bytes (another process's
/// allocation mapped here) are a device-to-device copy either way.
fn copy_to_device(stream: &Arc<CudaStream>, dst: &DeviceBuf, c: &weights::Copy) -> Result<()> {
    if let (Blob::Host(src), true) = (c.src, c.pitch == c.width) {
        let mut view = dst.view(c.dst as usize..(c.dst + c.width * c.rows) as usize)?;
        return Ok(stream.memcpy_htod(src, &mut view)?);
    }
    if let (Blob::Device { ptr, .. }, true) = (c.src, c.rows == 1 || c.pitch == c.width) {
        return copy_1d(stream.cu_stream(), dst.ptr + c.dst, ptr, c.width * c.rows);
    }
    copy_2d(
        stream.cu_stream(),
        (dst.ptr + c.dst, c.width, Space::Device),
        (c.src.ptr(), c.pitch, space(&c.src)),
        c.width,
        c.rows,
    )
}

/// The planned copies into a host weight (registered host memory every
/// rank maps). Host bytes are copied by the CPU; device bytes come down
/// over the stream, which is drained before the initializer returns.
fn copy_to_host(stream: &Arc<CudaStream>, dst: &mut [u8], copies: &[weights::Copy]) -> Result<()> {
    let base = dst.as_mut_ptr() as u64;
    for c in copies {
        match c.src {
            Blob::Host(src) => copy_rows(dst, c, src),
            Blob::Device { ptr, .. } => copy_2d(
                stream.cu_stream(),
                (base + c.dst, c.width, Space::Host),
                (ptr, c.pitch, Space::Device),
                c.width,
                c.rows,
            )?,
        }
    }
    Ok(stream.synchronize()?)
}

fn space(b: &Blob) -> Space {
    match b {
        Blob::Host(_) => Space::Host,
        Blob::Device { .. } => Space::Device,
    }
}

/// One copy of host bytes into a host buffer: a full-width segment is one
/// memcpy even for hundreds of millions of short rows; a rectangle keeps
/// its source pitch.
fn copy_rows(dst: &mut [u8], c: &weights::Copy, src: &[u8]) {
    let (width, dest) = (c.width as usize, c.dst as usize);
    if c.pitch == c.width {
        dst[dest..dest + src.len()].copy_from_slice(src);
    } else {
        for row in 0..c.rows as usize {
            let from = row * c.pitch as usize;
            dst[dest + row * width..dest + (row + 1) * width].copy_from_slice(&src[from..from + width]);
        }
    }
}

fn fmt_groups(t: &kern_manifest::types::Topology) -> String {
    t.groups.iter().map(|(g, n)| format!("{g}={n}")).collect::<Vec<_>>().join(", ")
}

/// The chunk the pooled states are backed in, from the smallest page or
/// slot object and the allocation granularity `g`.
fn chunk_size(m: &Manifest, page: u64, g: u64) -> u64 {
    let smallest = m
        .states
        .values()
        .filter_map(|s| match (s.bytes_per_token, s.bytes_per_seq) {
            (t, 0) if t > 0 => Some(t * page),
            (0, q) if q > 0 => Some(q),
            _ => None,
        })
        .min();
    chunk_for(smallest, g)
}

const CHUNK_MAX: u64 = 64 << 20;

/// A multiple of `g`, at most half the smallest object so an object spans
/// at least two chunks (one shared at a boundary is then one of many), at
/// most [`CHUNK_MAX`]. An object too small for even one granule per half
/// shares its chunk with many others whatever the chunk is, so the chunk
/// is then the largest: every driver call on the pool costs per chunk.
fn chunk_for(smallest: Option<u64>, g: u64) -> u64 {
    let cap = (CHUNK_MAX / g).max(1);
    let units = match smallest.map_or(0, |s| s / 2 / g) {
        0 => cap,
        n => n.min(cap),
    };
    units * g
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
mod chunk_tests {
    use super::chunk_for;

    const G: u64 = 2 << 20;

    #[test]
    fn half_the_smallest_object_in_granules_capped_at_64_mib() {
        assert_eq!(chunk_for(Some(48 << 20), G), 24 << 20);
        assert_eq!(chunk_for(Some(5 << 20), G), 2 << 20);
        assert_eq!(chunk_for(Some(1 << 30), G), 64 << 20);
    }

    #[test]
    fn an_object_below_two_granules_takes_the_largest_chunk() {
        assert_eq!(chunk_for(Some(1 << 10), G), 64 << 20);
        assert_eq!(chunk_for(Some(4 << 20), G), 2 << 20);
        assert_eq!(chunk_for(Some((4 << 20) - 1), G), 64 << 20);
        assert_eq!(chunk_for(None, G), 64 << 20);
    }
}

#[cfg(test)]
mod host_copy_tests {
    use super::*;

    fn copy(dst: &mut [u8], c: &weights::Copy) {
        let Blob::Host(src) = c.src else { unreachable!() };
        copy_rows(dst, c, src);
    }

    #[test]
    fn contiguous_and_strided_rectangles_preserve_destination_neighbors() {
        let contiguous = [90, 91, 1, 2, 3, 4, 5, 6, 92];
        let strided = [90, 1, 2, 80, 81, 3, 4, 82, 83, 5, 6, 84];
        for (src, pitch) in [(&contiguous[2..8], 2), (&strided[1..11], 4)] {
            let mut dst = [77; 12];
            copy(&mut dst, &weights::Copy { dst: 3, src: Blob::Host(src), width: 2, rows: 3, pitch });
            assert_eq!(dst, [77, 77, 77, 1, 2, 3, 4, 5, 6, 77, 77, 77]);
        }
    }

    #[test]
    fn mixed_segments_assemble_in_order() {
        let (a, b) = ([90, 1, 2, 3, 4, 91], [80, 5, 6, 81, 82, 7, 8, 83]);
        let copies = [
            weights::Copy { dst: 0, src: Blob::Host(&a[1..5]), width: 2, rows: 2, pitch: 2 },
            weights::Copy { dst: 4, src: Blob::Host(&b[1..7]), width: 2, rows: 2, pitch: 4 },
        ];
        let mut dst = [0; 8];
        copies.iter().for_each(|c| copy(&mut dst, c));
        assert_eq!(dst, [1, 2, 3, 4, 5, 6, 7, 8]);
    }
}
