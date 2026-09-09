//! Thin verifier-driven executor for kern manifests.
//!
//! The runtime knows nothing about models. It loads a verified manifest,
//! resolves each declared kernel against the cubins in a directory, allocates
//! every buffer/state, binds weight buffers by name from a safetensors blob,
//! and replays the program's call list. The only kernels it understands
//! natively are `extern:` ops (currently `extern:cublaslt_bf16_tn`).
//!
//! Names stop at load time: device pointers are static once buffers, states
//! and scratch are allocated, so [`Runtime::load`] lowers every program into
//! a flat launch list (see `compile`) whose slots are finished values or
//! var-indexed expressions. The name-keyed maps that remain exist only on
//! the caller API surface (`write_input("token_ids")`, `run("decode")`);
//! the execution path performs no name lookups.
//!
//! Same-name Triton kernels ship multiple constexpr instances with different
//! ABIs across modules; resolution picks the instance whose
//! `cuFuncGetParamInfo` layout matches the manifest's declared params — the
//! phase-2 ABI check doubles as instance selection.
//!
//! Every entry point binds the runtime's CUDA context to the calling
//! thread first, so one thread may drive several runtimes (a tray) and a
//! runtime may be loaded on one thread and driven from another.
//!
//! Two streams: every program and every pool copy runs on the compute
//! stream in order; the host tier's copies ([`Runtime::park`] out,
//! [`Runtime::wake`] in) run on a transfer stream, and the compute stream
//! never waits for them. The transfer stream starts each batch of copies
//! after everything the compute stream has enqueued; a parked checkpoint
//! stays held (its pages and slot out of the pool) until its copy has
//! landed, and a woken lease is a [`Waking`] until [`Runtime::awake`]
//! finds its copy landed, so no program can read pages still in flight.

mod chunks;
mod compile;
mod cubin;
mod device;
mod error;
mod exec;
mod harness;
mod host;
mod pages;
mod peers;
mod prefix;
pub mod profile;
pub mod values;
mod weights;

use std::collections::{BTreeMap, BTreeSet};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::Arc;
use std::thread::JoinHandle;

use cudarc::cublaslt::CudaBlasLT;
use cudarc::driver::{sys, CudaContext, CudaStream, PinnedHostSlice};
use kern_manifest::types::{BufferKind, Manifest, Provision, State};
use kern_manifest::Verified;

pub use chunks::{Kind, Remap};
use compile::{CompiledProgram, Dense};
pub use device::PeerHandle;
use device::{alloc, alloc_vmm, chunk_granularity, copy_2d, Arena, Blas, DeviceBuf, Mapper, Physical, Pinned, Share};
use error::{bail, cuda_check};
pub use error::{Error, Result};
pub use host::{Host, Parked};
pub use pages::{page_unit, Checkpoint, Copies, Denied, Lease, Pool, Pooled};
use peers::PeerSlot;
pub use prefix::{Chain, Hit, Kept, Prefix, Tier};

/// The CUDA API this binary binds (`13000` is 13.0): fixed by the `cudarc`
/// feature at build time, so a driver older than it fails to load the
/// libraries rather than at a random symbol later.
pub const CUDA_API: u32 = sys::CUDA_VERSION;

/// The host tier's block is handed out in these units.
const HOST_GRAIN: u64 = 1 << 16;

/// What the caller will hold in the pooled states at once: pages for
/// `tokens` tokens of every paged state and `seqs` sequences' slots of every
/// per-sequence state (plus the null slot and a spare). A number the caller
/// knows — its batch, its serving bound — never the manifest's var bounds,
/// which say what a step may address, not how many sequences live.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Capacity {
    /// Tokens of paged state; `None` takes what the device has left.
    pub tokens: Option<u64>,
    pub seqs: u64,
}

/// This rank's place in every group the manifest's `topology` declares.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Topology {
    pub groups: BTreeMap<String, GroupRank>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupRank {
    pub index: u64,
    pub size: u64,
}

impl Topology {
    /// A single group, e.g. `Topology::one("ep", 2, 4)`.
    pub fn one(group: &str, index: u64, size: u64) -> Topology {
        Topology { groups: BTreeMap::from([(group.to_string(), GroupRank { index, size })]) }
    }
}

pub struct Runtime {
    /// The manifest as loaded; verified, so nothing here checks it again.
    pub manifest: Verified,
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    /// The host tier's copies run here, off the compute stream.
    xfer: Arc<CudaStream>,
    /// The host tier: its accounting and its pinned block, once reserved.
    host: Option<(Arc<Host>, Pinned)>,
    /// Checkpoints whose park copies are still in flight: held here so
    /// their pages and slots stay out of the pool until the copy lands.
    parking: Vec<(Checkpoint, sys::CUevent)>,
    blt: CudaBlasLT,
    /// cuBLAS handle (with its own workspace) for the f32-result GEMM built-in.
    blas: Blas,
    /// What every `index_into` domain resolves against: the token slots a
    /// paged state's arena spans, the sequence slots a per-sequence one's.
    provision: Provision,
    /// Owner of the states' token slots: hands them out as leases.
    pool: Arc<pages::Pool>,
    /// The remap thread: runs the pool's plans off the serving thread.
    remaps: Remaps,
    remap_count: u64,
    /// Name-keyed because names are the caller API (`write_input`,
    /// `read_output`, weight binding); execution never looks these up —
    /// their device pointers are baked into `programs`.
    buffers: BTreeMap<String, DeviceBuf>,
    states: BTreeMap<String, DeviceBuf>,
    /// Persistent pinned staging, one per input buffer: H2D from pageable
    /// memory degrades to a synchronous driver-staged copy (tens of µs per
    /// call); through page-locked staging it is a true async DMA. The pinned
    /// slice's event guards reuse across steps.
    staging: BTreeMap<String, PinnedHostSlice<u8>>,
    /// Programs lowered to flat launch lists at load.
    programs: BTreeMap<String, CompiledProgram>,
    /// Owners of the impl-private scratch allocations whose pointers are
    /// baked into `programs`.
    #[allow(dead_code)]
    scratch: Vec<DeviceBuf>,
    /// Per kernel: the module each impl step resolved to (introspection).
    resolution: Vec<(String, Vec<String>)>,
    n_modules: usize,
    /// (program, dense var values) -> instantiated CUDA graph. Grid dims
    /// and scalar args are baked in at capture, so one program holds one
    /// graph per var assignment it was captured at (a batched decode keeps
    /// one per batch bucket).
    graphs: BTreeMap<(String, Dense), sys::CUgraphExec>,
    /// Debug override: [`Runtime::issue`] launches every program eagerly,
    /// whatever its manifest says.
    eager: bool,
    /// CUDA device ordinal, for the virtual-memory calls.
    gpu: usize,
    /// This rank's index per group (empty without a topology).
    ranks: BTreeMap<String, u64>,
    /// Every `peer` buffer, by name.
    peers: BTreeMap<String, PeerSlot>,
    /// Peer mappings kept alive as long as the addresses in `peers`.
    #[allow(dead_code)]
    imports: Vec<DeviceBuf>,
}

impl Drop for Runtime {
    fn drop(&mut self) {
        let _ = self.ctx.bind_to_thread();
        for exec in self.graphs.values() {
            unsafe { sys::cuGraphExecDestroy(*exec) };
        }
        // The thread owns the arenas: nothing on either stream may still
        // touch them when it unmaps.
        let _ = self.stream.synchronize();
        let _ = self.xfer.synchronize();
        for (_, ev) in self.parking.drain(..) {
            unsafe { sys::cuEventDestroy_v2(ev) };
        }
        let _ = self.remaps.jobs.send(Job::Stop);
        if let Some(t) = self.remaps.thread.take() {
            let _ = t.join();
        }
    }
}

/// What the remap thread is told: a plan to run once both streams have
/// passed `after` (recorded events, destroyed by the thread), or to stop.
enum Job {
    Run { plan: Remap, after: [u64; 2] },
    Stop,
}

/// The remap thread and the channels to it.
struct Remaps {
    jobs: Sender<Job>,
    done: Receiver<Result<Remap>>,
    thread: Option<JoinHandle<()>>,
}

/// Runs plans until told to stop, then lets the arenas and chunks go.
fn remap_thread(ctx: Arc<CudaContext>, mut mapper: Mapper, jobs: Receiver<Job>, done: Sender<Result<Remap>>) {
    if let Err(e) = ctx.bind_to_thread() {
        let _ = done.send(Err(e.into()));
        return;
    }
    while let Ok(job) = jobs.recv() {
        match job {
            Job::Run { plan, after } => {
                let evs = after.map(|e| e as sys::CUevent);
                let r = evs
                    .iter()
                    .try_for_each(|&ev| cuda_check(unsafe { sys::cuEventSynchronize(ev) }, "cuEventSynchronize"))
                    .and_then(|()| mapper.run(&plan));
                for ev in evs {
                    unsafe { sys::cuEventDestroy_v2(ev) };
                }
                if done.send(r.map(|()| plan)).is_err() {
                    break;
                }
            }
            Job::Stop => break,
        }
    }
    drop(mapper);
}

/// A fresh event recorded on `stream`; the caller destroys it.
fn record(stream: &CudaStream) -> Result<sys::CUevent> {
    let mut ev: sys::CUevent = std::ptr::null_mut();
    cuda_check(
        unsafe { sys::cuEventCreate(&mut ev, sys::CUevent_flags::CU_EVENT_DISABLE_TIMING as u32) },
        "cuEventCreate",
    )?;
    cuda_check(unsafe { sys::cuEventRecord(ev, stream.cu_stream()) }, "cuEventRecord")?;
    Ok(ev)
}

/// `stream` waits for `ev`, which is then destroyed.
fn wait_then_destroy(stream: &CudaStream, ev: sys::CUevent) -> Result<()> {
    let r = cuda_check(unsafe { sys::cuStreamWaitEvent(stream.cu_stream(), ev, 0) }, "cuStreamWaitEvent");
    unsafe { sys::cuEventDestroy_v2(ev) };
    r
}

/// Whether everything before `ev` on its stream has completed.
fn landed(ev: sys::CUevent) -> Result<bool> {
    match unsafe { sys::cuEventQuery(ev) } {
        sys::CUresult::CUDA_SUCCESS => Ok(true),
        sys::CUresult::CUDA_ERROR_NOT_READY => Ok(false),
        r => cuda_check(r, "cuEventQuery").map(|_| true),
    }
}

/// A checkpoint with host room found for it and nothing copied yet: what
/// [`Runtime::room`] hands out and [`Runtime::park`] spends. Dropping it
/// frees the room and drops the checkpoint.
pub struct Room {
    cp: Checkpoint,
    parked: Parked,
    plan: host::Park,
}

impl Room {
    /// Give the room back and keep the checkpoint.
    pub fn into_checkpoint(self) -> Checkpoint {
        self.cp
    }
}

/// A lease being woken: its pages are taken, their bytes still on the way
/// in. [`Runtime::awake`] turns it into the lease once they have landed;
/// dropping it earlier waits for them first, so the pages never return to
/// the pool with a copy still writing them.
pub struct Waking {
    lease: Option<Lease>,
    event: sys::CUevent,
}

// The event is only ever queried through the runtime that recorded it,
// on whatever thread drives that runtime; the runtime itself moves
// between threads the same way.
unsafe impl Send for Waking {}

impl Waking {
    /// Positions the lease will hold filled.
    pub fn prefix(&self) -> usize {
        self.lease.as_ref().map_or(0, Lease::prefix)
    }
}

impl Drop for Waking {
    fn drop(&mut self) {
        if !self.event.is_null() {
            unsafe {
                sys::cuEventSynchronize(self.event);
                sys::cuEventDestroy_v2(self.event);
            }
        }
    }
}

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
            let buf = if b.export {
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
                (aligned * token_bytes).div_ceil(chunk) + (first_slots * slot_bytes).div_ceil(chunk)
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
        let (jobs, job_rx) = mpsc::channel();
        let (done_tx, done) = mpsc::channel();
        let thread = std::thread::Builder::new()
            .name("kern-remap".into())
            .spawn({
                let ctx = Arc::clone(&ctx);
                move || remap_thread(ctx, mapper, job_rx, done_tx)
            })
            .map_err(|e| Error::Cuda(format!("spawning the remap thread: {e}")))?;
        let mut rt = Runtime {
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
            remaps: Remaps { jobs, done, thread: Some(thread) },
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

    pub fn module_count(&self) -> usize {
        self.n_modules
    }

    /// (name, class, allocated bytes) for every buffer.
    pub fn buffer_sizes(&self) -> Vec<(&str, BufferKind, u64)> {
        self.manifest.buffers.iter().map(|(n, b)| (n.as_str(), b.kind, self.buffers[n].bytes)).collect()
    }

    /// (name, declaration, allocated bytes) for every state.
    pub fn state_sizes(&self) -> Vec<(&str, &State, u64)> {
        self.manifest.states.iter().map(|(n, s)| (n.as_str(), s, self.states[n].bytes)).collect()
    }

    /// Per kernel: the module each impl step resolved to, in step order.
    pub fn op_resolution(&self) -> Vec<(String, Vec<String>)> {
        self.resolution.clone()
    }

    /// Assemble every `weight` buffer from the checkpoint tensors its
    /// `bind` names, out of one or more safetensors blobs (the model's
    /// shards, a draft's next to them). Only headers are parsed; each
    /// segment is one copy straight out of the blob. A tensor name that
    /// appears in more than one blob is ambiguous and refused.
    pub fn load_weights(&mut self, blobs: &[&[u8]]) -> Result<()> {
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
            for c in weights::plan(name, b, dst.bytes, lookup)? {
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
        Ok(())
    }

    /// Check `data` (a prefix of buffer `name`) against the buffer's declared
    /// domain, if any, at the given var values. Symbol-dependent bounds
    /// need `vars`; pass the values the next run will use.
    pub fn check_domain(&self, name: &str, data: &[u8], vars: &BTreeMap<String, u64>) -> Result<()> {
        let Some(b) = self.manifest.buffers.get(name) else {
            bail!(Api, "no buffer `{name}`");
        };
        let Some(d) = &b.domain else { return Ok(()) };
        let r = d
            .resolve(&self.manifest, vars, &self.provision)
            .map_err(|e| Error::Domain(format!("buffer `{name}`: {e}")))?;
        let vals = values::to_f64(b.dtype, data);
        let fmt_bound = |v: Option<f64>| v.map_or("∞".to_string(), |x| format!("{x}"));
        for (i, &v) in vals.iter().enumerate() {
            if !r.contains(v) {
                bail!(Domain, "buffer `{name}`[{i}] = {v} outside declared [{}, {}]", fmt_bound(r.lo), fmt_bound(r.hi));
            }
            if r.monotone && i > 0 && v < vals[i - 1] {
                bail!(Domain, "buffer `{name}` is declared monotone but [{i}] = {v} < [{}] = {}", i - 1, vals[i - 1]);
            }
        }
        Ok(())
    }

    /// Write an input buffer. The domain check needs the var values the
    /// next run will use; `write_input` checks against var upper bounds
    /// (the loosest valid reading), `write_input_at` against exact values.
    pub fn write_input(&mut self, name: &str, data: &[u8]) -> Result<()> {
        let vars_max: BTreeMap<_, _> = self.manifest.vars.iter().map(|(s, v)| (s.clone(), v.max)).collect();
        self.write_input_at(name, data, &vars_max)
    }

    pub fn write_input_at(&mut self, name: &str, data: &[u8], vars: &BTreeMap<String, u64>) -> Result<()> {
        self.ctx.bind_to_thread()?;
        let Some(b) = self.manifest.buffers.get(name) else {
            bail!(Api, "no buffer `{name}`");
        };
        if b.kind != BufferKind::Input {
            bail!(Api, "buffer `{name}` is {}, not input", b.kind);
        }
        if data.len() as u64 > self.buffers[name].bytes {
            bail!(Api, "input `{name}`: got {} bytes, buffer is {}", data.len(), self.buffers[name].bytes);
        }
        self.check_domain(name, data, vars)?;
        let dst = self.buffers.get_mut(name).unwrap();
        let pinned = self.staging.get_mut(name).unwrap();
        // Waits on the pinned slice's event: the previous step's DMA from
        // this staging must finish before we overwrite it. A prefix write
        // (variable-length inputs) still DMAs the whole buffer — the stale
        // tail is never read, grids are bounded by the vars.
        pinned.as_mut_slice()?[..data.len()].copy_from_slice(data);
        self.stream.memcpy_htod(pinned, dst)?;
        Ok(())
    }

    pub fn read_output(&self, name: &str) -> Result<Vec<u8>> {
        self.ctx.bind_to_thread()?;
        let Some(b) = self.manifest.buffers.get(name) else {
            bail!(Api, "no buffer `{name}`");
        };
        if b.kind != BufferKind::Output {
            bail!(Api, "buffer `{name}` is {}, not output", b.kind);
        }
        Ok(self.stream.clone_dtoh(&self.buffers[name])?)
    }

    // ---- token slots: the states are addressed only through leases.

    /// Lease the pages `tokens` slots need and, when the manifest has
    /// per-sequence states, a slot in each of them, all or nothing
    /// ([`Error::Denied`] says why not). The lease is the only source of
    /// `slot_mapping` values, page-table rows and line indices for the
    /// sequence; the slot is zeroed on the stream (a fresh sequence's
    /// recurrent state), and everything returns when the lease drops.
    pub fn lease(&mut self, tokens: usize) -> Result<Lease> {
        self.ctx.bind_to_thread()?;
        self.poll()?;
        let lease = match self.pool.lease(tokens) {
            Ok(l) => l,
            Err(d) => return self.denied(d),
        };
        for (name, st) in &self.manifest.states {
            let Some(range) = lease.seq_bytes(st.bytes_per_seq).filter(|_| st.is_per_seq()) else { continue };
            let s = self.states.get_mut(name).unwrap();
            let mut view = s.view(range)?;
            self.stream.memset_zeros(&mut view)?;
        }
        Ok(lease)
    }

    /// A slot in each per-sequence state and no pages, zeroed on the
    /// stream: this rank's share of a sequence whose positions live on
    /// another rank (a tensor-parallel peer holds a slice of every row's
    /// recurrent state). Empty when the manifest has no per-sequence state.
    pub fn lease_slot(&mut self) -> Result<Lease> {
        self.ctx.bind_to_thread()?;
        self.poll()?;
        let lease = match self.pool.lease_slot() {
            Ok(l) => l,
            Err(d) => return self.denied(d),
        };
        for (name, st) in &self.manifest.states {
            let Some(range) = lease.seq_bytes(st.bytes_per_seq).filter(|_| st.is_per_seq()) else { continue };
            let s = self.states.get_mut(name).unwrap();
            let mut view = s.view(range)?;
            self.stream.memset_zeros(&mut view)?;
        }
        Ok(lease)
    }

    /// The first `len` tokens of `lease` as a [`Checkpoint`] its
    /// sequence keeps running past: pages shared, the per-sequence state
    /// copied into a fresh slot ([`Error::Denied`] when none is free).
    pub fn checkpoint(&mut self, lease: &mut Lease, len: usize) -> Result<Checkpoint> {
        self.ctx.bind_to_thread()?;
        self.poll()?;
        let (cp, copies) = match self.pool.checkpoint(lease, len) {
            Ok(x) => x,
            Err(d) => return self.denied(d),
        };
        self.copy(&copies)?;
        Ok(cp)
    }

    /// The first `len` tokens of a finished sequence as a [`Checkpoint`]:
    /// its pages past `len` return, its state slot moves over as it is.
    pub fn retire(&mut self, lease: Lease, len: usize) -> Checkpoint {
        self.pool.retire(lease, len)
    }

    /// A sequence continuing from the first `len` tokens of `cp` with room
    /// for `tokens`: shares those pages (copying the one `len` ends
    /// inside), copies the state into a fresh slot, and hands out a lease
    /// that names positions from `len` on. `len` is the checkpoint's own
    /// length, or any whole number of its pages when it holds no state.
    /// [`Error::Denied`] as for [`Runtime::lease`].
    pub fn lease_from(&mut self, cp: &Checkpoint, len: usize, tokens: usize) -> Result<Lease> {
        self.ctx.bind_to_thread()?;
        self.poll()?;
        let (lease, copies) = match self.pool.restore(cp, len, tokens) {
            Ok(x) => x,
            Err(d) => return self.denied(d),
        };
        self.copy(&copies)?;
        Ok(lease)
    }

    /// A sequence branched off the first `len` tokens of `parent`, which
    /// keeps running: the whole pages shared, the page `len` ends inside
    /// copied, the parent's state copied into a fresh slot. With a
    /// recurrent state `len` is the parent's position — the state is the
    /// parent's as of now. The lease names positions from `len` on.
    pub fn fork(&mut self, parent: &mut Lease, len: usize, tokens: usize) -> Result<Lease> {
        self.ctx.bind_to_thread()?;
        self.poll()?;
        let (lease, copies) = match self.pool.fork(parent, len, tokens) {
            Ok(x) => x,
            Err(d) => return self.denied(d),
        };
        self.copy(&copies)?;
        Ok(lease)
    }

    // ---- the host tier: checkpoints parked in pinned DRAM.

    /// Reserve `bytes` of page-locked host memory for parked checkpoints
    /// (once; ~100 ms per GiB on GB300).
    pub fn reserve_host(&mut self, bytes: u64) -> Result<()> {
        self.ctx.bind_to_thread()?;
        if self.host.is_some() {
            bail!(Api, "the host tier is reserved already");
        }
        let pinned = Pinned::alloc(bytes, self.gpu as i32)?;
        self.host = Some((Arc::new(Host::new(bytes, HOST_GRAIN)), pinned));
        Ok(())
    }

    /// (bytes used, bytes reserved) of the host tier, when there is one.
    pub fn host_tier(&self) -> Option<(u64, u64)> {
        self.host.as_ref().map(|(h, p)| (h.used(), p.bytes()))
    }

    /// Bytes per host page (every paged state's page back to back) and per
    /// host slot, and each pooled state's offset inside its one.
    fn host_layout(&self) -> (u64, u64, Vec<u64>) {
        let (mut page, mut slot) = (0u64, 0u64);
        let offsets = self
            .pool
            .pooled()
            .iter()
            .map(|a| match a.kind {
                Kind::Page => {
                    page += a.object;
                    page - a.object
                }
                Kind::Slot => {
                    slot += a.object;
                    slot - a.object
                }
            })
            .collect();
        (page, slot, offsets)
    }

    /// Host room for `cp` — its pages not parked already (an earlier turn
    /// of the same session shares them there) and its slot — with nothing
    /// copied yet: the first half of a park, so a caller parking several
    /// checkpoints as one unit can find room for all of them before any
    /// byte moves. `Err(cp)` hands the checkpoint back when the block
    /// cannot hold it. Dropping a [`Room`] frees the room and the
    /// checkpoint with it; [`Room::into_checkpoint`] keeps the checkpoint.
    pub fn room(&mut self, cp: Checkpoint) -> Result<std::result::Result<Room, Checkpoint>> {
        self.ctx.bind_to_thread()?;
        self.poll()?;
        let Some((host, _)) = &self.host else {
            bail!(Api, "no host tier: reserve_host first");
        };
        let (page_bytes, slot_bytes, _) = self.host_layout();
        match host.park(&cp.nodes(), page_bytes, cp.seq_slot().map(|s| (s, slot_bytes)), cp.tokens()) {
            Ok((parked, plan)) => Ok(Ok(Room { cp, parked, plan })),
            Err(Denied::HostFull) => Ok(Err(cp)),
            Err(d) => Err(Error::Denied(d)),
        }
    }

    /// Copy a checkpoint into the room found for it. The checkpoint is
    /// held until the copies, on the transfer stream, have landed; then
    /// its device pages and slot return.
    pub fn park(&mut self, room: Room) -> Result<Parked> {
        self.ctx.bind_to_thread()?;
        let Room { cp, parked, plan } = room;
        self.transfer(&plan.pages, plan.slot, true)?;
        self.parking.push((cp, record(&self.xfer)?));
        Ok(parked)
    }

    /// A sequence continuing from the first `len` tokens of a parked
    /// checkpoint with room for `tokens`: fresh pages with those tokens'
    /// pages copied back in, a fresh slot with its state when `len` is the
    /// whole checkpoint (a parked state is usable at its length only). A
    /// slot-only checkpoint wakes to a slot-only lease. The copies run on
    /// the transfer stream; [`Runtime::awake`] hands out the lease once
    /// they have landed.
    pub fn wake(&mut self, p: &Parked, len: usize, tokens: usize) -> Result<Waking> {
        self.ctx.bind_to_thread()?;
        self.poll()?;
        if self.host.is_none() {
            bail!(Api, "no host tier: reserve_host first");
        }
        let unit = self.pool.unit() as usize;
        if len == 0 || len > p.tokens() || (len != p.tokens() && (p.has_slot() || !len.is_multiple_of(unit))) {
            bail!(Api, "waking {len} tokens of a parked checkpoint of {} ({p:?})", p.tokens());
        }
        let lease = match if p.paged() { self.pool.wake(len, tokens) } else { self.pool.wake_slot(len) } {
            Ok(l) => l,
            Err(d) => return self.denied(d),
        };
        let n = if p.paged() { len.div_ceil(unit) } else { 0 };
        let pairs: Vec<(i32, u64)> = lease.page_ids()[..n].iter().copied().zip(p.pages(n)).collect();
        let slot = if len == p.tokens() { lease.seq_slot().zip(p.slot()) } else { None };
        self.transfer(&pairs, slot, false)?;
        Ok(Waking { lease: Some(lease), event: record(&self.xfer)? })
    }

    /// Whether a wake's copies have landed. Does not block.
    pub fn landed(&self, w: &Waking) -> Result<bool> {
        self.ctx.bind_to_thread()?;
        landed(w.event)
    }

    /// The lease of a wake whose copies have landed; `Err(w)` while they
    /// are still in flight. Does not block.
    pub fn awake(&self, mut w: Waking) -> Result<std::result::Result<Lease, Waking>> {
        self.ctx.bind_to_thread()?;
        if !landed(w.event)? {
            return Ok(Err(w));
        }
        unsafe { sys::cuEventDestroy_v2(w.event) };
        w.event = std::ptr::null_mut();
        Ok(Ok(w.lease.take().expect("a waking lease")))
    }

    /// (device page, host offset) pages and a (device slot, host offset)
    /// slot between the pooled states and the host block, on the transfer
    /// stream after everything the compute stream has enqueued: one
    /// strided copy per run of consecutive pages per state.
    fn transfer(&self, pages: &[(i32, u64)], slot: Option<(i32, u64)>, to_host: bool) -> Result<()> {
        let Some((_, pinned)) = &self.host else {
            bail!(Api, "no host tier: reserve_host first");
        };
        let base = pinned.ptr();
        let (page_bytes, slot_bytes, offsets) = self.host_layout();
        wait_then_destroy(&self.xfer, record(&self.stream)?)?;
        let stream = self.xfer.cu_stream();
        for (a, ar) in self.pool.pooled().iter().enumerate() {
            let st = &self.states[&ar.state];
            match ar.kind {
                Kind::Page => {
                    for (p, o, n) in host::runs(pages, page_bytes) {
                        let dev = st.ptr + p as u64 * ar.object;
                        let hst = base + o + offsets[a];
                        copy_2d(stream, (dev, ar.object), (hst, page_bytes), ar.object, n as u64, to_host)?;
                    }
                }
                Kind::Slot => {
                    if let Some((s, o)) = slot {
                        let dev = st.ptr + s as u64 * ar.object;
                        let hst = base + o + offsets[a];
                        copy_2d(stream, (dev, ar.object), (hst, slot_bytes), ar.object, 1, to_host)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Run a pool decision's byte moves on the stream: whole pages of every
    /// paged state, whole slots of every per-sequence state.
    fn copy(&mut self, c: &Copies) -> Result<()> {
        let unit = self.pool.unit();
        for (name, st) in &self.manifest.states {
            let s = &self.states[name];
            let moves: Vec<(u64, u64, u64)> = if st.is_per_seq() {
                c.slot.iter().map(|&(a, b)| (a as u64, b as u64, st.bytes_per_seq)).collect()
            } else if st.bytes_per_token > 0 {
                c.pages.iter().map(|&(a, b)| (a as u64, b as u64, unit * st.bytes_per_token)).collect()
            } else {
                Vec::new()
            };
            for (a, b, bytes) in moves {
                let src = s.view((a * bytes) as usize..((a + 1) * bytes) as usize)?;
                let mut dst = s.view((b * bytes) as usize..((b + 1) * bytes) as usize)?;
                self.stream.memcpy_dtod(&src, &mut dst)?;
            }
        }
        Ok(())
    }

    /// A denial, after handing the remap it may have planned to the thread.
    fn denied<T>(&mut self, d: Denied) -> Result<T> {
        self.poll()?;
        Err(d.into())
    }

    /// Land the remaps the thread has run — their fresh chunks zeroed on
    /// the stream, so a page or a slot comes out of the pool as it did at
    /// load — and hand the thread the plan the pool has pending: it runs
    /// once the stream has passed everything enqueued so far.
    fn poll(&mut self) -> Result<()> {
        let mut parking = std::mem::take(&mut self.parking);
        for (cp, ev) in parking.drain(..) {
            if landed(ev)? {
                unsafe { sys::cuEventDestroy_v2(ev) };
                drop(cp);
            } else {
                self.parking.push((cp, ev));
            }
        }
        loop {
            match self.remaps.done.try_recv() {
                Ok(Ok(plan)) => {
                    self.zero_fresh(&plan)?;
                    self.pool.complete(plan);
                }
                Ok(Err(e)) => return Err(e),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => bail!(Cuda, "the remap thread is gone"),
            }
        }
        if let Some(plan) = self.pool.take_pending() {
            let after = [record(&self.stream)? as u64, record(&self.xfer)? as u64];
            self.remap_count += 1;
            tracing::debug!(unmap = plan.unmap.len(), map = plan.map.len(), "remap");
            if self.remaps.jobs.send(Job::Run { plan, after }).is_err() {
                bail!(Cuda, "the remap thread is gone");
            }
        }
        Ok(())
    }

    /// Zero every chunk `plan` mapped.
    fn zero_fresh(&mut self, plan: &Remap) -> Result<()> {
        let chunk = self.pool.chunk() as usize;
        for &(a, p, _) in &plan.map {
            let s = &self.states[&self.pool.pooled()[a].state];
            let mut v = s.view(p * chunk..(p + 1) * chunk)?;
            self.stream.memset_zeros(&mut v)?;
        }
        Ok(())
    }

    /// Remaps planned so far.
    pub fn remaps(&self) -> u64 {
        self.remap_count
    }

    /// Whether the manifest has per-sequence states (sequence slots at all).
    pub fn has_seq_state(&self) -> bool {
        self.pool.has_slots()
    }

    /// Sequence slots that exist now in every per-sequence state, slot 0
    /// among them (0 without one).
    pub fn seq_slots(&self) -> usize {
        self.pool.slots()
    }

    /// Sequence slots held by a lease or a checkpoint.
    pub fn seq_slots_used(&self) -> usize {
        self.pool.slots_used()
    }

    /// The line-table inputs (`index_into` a per-sequence state, shaped
    /// `[lines, seqs]` or `[lines, seqs, w]`), e.g. a hybrid model's
    /// `gdn.line_index`.
    pub fn seq_tables(&self) -> impl Iterator<Item = &str> {
        self.pool.seq_tables()
    }

    /// Pages the states hold in total.
    pub fn pages_total(&self) -> usize {
        self.pool.total()
    }

    /// Pages currently leased.
    pub fn pages_used(&self) -> usize {
        self.pool.used()
    }

    /// Longest sequence one page-table row can address (whole pages).
    pub fn max_seq_tokens(&self) -> usize {
        self.pool.max_seq_tokens()
    }

    /// The page-table inputs (`index_into` a state, constant row width),
    /// e.g. `block_table` and a speculative manifest's `draft_block_table`.
    pub fn page_tables(&self) -> impl Iterator<Item = &str> {
        self.pool.tables()
    }

    /// Token slots the paged states hold now (whole pages).
    pub fn capacity(&self) -> u64 {
        self.pool.total() as u64 * self.pool.unit()
    }

    /// The bounds `index_into` domains resolve against.
    pub fn provision(&self) -> Provision {
        self.provision
    }

    /// Page unit in tokens (1 if no state is paged).
    pub fn page(&self) -> u64 {
        self.pool.unit()
    }

    /// Wait for everything enqueued on both streams.
    pub fn synchronize(&self) -> Result<()> {
        self.ctx.bind_to_thread()?;
        self.stream.synchronize()?;
        self.xfer.synchronize()?;
        Ok(())
    }
}

fn fmt_groups(t: &kern_manifest::types::Topology) -> String {
    t.groups.iter().map(|(g, n)| format!("{g}={n}")).collect::<Vec<_>>().join(", ")
}

/// Device memory left untouched when the states are fitted to the device:
/// the driver's own allocations after load (captured graphs, module
/// lazy-loading, cuBLASLt algorithm state) and a margin for a neighbour.
pub const HEADROOM: u64 = 1 << 30;

/// Tokens one sequence of `m` can reach — the narrowest page table's row,
/// in whole pages — or `None` when nothing is paged per token. What a
/// single-sequence caller wants as its state capacity.
pub fn seq_capacity(m: &Manifest) -> Option<u64> {
    pages::row_tokens(m, page_unit(m))
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
