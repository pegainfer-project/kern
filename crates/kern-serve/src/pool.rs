//! The pooled tier's client: one Mooncake Store client per rank process,
//! moving bytes between this rank's pooled states and the pool on threads
//! of its own.
//!
//! The scheduler decides what to ask the pool ([`Act::Probe`], [`Act::Put`],
//! [`Act::Get`], [`Act::Mark`], `ranks.rs`) and never calls it: every act
//! is handed here by the rank's executor with the device [`Mark`] the
//! transfer must wait for, since a put reads pages the acts before it copy
//! into and a get lands in a slot the acts before it zero. RDMA bypasses
//! the compute stream, so the wait is the only ordering there is: a worker
//! polls the mark's pinned word (never a CUDA call on a side thread) and
//! then moves the bytes. Three workers, each a queue in act order: probes,
//! gets, and puts with their marks, so a long put never delays the answer
//! a waiting request needs.
//!
//! Every answer is a flag per key or object ([`Landed`]), collected by
//! [`Client::landed`] into the rank's next reply. The pool is never
//! trusted: a get whose stored size is not the object's is a miss, a put
//! the pool reports failed is not marked, and a client error puts the
//! tier in a cooldown during which every act fails at once, so a dead
//! master costs one timeout rather than one per request.
//!
//! Registration happens once at open, on the thread that drives the
//! device: every pooled state's allocation, whole. Mooncake's own CUDA
//! calls (the GPUDirect flush check) happen there and nowhere else.

#![allow(unsafe_code)]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use kern_runtime::{Mark, Span};
use mooncake_store::{MooncakeStore, ReplicateConfig};
use tracing::{debug, info, warn};

use kern_pool::Object;

use crate::ranks::Landed;

#[derive(Debug, Clone)]
pub struct Config {
    /// The master, `host:port`.
    pub master: String,
    /// The transfer engine's metadata server (`http://host:port/metadata`,
    /// or `P2PHANDSHAKE`).
    pub metadata: String,
    /// The RDMA device the transfer engine binds.
    pub device: String,
    /// This client's name in the pool, `host:port`, unique per process.
    pub hostname: String,
    /// The segment this tray's holder mounted, which a put prefers.
    pub segment: Option<String>,
    /// `libmooncake_store.so`, when not on the loader's path.
    pub library: Option<PathBuf>,
    /// How long every act fails outright after a client error.
    pub cooldown: Duration,
}

/// The local buffer Mooncake keeps for its own small transfers.
const LOCAL_BUFFER: u64 = 64 << 20;

/// Where a pooled object's bytes live on this rank: one range per state.
pub type Buffers = Vec<(u64, usize)>;

enum Job {
    Probe { id: u64, keys: Vec<String> },
    Put { id: u64, after: Mark, objects: Vec<(String, Buffers)> },
    Get { id: u64, after: Mark, objects: Vec<(String, Buffers)> },
    Mark { id: u64, key: String },
}

impl Job {
    fn id(&self) -> u64 {
        match self {
            Job::Probe { id, .. } | Job::Put { id, .. } | Job::Get { id, .. } | Job::Mark { id, .. } => *id,
        }
    }

    fn keys(&self) -> usize {
        match self {
            Job::Probe { keys, .. } => keys.len(),
            Job::Put { objects, .. } | Job::Get { objects, .. } => objects.len(),
            Job::Mark { .. } => 1,
        }
    }
}

/// What every worker shares: the store and the cooldown.
struct Shared {
    store: MooncakeStore,
    prefer: Option<ReplicateConfig>,
    cooldown: Duration,
    /// When the last client error happened, while its cooldown runs.
    failed: Mutex<Option<Instant>>,
    done: Mutex<Sender<Landed>>,
}

impl Shared {
    fn cooling(&self) -> bool {
        let mut f = self.failed.lock().expect("not poisoned");
        match *f {
            Some(t) if t.elapsed() < self.cooldown => true,
            Some(_) => {
                *f = None;
                info!("pool: cooldown over");
                false
            }
            None => false,
        }
    }

    fn fail(&self, what: &str, e: &dyn std::fmt::Display) {
        warn!(%what, error = %e, cooldown_s = self.cooldown.as_secs(), "pool: client error");
        *self.failed.lock().expect("not poisoned") = Some(Instant::now());
    }

    fn report(&self, id: u64, ok: Vec<bool>) {
        let _ = self.done.lock().expect("not poisoned").send(Landed { id, ok });
    }

    /// Run one job and report its flags.
    fn run(&self, job: Job) {
        let (id, n) = (job.id(), job.keys());
        if self.cooling() {
            return self.report(id, vec![false; n]);
        }
        let t0 = Instant::now();
        let ok = match job {
            Job::Probe { keys, .. } => {
                let refs: Vec<&str> = keys.iter().map(String::as_str).collect();
                match self.store.batch_is_exist(&refs) {
                    Ok(v) => v,
                    Err(e) => {
                        self.fail("probe", &e);
                        vec![false; n]
                    }
                }
            }
            Job::Mark { key, .. } => match self.store.put(&key, &[1], self.prefer.as_ref()) {
                Ok(()) => vec![true],
                Err(e) => {
                    self.fail("mark", &e);
                    vec![false]
                }
            },
            Job::Put { after, objects, .. } => match after.wait() {
                Ok(()) => self.put(&objects),
                Err(e) => {
                    self.fail("put", &e);
                    vec![false; n]
                }
            },
            Job::Get { after, objects, .. } => match after.wait() {
                Ok(()) => self.get(&objects),
                Err(e) => {
                    self.fail("get", &e);
                    vec![false; n]
                }
            },
        };
        debug!(
            id,
            keys = n,
            landed = ok.iter().filter(|&&b| b).count(),
            ms = t0.elapsed().as_millis() as u64,
            "pool: act"
        );
        self.report(id, ok);
    }

    /// An object with no bytes on this rank (a stage holding none of its
    /// states) is landed at once.
    fn put(&self, objects: &[(String, Buffers)]) -> Vec<bool> {
        let (keys, bufs) = flatten(objects);
        let bufs: Vec<&[(*mut std::ffi::c_void, usize)]> = bufs.iter().map(Vec::as_slice).collect();
        let got = match keys.is_empty() {
            true => Ok(Vec::new()),
            false => unsafe { self.store.batch_put_from_multi_buffers(&keys, &bufs, self.prefer.as_ref()) },
        };
        match got {
            Ok(rcs) => {
                refused("put", &rcs);
                spread(objects, rcs.iter().map(|&rc| rc == 0))
            }
            Err(e) => {
                self.fail("put", &e);
                vec![false; objects.len()]
            }
        }
    }

    /// An object landed when the pool held exactly its size.
    fn get(&self, objects: &[(String, Buffers)]) -> Vec<bool> {
        let (keys, bufs) = flatten(objects);
        let bufs: Vec<&[(*mut std::ffi::c_void, usize)]> = bufs.iter().map(Vec::as_slice).collect();
        let got = match keys.is_empty() {
            true => Ok(Vec::new()),
            false => unsafe { self.store.batch_get_into_multi_buffers(&keys, &bufs, true) },
        };
        match got {
            Ok(sizes) => {
                refused("get", &sizes.iter().map(|&n| i32::from(n < 0)).collect::<Vec<_>>());
                let want = objects
                    .iter()
                    .filter(|(_, b)| !b.is_empty())
                    .map(|(_, b)| b.iter().map(|&(_, n)| n as u64).sum::<u64>());
                spread(objects, sizes.iter().zip(want).map(|(&got, want)| got >= 0 && got as u64 == want))
            }
            Err(e) => {
                self.fail("get", &e);
                vec![false; objects.len()]
            }
        }
    }
}

/// A batch the pool answered object by object: the refusals, with the
/// first code (the master's log says why, say insufficient space).
fn refused(what: &str, rcs: &[i32]) {
    let failed = rcs.iter().filter(|&&rc| rc != 0).count();
    if failed > 0 {
        debug!(
            what,
            failed,
            of = rcs.len(),
            rc = rcs.iter().find(|&&rc| rc != 0).copied().unwrap_or(0),
            "pool refused"
        );
    }
}

/// `ok`, one flag per object that has bytes here, as one flag per object.
fn spread(objects: &[(String, Buffers)], mut ok: impl Iterator<Item = bool>) -> Vec<bool> {
    objects.iter().map(|(_, b)| b.is_empty() || ok.next().unwrap_or(false)).collect()
}

/// The objects with bytes here, as the batch calls take them.
#[allow(clippy::type_complexity)]
fn flatten(objects: &[(String, Buffers)]) -> (Vec<&str>, Vec<Vec<(*mut std::ffi::c_void, usize)>>) {
    let with = objects.iter().filter(|(_, b)| !b.is_empty());
    let keys = with.clone().map(|(k, _)| k.as_str()).collect();
    let bufs = with.map(|(_, b)| b.iter().map(|&(p, n)| (p as *mut std::ffi::c_void, n)).collect()).collect();
    (keys, bufs)
}

/// This rank's client: its pooled states registered, three workers.
pub struct Client {
    probes: Sender<Job>,
    gets: Sender<Job>,
    puts: Sender<Job>,
    done: Receiver<Landed>,
    spans: BTreeMap<String, Span>,
}

impl Client {
    /// Connect to the pool and register every span, whole. Called on the
    /// thread that drives the device, once it is loaded.
    pub fn open(cfg: &Config, spans: Vec<Span>) -> Result<Client> {
        if let Some(lib) = &cfg.library {
            mooncake_store::load_library(lib).with_context(|| format!("loading {}", lib.display()))?;
        }
        let store = MooncakeStore::new().context("creating the Mooncake Store client")?;
        store
            .setup(&cfg.hostname, &cfg.metadata, 0, LOCAL_BUFFER, "rdma", &cfg.device, &cfg.master)
            .with_context(|| format!("pool: connecting to master {} as {}", cfg.master, cfg.hostname))?;
        let mut registered = 0u64;
        for s in &spans {
            let bytes = (s.object * s.objects) as usize;
            unsafe { store.register_buffer(s.ptr as *mut std::ffi::c_void, bytes) }
                .with_context(|| format!("pool: registering `{}` ({bytes} bytes)", s.state))?;
            registered += bytes as u64;
        }
        if spans.is_empty() {
            bail!("pool: the manifest has no pooled state to share");
        }
        info!(
            master = %cfg.master,
            hostname = %cfg.hostname,
            device = %cfg.device,
            segment = ?cfg.segment,
            states = spans.len(),
            registered_gib = registered as f64 / (1u64 << 30) as f64,
            "pool: client open"
        );
        let prefer = cfg.segment.clone().map(|s| ReplicateConfig {
            replica_num: 1,
            preferred_segments: vec![s],
            ..ReplicateConfig::default()
        });
        let (done_tx, done) = channel();
        let shared = Arc::new(Shared {
            store,
            prefer,
            cooldown: cfg.cooldown,
            failed: Mutex::new(None),
            done: Mutex::new(done_tx),
        });
        let worker = |name: &str| -> Result<Sender<Job>> {
            let (tx, rx) = channel::<Job>();
            let sh = Arc::clone(&shared);
            std::thread::Builder::new()
                .name(format!("kern-pool-{name}"))
                .spawn(move || {
                    while let Ok(job) = rx.recv() {
                        sh.run(job);
                    }
                })
                .with_context(|| format!("spawning the pool {name} worker"))?;
            Ok(tx)
        };
        let (probes, gets, puts) = (worker("probe")?, worker("get")?, worker("put")?);
        let spans = spans.into_iter().map(|s| (s.state.clone(), s)).collect();
        Ok(Client { probes, gets, puts, done, spans })
    }

    /// Where `o`'s bytes live on this rank, state by state.
    fn buffers(&self, o: &Object) -> Buffers {
        self.spans
            .values()
            .flat_map(|s| {
                let at = |i: i32| (s.ptr + i as u64 * s.object, s.object as usize);
                match s.kind {
                    kern_pool::Kind::Page => o.pages.iter().map(|&p| at(p)).collect::<Vec<_>>(),
                    kern_pool::Kind::Slot => o.slot.map(at).into_iter().collect(),
                }
            })
            .collect()
    }

    fn objects(&self, objects: &[Object]) -> Vec<(String, Buffers)> {
        objects.iter().map(|o| (o.key.clone(), self.buffers(o))).collect()
    }

    pub fn probe(&self, id: u64, keys: Vec<String>) {
        let _ = self.probes.send(Job::Probe { id, keys });
    }

    pub fn put(&self, id: u64, after: Mark, objects: &[Object]) {
        let _ = self.puts.send(Job::Put { id, after, objects: self.objects(objects) });
    }

    pub fn get(&self, id: u64, after: Mark, objects: &[Object]) {
        let _ = self.gets.send(Job::Get { id, after, objects: self.objects(objects) });
    }

    pub fn mark(&self, id: u64, key: String) {
        let _ = self.puts.send(Job::Mark { id, key });
    }

    /// Every act answered since the last call.
    pub fn landed(&self) -> Vec<Landed> {
        self.done.try_iter().collect()
    }

    /// Hand every answer to `f` on a thread of its own instead, as it
    /// comes, until `f` says to stop.
    pub fn forward(&mut self, mut f: impl FnMut(Landed) -> bool + Send + 'static) -> Result<()> {
        let (_, rx) = channel();
        let rx = std::mem::replace(&mut self.done, rx);
        std::thread::Builder::new()
            .name("kern-pool-landed".into())
            .spawn(move || {
                for l in rx {
                    if !f(l) {
                        return;
                    }
                }
            })
            .context("spawning the pool answer thread")?;
        Ok(())
    }
}
