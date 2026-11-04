//! Kernel timestamps from CUPTI's hardware event system, each kernel named
//! by the manifest call that launched it.
//!
//! [`start`] subscribes the process: it must run before any CUDA context
//! exists, because the hardware event system (HES) is chosen once per
//! process, and once on it stays on. On Blackwell the GPU writes a kernel's
//! start and end itself, so a captured graph pays about a tenth of a
//! microsecond per kernel node for its timestamps and an eager launch the
//! CUPTI launch hook; nothing is inserted between kernels.
//!
//! While tracing, every issue carries a tag as CUPTI's external
//! correlation id: its high bits the issue's sequence number, its low bits
//! the call plus one for an eager program, zero for a graph's single
//! launch. A program is captured off one stream, so its graph is a chain
//! and its kernels run in the order their nodes were added: capture keeps,
//! per kernel node, the call that added it, and a graph's records sorted
//! by start are that list. [`collect`] flushes CUPTI and returns every
//! [`Step`] whose kernels have all arrived, and the kernels no issue
//! claims (the runtime's own copies and fills between programs). An issue
//! is complete once a graph's every node has a record, a later issue on
//! its device has kernels, or [`PATIENCE`] collects brought it nothing.
//!
//! An issue also carries its context: the sum of the `seq_len` fill the
//! caller wrote before it, the KV rows its sequences read.
//!
//! The joining is `Ledger::fold`, a pure function of the records; the
//! CUPTI callbacks only copy records out of their buffers.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::CStr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use cudarc::cupti::result::activity;
use cudarc::cupti::sys as cupti;

use crate::{Error, Result};

const BUFFER: usize = 8 << 20;
const ALIGN: usize = 8;
const CALL_BITS: u32 = 16;
const CALL_MASK: u64 = (1 << CALL_BITS) - 1;
/// Folds a kernel may wait for its tag before it counts as unclaimed: the
/// correlation record can land in a later buffer than the kernel's.
const PATIENCE: u32 = 4;
/// Issues kept waiting for their kernels; past this the oldest go.
const PENDING: usize = 4096;
const KIND: cupti::CUpti_ExternalCorrelationKind =
    cupti::CUpti_ExternalCorrelationKind::CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0;

pub(crate) const NO_KV: u64 = u64::MAX;

static ON: AtomicBool = AtomicBool::new(false);
static DROPPED: AtomicU64 = AtomicU64::new(0);
static SEQ: AtomicU64 = AtomicU64::new(1);
static SINK: Mutex<Sink> = Mutex::new(Sink { kernels: Vec::new(), links: Vec::new(), names: BTreeMap::new() });
/// Issues registered since the last collect: the issuing thread only ever
/// pushes here, so it never waits on a fold.
static ISSUED: Mutex<Vec<(u64, Issue)>> = Mutex::new(Vec::new());
/// Only `collect` touches the ledger.
static LEDGER: Mutex<Ledger> = Mutex::new(Ledger::new());

/// One kernel as the GPU ran it; timestamps in nanoseconds.
#[derive(Debug, Clone)]
pub struct Kernel {
    pub correlation: u32,
    pub device: u32,
    pub start: u64,
    pub end: u64,
    pub name: Arc<str>,
}

/// What the runtime said a tag is when it issued it.
#[derive(Debug, Clone)]
pub struct Issue {
    pub program: String,
    /// The vars the program reads, with this issue's values.
    pub vars: Vec<(String, u64)>,
    pub graph: bool,
    /// The sum of the `seq_len` fill written for this issue.
    pub kv: Option<u64>,
    /// A graph's kernels in run order, each the index of the call that
    /// added it; empty for an eager issue.
    pub kernels: Arc<[u32]>,
}

#[derive(Debug, Clone)]
pub struct Span {
    /// Index into the program's calls; `None` where a graph's records did
    /// not match its capture one to one.
    pub call: Option<u32>,
    pub name: Arc<str>,
    pub start: u64,
    pub end: u64,
}

/// One issue with its kernels, in start order.
#[derive(Debug, Clone)]
pub struct Step {
    pub seq: u64,
    pub issue: Issue,
    pub spans: Vec<Span>,
}

#[derive(Debug, Default)]
pub struct Collected {
    pub steps: Vec<Step>,
    pub unclaimed: Vec<Kernel>,
}

struct Pending {
    issue: Issue,
    kernels: Vec<(Kernel, Option<u32>)>,
    /// Folds since a kernel last arrived for it.
    quiet: u32,
}

/// Issues waiting for their kernels and kernels waiting for their tags.
struct Ledger {
    issues: BTreeMap<u64, Pending>,
    links: BTreeMap<u32, u64>,
    waiting: Vec<(Kernel, u32)>,
    /// Per device, the newest issue any kernel has arrived for: a stream
    /// runs in order, so every issue before it on that device is done.
    newest: BTreeMap<u32, u64>,
}

impl Ledger {
    const fn new() -> Self {
        Self { issues: BTreeMap::new(), links: BTreeMap::new(), waiting: Vec::new(), newest: BTreeMap::new() }
    }

    fn issue(&mut self, seq: u64, issue: Issue) {
        self.issues.insert(seq, Pending { issue, kernels: Vec::new(), quiet: 0 });
        while self.issues.len() > PENDING {
            self.issues.pop_first();
        }
    }

    /// Joins kernel records to their issues through the correlation
    /// records; returns the issues now complete and the kernels no issue
    /// claims.
    fn fold(&mut self, kernels: Vec<Kernel>, links: Vec<(u32, u64)>) -> Collected {
        self.links.extend(links);
        let mut unclaimed = Vec::new();
        let mut touched = BTreeSet::new();
        let arrived = std::mem::take(&mut self.waiting).into_iter().chain(kernels.into_iter().map(|k| (k, 0)));
        for (k, waited) in arrived {
            match self.links.get(&k.correlation).copied() {
                Some(tag) => {
                    let seq = tag >> CALL_BITS;
                    match self.issues.get_mut(&seq) {
                        Some(p) => {
                            let newest = self.newest.entry(k.device).or_default();
                            *newest = (*newest).max(seq);
                            p.kernels.push((k, ((tag & CALL_MASK) as u32).checked_sub(1)));
                            touched.insert(seq);
                        }
                        None => unclaimed.push(k),
                    }
                }
                None if waited < PATIENCE => self.waiting.push((k, waited + 1)),
                None => unclaimed.push(k),
            }
        }
        for (seq, p) in self.issues.iter_mut().filter(|(_, p)| !p.kernels.is_empty()) {
            p.quiet = if touched.contains(seq) { 0 } else { p.quiet + 1 };
        }
        let done: Vec<u64> = self
            .issues
            .iter()
            .filter(|(seq, p)| {
                let full = p.issue.graph && p.kernels.len() >= p.issue.kernels.len();
                let passed =
                    p.kernels.first().is_some_and(|(k, _)| self.newest.get(&k.device).is_some_and(|n| n > seq));
                full || passed || p.quiet >= PATIENCE
            })
            .map(|(seq, _)| *seq)
            .collect();
        let steps = done.into_iter().filter_map(|seq| self.issues.remove(&seq).map(|p| finish(seq, p))).collect();
        let live: BTreeSet<u64> = self.issues.keys().copied().collect();
        self.links.retain(|_, tag| live.contains(&(*tag >> CALL_BITS)));
        Collected { steps, unclaimed }
    }
}

fn finish(seq: u64, p: Pending) -> Step {
    let mut kernels = p.kernels;
    kernels.sort_by_key(|(k, _)| (k.start, k.end));
    let matched = kernels.len() == p.issue.kernels.len();
    let spans = kernels
        .iter()
        .enumerate()
        .map(|(i, (k, call))| Span {
            call: if p.issue.graph { matched.then(|| p.issue.kernels[i]) } else { *call },
            name: k.name.clone(),
            start: k.start,
            end: k.end,
        })
        .collect();
    Step { seq, issue: p.issue, spans }
}

struct Sink {
    kernels: Vec<Kernel>,
    links: Vec<(u32, u64)>,
    /// CUPTI's interned kernel names, by address.
    names: BTreeMap<usize, Arc<str>>,
}

fn check(r: std::result::Result<(), cudarc::cupti::result::CuptiError>, what: &str) -> Result<()> {
    r.map_err(|e| Error::Cuda(format!("CUPTI {what}: {e:?}")))
}

/// Subscribes the process to kernel records through HES; before any CUDA
/// context exists. `flush_ms` is how often CUPTI hands buffers over on its
/// own (HES does not flush unless asked).
pub fn start(flush_ms: u32) -> Result<()> {
    hes_attribute("cuptiActivitySetAttribute", 1)?;
    check(activity::register_callbacks(Some(requested), Some(completed)), "register callbacks")?;
    check(activity::enable(cupti::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL), "enable kernels")?;
    // A correlation record is written beside the API record of the call
    // it tags, so API records must be on (cuBLAS launches some kernels
    // through the runtime API); they are dropped unread.
    check(activity::enable(cupti::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_DRIVER), "enable driver records")?;
    check(activity::enable(cupti::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_RUNTIME), "enable runtime records")?;
    check(
        activity::enable(cupti::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION),
        "enable external correlation",
    )?;
    check(activity::flush_period(flush_ms), "flush period")?;
    ON.store(true, Ordering::Release);
    Ok(())
}

/// Whether HES is what CUPTI uses: meaningful once a context exists, since
/// CUPTI falls back to its software timestamps silently.
pub fn hes() -> Result<bool> {
    Ok(hes_attribute("cuptiActivityGetAttribute", 0)? == 1)
}

/// `CUPTI_ACTIVITY_ATTR_ENABLE_HES` through `f` (the set or the get), by
/// its value: the attribute is newer than the CUDA API this binary binds.
fn hes_attribute(f: &str, value: u8) -> Result<u8> {
    const ENABLE_HES: u32 = 13;
    type Attribute = unsafe extern "C" fn(u32, *mut usize, *mut std::ffi::c_void) -> u32;
    let mut v = value;
    let mut size = std::mem::size_of::<u8>();
    let r = unsafe {
        let call = cupti::culib()
            .get::<Attribute>(format!("{f}\0").as_bytes())
            .map_err(|e| Error::Cuda(format!("CUPTI {f}: {e}")))?;
        call(ENABLE_HES, &mut size, &mut v as *mut u8 as *mut _)
    };
    match r {
        0 => Ok(v),
        r => Err(Error::Cuda(format!("CUPTI {f}(ENABLE_HES): error {r}"))),
    }
}

/// CUPTI's clock, the one kernel records are stamped in, in nanoseconds.
pub fn now() -> Result<u64> {
    let mut t = 0;
    match unsafe { cupti::cuptiGetTimestamp(&mut t) } {
        cupti::CUptiResult::CUPTI_SUCCESS => Ok(t),
        r => Err(Error::Cuda(format!("CUPTI timestamp: {r:?}"))),
    }
}

/// Records CUPTI dropped for want of buffer space since [`start`].
pub fn dropped() -> u64 {
    DROPPED.load(Ordering::Relaxed)
}

pub(crate) fn on() -> bool {
    ON.load(Ordering::Acquire)
}

/// Registers an issue and returns its sequence number.
pub(crate) fn begin(issue: Issue) -> u64 {
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    ISSUED.lock().unwrap_or_else(|e| e.into_inner()).push((seq, issue));
    seq
}

/// Runs `f` with `seq`'s tag for `call` (`None`: a whole graph) on this
/// thread's correlation stack.
pub(crate) fn tagged<T>(seq: u64, call: Option<usize>, f: impl FnOnce() -> Result<T>) -> Result<T> {
    let tag = seq << CALL_BITS | call.map_or(0, |c| c as u64 + 1);
    check(activity::push_external_correlation_id(KIND, tag), "push correlation")?;
    let r = f();
    let mut last = 0;
    check(unsafe { activity::pop_external_correlation_id(KIND, &mut last) }, "pop correlation")?;
    r
}

/// Flushes CUPTI's finished buffers and joins what arrived.
pub fn collect() -> Result<Collected> {
    check(activity::flush_all(0), "flush")?;
    // The sink before the issues: a kernel is launched after its issue is
    // registered, so every kernel taken here finds its issue below.
    let (kernels, links) = {
        let mut s = SINK.lock().unwrap_or_else(|e| e.into_inner());
        (std::mem::take(&mut s.kernels), std::mem::take(&mut s.links))
    };
    let issued = std::mem::take(&mut *ISSUED.lock().unwrap_or_else(|e| e.into_inner()));
    let mut ledger = LEDGER.lock().unwrap_or_else(|e| e.into_inner());
    issued.into_iter().for_each(|(seq, issue)| ledger.issue(seq, issue));
    Ok(ledger.fold(kernels, links))
}

fn layout() -> std::alloc::Layout {
    std::alloc::Layout::from_size_align(BUFFER, ALIGN).expect("constant layout")
}

unsafe extern "C" fn requested(buffer: *mut *mut u8, size: *mut usize, max_records: *mut usize) {
    unsafe {
        *buffer = std::alloc::alloc(layout());
        *size = if (*buffer).is_null() { 0 } else { BUFFER };
        *max_records = 0;
    }
}

unsafe extern "C" fn completed(
    ctx: cudarc::driver::sys::CUcontext,
    stream: u32,
    buffer: *mut u8,
    _: usize,
    valid: usize,
) {
    let mut lost = 0;
    if unsafe { cupti::cuptiActivityGetNumDroppedRecords(ctx as _, stream, &mut lost) }
        == cupti::CUptiResult::CUPTI_SUCCESS
    {
        DROPPED.fetch_add(lost as u64, Ordering::Relaxed);
    }
    if buffer.is_null() {
        return;
    }
    let mut s = SINK.lock().unwrap_or_else(|e| e.into_inner());
    let mut record: *mut cupti::CUpti_Activity = std::ptr::null_mut();
    while unsafe { activity::get_next_record(buffer, valid, &mut record) }.is_ok() {
        match unsafe { (*record).kind } {
            cupti::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL => {
                // Newer CUPTI versions append fields; the prefix is this one.
                let k = unsafe { std::ptr::read_unaligned(record as *const cupti::CUpti_ActivityKernel9) };
                let name = s
                    .names
                    .entry(k.name as usize)
                    .or_insert_with(|| match k.name.is_null() {
                        true => Arc::from("?"),
                        false => Arc::from(unsafe { CStr::from_ptr(k.name) }.to_string_lossy().as_ref()),
                    })
                    .clone();
                s.kernels.push(Kernel {
                    correlation: k.correlationId,
                    device: k.deviceId,
                    start: k.start,
                    end: k.end,
                    name,
                });
            }
            cupti::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION => {
                let c = unsafe { std::ptr::read_unaligned(record as *const cupti::CUpti_ActivityExternalCorrelation) };
                if c.externalKind == KIND {
                    s.links.push((c.correlationId, c.externalId));
                }
            }
            _ => {}
        }
    }
    drop(s);
    unsafe { std::alloc::dealloc(buffer, layout()) };
}
