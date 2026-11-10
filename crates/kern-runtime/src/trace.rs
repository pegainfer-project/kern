//! Kernel timestamps from CUPTI's hardware event system, each kernel named
//! by the manifest call that launched it.
//!
//! [`start`] subscribes the process: it must run before any CUDA context
//! exists, because the hardware event system (HES) is chosen once per
//! process, and once on it stays on. On Blackwell the GPU writes a kernel's
//! start and end itself, so a captured graph pays about a tenth of a
//! microsecond per kernel node for its timestamps and an eager launch the
//! CUPTI launch hook; nothing is inserted between kernels. Every call is
//! CUPTI's subscriber-scoped (V2) API, which libcupti 13.3 and newer have.
//!
//! While tracing, every launch carries a tag as CUPTI's external
//! correlation id: its high bits an issue's sequence number, its low bits
//! what was launched (`Launch`): one call of an eager program, the
//! issue's own graph, or a graph of the host's. Capture keeps, per kernel
//! node, the call that added it, keyed by the node's local id: CUPTI's
//! graph node id is the graph's id in its high half and the node's local
//! id in its low half, and instantiation keeps the local ids, so a kernel
//! record of the graph exec names its call exactly, whatever order a
//! program's lanes ran in.
//!
//! A graph the host captured holds the programs it enqueued while
//! capturing and nodes of its own, and the host launches it without the
//! runtime. Each of those programs kept its nodes at capture
//! ([`crate::HostGraph`]); the host's launch registers one issue per
//! program, numbered in a row, under one tag that says how many, and a
//! kernel record goes to the issue whose nodes name it. A node none of
//! them added is the host's own kernel, and no issue claims it.
//!
//! [`collect`] flushes CUPTI and returns every [`Step`] whose kernels have
//! all arrived, and the kernels no issue claims (the runtime's own copies
//! and fills between programs, everything a host runs itself). An issue
//! is complete once a graph's every node has a record, a later launch on
//! its device has kernels, or [`PATIENCE`] collects brought it nothing.
//! CUPTI delivers a graph launch's correlation record behind its kernel
//! records, so a launch that stalls midway (a kernel waiting on a peer
//! that is late) has its first kernels arrive long before their tag: a
//! graph's kernel without a tag waits while a graph issued has none yet.
//!
//! CUPTI hands a buffer over only when asked; a periodic flush would hand
//! over only full ones, and a buffer of eight megabytes takes seconds to
//! fill. `collect` asks twice: under HES a flush seals the buffer eager
//! launches are filling while its newest records still wait for their
//! timestamps, holds it back, and gives it to the next flush, so without
//! the second ask a kernel waits a whole collect longer. [`finish`] is
//! the last collect: CUPTI gives up every buffer, a kernel record with no
//! end yet is dropped, and every issue still waiting is returned with the
//! kernels it has.
//!
//! An issue also carries its context: the sum of the `seq_len` fill the
//! caller wrote before it, the KV rows its sequences read.
//!
//! The joining is `Ledger::fold`, a pure function of the records; the
//! CUPTI callbacks only copy records out of their buffers.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::CStr;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use cudarc::cupti::sys as cupti;

use crate::{Error, Result};

const BUFFER: usize = 8 << 20;
const ALIGN: usize = 8;
const LAUNCH_BITS: u32 = 24;
/// Folds a kernel may wait for its tag before it counts as unclaimed: the
/// correlation record can land in a later buffer than the kernel's.
const PATIENCE: u32 = 4;
/// Folds a graph's kernel may wait for its tag while a graph issued has
/// none yet.
const STALLED: u32 = 4096;
/// Issues kept waiting for their kernels; past this the oldest go.
const PENDING: usize = 4096;
const KIND: cupti::CUpti_ExternalCorrelationKind =
    cupti::CUpti_ExternalCorrelationKind::CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0;

pub(crate) const NO_KV: u64 = u64::MAX;

static ON: AtomicBool = AtomicBool::new(false);
static SUBSCRIBER: AtomicPtr<cupti::CUpti_Subscriber_st> = AtomicPtr::new(std::ptr::null_mut());
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
    /// The local id of the graph node that launched it; `None` for an
    /// eager launch.
    pub node: Option<u32>,
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
    /// A graph's kernel nodes by local id, each to the index of the call
    /// that added it; empty for an eager issue.
    pub kernels: Arc<BTreeMap<u32, u32>>,
}

#[derive(Debug, Clone)]
pub struct Span {
    /// Index into the program's calls; `None` for a graph record whose
    /// node the capture did not see.
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
    /// Whether a tag that carries it has arrived.
    tagged: bool,
}

/// What a tag says was launched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Launch {
    /// One call of an eager issue.
    Call(u32),
    /// The issue's own graph.
    Graph,
    /// A graph of the host's carrying this many issues, the tagged one
    /// first.
    Host(u64),
}

/// The low bits are odd for a call, zero for an issue's own graph, and
/// otherwise twice the issues a graph of the host's carries.
fn tag(seq: u64, launch: Launch) -> u64 {
    let low = match launch {
        Launch::Call(c) => u64::from(c) << 1 | 1,
        Launch::Graph => 0,
        Launch::Host(issues) => issues << 1,
    };
    seq << LAUNCH_BITS | low
}

fn untag(tag: u64) -> (u64, Launch) {
    let low = tag & ((1 << LAUNCH_BITS) - 1);
    let launch = match (low & 1, low >> 1) {
        (1, call) => Launch::Call(call as u32),
        (_, 0) => Launch::Graph,
        (_, issues) => Launch::Host(issues),
    };
    (tag >> LAUNCH_BITS, launch)
}

/// The issues a tag's launch carries, by sequence number.
fn carried(tag: u64) -> std::ops::Range<u64> {
    match untag(tag) {
        (seq, Launch::Host(issues)) => seq..seq + issues,
        (seq, _) => seq..seq + 1,
    }
}

/// Issues waiting for their kernels and kernels waiting for their tags.
struct Ledger {
    issues: BTreeMap<u64, Pending>,
    links: BTreeMap<u32, u64>,
    waiting: Vec<(Kernel, u32)>,
    /// Per device, the newest launch any kernel has arrived for, by its
    /// first issue: a stream runs in order, so every issue before it on
    /// that device is done, and the issues of one launch never pass each
    /// other.
    newest: BTreeMap<u32, u64>,
}

impl Ledger {
    const fn new() -> Self {
        Self { issues: BTreeMap::new(), links: BTreeMap::new(), waiting: Vec::new(), newest: BTreeMap::new() }
    }

    fn issue(&mut self, seq: u64, issue: Issue) {
        self.issues.insert(seq, Pending { issue, kernels: Vec::new(), quiet: 0, tagged: false });
        while self.issues.len() > PENDING {
            self.issues.pop_first();
        }
    }

    /// Joins kernel records to their issues through the correlation
    /// records; returns the issues now complete and the kernels no issue
    /// claims.
    fn fold(&mut self, kernels: Vec<Kernel>, links: Vec<(u32, u64)>) -> Collected {
        for (_, tag) in &links {
            self.issues.range_mut(carried(*tag)).for_each(|(_, p)| p.tagged = true);
        }
        self.links.extend(links);
        let launching = self.issues.values().any(|p| p.issue.graph && !p.tagged);
        let mut unclaimed = Vec::new();
        let mut touched = BTreeSet::new();
        let arrived = std::mem::take(&mut self.waiting).into_iter().chain(kernels.into_iter().map(|k| (k, 0)));
        for (k, waited) in arrived {
            let Some(tag) = self.links.get(&k.correlation).copied() else {
                match waited < PATIENCE || (launching && k.node.is_some() && waited < STALLED) {
                    true => self.waiting.push((k, waited + 1)),
                    false => unclaimed.push(k),
                }
                continue;
            };
            let (seq, launch) = untag(tag);
            let named =
                |s: &u64| k.node.is_some_and(|n| self.issues.get(s).is_some_and(|p| p.issue.kernels.contains_key(&n)));
            let (owner, call) = match launch {
                Launch::Call(c) => (Some(seq), Some(c)),
                Launch::Graph => (Some(seq), None),
                Launch::Host(_) => (carried(tag).find(named), None),
            };
            match owner.and_then(|s| Some((s, self.issues.get_mut(&s)?))) {
                Some((s, p)) => {
                    let newest = self.newest.entry(k.device).or_default();
                    *newest = (*newest).max(seq);
                    p.kernels.push((k, call));
                    touched.insert(s);
                }
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
        let steps = done.into_iter().filter_map(|seq| self.issues.remove(&seq).map(|p| step(seq, p))).collect();
        let live = &self.issues;
        self.links.retain(|_, tag| live.range(carried(*tag)).next().is_some());
        Collected { steps, unclaimed }
    }

    /// Everything still waiting: the issues with kernels as steps, the
    /// kernels still without a tag as unclaimed.
    fn drain(&mut self) -> Collected {
        let unclaimed = std::mem::take(&mut self.waiting).into_iter().map(|(k, _)| k).collect();
        let steps = std::mem::take(&mut self.issues)
            .into_iter()
            .filter(|(_, p)| !p.kernels.is_empty())
            .map(|(seq, p)| step(seq, p))
            .collect();
        self.links.clear();
        Collected { steps, unclaimed }
    }
}

fn step(seq: u64, p: Pending) -> Step {
    let mut kernels = p.kernels;
    kernels.sort_by_key(|(k, _)| (k.start, k.end));
    let spans = kernels
        .iter()
        .map(|(k, call)| Span {
            call: match p.issue.graph {
                true => k.node.and_then(|n| p.issue.kernels.get(&n).copied()),
                false => *call,
            },
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

/// The libcupti the subscriber-scoped (V2) API needs: 13.3, where it left
/// beta. An older one is refused at [`start`] rather than missing a symbol
/// mid-serve.
const CUPTI_MIN: u32 = 130300;
const ENABLE_HES: cupti::CUpti_ActivityAttribute = cupti::CUpti_ActivityAttribute::CUPTI_ACTIVITY_ATTR_ENABLE_HES;

fn ok(r: cupti::CUptiResult, what: &str) -> Result<()> {
    match r {
        cupti::CUptiResult::CUPTI_SUCCESS => Ok(()),
        r => Err(Error::Cuda(format!("CUPTI {what}: {r:?}"))),
    }
}

fn subscriber() -> cupti::CUpti_SubscriberHandle {
    SUBSCRIBER.load(Ordering::Acquire)
}

/// Subscribes the process to kernel records through HES; before any CUDA
/// context exists.
pub fn start() -> Result<()> {
    let mut version = 0;
    ok(unsafe { cupti::cuptiGetVersion(&mut version) }, "version")?;
    if version < CUPTI_MIN {
        return Err(Error::Cuda(format!(
            "CUPTI: tracing needs libcupti API {CUPTI_MIN} (CUDA 13.3) or newer, got {version}"
        )));
    }
    let mut params = cupti::CUpti_SubscriberParams {
        structSize: std::mem::size_of::<cupti::CUpti_SubscriberParams>(),
        subscriberName: c"kern".as_ptr(),
        oldSubscriberName: std::ptr::null_mut(),
        oldSubscriberSize: 0,
        allowMultipleSubscribers: 0,
        padding: [0; 7],
    };
    let mut s = std::ptr::null_mut();
    ok(unsafe { cupti::cuptiSubscribe_v2(&mut s, None, std::ptr::null_mut(), &mut params) }, "subscribe")?;
    SUBSCRIBER.store(s, Ordering::Release);
    // HES is chosen before any kind is enabled.
    let (mut on, mut size) = (1u8, 1usize);
    ok(
        unsafe { cupti::cuptiActivitySetAttribute_v2(s, ENABLE_HES, &mut size, &mut on as *mut u8 as *mut _) },
        "set ENABLE_HES",
    )?;
    ok(unsafe { cupti::cuptiActivityRegisterCallbacks_v2(s, Some(requested), Some(completed)) }, "register callbacks")?;
    use cupti::CUpti_ActivityKind::*;
    // A correlation record is written beside the API record of the call
    // it tags, so API records must be on (cuBLAS launches some kernels
    // through the runtime API); they are dropped unread.
    [
        (CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL, "enable kernels"),
        (CUPTI_ACTIVITY_KIND_DRIVER, "enable driver records"),
        (CUPTI_ACTIVITY_KIND_RUNTIME, "enable runtime records"),
        (CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION, "enable external correlation"),
    ]
    .into_iter()
    .try_for_each(|(kind, what)| ok(unsafe { cupti::cuptiActivityEnable_v2(s, kind, std::ptr::null_mut()) }, what))?;
    ON.store(true, Ordering::Release);
    Ok(())
}

/// Whether HES is what CUPTI uses: meaningful once a context exists, since
/// CUPTI falls back to its software timestamps silently.
pub fn hes() -> Result<bool> {
    let (mut v, mut size) = (0u8, 1usize);
    ok(
        unsafe {
            cupti::cuptiActivityGetAttribute_v2(subscriber(), ENABLE_HES, &mut size, &mut v as *mut u8 as *mut _)
        },
        "get ENABLE_HES",
    )?;
    Ok(v == 1)
}

/// CUPTI's clock, the one kernel records are stamped in, in nanoseconds.
pub fn now() -> Result<u64> {
    let mut t = 0;
    ok(unsafe { cupti::cuptiGetTimestamp_v2(subscriber(), &mut t) }, "timestamp")?;
    Ok(t)
}

/// Kernel records lost since [`start`]: dropped by CUPTI for want of
/// buffer space, or handed over by [`finish`] before the kernel ended.
pub fn dropped() -> u64 {
    DROPPED.load(Ordering::Relaxed)
}

/// Whether [`start`] has subscribed the process.
pub fn on() -> bool {
    ON.load(Ordering::Acquire)
}

/// Registers the issues of one launch and returns the first's sequence
/// number; the rest follow it in a row.
pub(crate) fn begin(issues: Vec<Issue>) -> u64 {
    let first = SEQ.fetch_add(issues.len() as u64, Ordering::Relaxed);
    ISSUED.lock().unwrap_or_else(|e| e.into_inner()).extend((first..).zip(issues));
    first
}

fn push(tag: u64) -> Result<()> {
    ok(unsafe { cupti::cuptiActivityPushExternalCorrelationId_v2(subscriber(), KIND, tag) }, "push correlation")
}

fn pop() -> Result<()> {
    let mut last = 0;
    ok(unsafe { cupti::cuptiActivityPopExternalCorrelationId_v2(subscriber(), KIND, &mut last) }, "pop correlation")
}

/// Runs `f` with `seq`'s tag for `call` (`None`: a whole graph) on this
/// thread's correlation stack.
pub(crate) fn tagged<T>(seq: u64, call: Option<usize>, f: impl FnOnce() -> Result<T>) -> Result<T> {
    push(tag(seq, call.map_or(Launch::Graph, |c| Launch::Call(c as u32))))?;
    let r = f();
    pop()?;
    r
}

/// Tags this thread for a graph of the host's that carries `issues` issues
/// from `seq` on, until [`host_launched`].
pub(crate) fn host_launch(seq: u64, issues: u64) -> Result<()> {
    push(tag(seq, Launch::Host(issues)))
}

pub(crate) fn host_launched() -> Result<()> {
    pop()
}

/// A graph node's local id, the half of CUPTI's id its records keep
/// through instantiation.
pub(crate) fn node(node: cudarc::driver::sys::CUgraphNode) -> Result<u32> {
    let mut id = 0u64;
    ok(unsafe { cupti::cuptiGetGraphNodeId(node, &mut id) }, "graph node id")?;
    Ok(id as u32)
}

fn flush(flag: u32) -> Result<()> {
    ok(unsafe { cupti::cuptiActivityFlushAll(flag) }, "flush")
}

/// Flushes CUPTI's finished buffers and joins what arrived.
pub fn collect() -> Result<Collected> {
    flush(0)?;
    flush(0)?;
    Ok(join(false))
}

/// The last collect, once nothing more is launched: every buffer CUPTI
/// holds and every issue still waiting.
pub fn finish() -> Result<Collected> {
    flush(0)?;
    flush(0)?;
    flush(cupti::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_FLUSH_FORCED as u32)?;
    Ok(join(true))
}

fn join(last: bool) -> Collected {
    // The sink before the issues: a kernel is launched after its issue is
    // registered, so every kernel taken here finds its issue below.
    let (kernels, links) = {
        let mut s = SINK.lock().unwrap_or_else(|e| e.into_inner());
        (std::mem::take(&mut s.kernels), std::mem::take(&mut s.links))
    };
    let issued = std::mem::take(&mut *ISSUED.lock().unwrap_or_else(|e| e.into_inner()));
    let mut ledger = LEDGER.lock().unwrap_or_else(|e| e.into_inner());
    issued.into_iter().for_each(|(seq, issue)| ledger.issue(seq, issue));
    let mut c = ledger.fold(kernels, links);
    if last {
        let rest = ledger.drain();
        c.steps.extend(rest.steps);
        c.unclaimed.extend(rest.unclaimed);
    }
    c
}

fn layout() -> std::alloc::Layout {
    std::alloc::Layout::from_size_align(BUFFER, ALIGN).expect("constant layout")
}

unsafe extern "C" fn requested(
    buffer: *mut *mut u8,
    size: *mut usize,
    max_records: *mut usize,
    _: *mut cupti::CUpti_BufferCallbackRequestInfo,
) {
    unsafe {
        *buffer = std::alloc::alloc(layout());
        *size = if (*buffer).is_null() { 0 } else { BUFFER };
        *max_records = 0;
    }
}

unsafe extern "C" fn completed(
    buffer: *mut u8,
    _: usize,
    valid: usize,
    _: *mut cupti::CUpti_BufferCallbackCompleteInfo,
) {
    let (s, mut lost) = (subscriber(), 0);
    if unsafe { cupti::cuptiActivityGetNumDroppedRecords_v2(s, std::ptr::null_mut(), 0, &mut lost) }
        == cupti::CUptiResult::CUPTI_SUCCESS
    {
        DROPPED.fetch_add(lost as u64, Ordering::Relaxed);
    }
    if buffer.is_null() {
        return;
    }
    let mut sink = SINK.lock().unwrap_or_else(|e| e.into_inner());
    let mut record: *mut cupti::CUpti_Activity = std::ptr::null_mut();
    while unsafe { cupti::cuptiActivityGetNextRecord_v2(s, buffer, valid, &mut record) }
        == cupti::CUptiResult::CUPTI_SUCCESS
    {
        match unsafe { (*record).kind } {
            cupti::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL => {
                // Newer CUPTI versions append fields; the prefix is this one.
                let k = unsafe { std::ptr::read_unaligned(record as *const cupti::CUpti_ActivityKernel9) };
                // Only a forced flush hands over a kernel still running.
                if k.end < k.start || k.end == 0 {
                    DROPPED.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                let name = sink
                    .names
                    .entry(k.name as usize)
                    .or_insert_with(|| match k.name.is_null() {
                        true => Arc::from("?"),
                        false => Arc::from(unsafe { CStr::from_ptr(k.name) }.to_string_lossy().as_ref()),
                    })
                    .clone();
                sink.kernels.push(Kernel {
                    correlation: k.correlationId,
                    device: k.deviceId,
                    start: k.start,
                    end: k.end,
                    name,
                    // Zero for an eager launch: a graph's id, the high half, starts at one.
                    node: (k.graphNodeId != 0).then_some(k.graphNodeId as u32),
                });
            }
            cupti::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION => {
                let c = unsafe { std::ptr::read_unaligned(record as *const cupti::CUpti_ActivityExternalCorrelation) };
                if c.externalKind == KIND {
                    sink.links.push((c.correlationId, c.externalId));
                }
            }
            _ => {}
        }
    }
    drop(sink);
    unsafe { std::alloc::dealloc(buffer, layout()) };
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kernel(correlation: u32, start: u64) -> Kernel {
        Kernel { correlation, device: 0, start, end: start + 1, name: Arc::from("k"), node: None }
    }

    fn eager() -> Issue {
        Issue { program: "p".into(), vars: Vec::new(), graph: false, kv: None, kernels: Arc::default() }
    }

    #[test]
    fn a_graph_record_names_its_call_by_node_whatever_order_the_lanes_ran_in() {
        let mut l = Ledger::new();
        // Nodes 0..3 from calls 0, 1, 1, 2; call 1 ran on a side lane and finished first.
        l.issue(1, graph(&[(0, 0), (1, 1), (2, 1), (3, 2)]));
        let records = vec![node(3, 40), node(1, 10), node(0, 20), node(2, 15), node(9, 50)];
        let c = l.fold(records, vec![(5, tag(1, Launch::Graph))]);
        let calls: Vec<(u64, Option<u32>)> = c.steps[0].spans.iter().map(|s| (s.start, s.call)).collect();
        assert_eq!(calls, vec![(10, Some(1)), (15, Some(1)), (20, Some(0)), (40, Some(2)), (50, None)]);
    }

    fn graph(nodes: &[(u32, u32)]) -> Issue {
        Issue { graph: true, kernels: Arc::new(nodes.iter().copied().collect()), ..eager() }
    }

    fn node(n: u32, start: u64) -> Kernel {
        Kernel { node: Some(n), ..kernel(5, start) }
    }

    fn calls(c: &Collected) -> Vec<(u64, Vec<Option<u32>>)> {
        c.steps.iter().map(|s| (s.seq, s.spans.iter().map(|k| k.call).collect())).collect()
    }

    #[test]
    fn a_tag_reads_back_as_what_was_launched() {
        let launches = [Launch::Call(0), Launch::Call(1), Launch::Call(70_000), Launch::Graph, Launch::Host(1)];
        for seq in [1, 2, 1 << 30] {
            for launch in launches.into_iter().chain([Launch::Host(2), Launch::Host(4096)]) {
                assert_eq!(untag(tag(seq, launch)), (seq, launch));
            }
        }
        assert_eq!((carried(tag(7, Launch::Call(3))), carried(tag(7, Launch::Host(3)))), (7..8, 7..10));
    }

    #[test]
    fn a_host_launch_gives_each_issue_the_nodes_it_added_and_no_issue_the_hosts_own() {
        let mut l = Ledger::new();
        l.issue(1, graph(&[(10, 0), (11, 1)]));
        l.issue(2, graph(&[(20, 0)]));
        // Node 99 is a kernel the host captured itself, between the two programs.
        let c =
            l.fold(vec![node(20, 40), node(10, 10), node(99, 30), node(11, 20)], vec![(5, tag(1, Launch::Host(2)))]);
        let unclaimed: Vec<Option<u32>> = c.unclaimed.iter().map(|k| k.node).collect();
        assert_eq!((calls(&c), unclaimed), (vec![(1, vec![Some(0), Some(1)]), (2, vec![Some(0)])], vec![Some(99)]));
    }

    #[test]
    fn the_issues_of_one_launch_wait_for_a_later_launch_not_for_each_other() {
        let mut l = Ledger::new();
        l.issue(1, graph(&[(10, 0), (11, 0)]));
        l.issue(2, graph(&[(20, 0), (21, 0)]));
        l.issue(3, eager());
        // Issue 2 has a kernel while issue 1 still misses one: neither is done.
        let c = l.fold(vec![node(20, 30), node(10, 10)], vec![(5, tag(1, Launch::Host(2)))]);
        assert_eq!(calls(&c), vec![]);
        let c = l.fold(vec![kernel(6, 50)], vec![(6, tag(3, Launch::Call(0)))]);
        assert_eq!(calls(&c), vec![(1, vec![Some(0)]), (2, vec![Some(0)])]);
    }

    #[test]
    fn a_host_launch_keeps_its_tag_while_any_of_its_issues_waits() {
        let mut l = Ledger::new();
        l.issue(1, graph(&[(10, 0)]));
        l.issue(2, graph(&[(20, 0), (21, 1)]));
        let c = l.fold(vec![node(10, 10), node(20, 20)], vec![(5, tag(1, Launch::Host(2)))]);
        assert_eq!(calls(&c), vec![(1, vec![Some(0)])]);
        let c = l.fold(vec![node(21, 30)], Vec::new());
        assert_eq!((calls(&c), c.unclaimed.len()), (vec![(2, vec![Some(0), Some(1)])], 0));
        assert_eq!(l.links.len(), 0);
    }

    #[test]
    fn a_graphs_kernel_waits_for_the_tag_of_a_launch_that_stalled() {
        let mut l = Ledger::new();
        l.issue(1, graph(&[(10, 0), (11, 1)]));
        l.issue(2, graph(&[(20, 0)]));
        // Node 10 ran, node 11 waits on a peer: the launch's tag is behind node 11's record.
        // Kernel 9 is the host's own, eager, and never tagged.
        let c = l.fold(vec![node(10, 10), kernel(9, 5)], Vec::new());
        assert_eq!((c.steps.len(), c.unclaimed.len()), (0, 0));
        let quiet: Vec<usize> = (0..2 * PATIENCE).map(|_| l.fold(Vec::new(), Vec::new()).unclaimed.len()).collect();
        assert_eq!((quiet.iter().sum::<usize>(), l.waiting.len()), (1, 1));
        let c = l.fold(vec![node(11, 9000), node(20, 9010)], vec![(5, tag(1, Launch::Host(2)))]);
        assert_eq!((calls(&c), c.unclaimed.len()), (vec![(1, vec![Some(0), Some(1)]), (2, vec![Some(0)])], 0));
    }

    #[test]
    fn a_graphs_kernel_nobody_tagged_is_unclaimed_once_every_graph_issued_has_its_tag() {
        let mut l = Ledger::new();
        l.issue(1, graph(&[(10, 0)]));
        // Correlation 6 is a graph the host launched untagged.
        let other = Kernel { correlation: 6, ..node(30, 20) };
        let c = l.fold(vec![node(10, 10), other], vec![(5, tag(1, Launch::Graph))]);
        assert_eq!((calls(&c), c.unclaimed.len()), (vec![(1, vec![Some(0)])], 0));
        let later: Vec<usize> = (0..PATIENCE).map(|_| l.fold(Vec::new(), Vec::new()).unclaimed.len()).collect();
        assert_eq!(later, vec![0, 0, 0, 1]);
    }

    #[test]
    fn the_last_collect_returns_what_a_periodic_one_still_waits_for() {
        let mut l = Ledger::new();
        l.issue(1, eager());
        l.issue(2, eager());
        // Issue 1 has a kernel, issue 2 none; kernel 9's tag has not arrived.
        let c = l.fold(vec![kernel(7, 10), kernel(9, 20)], vec![(7, tag(1, Launch::Call(0)))]);
        assert_eq!((c.steps.len(), c.unclaimed.len()), (0, 0));
        let c = l.drain();
        let steps: Vec<(u64, Vec<Option<u32>>)> =
            c.steps.iter().map(|s| (s.seq, s.spans.iter().map(|k| k.call).collect())).collect();
        let unclaimed: Vec<u32> = c.unclaimed.iter().map(|k| k.correlation).collect();
        assert_eq!((steps, unclaimed), (vec![(1, vec![Some(0)])], vec![9]));
        let c = l.drain();
        assert_eq!((c.steps.len(), c.unclaimed.len(), l.links.len()), (0, 0, 0));
    }
}
