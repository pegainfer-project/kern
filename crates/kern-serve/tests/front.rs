//! A pipeline head's front under pegainfer's own driver, its head a thread
//! busy with half-second items and its decode tier a thread stepping every
//! 25 ms: D's tokens reach the client a step after D makes them whatever
//! the head is doing, and the clients that leave or stop mid-decode free
//! their seats at once, the front's aborts written by the head.

use std::collections::BTreeMap;
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::anyhow;
use kern_serve::front::{Call, Front, Note};
use kern_serve::pd::{Ask, Relay, Req};
use pegainfer_frontend::engine::{
    spawn_scheduler, EosPolicy, FinishReason, LiveScheduler, Request, RequestId, StepReceiver, StopPolicy, Terminal,
};
use pegainfer_frontend::sampler::SamplingParams;

const STEP: Duration = Duration::from_millis(25);
const ITEM: Duration = Duration::from_millis(500);

/// The first token, the prefill's; D's `n`th after it is `100 + n`.
const FIRST: u32 = 1;

fn request(max_tokens: usize, stop: StopPolicy) -> Request {
    Request {
        prompt_tokens: vec![3; 16],
        params: SamplingParams::default(),
        history_tokens: None,
        stop_policy: stop,
        max_tokens,
        lora_adapter: None,
        kv_transfer_params: None,
        logprobs: None,
        prompt_logprobs: None,
        trace_parent: None,
        client_label: None,
    }
}

/// When things happened, by request.
#[derive(Default)]
struct Log {
    /// D's tokens as it sent them.
    sent: BTreeMap<Req, Vec<Instant>>,
    /// The aborts the head wrote to D for the front.
    aborted: BTreeMap<Req, Instant>,
    /// The clients gone the head heard of.
    gone: Vec<RequestId>,
}

type Shared = Arc<Mutex<Log>>;

/// A head busy with item after item, each `ITEM` long, hearing the front's
/// submissions and departures only between them: a submission whose
/// client is still there is admitted, gets its first token and goes to D;
/// one gone is retired. The front's aborts it writes to D at once, as the
/// head does while it waits on its pipeline.
fn head(calls: mpsc::Receiver<Call>, notes: mpsc::Sender<Note>, d: mpsc::Sender<Ask>, log: Shared) {
    let mut next = Instant::now() + ITEM;
    let mut heard = Vec::new();
    loop {
        match calls.recv_timeout(next.saturating_duration_since(Instant::now())) {
            Ok(Call::Abort { tier, req }) => {
                assert_eq!(tier, 0);
                log.lock().unwrap().aborted.insert(req, Instant::now());
                d.send(Ask::Abort { req }).unwrap();
                continue;
            }
            Ok(c) => {
                heard.push(c);
                continue;
            }
            Err(mpsc::RecvTimeoutError::Timeout) => next += ITEM,
            Err(mpsc::RecvTimeoutError::Disconnected) => return,
        }
        let gone: Vec<RequestId> =
            heard.iter().filter_map(|c| if let Call::Gone(id) = c { Some(*id) } else { None }).collect();
        log.lock().unwrap().gone.extend(&gone);
        for c in std::mem::take(&mut heard) {
            let Call::Submit(q) = c else { continue };
            let (id, max) = (q.id, q.request.max_tokens);
            if gone.contains(&id) {
                notes.send(Note::Retire(id)).unwrap();
                continue;
            }
            notes.send(Note::Admit { id, cached: 0 }).unwrap();
            notes.send(Note::Emit { id, tok: FIRST }).unwrap();
            notes
                .send(Note::Decode { id, relay: Relay::new(max - 1, q.request.stop_policy.clone()), tier: 0 })
                .unwrap();
            d.send(Ask::Start { req: id.raw(), first: FIRST, len: 16, max_tokens: max - 1 }).unwrap();
        }
    }
}

/// A decode tier stepping every `STEP` on its own clock: each running
/// request's next token, its last at its `max_tokens`.
fn tier(asks: mpsc::Receiver<Ask>, notes: mpsc::Sender<Note>, log: Shared) {
    let t0 = Instant::now();
    let mut running: BTreeMap<Req, (u32, usize)> = BTreeMap::new();
    for k in 1.. {
        std::thread::sleep((t0 + STEP * k).saturating_duration_since(Instant::now()));
        loop {
            match asks.try_recv() {
                Ok(Ask::Start { req, max_tokens, .. }) => {
                    running.insert(req, (0, max_tokens));
                }
                Ok(Ask::Abort { req }) => {
                    running.remove(&req);
                }
                Ok(a) => panic!("the tier was asked {a:?}"),
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) if running.is_empty() => return,
                Err(mpsc::TryRecvError::Disconnected) => break,
            }
        }
        if running.is_empty() {
            continue;
        }
        let mut step = Vec::new();
        for (&req, (n, max)) in &mut running {
            *n += 1;
            step.push((req, 100 + *n, *n as usize == *max));
        }
        running.retain(|_, (n, max)| (*n as usize) < *max);
        let now = Instant::now();
        let mut l = log.lock().unwrap();
        for &(req, ..) in &step {
            l.sent.entry(req).or_default().push(now);
        }
        drop(l);
        notes.send(Note::Tokens(step)).unwrap();
    }
}

/// The front over a fake head and tier, driven as pegainfer drives it.
fn stack() -> (LiveScheduler, StepReceiver, Shared) {
    let log = Shared::default();
    let (notes, front_notes) = mpsc::channel();
    let (calls_tx, calls) = mpsc::channel();
    let (asks_tx, asks) = mpsc::channel();
    let to_head = move |c: Call| calls_tx.send(c).map_err(|_| anyhow!("head gone"));
    std::thread::spawn({
        let (notes, log) = (notes.clone(), log.clone());
        move || head(calls, notes, asks_tx, log)
    });
    std::thread::spawn({
        let log = log.clone();
        move || tier(asks, notes, log)
    });
    let mut live = spawn_scheduler("front", Front::new(front_notes, to_head, 100));
    let steps = live.handle.take_steps().unwrap();
    (live, steps, log)
}

/// What the client saw of each request: when each token arrived, and how
/// it ended.
#[derive(Default)]
struct Seen {
    tokens: BTreeMap<RequestId, Vec<(u32, Instant)>>,
    ends: BTreeMap<RequestId, Terminal>,
}

impl Seen {
    /// Read the step stream until `done` or `patience` runs out.
    fn read(&mut self, steps: &mut StepReceiver, patience: Duration, mut done: impl FnMut(&Seen) -> bool) {
        let t0 = Instant::now();
        while !done(self) && t0.elapsed() < patience {
            match steps.try_recv() {
                Ok(s) => {
                    let now = Instant::now();
                    for u in s.updates {
                        self.tokens.entry(u.id).or_default().extend(u.tokens.iter().map(|&t| (t, now)));
                        if let Some(t) = u.terminal {
                            self.ends.insert(u.id, t);
                        }
                    }
                }
                Err(_) => std::thread::sleep(Duration::from_micros(200)),
            }
        }
    }

    fn ids(&self, id: RequestId) -> Vec<u32> {
        self.tokens.get(&id).map_or(Vec::new(), |v| v.iter().map(|&(t, _)| t).collect())
    }
}

fn finished(t: Option<&Terminal>) -> Option<(FinishReason, usize)> {
    match t {
        Some(Terminal::Finished { reason, completion_tokens, .. }) => Some((*reason, *completion_tokens)),
        _ => None,
    }
}

/// Four requests at once and a fifth while the head is mid-item: past the
/// first token, every token reaches the client within a step of D sending
/// it and no two arrive more than two steps apart, though the head hears
/// nothing for half a second at a time.
#[test]
fn a_busy_head_holds_no_decode_token_back() {
    let (live, mut steps, log) = stack();
    let max = 41;
    let mut ids: Vec<RequestId> =
        (0..4).map(|_| live.handle.submit(request(max, StopPolicy::default())).id()).collect();
    let mut seen = Seen::default();
    seen.read(&mut steps, ITEM + ITEM / 4, |_| false);
    ids.push(live.handle.submit(request(max, StopPolicy::default())).id());
    seen.read(&mut steps, Duration::from_secs(10), |s| s.ends.len() == ids.len());

    let l = log.lock().unwrap();
    let (mut worst_late, mut worst_gap, mut gaps) = (Duration::ZERO, Duration::ZERO, Vec::new());
    for id in &ids {
        let want: Vec<u32> = std::iter::once(FIRST).chain((1..max as u32).map(|n| 100 + n)).collect();
        assert_eq!((seen.ids(*id), finished(seen.ends.get(id))), (want, Some((FinishReason::Length, max))), "{id}");
        let got = &seen.tokens[id][1..];
        let sent = &l.sent[&id.raw()];
        for ((_, at), s) in got.iter().zip(sent) {
            worst_late = worst_late.max(at.saturating_duration_since(*s));
        }
        for w in got.windows(2) {
            let g = w[1].1 - w[0].1;
            worst_gap = worst_gap.max(g);
            gaps.push(g);
        }
    }
    gaps.sort();
    eprintln!(
        "decode tokens {}: gap p50 {:?} max {:?}, latest {:?} after D sent it",
        gaps.len() + ids.len(),
        gaps[gaps.len() / 2],
        worst_gap,
        worst_late
    );
    assert!(worst_late <= STEP, "a token reached the client {worst_late:?} after D sent it");
    assert!(worst_gap <= 2 * STEP, "two tokens arrived {worst_gap:?} apart");
    drop(l);
    drop(live.handle);
    live.join.join().unwrap();
}

/// A stop token D cannot see ends the request at the front and frees its
/// seats at once; a client leaving mid-decode frees them from the front
/// within a step, the head never hearing of it; a client leaving before the head
/// took its request is told to the head once, and the head retires it.
/// Every account closes: the driver drains and exits.
#[test]
fn the_front_ends_what_decodes_and_tells_the_head_the_rest() {
    let (live, mut steps, log) = stack();
    let stop = StopPolicy::new(EosPolicy::Ignore, vec![105]);
    let a = live.handle.submit(request(64, stop)).id();
    let b = live.handle.submit(request(64, StopPolicy::default()));
    let c = live.handle.submit(request(64, StopPolicy::default()));
    c.abort();
    let mut seen = Seen::default();
    seen.read(&mut steps, Duration::from_secs(5), |s| s.ids(b.id()).len() >= 4);
    let left = Instant::now();
    b.abort();
    seen.read(&mut steps, Duration::from_secs(5), |s| s.ends.contains_key(&a));
    seen.read(&mut steps, 4 * STEP, |_| false);

    let l = log.lock().unwrap();
    let ended = l.sent[&a.raw()][4];
    assert_eq!(
        (seen.ids(a), finished(seen.ends.get(&a))),
        (vec![FIRST, 101, 102, 103, 104, 105], Some((FinishReason::Stop, 6)))
    );
    assert!(
        l.aborted[&a.raw()] - ended <= STEP,
        "a stopped request's seats freed {:?} late",
        l.aborted[&a.raw()] - ended
    );
    assert!(
        l.aborted[&b.id().raw()] - left <= 2 * STEP,
        "a gone client's seats freed {:?} late",
        l.aborted[&b.id().raw()] - left
    );
    assert!(seen.ids(b.id()).len() <= 6 && !seen.ends.contains_key(&b.id()), "{:?}", seen.ids(b.id()));
    assert_eq!((l.gone.clone(), seen.ids(c.id()), l.sent.contains_key(&c.id().raw())), (vec![c.id()], vec![], false));
    drop(l);
    drop(live.handle);
    live.join.join().unwrap();
}
