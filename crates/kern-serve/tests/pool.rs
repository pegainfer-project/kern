//! The scheduler over a pooled tier, driven as infront drives it, its
//! ranks a fake that answers every pooled act a few replies late out of
//! one shared in-memory pool: a turn kept on one instance is fetched by
//! another and prefilled from where it ends, and a pool that loses puts,
//! gets or probes at random changes what the client gets by nothing.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use infront_engine::SamplingParams;
use infront_engine::{
    spawn_scheduler, EosPolicy, LiveScheduler, Request, RequestId, StepReceiver, StopPolicy, Terminal,
};
use kern_manifest::types::Manifest;
use kern_manifest::{verify, Protocol};
use kern_pool::{Keys, Pool};
use kern_serve::pd::Run;
use kern_serve::ranks::{Act, Answer, Groups, Landed, Order, Ranks};
use kern_serve::scheduler::{KernScheduler, Policy};

mod common;
use common::Rand;

/// The fake model: a row fed `t` hands back `next(t)`.
fn next(t: i64) -> i64 {
    t * 7 % 97 + 1
}

/// kern-manifest's plain contract: 16-token pages, 8-token chunks, a
/// recurrent slot, a prefill that only writes state.
fn manifest() -> Manifest {
    Manifest::from_json(include_str!("../../kern-manifest/tests/fixtures/plain.json")).unwrap()
}

/// The pool every instance shares, and how often it loses an act.
#[derive(Default)]
struct Store {
    keys: BTreeSet<String>,
    /// Percent of puts, gets and probes that fail.
    lose: usize,
    rand: Option<Rand>,
}

impl Store {
    fn lost(&mut self) -> bool {
        self.rand.as_mut().is_some_and(|r| r.next(100) < self.lose)
    }
}

/// What one instance's rank saw.
#[derive(Default)]
struct Log {
    /// Each prefill item: the first position fed and how many.
    prefills: Vec<(i64, usize)>,
    /// Pooled acts as they were issued, by kind.
    acts: Vec<&'static str>,
}

/// One rank answering every row with [`next`], its pooled acts out of
/// `store` a few replies later.
struct Fake {
    store: Arc<Mutex<Store>>,
    log: Arc<Mutex<Log>>,
    rand: Rand,
    /// Answers not yet given, each with the replies still to wait.
    later: VecDeque<(usize, Landed)>,
}

impl Fake {
    fn act(&mut self, a: &Act) {
        let mut st = self.store.lock().unwrap();
        let (kind, landed) = match a {
            Act::Probe { id, keys } => {
                let lost = st.lost();
                ("probe", Landed { id: *id, ok: keys.iter().map(|k| !lost && st.keys.contains(k)).collect() })
            }
            Act::Put { id, objects } => {
                let ok = objects
                    .iter()
                    .map(|o| {
                        let ok = !st.lost();
                        if ok {
                            st.keys.insert(o.key.clone());
                        }
                        ok
                    })
                    .collect();
                ("put", Landed { id: *id, ok })
            }
            Act::Get { id, objects } => {
                let ok = objects.iter().map(|o| st.keys.contains(&o.key) && !st.lost()).collect();
                ("get", Landed { id: *id, ok })
            }
            Act::Mark { id, key } => {
                st.keys.insert(key.clone());
                ("mark", Landed { id: *id, ok: vec![true] })
            }
            Act::Zero(_) | Act::Copy(_) => return,
        };
        self.log.lock().unwrap().acts.push(kind);
        let wait = self.rand.next(3);
        self.later.push_back((wait, landed));
    }
}

impl Run for Fake {
    fn run(&mut self, orders: Vec<Order>) -> anyhow::Result<Vec<Answer>> {
        assert_eq!(orders.len(), 1);
        let o = &orders[0];
        for a in &o.acts {
            self.act(a);
        }
        let tokens = o.item.as_ref().map_or(Vec::new(), |i| {
            let ids = i.rows["token_ids"].ints().unwrap();
            if i.program == "prefill" {
                let pos = i.rows["positions"].ints().unwrap();
                self.log.lock().unwrap().prefills.push((pos[0], ids.len()));
            }
            ids.iter().map(|&t| next(t)).collect()
        });
        let mut landed = Vec::new();
        for (wait, l) in std::mem::take(&mut self.later) {
            match wait {
                0 => landed.push(l),
                n => self.later.push_back((n - 1, l)),
            }
        }
        Ok(vec![Answer { tokens, counts: Vec::new(), landed }])
    }
}

struct Instance {
    live: LiveScheduler,
    steps: StepReceiver,
    log: Arc<Mutex<Log>>,
}

/// A scheduler over one rank of 32 pages and 8 slots, on `store` when
/// `pooled`.
fn instance(store: &Arc<Mutex<Store>>, seed: u64, pooled: bool) -> Instance {
    let m = manifest();
    let p = Protocol::check(&verify(m.clone()).unwrap()).unwrap();
    let ranks = Ranks::new(p, Groups::new(1, 1).unwrap(), vec![Pool::new(&m, 32, 8).unwrap()]).unwrap();
    let log = Arc::new(Mutex::new(Log::default()));
    let fake = Fake { store: Arc::clone(store), log: Arc::clone(&log), rand: Rand(seed | 1), later: VecDeque::new() };
    let keys = pooled.then(|| Keys::new(&[b"plain", b"weights"]));
    let sched = KernScheduler::new(ranks, fake, Policy { chunk: None, max_seqs: 4, rows: None }, keys).unwrap();
    let mut live = spawn_scheduler("pooled", sched);
    let steps = live.handle.take_steps().unwrap();
    Instance { live, steps, log }
}

impl Instance {
    fn submit(&self, prompt: Vec<u32>, history: (Option<usize>, Vec<usize>), max_tokens: usize) -> RequestId {
        let request = Request {
            prompt_tokens: prompt,
            params: SamplingParams::default(),
            history_tokens: history.0,
            history_marks: history.1,
            stop_policy: StopPolicy::new(EosPolicy::Ignore, Vec::new()),
            max_tokens,
            logprobs: None,
            prompt_logprobs: None,
        };
        self.live.handle.submit(request).id()
    }

    /// Every request's tokens once all of `ids` ended.
    fn finish(&mut self, ids: &[RequestId]) -> BTreeMap<RequestId, Vec<u32>> {
        let mut tokens: BTreeMap<RequestId, Vec<u32>> = BTreeMap::new();
        let mut ended = BTreeSet::new();
        let t0 = Instant::now();
        while ended.len() < ids.len() {
            assert!(t0.elapsed() < Duration::from_secs(20), "ended {ended:?} of {ids:?}");
            match self.steps.try_recv() {
                Ok(s) => {
                    for u in s.updates {
                        tokens.entry(u.id).or_default().extend(u.tokens.iter().copied());
                        if let Some(t) = u.terminal {
                            assert!(matches!(t, Terminal::Finished { .. }), "{}: {t:?}", u.id);
                            ended.insert(u.id);
                        }
                    }
                }
                Err(_) => std::thread::sleep(Duration::from_micros(200)),
            }
        }
        tokens
    }
}

/// A conversation: turn `k` is the turns before it, each a prompt, the
/// two-token generation prompt and the reply, then `k`'s own tokens; the
/// history is where the generation prompt begins, the marks where the
/// earlier turns' did, newest first.
fn turns(seed: u64, n: usize) -> Vec<(Vec<u32>, (Option<usize>, Vec<usize>))> {
    let mut r = Rand(seed * 977 + 13);
    let mut prompt: Vec<u32> = (0..6 + r.next(7)).map(|_| 2 + r.next(90) as u32).collect();
    let mut out: Vec<(Vec<u32>, (Option<usize>, Vec<usize>))> = Vec::new();
    for _ in 0..n {
        let mark = prompt.len();
        prompt.extend([1, 1]);
        let earlier = out.iter().rev().filter_map(|(_, (m, _))| *m).collect();
        out.push((prompt.clone(), (Some(mark), earlier)));
        // The reply the fake gives: the greedy chain from the last token.
        let mut t = 1;
        for _ in 0..3 {
            t = next(t);
            prompt.push(t as u32);
        }
        prompt.extend((0..3 + r.next(6)).map(|_| 2 + r.next(90) as u32));
    }
    out
}

#[test]
fn a_turn_kept_on_one_instance_is_fetched_by_the_next() {
    let store = Arc::new(Mutex::new(Store::default()));
    let conversation = turns(1, 2);
    let (first, mark) = conversation[0].clone();
    let m = mark.0.unwrap();

    let mut a = instance(&store, 1, true);
    let id = a.submit(first.clone(), mark, 3);
    let got_a = a.finish(&[id]);
    {
        let log = a.log.lock().unwrap();
        // Prefilled from 0 to the mark, then on past it; the snapshot at
        // the mark was put and marked.
        assert_eq!(log.prefills[0].0, 0);
        assert!(log.prefills.iter().any(|&(p, n)| p as usize + n == m), "{:?}", log.prefills);
        assert_eq!(log.acts, ["probe", "put", "mark"]);
    }
    // One page per whole page, the state, the mark.
    assert_eq!(store.lock().unwrap().keys.len(), m / 16 + 2);

    // The next turn on another instance, sharing nothing but the pool.
    let (second, mark2) = conversation[1].clone();
    assert_eq!(mark2.1, [m]);
    let mut b = instance(&store, 2, true);
    let id_b = b.submit(second.clone(), mark2.clone(), 3);
    let got_b = b.finish(&[id_b]);
    {
        let log = b.log.lock().unwrap();
        assert_eq!(&log.acts[..2], ["probe", "get"]);
        // Nothing before the mark was prefilled.
        assert_eq!(log.prefills[0].0 as usize, m, "{:?}", log.prefills);
    }
    // The same tokens as an instance without a pool gives.
    let mut c = instance(&store, 3, false);
    let id_c = c.submit(second, mark2, 3);
    let got_c = c.finish(&[id_c]);
    assert_eq!((got_b[&id_b].clone(), got_a[&id].len()), (got_c[&id_c].clone(), 3));
    assert_eq!(c.log.lock().unwrap().prefills[0].0, 0);
}

#[test]
fn a_pool_that_loses_acts_changes_nothing_the_client_sees() {
    let mut fetched = 0;
    for seed in 0..40u64 {
        let plain = Arc::new(Mutex::new(Store::default()));
        let lossy = Arc::new(Mutex::new(Store {
            lose: 10 + (seed as usize % 4) * 25,
            rand: Some(Rand(seed + 7)),
            ..Store::default()
        }));
        let conversations: Vec<_> = (0..3).map(|c| turns(seed * 10 + c, 3)).collect();
        let mut want = Vec::new();
        for (k, store) in [(0, &plain), (1, &lossy)] {
            let mut inst = instance(store, seed * 3 + k, true);
            let mut got = Vec::new();
            // Every conversation's turns in order, the three interleaved.
            for turn in 0..3 {
                let ids: Vec<RequestId> =
                    conversations.iter().map(|c| inst.submit(c[turn].0.clone(), c[turn].1.clone(), 3)).collect();
                let tokens = inst.finish(&ids);
                got.extend(ids.iter().map(|id| tokens[id].clone()));
            }
            match k {
                0 => want = got,
                _ => assert_eq!(got, want, "seed {seed}"),
            }
            let metrics = inst.live.handle.metrics();
            assert_eq!((metrics.num_running_reqs, metrics.num_waiting_reqs), (0, 0), "seed {seed}");
        }
        // A second instance over the lossy pool: whatever landed is used,
        // whatever did not is prefilled.
        let mut inst = instance(&lossy, seed * 3 + 2, true);
        let ids: Vec<RequestId> =
            conversations.iter().map(|c| inst.submit(c[2].0.clone(), c[2].1.clone(), 3)).collect();
        let tokens = inst.finish(&ids);
        let got: Vec<Vec<u32>> = ids.iter().map(|id| tokens[id].clone()).collect();
        assert_eq!(got, want[6..], "seed {seed}");
        fetched += inst.log.lock().unwrap().acts.iter().filter(|&&a| a == "get").count();
    }
    assert!(fetched > 20, "{fetched} fetches over every seed");
}
