//! The P/D handoff through its public API: the shape check, the push plan,
//! the D head's seats and decode loop over real pools of kern-manifest's
//! `dealt.json` and a fake set of ranks, and the P head's handoffs and its
//! front's relays driven against that decoder through every interleaving a
//! random schedule finds.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Arc;

use anyhow::Result;
use kern_manifest::types::Manifest;
use kern_pool::{Lease, Pool};
use kern_serve::pd::{
    cover, moves, pick, shapes, Ask, Decoder, Handoffs, Move, Out, Pools, Push, Refusal, Relay, Relays, Req, Run, Seat,
    Shape, Standing, Tell,
};
use kern_serve::ranks::{Act, Answer, Order};
use pegainfer_frontend::engine::{EosPolicy, FinishReason, StopPolicy};

mod common;
use common::{dealt, dealt_with, replicated, replicated_of, replicated_with, Rand};

/// The fake model: a row fed `t` hands back `next(t)`, whatever its position.
fn next(t: i64) -> i64 {
    t * 7 % 97 + 1
}

/// Ranks that land every remap the moment they hear it, answer every
/// row with [`next`], and remember every slot they were told to zero.
#[derive(Default)]
struct Fake {
    zeroed: Vec<(usize, i32)>,
}

impl Run for Fake {
    fn run(&mut self, orders: Vec<Order>) -> Result<Vec<Answer>> {
        Ok(orders
            .iter()
            .enumerate()
            .map(|(q, o)| {
                self.zeroed
                    .extend(o.acts.iter().filter_map(|a| if let Act::Zero(s) = a { Some((q, *s)) } else { None }));
                let remaps = o.acts.iter().filter(|a| matches!(a, Act::Remap(_))).count() as u64;
                let tokens =
                    o.item.as_ref().map_or(Vec::new(), |i| i.rows["token_ids"].iter().map(|&t| next(t)).collect());
                Answer { tokens, remaps, ..Answer::default() }
            })
            .collect())
    }
}

fn paged(row: u64) -> Shape {
    Shape::Paged { row }
}

fn seq(bytes: u64) -> Shape {
    Shape::Seq { bytes }
}

/// The decode tier of `dealt.json` over four members as a P head hears it.
fn pools4() -> Pools {
    let d = Decoder::new(replicated(4, 1), Fake::default(), 8).unwrap();
    d.pools(vec![String::new(); 4], shapes(&dealt(4)))
}

#[test]
fn the_decode_tier_says_what_a_stage_mirrors_and_pushes() {
    let p = pools4();
    let states = BTreeMap::from([("kv.0".into(), paged(16)), ("kv.1".into(), paged(16)), ("kda".into(), seq(24))]);
    assert_eq!((p.group, p.dealt, p.page, p.max_seq_tokens, p.placements.len()), (4, true, 16, 192, 4));
    assert_eq!((p.states, p.heads), (states, BTreeMap::from([("kda".into(), vec![(1, 64), (2, 16)])])));
}

#[test]
fn stages_cover_every_decode_state_once_or_every_mismatch_is_named() {
    let d = pools4();
    let s = |v: &[(&str, Shape)]| -> BTreeMap<String, Shape> { v.iter().map(|(n, x)| (n.to_string(), *x)).collect() };
    // A slot of the whole sequence: four members' 24 bytes, one line of its segments.
    assert!(cover(&[s(&[("kv.0", paged(16)), ("kda", seq(96))]), s(&[("kv.1", paged(16))])], &d).is_ok());
    let e = cover(&[s(&[("kv.0", paged(32)), ("kda", seq(24))]), s(&[("kv.0", paged(16)), ("mtp", paged(8))])], &d)
        .unwrap_err()
        .to_string();
    for want in [
        "`kv.0`: stages [0, 1] all hold it",
        "`kv.1`: the decode tier holds it, no prefill stage does",
        "`kda`: 24 bytes per slot on stage 0; expected 96",
        "`mtp`: stage 1 holds it, the decode tier does not",
    ] {
        assert!(e.contains(want), "{want}\n{e}");
    }
    // A row's bytes, a kind, and a slot nobody splits over a group of four.
    let mut d2 = d.clone();
    d2.heads.clear();
    let e = cover(&[s(&[("kv.0", paged(32)), ("kv.1", seq(16)), ("kda", seq(24))])], &d2).unwrap_err().to_string();
    for want in [
        "`kv.0`: 32 bytes per token on stage 0, 16 on the decode tier",
        "`kv.1`: Seq { bytes: 16 } on stage 0, Paged { row: 16 } on the decode tier",
        "`kda`: 24 bytes per slot on stage 0, 24 on each of 4 decode members and not split by heads",
    ] {
        assert!(e.contains(want), "{want}\n{e}");
    }
}

/// Distinct page ids, shuffled, `n` of them out of `0..4n`.
fn scattered(rand: &mut Rand, n: usize) -> Vec<i32> {
    let mut ids: Vec<i32> = (0..4 * n as i32).collect();
    for i in (1..ids.len()).rev() {
        ids.swap(i, rand.next(i + 1));
    }
    ids.truncate(n);
    ids
}

/// Random pushes over (w, units, length, scattered pages on both sides):
/// every position `p < len` of the source lands once, on member `p % w`
/// at its local `p / w`, through the same deal for every paged state;
/// every member gets its heads of the slot. Undealt, the member with pages
/// takes every row in order and a group of one the slot whole.
#[test]
fn a_push_lands_every_position_once_on_the_member_that_holds_it() {
    let mut rand = Rand(0xD1B5_4A32_D192_ED03);
    for case in 0..500 {
        let w = 1 + rand.next(8);
        let dealt = case % 5 != 0 || w == 1;
        let w = if dealt { w } else { 1 + rand.next(2) };
        let (unit, page) = ([4u64, 16, 64][rand.next(3)], [4u64, 16][rand.next(2)]);
        let len = 1 + rand.next(300);
        let src = scattered(&mut rand, len.div_ceil(unit as usize));
        let local = |m: usize| {
            if dealt {
                len.div_ceil(w)
            } else if m == 0 {
                len
            } else {
                0
            }
        };
        let seats: Vec<Seat> = (0..w)
            .map(|m| Seat {
                rank: 10 + m,
                pages: scattered(&mut rand, local(m).div_ceil(page as usize)),
                slot: Some(3 + m as i32),
                epoch: 0,
            })
            .collect();
        let segs = vec![(1, 8 * w as u64), (3, 4 * w as u64)];
        let d = Pools {
            handles: vec![],
            placements: vec![],
            states: BTreeMap::new(),
            group: w,
            dealt,
            heads: if w > 1 { BTreeMap::from([("kda".into(), segs.clone())]) } else { BTreeMap::new() },
            page,
            max_seq_tokens: 0,
        };
        let mine = BTreeMap::from([
            ("kv.0".into(), paged(32)),
            ("kv.1".into(), paged(32)),
            ("kda".into(), seq(20 * w as u64)),
        ]);
        let push = Push { req: 1, tier: 0, pages: src.clone(), slot: Some(2), len, seats: seats.clone() };
        let mv = moves(&push, &d, &mine, unit).unwrap();
        let mut landed = BTreeMap::new();
        let mut deals: BTreeMap<usize, Vec<_>> = BTreeMap::new();
        for x in &mv {
            match x {
                Move::Rows { state, rank, deal } => {
                    let m = rank - 10;
                    deals.entry(m).or_default().push((state.clone(), deal.clone()));
                    if state != "kv.0" {
                        continue;
                    }
                    for r in deal.runs() {
                        for i in 0..r.rows {
                            let (s, t) = (r.src_row + i * r.src_stride, r.dst_row + i);
                            let p = src.iter().position(|&id| id as u64 == s / unit).unwrap() as u64 * unit + s % unit;
                            let l = seats[m].pages.iter().position(|&id| id as u64 == t / page).unwrap() as u64 * page
                                + t % page;
                            assert!(landed.insert(p, (m, l)).is_none(), "case {case}: position {p} twice");
                        }
                    }
                }
                Move::Part { rank, from, into, part, .. } => {
                    assert_eq!((w > 1, *from, *into, *part), (true, 2, 3 + (rank - 10) as i32, (rank - 10) as u64))
                }
                Move::Slot { rank, from, into, .. } => assert_eq!((w, *rank, *from, *into), (1, 10, 2, 3)),
            }
        }
        let want: BTreeMap<u64, (usize, u64)> = (0..len as u64)
            .map(|p| (p, if dealt { ((p % w as u64) as usize, p / w as u64) } else { (0, p) }))
            .collect();
        assert_eq!(landed, want, "case {case}: w {w} dealt {dealt} unit {unit} page {page} len {len}");
        for v in deals.values() {
            assert_eq!((v.len(), &v[0].1), (2, &v[1].1), "case {case}: one deal for every paged state");
        }
        let parts = mv.iter().filter(|x| !matches!(x, Move::Rows { .. })).count();
        assert_eq!(parts, w, "case {case}: one slot move per member");
        // A stage holding no D state has no pages of its own: nothing moves.
        assert_eq!(moves(&push, &d, &BTreeMap::new(), 1).unwrap(), vec![], "case {case}");
    }
}

#[test]
fn a_seat_is_adopted_on_every_member_and_decodes_to_max_tokens() {
    let mut d = Decoder::new(replicated(4, 1), Fake::default(), 8).unwrap();
    // ⌈40 / 4⌉ = 10 local positions: a page and a slot on each member.
    let t = d.ask(Ask::Reserve { req: 1, tokens: 40 }).unwrap();
    let [Tell::Grant { req: 1, seats, load }] = t.as_slice() else { panic!("{t:?}") };
    let shape: Vec<(usize, usize, bool, u64)> =
        seats.iter().map(|s| (s.rank, s.pages.len(), s.slot.is_some(), s.epoch)).collect();
    assert_eq!(
        (shape, d.seated(), (load.seats, load.pages)),
        (vec![(0, 1, true, 0), (1, 1, true, 0), (2, 1, true, 0), (3, 1, true, 0)], vec![1], (1, 4))
    );
    // Nothing runs before the start; the pads' zeroing lands.
    assert_eq!(d.step().unwrap(), vec![]);
    d.ask(Ask::Start { req: 1, first: 5, len: 30, max_tokens: 3 }).unwrap();
    let (a, b) = (next(5) as u32, next(next(5)) as u32);
    let c = next(b as i64) as u32;
    let steps: Vec<_> = (0..3)
        .map(|_| match d.step().unwrap().as_slice() {
            [Tell::Tokens { step, load }] => (step.clone(), load.seats, load.pages),
            t => panic!("{t:?}"),
        })
        .collect();
    // The row holds its seat and pages until its last token is out.
    assert_eq!(steps, [(vec![(1, a, false)], 1, 4), (vec![(1, b, false)], 1, 4), (vec![(1, c, true)], 0, 0)]);
    // Only the pads' slots were ever zeroed: a pushed slot is the push's.
    assert_eq!((d.seated(), d.busy(), d.step().unwrap()), (vec![], false, vec![]));
    let seat_slots: Vec<(usize, i32)> = seats.iter().map(|s| (s.rank, s.slot.unwrap())).collect();
    assert!(seat_slots.iter().all(|s| !d.exec().zeroed.contains(s)), "{:?}", d.exec().zeroed);
}

#[test]
fn a_full_tier_is_busy_until_a_seat_frees_or_a_remap_lands() {
    let mut d = Decoder::new(replicated(2, 1), Fake::default(), 1).unwrap();
    assert!(matches!(d.ask(Ask::Reserve { req: 1, tokens: 20 }).unwrap()[..], [Tell::Grant { req: 1, .. }]));
    assert_eq!(d.ask(Ask::Reserve { req: 2, tokens: 20 }).unwrap(), vec![Tell::Busy { req: 2 }]);
    assert_eq!(d.step().unwrap(), vec![]);
    assert_eq!(d.ask(Ask::Abort { req: 1 }).unwrap(), vec![Tell::Room]);
    // An abort of what already ended holds nothing and owes nothing.
    assert_eq!(d.ask(Ask::Abort { req: 1 }).unwrap(), vec![]);
    // Two members of 48 local positions hold 96.
    let refused = Tell::Refuse { req: 3, why: Refusal::ContextLength { limit: 96 } };
    assert_eq!(d.ask(Ask::Reserve { req: 3, tokens: 97 }).unwrap(), vec![refused]);
    assert!(matches!(d.ask(Ask::Reserve { req: 4, tokens: 20 }).unwrap()[..], [Tell::Grant { req: 4, .. }]));
    assert!(d.ask(Ask::Start { req: 4, first: 1, len: 15, max_tokens: 6 }).is_err(), "past the seat");
    assert!(d.ask(Ask::Start { req: 9, first: 1, len: 1, max_tokens: 1 }).is_err(), "no seat");
    assert!(d.ask(Ask::Reserve { req: 4, tokens: 20 }).is_err(), "reserved twice");

    // Three slots a member (the null one, the pad's, one more), each three
    // chunks: the second seat waits for a slot carved out of free pages, a
    // member at a time (an adopt stops at the first member that has to
    // wait), each remap's landing makes room, and the grant follows both
    // ranks' next placements, its seats in them.
    let mut d = Decoder::new(replicated_of(dealt_with(2, 768), 2, 1, 3), Fake::default(), 8).unwrap();
    assert!(matches!(d.ask(Ask::Reserve { req: 1, tokens: 20 }).unwrap()[..], [Tell::Grant { .. }]));
    for _ in 0..2 {
        assert_eq!(
            (d.ask(Ask::Reserve { req: 2, tokens: 20 }).unwrap(), d.busy()),
            (vec![Tell::Busy { req: 2 }], true)
        );
        assert_eq!(d.step().unwrap(), vec![Tell::Room]);
    }
    let t = d.ask(Ask::Reserve { req: 2, tokens: 20 }).unwrap();
    let [Tell::Placed { rank: 0, epoch: 1, .. }, Tell::Placed { rank: 1, epoch: 1, .. }, Tell::Grant { seats, .. }] =
        t.as_slice()
    else {
        panic!("{t:?}")
    };
    assert_eq!(seats.iter().map(|s| (s.rank, s.epoch)).collect::<Vec<_>>(), [(0, 1), (1, 1)]);
    // A placement that did not move is not told again.
    let t =
        d.ask(Ask::Abort { req: 2 }).unwrap().into_iter().chain(d.ask(Ask::Reserve { req: 3, tokens: 20 }).unwrap());
    assert!(matches!(t.collect::<Vec<_>>()[..], [Tell::Grant { .. }]));
}

#[test]
fn a_tier_takes_one_reserve_at_a_time_none_while_busy_and_names_itself_in_its_pushes() {
    let mut d = Decoder::new(replicated(2, 1), Fake::default(), 1).unwrap();
    let mut h = Handoffs::new(STAGES, 1);
    let mut hear =
        |h: &mut Handoffs, a: Ask| d.ask(a).unwrap().into_iter().for_each(|t| assert_eq!(h.hear(t).unwrap(), vec![]));
    let a = h.reserve(1, 20, Relay::new(5, stop())).unwrap();
    assert_eq!((h.ready(), h.reserve(2, 20, Relay::new(5, stop()))), (false, None));
    hear(&mut h, a);
    let a = h.reserve(2, 20, Relay::new(5, stop())).unwrap();
    hear(&mut h, a);
    assert_eq!((h.ready(), h.standing(1), h.standing(2)), (false, Standing::Granted, Standing::Unasked));
    assert_eq!(h.hear(Tell::Room).unwrap(), vec![]);
    assert!(h.ready());
    let plain = Manifest::from_json(include_str!("../../kern-manifest/tests/fixtures/plain.json")).unwrap();
    let pool = Arc::new(Pool::new(&plain, 64, 1024, 16, None).unwrap().0);
    h.admitted(1);
    let outs = h.prefilled(1, &pool.lease(10).unwrap(), 10);
    assert!(matches!(&outs[..], [Out::Push(p)] if (p.req, p.tier, p.len) == (1, 1, 10)), "{outs:?}");
}

#[test]
fn a_request_goes_to_the_lighter_of_two_tiers_it_draws_and_never_the_heaviest() {
    // Every assignment of 0..=4 tiers, each busy or holding 0..=2 seats, against 64 requests.
    let choices = [None, Some(0), Some(1), Some(2)];
    let mut ties = [0; 2];
    for n in 0..=4u32 {
        for code in 0..4usize.pow(n) {
            let seats: Vec<Option<usize>> = (0..n).map(|i| choices[code / 4usize.pow(i) % 4]).collect();
            let ready: Vec<usize> = (0..seats.len()).filter(|&i| seats[i].is_some()).collect();
            for req in 0..64 {
                let got = pick(req, &seats);
                assert_eq!(got, pick(req, &seats), "the same request, the same tier");
                let Some(i) = got else {
                    assert!(ready.is_empty(), "{seats:?}: none picked of {ready:?}");
                    continue;
                };
                let s = seats[i].expect("a ready tier");
                let as_heavy_elsewhere = ready.iter().any(|&j| j != i && seats[j].unwrap() >= s);
                assert!(ready.len() == 1 || as_heavy_elsewhere, "{seats:?}: {i} is the heaviest alone");
                if ready.len() == 2 {
                    let other = seats[ready[0] + ready[1] - i].unwrap();
                    assert!(s <= other, "{seats:?}: of two, {i} is heavier");
                    if s == other && seats == [Some(0), Some(0)] {
                        ties[i] += 1;
                    }
                }
            }
        }
    }
    // A tie goes to the tier drawn first, which the request's hash spreads.
    assert!(ties[0] > 16 && ties[1] > 16, "{ties:?}");
}

const STAGES: usize = 3;
const STOP: u32 = 7;
/// Prompts P seats; longer ones it rejects after their grant.
const P_LIMIT: usize = 64;

fn first_token(req: Req) -> u32 {
    (req * 13 % 97 + 1) as u32
}

fn stop() -> StopPolicy {
    StopPolicy::new(EosPolicy::Ignore, vec![STOP])
}

/// What a client finally got.
#[derive(Debug, Clone, PartialEq, Eq)]
enum End {
    Finished(Vec<u32>, FinishReason),
    Rejected,
    Retired,
}

struct Client {
    prompt: usize,
    max_tokens: usize,
    tokens: Vec<u32>,
    end: Option<End>,
    gone: bool,
}

/// A P head, its stages and the client, as the head would drive the
/// handoffs, against a real decoder: every message is a queue entry, and
/// a schedule picks which one moves next.
struct World {
    h: Handoffs,
    r: Relays,
    d: Decoder<Fake>,
    pool: Arc<Pool>,
    clients: BTreeMap<Req, Client>,
    waiting: VecDeque<Req>,
    prefilling: BTreeMap<Req, Lease>,
    to_d: VecDeque<Ask>,
    to_p: VecDeque<Tell>,
    pushes: Vec<(Req, usize)>,
    firsts: Vec<Req>,
    /// Pushes issued and not landed on every stage, by request: the stages still pushing.
    in_flight: BTreeMap<Req, usize>,
    first_in: BTreeSet<Req>,
    moved: u64,
}

impl World {
    fn new(rand: &mut Rand) -> World {
        // kern-manifest's plain contract with room for 128 tokens a row.
        let plain =
            include_str!("../../kern-manifest/tests/fixtures/plain.json").replace("[\"seqs\", 3]", "[\"seqs\", 8]");
        let plain = Manifest::from_json(&plain).unwrap();
        let mut clients = BTreeMap::new();
        for req in 1..=8 {
            let prompt = match rand.next(10) {
                0 => 97,
                1 => P_LIMIT + 1,
                _ => 1 + rand.next(60),
            };
            let max_tokens = [1, 2, 3, 5, 9, 30][rand.next(6)];
            clients.insert(req, Client { prompt, max_tokens, tokens: Vec::new(), end: None, gone: false });
        }
        World {
            h: Handoffs::new(STAGES, 0),
            r: Relays::new(),
            d: Decoder::new(replicated_with(2, 1, 6), Fake::default(), 3).unwrap(),
            pool: Arc::new(Pool::new(&plain, 64, 1024, 16, None).unwrap().0),
            waiting: clients.keys().copied().collect(),
            clients,
            prefilling: BTreeMap::new(),
            to_d: VecDeque::new(),
            to_p: VecDeque::new(),
            pushes: Vec::new(),
            firsts: Vec::new(),
            in_flight: BTreeMap::new(),
            first_in: BTreeSet::new(),
            moved: 0,
        }
    }

    fn apply(&mut self, outs: Vec<Out>) {
        for o in outs {
            match o {
                Out::Ask(a) => {
                    match &a {
                        Ask::Abort { req } => assert!(!self.in_flight.contains_key(req), "abort of {req} mid-push"),
                        Ask::Start { req, first, len, max_tokens } => {
                            let c = &self.clients[req];
                            assert!(
                                !self.in_flight.contains_key(req) && self.first_in.contains(req),
                                "early start of {req}"
                            );
                            assert!(self.r.holds(*req), "{req} starts before the front relays it");
                            assert_eq!((*first, *len, *max_tokens), (first_token(*req), c.prompt, c.max_tokens - 1));
                        }
                        Ask::Reserve { req, tokens } => {
                            let c = &self.clients[req];
                            assert_eq!((*tokens, Relay::hands_off(c.max_tokens)), (c.prompt + c.max_tokens, true));
                        }
                    }
                    self.to_d.push_back(a);
                }
                Out::Push(p) => {
                    assert_eq!((p.len, p.seats.len()), (self.clients[&p.req].prompt, 2));
                    self.in_flight.insert(p.req, STAGES);
                    self.pushes.extend((0..STAGES).map(|s| (p.req, s)));
                    self.firsts.push(p.req);
                }
                Out::Emit { req, tok } => {
                    let c = self.clients.get_mut(&req).unwrap();
                    assert!(c.end.is_none() && !c.gone, "a token for {req} after its end");
                    c.tokens.push(tok);
                }
                Out::Finish { req, why, .. } => {
                    let c = self.clients.get_mut(&req).unwrap();
                    assert!(c.end.is_none(), "{req} finished twice");
                    c.end = Some(End::Finished(c.tokens.clone(), why));
                }
                Out::Decode { req, relay } => {
                    assert!(self.h.standing(req) == Standing::Unasked && !self.r.holds(req), "{req} handed over twice");
                    self.r.start(req, relay);
                }
                // Its decoder's 24-byte slots never move a chunk.
                Out::Place { .. } => panic!("a placement moved"),
                Out::Reject { req, why } => {
                    assert_eq!(self.waiting.pop_front(), Some(req), "a reject of {req}, not the front");
                    let c = self.clients.get_mut(&req).unwrap();
                    assert_eq!((why, c.end.is_none()), (Refusal::ContextLength { limit: 96 }, true));
                    c.end = Some(End::Rejected);
                }
            }
        }
        self.moved += 1;
    }

    /// The decoder's answers, checking that a seat it lets go of has no
    /// push in flight.
    fn tier(&mut self, tells: Vec<Tell>, before: Vec<Req>) {
        let after: BTreeSet<Req> = self.d.seated().into_iter().collect();
        for r in before.iter().filter(|r| !after.contains(r)) {
            assert!(!self.in_flight.contains_key(r), "seat of {r} freed mid-push");
        }
        self.moved += !tells.is_empty() as u64;
        self.to_p.extend(tells);
    }

    /// The head's admission, once, on the waiting front.
    fn admit(&mut self, rand: &mut Rand) {
        let Some(&req) = self.waiting.front() else { return };
        let c = &self.clients[&req];
        if c.gone {
            self.waiting.pop_front();
            self.clients.get_mut(&req).unwrap().end = Some(End::Retired);
            let outs = self.h.abort(req);
            return self.apply(outs);
        }
        if !Relay::hands_off(c.max_tokens) {
            // One token: the prefill's own, and D never hears of it.
            self.waiting.pop_front();
            let c = self.clients.get_mut(&req).unwrap();
            c.tokens = vec![first_token(req)];
            let why = if first_token(req) == STOP { FinishReason::Stop } else { FinishReason::Length };
            c.end = Some(End::Finished(c.tokens.clone(), why));
            self.moved += 1;
            return;
        }
        match self.h.standing(req) {
            Standing::Unasked => {
                if let Some(a) = self.h.reserve(req, c.prompt + c.max_tokens, Relay::new(c.max_tokens, stop())) {
                    self.apply(vec![Out::Ask(a)]);
                }
            }
            Standing::Reserving => {}
            // P's own pool is busy now and then: the grant waits with it.
            Standing::Granted if rand.next(4) == 0 => {}
            Standing::Granted if c.prompt > P_LIMIT => {
                self.waiting.pop_front();
                self.clients.get_mut(&req).unwrap().end = Some(End::Rejected);
                let outs = self.h.release(req);
                self.apply(outs);
            }
            Standing::Granted => {
                let lease = self.pool.lease(c.prompt).unwrap();
                self.waiting.pop_front();
                self.h.admitted(req);
                self.prefilling.insert(req, lease);
                self.moved += 1;
            }
            Standing::Underway => panic!("{req} underway while waiting"),
        }
    }

    /// One thing happens: `k` picks what, `rand` among its candidates.
    fn act(&mut self, k: usize, rand: &mut Rand) {
        match k {
            0 => self.admit(rand),
            1 if !self.prefilling.is_empty() => {
                let req = *self.prefilling.keys().nth(rand.next(self.prefilling.len())).unwrap();
                let lease = self.prefilling.remove(&req).unwrap();
                let outs = self.h.prefilled(req, &lease, self.clients[&req].prompt);
                // The head retires the lease into its cache right after.
                drop(lease);
                self.apply(outs);
            }
            2 if !self.to_d.is_empty() => {
                let before = self.d.seated();
                let a = self.to_d.pop_front().unwrap();
                let tells = self.d.ask(a).unwrap();
                self.moved += 1;
                self.tier(tells, before);
            }
            3 => {
                let before = self.d.seated();
                let tells = self.d.step().unwrap();
                self.tier(tells, before);
            }
            4 if !self.to_p.is_empty() => {
                // The connection's reader hands the tokens to the front, the rest to the head.
                let outs = match self.to_p.pop_front().unwrap() {
                    Tell::Tokens { step, .. } => self.r.hear(step).unwrap(),
                    t => self.h.hear(t).unwrap(),
                };
                self.apply(outs);
            }
            5 if !self.pushes.is_empty() => {
                let (req, s) = self.pushes.swap_remove(rand.next(self.pushes.len()));
                let left = self.in_flight.get_mut(&req).unwrap();
                *left -= 1;
                if *left == 0 {
                    self.in_flight.remove(&req);
                }
                let outs = self.h.pushed(req, s).unwrap();
                self.apply(outs);
            }
            6 if !self.firsts.is_empty() => {
                let req = self.firsts.swap_remove(rand.next(self.firsts.len()));
                self.first_in.insert(req);
                let outs = self.h.first(req, first_token(req)).unwrap();
                self.apply(outs);
            }
            7 if rand.next(8) == 0 => {
                let live: Vec<Req> =
                    self.clients.iter().filter(|(_, c)| c.end.is_none() && !c.gone).map(|(&r, _)| r).collect();
                let Some(&req) = live.get(rand.next(live.len().max(1))) else { return };
                self.clients.get_mut(&req).unwrap().gone = true;
                // Decoding: the front retires it and frees its seats; admitted: the head retires it
                // and tells the handoffs; waiting: admission does.
                let outs = match (self.r.holds(req), self.h.open().contains(&req)) {
                    (true, _) => self.r.abort(req),
                    (false, true) => self.h.abort(req),
                    (false, false) => return,
                };
                self.clients.get_mut(&req).unwrap().end = Some(End::Retired);
                self.apply(outs);
            }
            _ => {}
        }
    }
}

/// Random schedules of eight requests through a P head with three stages
/// and a decode tier of two members, three seats and slots to carve: no
/// seat is freed and no abort is sent while a push is in flight, no start
/// precedes the last push, the first token and the front's relay, D's
/// tokens are the front's alone, a busy tier is asked again
/// once it has room, a one-token request never reaches D, and every
/// client ends exactly once, a finished one with the greedy chain up to
/// its stop or `max_tokens` whatever the order things arrived in.
#[test]
fn every_schedule_hands_off_in_order_and_ends_every_request_once() {
    let mut rand = Rand(0x2545_F491_4F6C_DD1D);
    for case in 0..200 {
        let mut w = World::new(&mut rand);
        for _ in 0..300 {
            let k = rand.next(8);
            w.act(k, &mut rand);
        }
        // Then everything that can move moves, in a fixed order, until nothing does.
        for round in 0.. {
            assert!(round < 10_000, "case {case}: no end in sight");
            let moved = w.moved;
            for k in 0..7 {
                w.act(k, &mut rand);
            }
            if w.moved == moved && w.prefilling.is_empty() && w.waiting.is_empty() && !w.d.busy() {
                break;
            }
        }
        for (req, c) in &w.clients {
            let mut chain = vec![first_token(*req)];
            while chain.len() < c.max_tokens && *chain.last().unwrap() != STOP {
                chain.push(next(*chain.last().unwrap() as i64) as u32);
            }
            let why = if *chain.last().unwrap() == STOP { FinishReason::Stop } else { FinishReason::Length };
            match &c.end {
                Some(End::Finished(t, r)) => assert_eq!((t, *r), (&chain, why), "case {case}: request {req}"),
                Some(End::Rejected) => assert!(c.prompt > P_LIMIT, "case {case}: request {req} rejected"),
                Some(End::Retired) => assert!(c.gone),
                None => panic!("case {case}: request {req} never ended: {:?}", w.h.standing(*req)),
            }
        }
        assert_eq!((w.h.live(), w.r.live(), w.d.seated(), w.in_flight.len()), (0, 0, vec![], 0), "case {case}");
    }
}
