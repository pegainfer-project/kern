//! The ranks' shell: every GPU is one process, running what the head's
//! [`Ranks`] decided for it.
//!
//! Rank 0 is the head: the HTTP front end, the scheduler, every rank's
//! accounting, and the head's own GPU, which it drives in-process. Every
//! other rank is a process started with `--head host:port --rank q`; it
//! loads its place in the topology, connects and says [`Hello`]: its rank,
//! the budget its pool was cut from, its [`Offer`] (the manifest it loaded,
//! its exported allocations, the id of every communicator it leads) and
//! the handle of its pool (which a prefill tier pushes into when these
//! ranks decode for one, `pd`). Once every rank has said hello the head
//! answers each with a [`Table`]: every group's members' offers in member
//! order, and the host tier to reserve. Each rank then connects
//! ([`Runtime::connect`]: maps its peers, joins its communicators), runs
//! what the manifest runs once and reserves its host tier.
//!
//! From then on the head sends every rank an [`Order`] and reads back its
//! [`Answer`], over one TCP stream of frames (a `u32` length, then JSON) as
//! the pipeline's connections are. The remote orders go out on threads of
//! their own while the head runs its own, and every answer is read after:
//! a rank's kernels wait on its peers' (an EP dispatch, the group
//! collectives), so the head must never wait on its device with an order
//! still unsent, and sending them one by one ahead of its own kept every
//! device idle for as long as the encoding took. A rank answers once its device is done with the order's item,
//! having checked the manifest's error word; a rank that fails exits, and
//! once the tables are out a closed connection ends the process on the
//! other side ([`crate::watch`]): a member waiting on a dead one's
//! collective would otherwise wait forever.
//!
//! Every item leaves one debug line per rank, `rank item`: the rank's
//! wall-clock microseconds at issue and at done (`kern_serve=debug`), and
//! the head adds `ranks step` with how long it waited on the slowest
//! remote answer after its own was done, so a step's time splits into
//! the ranks' device time and the wire.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{bail, ensure, Context, Result};
use kern_manifest::types::Fill;
use kern_manifest::{Protocol, Verified};
use kern_run::Weights;
use kern_runtime::{Budget, Capacity, GroupRank, Landing, Offer, Runtime, Topology};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::logline;
use crate::pp::{connect, recv, send};
use crate::ranks::{Act, Answer, Groups, Order, Ranks};

/// How long a rank keeps trying to reach a head that is not up yet.
const CONNECT: Duration = Duration::from_secs(600);

/// How every rank loads: the same flags on every process.
pub struct Load {
    pub gpu: usize,
    pub kernels: Option<PathBuf>,
    pub capacity: Capacity,
    pub eager: bool,
}

/// A rank introducing itself to the head.
#[derive(Debug, Serialize, Deserialize)]
pub struct Hello {
    pub rank: usize,
    pub chunk: u64,
    pub chunks: u32,
    pub first_slots: usize,
    pub tokens: Option<u64>,
    /// The rank's [`Offer`], hex.
    pub offer: String,
    /// The pool's `PoolHandle`, hex; none without fabric handles.
    pub pool: Option<String>,
}

/// The head's answer: by group, every member's offer (hex) in member
/// order, and the host tier to reserve as (host pages, host slots), the
/// units the head's [`Host`](kern_pool::Host) counts.
#[derive(Debug, Serialize, Deserialize)]
pub struct Table {
    pub peers: BTreeMap<String, Vec<String>>,
    pub host: (u64, u64),
}

/// The ranks a manifest runs as: the size its groups other than `tp`
/// share (`tp` alone: its size), split into `tp` group batches; one rank
/// without a topology.
pub fn world(m: &Verified) -> Result<Groups> {
    let t = m.group_size("tp").unwrap_or(1) as usize;
    let others: BTreeSet<u64> =
        m.topology.iter().flat_map(|x| &x.groups).filter(|(g, _)| *g != "tp").map(|(_, &n)| n).collect();
    let n = match others.into_iter().collect::<Vec<_>>().as_slice() {
        [] => t,
        [n] => *n as usize,
        sizes => {
            bail!("the manifest's topology groups other than `tp` differ in size ({sizes:?}); one world spans them all")
        }
    };
    Groups::new(n, t)
}

/// Rank `q`'s place in every group of the topology: member `q % t` of
/// its `tp` group, `q` of every other.
fn topology(m: &Verified, groups: Groups, q: usize) -> Option<Topology> {
    let t = m.topology.as_ref()?;
    let place = |g: &str| if g == "tp" { groups.member(q) } else { q } as u64;
    Some(Topology {
        groups: t.groups.iter().map(|(g, &size)| (g.clone(), GroupRank { index: place(g), size })).collect(),
    })
}

fn load(m: &Verified, weights: &Weights, groups: Groups, q: usize, o: &Load) -> Result<Runtime> {
    let t0 = Instant::now();
    let topo = topology(m, groups, q);
    let mut rt = Runtime::load(m, o.kernels.as_deref(), o.gpu, Some(o.capacity), topo.as_ref())
        .with_context(|| format!("rank {q} on gpu {}", o.gpu))?;
    rt.set_eager(o.eager);
    weights.bind(&mut rt).with_context(|| format!("rank {q}: binding weights"))?;
    info!(rank = q, gpu = o.gpu, pages = rt.pages_total(), load_s = logline::secs(t0.elapsed()), "rank loaded");
    Ok(rt)
}

/// This runtime's pool as another process maps it; `None` when its chunks
/// have no fabric handle.
fn export_pool(rt: &Runtime) -> Result<Option<String>> {
    Ok(rt.export_pool()?.to_bytes().map(hex::encode))
}

fn offer_hex(o: &Offer) -> Result<String> {
    Ok(hex::encode(o.to_bytes()?))
}

fn offer_of(x: &str) -> Result<Offer> {
    Ok(Offer::from_bytes(&hex::decode(x)?)?)
}

/// Rank `q`'s groups, each as its members' offers in member order: its
/// own `tp` group, every rank for any other group.
fn members<T: Clone>(m: &Verified, groups: Groups, q: usize, offers: &[T]) -> BTreeMap<String, Vec<T>> {
    let ranks = |g: &str| if g == "tp" { groups.members(q) } else { 0..groups.ranks() };
    m.topology
        .iter()
        .flat_map(|t| t.groups.keys())
        .map(|g| (g.clone(), ranks(g).map(|r| offers[r].clone()).collect()))
        .collect()
}

/// Connect, run what runs once, reserve the host tier.
fn join(
    rt: &mut Runtime,
    p: &Protocol,
    groups: Groups,
    peers: &BTreeMap<String, Vec<Offer>>,
    host: (u64, u64),
) -> Result<()> {
    rt.connect(peers).context("connecting to the groups")?;
    // Once after load, the peers mapped: a group manifest's setup (the
    // allreduce's Lamport stages are poisoned, not zeroed).
    let vars = p.vars(1, 1, groups.size() as u64, 1);
    for prog in &p.once {
        rt.run(prog, &vars).with_context(|| format!("`{prog}`"))?;
    }
    if host != (0, 0) {
        let t0 = Instant::now();
        let (pages, slots) = host;
        rt.reserve_host(pages, slots).context("reserving the host tier")?;
        info!(pages, slots, reserve_s = logline::secs(t0.elapsed()), "host tier reserved");
    }
    Ok(())
}

/// Load rank 0, wait at `listen` for every other rank (none for a
/// manifest of one), connect them all, and build every rank's
/// accounting from the budgets they report.
pub fn head(
    m: &Verified,
    weights: &Weights,
    o: &Load,
    listen: Option<&str>,
    (host_bytes, host_turn): (u64, u64),
) -> Result<(Ranks, Executors)> {
    let groups = world(m)?;
    let n = groups.ranks();
    let listener = match (n, listen) {
        (1, _) => None,
        (_, Some(a)) => Some(TcpListener::bind(a).with_context(|| format!("binding {a}"))?),
        (_, None) => bail!(
            "the manifest spans {n} ranks, one process each: the head takes `--listen host:port`, every other rank `--head host:port --rank <q>`"
        ),
    };
    let p = Protocol::check(m)?;
    let mut rt = load(m, weights, groups, 0, o)?;
    // Every rank's states are the head's manifest's, so one runtime's split holds for all.
    let host = rt.host_units(host_bytes, host_turn);
    let own = rt.offer()?;
    let mut hellos: Vec<Option<(Hello, TcpStream)>> = (0..n).map(|_| None).collect();
    let mut offers = vec![own.clone(); n];
    if let Some(l) = &listener {
        info!(listen, ranks = n, group = groups.size(), "waiting for the ranks");
        while hellos[1..].iter().any(Option::is_none) {
            let (mut s, addr) = l.accept()?;
            let _ = s.set_nodelay(true);
            let h: Hello = recv(&mut s)?.with_context(|| format!("{addr} closed before its hello"))?;
            let o = offer_of(&h.offer).with_context(|| format!("{addr}: rank {}'s offer", h.rank))?;
            ensure!(
                o.manifest == own.manifest,
                "{addr}: rank {} loaded another manifest ({:016x}, the head's {:016x})",
                h.rank,
                o.manifest,
                own.manifest
            );
            ensure!(
                (1..n).contains(&h.rank) && hellos[h.rank].is_none(),
                "{addr}: rank {} is not a rank still missing of 1..{n}",
                h.rank
            );
            info!(rank = h.rank, %addr, "rank joined");
            let q = h.rank;
            offers[q] = o;
            hellos[q] = Some((h, s));
        }
    }
    let wire = match n {
        1 => Vec::new(),
        _ => offers.iter().map(offer_hex).collect::<Result<Vec<_>>>()?,
    };
    let mut budgets = vec![rt.budget()];
    let mut exported = vec![export_pool(&rt)?];
    let mut remote = Vec::with_capacity(n - 1);
    for (q, (h, mut s)) in hellos.into_iter().enumerate().skip(1).map(|(q, x)| (q, x.expect("every rank said hello"))) {
        send(&mut s, &Table { peers: members(m, groups, q, &wire), host })?;
        budgets.push(Budget { chunk: h.chunk, chunks: h.chunks, first_slots: h.first_slots, tokens: h.tokens });
        exported.push(h.pool);
        crate::watch::exit_with(&s, format!("rank {q}"))?;
        remote.push(s);
    }
    join(&mut rt, &p, groups, &members(m, groups, 0, &offers), host)?;
    let pools = budgets.iter().map(|b| b.pool(m)).collect::<kern_pool::Result<Vec<_>>>()?;
    let ranks = Ranks::new(p.clone(), groups, pools, host)?;
    info!(ranks = n, group = groups.size(), "ranks connected");
    Ok((ranks, Executors { local: Executor::new(0, rt, p), remote, pools: exported }))
}

/// Serve rank `q` for the head at `head` until the head is gone.
pub fn rank(m: &Verified, weights: &Weights, o: &Load, head: &str, q: usize) -> Result<()> {
    let groups = world(m)?;
    ensure!(
        (1..groups.ranks()).contains(&q),
        "--rank {q}: the manifest spans ranks 0..{}, rank 0 the head",
        groups.ranks()
    );
    let p = Protocol::check(m)?;
    let mut rt = load(m, weights, groups, q, o)?;
    let mut s = connect(head, CONNECT)?;
    let b = rt.budget();
    let hello = Hello {
        rank: q,
        chunk: b.chunk,
        chunks: b.chunks,
        first_slots: b.first_slots,
        tokens: b.tokens,
        offer: offer_hex(&rt.offer()?)?,
        pool: export_pool(&rt)?,
    };
    send(&mut s, &hello)?;
    let t: Table = recv(&mut s)?.context("the head closed before its table")?;
    crate::watch::exit_with(&s, "the head".into())?;
    let peers = t
        .peers
        .iter()
        .map(|(g, v)| Ok((g.clone(), v.iter().map(|x| offer_of(x)).collect::<Result<_>>()?)))
        .collect::<Result<_>>()?;
    join(&mut rt, &p, groups, &peers, t.host)?;
    info!(rank = q, head, "rank serving");
    let mut x = Executor::new(q, rt, p);
    while let Some(order) = recv::<Order>(&mut s)? {
        let answer = x.run(&order).with_context(|| format!("rank {q}"))?;
        send(&mut s, &answer)?;
    }
    info!(rank = q, "the head closed its connection");
    Ok(())
}

/// One rank's runtime, running orders: acts in order, then the item.
pub struct Executor {
    rank: usize,
    rt: Runtime,
    p: Protocol,
    /// Host transfers in flight, in issue order (one stream: they land in it).
    moving: VecDeque<Landing>,
    landed: u64,
    remapping: bool,
}

impl Executor {
    fn new(rank: usize, rt: Runtime, p: Protocol) -> Executor {
        Executor { rank, rt, p, moving: VecDeque::new(), landed: 0, remapping: false }
    }

    fn run(&mut self, o: &Order) -> Result<Answer> {
        let issued_us = logline::now_us();
        let launched_us = self.issue(o)?;
        let a = self.answer(o)?;
        if let Some(item) = &o.item {
            debug!(rank = self.rank, program = %item.program, acts = o.acts.len(), issued_us, launched_us, done_us = logline::now_us(), "rank item");
        }
        Ok(a)
    }

    /// Queue the acts and the item; the wall clock once the item's inputs are written.
    fn issue(&mut self, o: &Order) -> Result<u64> {
        for a in &o.acts {
            match a {
                Act::Remap(m) => {
                    self.rt.remap(m.clone().remap())?;
                    self.remapping = true;
                }
                Act::Zero(s) => self.rt.zero_slot(*s)?,
                Act::Copy(c) => self.rt.copy(&c.pages, c.slot.as_slice())?,
                Act::Park(c) => self.moving.push_back(self.rt.host_out(&c.clone().into())?),
                Act::Wake(c) => self.moving.push_back(self.rt.host_in(&c.clone().into())?),
            }
        }
        if let Some(item) = &o.item {
            for (name, v) in &item.rows {
                kern_run::write_named(&mut self.rt, &self.p, name, v, &item.vars)?;
            }
            let written_us = logline::now_us();
            self.rt.issue(&item.program, &item.vars).with_context(|| format!("`{}`", item.program))?;
            return Ok(written_us);
        }
        Ok(logline::now_us())
    }

    /// Once the item is done: its error word, its outputs, and what landed.
    fn answer(&mut self, o: &Order) -> Result<Answer> {
        let (tokens, counts) = match &o.item {
            Some(item) => {
                self.rt.synchronize()?;
                if let Some(e) = self.p.any(Fill::Error) {
                    let err = e.decode(&self.rt.read_output(&e.name)?)[0];
                    if err != 0 {
                        bail!("`{}` reports collective error {err}", item.program);
                    }
                }
                let f = self.p.forwards.iter().find(|f| f.name == item.program);
                let read = |i: Option<usize>| -> Result<Vec<i64>> {
                    let Some(fill) = i.map(|i| &self.p.fills[i]) else { return Ok(Vec::new()) };
                    Ok(fill.decode(&self.rt.read_output(&fill.name)?))
                };
                (read(f.and_then(|f| f.emits))?, read(f.and_then(|f| f.count))?)
            }
            None => (Vec::new(), Vec::new()),
        };
        let remaps = self.rt.remapped(o.wait && self.remapping)?;
        if remaps > 0 {
            self.remapping = false;
        }
        if o.wait {
            // A landing dropped is a landing waited for.
            self.landed += self.moving.len() as u64;
            self.moving.clear();
        }
        while self.moving.front().map_or(Ok(false), Landing::landed)? {
            self.moving.pop_front();
            self.landed += 1;
        }
        Ok(Answer { tokens, counts, remaps, landed: self.landed })
    }
}

/// The head's side of the ranks: its own executor, a stream to every
/// other rank, in rank order, and every rank's pool handle.
pub struct Executors {
    local: Executor,
    remote: Vec<TcpStream>,
    pools: Vec<Option<String>>,
}

// Built on the scheduler thread and used there only.
#[allow(unsafe_code)]
unsafe impl Send for Executors {}

impl crate::pd::Run for Executors {
    fn run(&mut self, orders: Vec<Order>) -> Result<Vec<Answer>> {
        Executors::run(self, orders)
    }
}

impl Executors {
    /// Every rank's pool handle (hex), for a prefill tier to map; refused
    /// when a rank's chunks have no fabric handle.
    pub fn pools(&self) -> Result<Vec<String>> {
        self.pools
            .iter()
            .enumerate()
            .map(|(q, p)| p.clone().with_context(|| format!("rank {q}'s pool has no fabric handle to push into")))
            .collect()
    }

    /// Run every rank's order (rank 0's here, the rest on their own
    /// processes) and collect the answers in rank order. Each remote
    /// order is encoded and sent on a thread of its own while rank 0 runs
    /// its own: a step's orders are large (every row's page table), and
    /// sent one after another they kept every GPU idle for as long.
    pub fn run(&mut self, orders: Vec<Order>) -> Result<Vec<Answer>> {
        ensure!(orders.len() == self.remote.len() + 1, "{} orders for {} ranks", orders.len(), self.remote.len() + 1);
        let mut orders = orders.into_iter();
        let own = orders.next().expect("counted");
        let item = own.item.is_some();
        let (remote, local) = (&mut self.remote, &mut self.local);
        let (first, send_us) = std::thread::scope(|sc| {
            let t0 = Instant::now();
            let sends: Vec<_> = remote
                .iter_mut()
                .zip(orders)
                .enumerate()
                .map(|(q, (s, o))| {
                    sc.spawn(move || {
                        send(s, &o).with_context(|| format!("rank {}", q + 1))?;
                        Ok::<_, anyhow::Error>(t0.elapsed().as_micros() as u64)
                    })
                })
                .collect();
            let first = local.run(&own).context("rank 0");
            let sent = sends
                .into_iter()
                .map(|h| h.join().expect("a send thread does not panic"))
                .collect::<Result<Vec<u64>>>();
            (first, sent.map(|v| v.into_iter().max().unwrap_or(0)))
        });
        let (first, send_us) = (first?, send_us?);
        let t = Instant::now();
        let rest = self.remote.iter_mut().enumerate().map(|(q, s)| {
            recv::<Answer>(s)
                .and_then(|a| a.context("the connection closed: the rank is gone"))
                .with_context(|| format!("rank {}", q + 1))
        });
        let answers = std::iter::once(Ok(first)).chain(rest).collect();
        if item {
            debug!(send_us, remote_wait_us = t.elapsed().as_micros() as u64, "ranks step");
        }
        answers
    }
}
