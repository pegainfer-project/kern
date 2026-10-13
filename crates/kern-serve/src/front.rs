//! A pipeline head's front: the pegainfer [`Scheduler`] on the driver
//! thread, apart from the head that feeds the pipeline.
//!
//! The head waits on its device and on stage 1 for as long as an item
//! takes (a cold long prompt's chunk is half a second on K3), and the
//! driver ships what a step wrote to the ledger only once the step returns.
//! A head that were the scheduler would hold every token of the process
//! behind that wait, the decode tier's among them, and clients would get
//! them in bursts an item apart. So the head runs on a thread of its own
//! and writes [`Note`]s, in order, into the front's one channel; the decode
//! tier's `Tokens` go into the same channel straight from its connection's
//! reader. The front applies both to the ledger as they come and relays each
//! decoding request's tokens itself ([`Relays`]): the head hands a request
//! over ([`Note::Decode`]) before it sends D the `Start`, so in the one
//! channel the request's tokens always come after it. With several decode
//! tiers each request is on one, which the hand-over names, and every
//! tier's reader writes into the same channel.
//!
//! Who answers a client: the head, from submission until it finishes,
//! rejects or retires the request or hands it over; the front after. A
//! client that leaves is the front's to see (the ledger's abort flag): it
//! retires a decoding request at once and has the head free its seats
//! ([`Call::Abort`], as for a stop the relays see first: the head is the
//! one writer to a decode tier, and the hand-over came before the head's
//! `Start`, so the abort is behind it), and tells the head once of any
//! other ([`Call::Gone`]), which retires it at its next touch.

use std::collections::BTreeMap;
use std::sync::mpsc;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use pegainfer_frontend::engine::{
    FinishReason, QueuedRequest, RejectReason, RequestId, RequestLedger, Scheduler, SchedulerMetrics, StopCause,
};

use crate::pd::{Ask, Out, Relay, Relays, Req};

/// How long a step waits for a note before handing the driver back: the
/// longest a new submission waits to reach the head.
const WAIT: Duration = Duration::from_millis(1);

/// What reaches the front, in order: the head's writes against its
/// requests, and every D step's tokens.
#[derive(Debug)]
pub enum Note {
    Admit {
        id: RequestId,
        cached: usize,
    },
    Reject {
        id: RequestId,
        why: RejectReason,
    },
    Emit {
        id: RequestId,
        tok: u32,
    },
    Finish {
        id: RequestId,
        why: FinishReason,
        cause: Option<StopCause>,
    },
    Retire(RequestId),
    /// Starting on decode tier `tier`: its tokens are the front's to relay.
    Decode {
        id: RequestId,
        relay: Relay,
        tier: usize,
    },
    /// The head's pages in use.
    Pages(u64),
    Tokens(Vec<(Req, u32, bool)>),
}

/// What the front tells the head.
pub enum Call {
    Submit(Box<QueuedRequest>),
    /// The client left a request the head may still hold.
    Gone(RequestId),
    /// A request decoding on tier `tier` ended here: free its seats.
    Abort {
        tier: usize,
        req: Req,
    },
}

type Link<T> = Box<dyn FnMut(T) -> Result<()> + Send>;

/// What the front's decisions come to, in order: a write to a request's
/// ledger account, or a word to the head.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Write {
    Admit {
        id: RequestId,
        cached: usize,
    },
    Reject {
        id: RequestId,
        why: RejectReason,
    },
    Tokens {
        id: RequestId,
        tok: u32,
    },
    Finish {
        id: RequestId,
        why: FinishReason,
        cause: Option<StopCause>,
    },
    Retire(RequestId),
    /// Tell the head the client left (`Call::Gone`).
    Gone(RequestId),
    /// Have the head free a decoding request's seats (`Call::Abort`).
    Abort {
        tier: usize,
        req: Req,
    },
}

/// A request submitted and not yet answered.
#[derive(Debug, Clone, Copy, Default)]
struct Open {
    admitted: bool,
    /// The head was told the client left.
    told: bool,
}

/// The requests the front answers, as a pure state machine: what each note
/// and each departed client comes to ([`Write`]s), the ledger and the head
/// left to the [`Front`] that carries them out.
#[derive(Debug, Clone, Default)]
pub struct Clients {
    relays: Relays,
    /// The decode tier of every request the relays hold.
    tiers: BTreeMap<Req, usize>,
    open: BTreeMap<RequestId, Open>,
    pages: u64,
    pages_total: u64,
}

impl Clients {
    /// The clients of a head whose pool is `pages_total` pages.
    pub fn new(pages_total: u64) -> Clients {
        Clients { pages_total, ..Clients::default() }
    }

    /// `id` was submitted: the front answers it until it ends.
    pub fn submit(&mut self, id: RequestId) {
        self.open.insert(id, Open::default());
    }

    /// Requests submitted and not yet answered.
    pub fn open(&self) -> usize {
        self.open.len()
    }

    /// Requests whose tokens the relays hold.
    pub fn decoding(&self) -> usize {
        self.relays.live()
    }

    pub fn note(&mut self, n: Note) -> Result<Vec<Write>> {
        Ok(match n {
            Note::Admit { id, cached } => {
                self.open.entry(id).or_default().admitted = true;
                vec![Write::Admit { id, cached }]
            }
            Note::Reject { id, why } => {
                self.open.remove(&id);
                vec![Write::Reject { id, why }]
            }
            Note::Emit { id, tok } => vec![Write::Tokens { id, tok }],
            Note::Finish { id, why, cause } => {
                self.open.remove(&id);
                vec![Write::Finish { id, why, cause }]
            }
            Note::Retire(id) => {
                self.open.remove(&id);
                vec![Write::Retire(id)]
            }
            Note::Decode { id, relay, tier } => {
                self.relays.start(id.raw(), relay);
                self.tiers.insert(id.raw(), tier);
                Vec::new()
            }
            Note::Pages(n) => {
                self.pages = n;
                Vec::new()
            }
            Note::Tokens(step) => {
                let outs = self.relays.hear(step)?;
                self.relay(outs)?
            }
        })
    }

    fn relay(&mut self, outs: Vec<Out>) -> Result<Vec<Write>> {
        let mut w = Vec::with_capacity(outs.len());
        for o in outs {
            w.push(match o {
                Out::Emit { req, tok } => Write::Tokens { id: RequestId::new(req), tok },
                Out::Finish { req, why, cause } => {
                    let id = RequestId::new(req);
                    self.open.remove(&id);
                    Write::Finish { id, why, cause }
                }
                Out::Ask(Ask::Abort { req }) => {
                    let tier = *self.tiers.get(&req).context("an abort of a request on no decode tier")?;
                    Write::Abort { tier, req }
                }
                o => bail!("the relays asked for {o:?}"),
            });
        }
        let relays = &self.relays;
        self.tiers.retain(|&r, _| relays.holds(r));
        Ok(w)
    }

    /// The clients `aborted` says left: a decoding one's request ends here,
    /// any other's is the head's to end, told once.
    pub fn sweep(&mut self, aborted: impl Fn(RequestId) -> bool) -> Result<Vec<Write>> {
        let gone: Vec<RequestId> = self
            .open
            .iter()
            .filter(|&(id, o)| (!o.told || self.relays.holds(id.raw())) && aborted(*id))
            .map(|(&id, _)| id)
            .collect();
        let mut w = Vec::new();
        for id in gone {
            if self.relays.holds(id.raw()) {
                self.open.remove(&id);
                w.push(Write::Retire(id));
                let outs = self.relays.abort(id.raw());
                w.extend(self.relay(outs)?);
            } else {
                self.open.get_mut(&id).expect("listed").told = true;
                w.push(Write::Gone(id));
            }
        }
        Ok(w)
    }

    pub fn metrics(&self) -> SchedulerMetrics {
        let running = self.open.values().filter(|o| o.admitted).count();
        SchedulerMetrics {
            kv_used_blocks: self.pages,
            kv_total_blocks: self.pages_total,
            num_running_reqs: running as u64,
            num_waiting_reqs: (self.open.len() - running) as u64,
            spec_decode: None,
        }
    }
}

/// The front as pegainfer drives it: the head's notes in, [`Clients`]'
/// writes carried out against the ledger and the head.
pub struct Front {
    notes: mpsc::Receiver<Note>,
    head: Link<Call>,
    clients: Clients,
}

impl Front {
    /// The front of a head that reads `head` and writes `notes`, its pool
    /// `pages_total` pages.
    pub fn new(
        notes: mpsc::Receiver<Note>,
        head: impl FnMut(Call) -> Result<()> + Send + 'static,
        pages_total: u64,
    ) -> Front {
        Front { notes, head: Box::new(head), clients: Clients::new(pages_total) }
    }

    fn carry(&mut self, writes: Vec<Write>, ledger: &mut RequestLedger) -> Result<()> {
        for w in writes {
            match w {
                Write::Admit { id, cached } => {
                    ledger.admit(id);
                    ledger.set_cached_tokens(id, cached);
                }
                Write::Reject { id, why } => ledger.reject(id, why),
                Write::Tokens { id, tok } => ledger.push_tokens(id, &[tok], &[]),
                Write::Finish { id, why, cause } => ledger.finish_with_cause(id, why, cause),
                Write::Retire(id) => ledger.retire(id),
                Write::Gone(id) => (self.head)(Call::Gone(id))?,
                Write::Abort { tier, req } => (self.head)(Call::Abort { tier, req })?,
            }
        }
        Ok(())
    }
}

impl Scheduler for Front {
    fn submit(&mut self, request: QueuedRequest) {
        self.clients.submit(request.id);
        // A head gone ends the process; the request is answered by the ledger's write-off.
        let _ = (self.head)(Call::Submit(Box::new(request)));
    }

    fn step(&mut self, ledger: &mut RequestLedger) -> Result<()> {
        let mut next = match self.notes.recv_timeout(WAIT) {
            Ok(n) => Some(n),
            Err(mpsc::RecvTimeoutError::Timeout) => None,
            Err(mpsc::RecvTimeoutError::Disconnected) => bail!("the pipeline head is gone"),
        };
        while let Some(n) = next {
            let w = self.clients.note(n)?;
            self.carry(w, ledger)?;
            next = self.notes.try_recv().ok();
        }
        let w = self.clients.sweep(|id| ledger.is_aborted(id))?;
        self.carry(w, ledger)
    }

    fn metrics(&self) -> SchedulerMetrics {
        self.clients.metrics()
    }
}
