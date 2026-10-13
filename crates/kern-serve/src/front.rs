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

/// A request submitted and not yet answered.
#[derive(Debug, Clone, Copy, Default)]
struct Open {
    admitted: bool,
    /// The head was told the client left.
    told: bool,
}

pub struct Front {
    notes: mpsc::Receiver<Note>,
    head: Link<Call>,
    relays: Relays,
    /// The decode tier of every request the relays hold.
    tiers: BTreeMap<Req, usize>,
    open: BTreeMap<RequestId, Open>,
    pages: u64,
    pages_total: u64,
}

impl Front {
    /// The front of a head that reads `head` and writes `notes`, its pool
    /// `pages_total` pages.
    pub fn new(
        notes: mpsc::Receiver<Note>,
        head: impl FnMut(Call) -> Result<()> + Send + 'static,
        pages_total: u64,
    ) -> Front {
        Front {
            notes,
            head: Box::new(head),
            relays: Relays::new(),
            tiers: BTreeMap::new(),
            open: BTreeMap::new(),
            pages: 0,
            pages_total,
        }
    }

    fn note(&mut self, n: Note, ledger: &mut RequestLedger) -> Result<()> {
        match n {
            Note::Admit { id, cached } => {
                ledger.admit(id);
                ledger.set_cached_tokens(id, cached);
                self.open.entry(id).or_default().admitted = true;
            }
            Note::Reject { id, why } => {
                ledger.reject(id, why);
                self.open.remove(&id);
            }
            Note::Emit { id, tok } => ledger.push_tokens(id, &[tok], &[]),
            Note::Finish { id, why, cause } => {
                ledger.finish_with_cause(id, why, cause);
                self.open.remove(&id);
            }
            Note::Retire(id) => {
                ledger.retire(id);
                self.open.remove(&id);
            }
            Note::Decode { id, relay, tier } => {
                self.relays.start(id.raw(), relay);
                self.tiers.insert(id.raw(), tier);
            }
            Note::Pages(n) => self.pages = n,
            Note::Tokens(step) => {
                let outs = self.relays.hear(step)?;
                self.relay(outs, ledger)?;
            }
        }
        Ok(())
    }

    fn relay(&mut self, outs: Vec<Out>, ledger: &mut RequestLedger) -> Result<()> {
        for o in outs {
            match o {
                Out::Emit { req, tok } => ledger.push_tokens(RequestId::new(req), &[tok], &[]),
                Out::Finish { req, why, cause } => {
                    let id = RequestId::new(req);
                    ledger.finish_with_cause(id, why, cause);
                    self.open.remove(&id);
                }
                Out::Ask(Ask::Abort { req }) => {
                    let tier = *self.tiers.get(&req).context("an abort of a request on no decode tier")?;
                    (self.head)(Call::Abort { tier, req })?
                }
                o => bail!("the relays asked for {o:?}"),
            }
        }
        let relays = &self.relays;
        self.tiers.retain(|&r, _| relays.holds(r));
        Ok(())
    }

    /// The clients gone since the last look: a decoding one's request ends
    /// here, any other's is the head's to end.
    fn sweep(&mut self, ledger: &mut RequestLedger) -> Result<()> {
        let gone: Vec<RequestId> = self
            .open
            .iter()
            .filter(|&(id, o)| (!o.told || self.relays.holds(id.raw())) && ledger.is_aborted(*id))
            .map(|(&id, _)| id)
            .collect();
        for id in gone {
            if self.relays.holds(id.raw()) {
                ledger.retire(id);
                self.open.remove(&id);
                let outs = self.relays.abort(id.raw());
                self.relay(outs, ledger)?;
            } else {
                self.open.get_mut(&id).expect("listed").told = true;
                (self.head)(Call::Gone(id))?;
            }
        }
        Ok(())
    }
}

impl Scheduler for Front {
    fn submit(&mut self, request: QueuedRequest) {
        self.open.insert(request.id, Open::default());
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
            self.note(n, ledger)?;
            next = self.notes.try_recv().ok();
        }
        self.sweep(ledger)
    }

    fn metrics(&self) -> SchedulerMetrics {
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
