//! A cut manifest's pipeline as the prefill instance of SGLang's own P/D:
//! every stage one of SGLang's prefill pp ranks, sending its layers'
//! pages to an unmodified SGLang decode instance through SGLang's transfer
//! code (Mooncake over RDMA), and the head answering SGLang's router.
//!
//! The split is transport in Python, scheduling in Rust. Every stage
//! process embeds Python (feature `sglang-pd`) and drives
//! `kern_sglang.prefill.Stage` from one thread of its own ([`start`]), which
//! holds the GIL only for a call: the stage's issuing thread and its watch
//! hand it messages ([`Msg`]) and never wait on it. What is decided about a
//! request is the head's, in a pure state machine ([`crate::rooms`]): when
//! its room opens on every stage, when every stage sends, and when its
//! lease goes back. The stage's thread only keeps, per request, what it
//! has been told until the send can go (the pages, the item landed, the
//! first token on the last stage), and asks Python how it stands.
//!
//! The head serves SGLang's HTTP instead of the OpenAI one (`http`): a
//! `/generate` with `input_ids` and the bootstrap room the router assigned
//! answers with the first token once every stage's send ended, a 5xx if any
//! failed; `/health`, `/server_info` and `/model_info` are what the router
//! probes when it registers a worker.

// Built without the feature, the transport is a stub that refuses to start,
// and what it would read goes unread.
#![cfg_attr(not(feature = "sglang-pd"), allow(dead_code))]

#[cfg(feature = "sglang-pd")]
mod endpoint;
#[cfg(feature = "sglang-pd")]
pub(crate) mod http;

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::pd::Req;
use crate::rooms::Transfer;

/// Where a stage's SGLang transport lives: the address its transfer engine
/// (and, on the head, the bootstrap server, at `bootstrap_port`) listens
/// on, its RDMA device (SGLang picks one when none is named), and the
/// checkpoint directory SGLang reads the model's config from.
#[derive(Debug, Clone)]
pub struct Sglang {
    pub ip: String,
    pub ib_device: Option<String>,
    pub model_path: PathBuf,
    pub bootstrap_port: u16,
}

/// What the head's table tells each stage: the bootstrap server's port,
/// what pp rank 0 broadcast to the others when it started, and how many
/// sends may be in flight at once.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bootstrap {
    pub port: u16,
    pub consensus: Vec<serde_json::Value>,
    pub slots: usize,
}

/// What a stage's transport thread hears, in order.
#[derive(Debug)]
pub(crate) enum Msg {
    /// Request `req` was admitted for room `room`.
    Open { req: Req, room: u64 },
    /// Its pages go once the item before has landed.
    Transfer(Transfer),
    /// The item before `req`'s transfer has landed on this stage's device.
    Landed(Req),
    /// The last stage's answer to an item, in item order.
    Tokens(Vec<i64>),
}

/// Where a stage's transport thread says how a send ended.
pub(crate) type Sent = Box<dyn FnMut(Req, Option<String>) -> anyhow::Result<()> + Send>;

/// A stage's place among SGLang's prefill pp ranks.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Rank {
    pub stage: usize,
    pub stages: usize,
}

#[cfg(feature = "sglang-pd")]
pub(crate) use endpoint::start;

#[cfg(not(feature = "sglang-pd"))]
pub(crate) fn start(
    _: &Sglang,
    _: &kern_runtime::Runtime,
    _: Rank,
    _: &Bootstrap,
    _: Sent,
) -> anyhow::Result<(std::sync::mpsc::Sender<Msg>, Vec<serde_json::Value>)> {
    anyhow::bail!("this kern-serve is built without the `sglang-pd` feature")
}
