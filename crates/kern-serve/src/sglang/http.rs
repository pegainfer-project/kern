//! The head's HTTP as SGLang's router sees a prefill worker.
//!
//! The router sends the client's `/generate` body to the prefill and the
//! decode worker at once, with the bootstrap room it chose, and reads the
//! decode worker's answer only once the prefill one answered: so the answer
//! here comes when the request's room ended on every stage, with the first
//! token, as SGLang's prefill answers (one token, whatever was asked). The
//! same goes for `/v1/completions` and `/v1/chat/completions`, which the
//! router hands both workers as the client sent them: the decode side
//! tokenizes the request itself, so a prompt that is not already token ids
//! (`input_ids`) is turned into them by SGLang's own server code, embedded
//! (`kern_sglang.prompt`), and P and D read the same ids. Every request the
//! head admits ends once ([`crate::rooms`]); a client that leaves only
//! stops waiting for it.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use pegainfer_frontend::engine::{EosPolicy, QueuedRequest, Request, RequestId, StopPolicy};
use pegainfer_frontend::sampler::SamplingParams;
use pyo3::prelude::*;
use serde::Serialize;
use serde_json::{json, Value};
use tokio::sync::oneshot;
use tracing::{error, info};

use crate::front::{Call, Note};
use crate::pp::head::Fed;

/// What the router reads off a worker when it registers it.
#[derive(Debug, Clone)]
pub struct Info {
    pub served_model_name: String,
    pub model_path: PathBuf,
    pub stages: usize,
}

/// Why a request is refused, and the status that says so.
type Refusal = (StatusCode, String);

type Answer = std::result::Result<Reply, Refusal>;

/// This worker's part in SGLang's P/D, as `/server_info` names it.
#[derive(Serialize)]
#[serde(rename_all = "lowercase")]
enum Mode {
    Prefill,
}

/// Why a request got no answer.
#[derive(Serialize)]
struct Failure {
    error: String,
}

#[derive(Debug, Clone, Copy)]
struct Reply {
    tok: u32,
    prompt: usize,
    cached: usize,
}

/// A request waiting for its room to end.
struct Waiter {
    tx: oneshot::Sender<Answer>,
    reply: Reply,
    room: u64,
}

struct Door {
    fed: Mutex<crate::pp::head::Caller>,
    /// `kern_sglang.prompt.Prompts`.
    prompts: Py<PyAny>,
    waiting: Mutex<BTreeMap<RequestId, Waiter>>,
    /// Rooms of requests not yet answered: a room is one request's at a time.
    rooms: Mutex<BTreeSet<u64>>,
    next: AtomicU64,
    server_info: Value,
    model_info: Value,
}

/// Load the head with `load`, then answer SGLang's router on `port` until
/// the process ends.
pub fn serve(load: impl FnOnce() -> Result<Fed>, port: u16, info: Info) -> Result<()> {
    let fed = load()?;
    let prompts = Python::attach(|py| -> PyResult<Py<PyAny>> {
        Ok(py.import("kern_sglang.prompt")?.getattr("Prompts")?.call0()?.unbind())
    })
    .map_err(|e| anyhow::anyhow!("loading SGLang's prompt handling: {e}"))?;
    let config: Value = std::fs::read_to_string(info.model_path.join("config.json"))
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or(Value::Null);
    let path = info.model_path.to_string_lossy().into_owned();
    let door = Arc::new(Door {
        fed: Mutex::new(fed.caller),
        prompts,
        waiting: Mutex::new(BTreeMap::new()),
        rooms: Mutex::new(BTreeSet::new()),
        next: AtomicU64::new(1),
        server_info: json!({
            "model_path": path, "served_model_name": info.served_model_name, "tp_size": 1, "dp_size": 1,
            "pp_size": info.stages, "disaggregation_mode": Mode::Prefill, "page_size": fed.facts.block_size,
            "max_total_tokens": fed.facts.total_blocks * fed.facts.block_size,
        }),
        model_info: json!({
            "model_path": path, "served_model_name": info.served_model_name, "tokenizer_path": path,
            "is_generation": true, "model_type": config["model_type"], "architectures": config["architectures"],
        }),
    });
    std::thread::Builder::new().name("kern-sglang-notes".into()).spawn({
        let door = door.clone();
        move || {
            for n in fed.notes {
                door.note(n);
            }
            error!("the pipeline head is gone");
            std::process::exit(1);
        }
    })?;
    let app = Router::new()
        .route("/generate", post(|State(d), Json(b)| answer(d, "generate", b)))
        .route("/v1/completions", post(|State(d), Json(b)| answer(d, "completion", b)))
        .route("/v1/chat/completions", post(|State(d), Json(b)| answer(d, "chat", b)))
        .route("/health", get(|| async { StatusCode::OK }))
        .route("/server_info", get(server_info))
        .route("/get_server_info", get(server_info))
        .route("/model_info", get(model_info))
        .route("/get_model_info", get(model_info))
        .route("/v1/models", get(models))
        .with_state(door);
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await.context("binding the HTTP port")?;
        info!(port, "serving as sglang's prefill");
        axum::serve(listener, app).await.context("serving HTTP")
    })
}

impl Door {
    /// One of the head's notes: a waiting request admitted, answered, or failed.
    fn note(&self, n: Note) {
        let mut waiting = self.waiting.lock().expect("not poisoned");
        let (id, end) = match n {
            Note::Admit { id, cached } => {
                if let Some(w) = waiting.get_mut(&id) {
                    w.reply.cached = cached;
                }
                return;
            }
            Note::Emit { id, tok } => {
                if let Some(w) = waiting.get_mut(&id) {
                    w.reply.tok = tok;
                }
                return;
            }
            Note::Finish { id, .. } => (id, None),
            Note::Fail { id, why } => (id, Some((StatusCode::BAD_GATEWAY, why))),
            Note::Reject { id, why } => (id, Some((StatusCode::BAD_REQUEST, format!("{why:?}")))),
            Note::Retire(id) => (id, Some((StatusCode::INTERNAL_SERVER_ERROR, "retired".into()))),
            Note::Decode { .. } | Note::Pages(_) | Note::Tokens(_) => return,
        };
        let Some(w) = waiting.remove(&id) else { return };
        self.rooms.lock().expect("not poisoned").remove(&w.room);
        let _ = w.tx.send(end.map_or(Ok(w.reply), Err));
    }
}

fn bad(why: impl Into<String>) -> Refusal {
    (StatusCode::BAD_REQUEST, why.into())
}

/// A request's prompt as token ids: its `input_ids` when it gives them,
/// else what SGLang's server code makes of it on `route` (see
/// `kern_sglang.prompt`).
async fn prompt(d: &Arc<Door>, route: &'static str, body: &Value) -> std::result::Result<Vec<u32>, Refusal> {
    if let Some(ids) = body["input_ids"].as_array() {
        return ids
            .iter()
            .map(|t| t.as_u64().and_then(|t| u32::try_from(t).ok()))
            .collect::<Option<Vec<u32>>>()
            .ok_or_else(|| bad("`input_ids` is one prompt: a list of token ids"));
    }
    let (d, text) = (d.clone(), body.to_string());
    let ids = tokio::task::spawn_blocking(move || {
        Python::attach(|py| -> PyResult<Vec<u32>> { d.prompts.bind(py).call_method1("ids", (route, text))?.extract() })
            .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    ids.map_err(bad)
}

/// Prefill `ids` into the room the router assigned in `body`.
async fn prefill(d: &Arc<Door>, body: &Value, ids: Vec<u32>) -> std::result::Result<(Reply, String), Refusal> {
    let Some(room) = body["bootstrap_room"].as_u64() else {
        return Err(bad("a prefill takes the `bootstrap_room` the router assigns"));
    };
    if !d.rooms.lock().expect("not poisoned").insert(room) {
        return Err(bad(format!("bootstrap room {room} is already in use")));
    }
    let id = RequestId::new(d.next.fetch_add(1, Ordering::Relaxed));
    let (tx, rx) = oneshot::channel();
    let prompt = ids.len();
    d.waiting.lock().expect("not poisoned").insert(id, Waiter { tx, reply: Reply { tok: 0, prompt, cached: 0 }, room });
    let rid = body["rid"].as_str().map(Arc::from);
    let request = Request {
        prompt_tokens: ids,
        params: SamplingParams::default(),
        history_tokens: None,
        stop_policy: StopPolicy::new(EosPolicy::Ignore, Vec::new()),
        max_tokens: 1,
        lora_adapter: None,
        kv_transfer_params: Some(json!({ "bootstrap_room": room })),
        logprobs: None,
        prompt_logprobs: None,
        trace_parent: None,
        client_label: rid.clone(),
    };
    let submitted = {
        let mut call = d.fed.lock().expect("not poisoned");
        (*call)(Call::Submit(Box::new(QueuedRequest { id, request })))
    };
    if let Err(e) = submitted {
        d.waiting.lock().expect("not poisoned").remove(&id);
        d.rooms.lock().expect("not poisoned").remove(&room);
        return Err((StatusCode::SERVICE_UNAVAILABLE, format!("{e:#}")));
    }
    let name = rid.as_deref().map_or_else(|| id.raw().to_string(), str::to_string);
    match rx.await {
        Ok(Ok(r)) => Ok((r, name)),
        Ok(Err(refused)) => Err(refused),
        Err(_) => Err((StatusCode::INTERNAL_SERVER_ERROR, "the head dropped the request".into())),
    }
}

/// Answer a prefill on `route` with its first token, in the route's shape.
async fn answer(d: Arc<Door>, route: &'static str, body: Value) -> Response {
    let done = match prompt(&d, route, &body).await {
        Ok(ids) => prefill(&d, &body, ids).await,
        Err(refused) => Err(refused),
    };
    let (r, id) = match done {
        Ok(done) => done,
        Err((status, error)) => return (status, Json(Failure { error })).into_response(),
    };
    let usage = json!({"prompt_tokens": r.prompt, "completion_tokens": 1, "total_tokens": r.prompt + 1,
                       "prompt_tokens_details": {"cached_tokens": r.cached}});
    Json(match route {
        "chat" => json!({
            "id": id, "object": "chat.completion", "model": body["model"], "usage": usage,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "length"}],
        }),
        "completion" => json!({
            "id": id, "object": "text_completion", "model": body["model"], "usage": usage,
            "choices": [{"index": 0, "text": "", "finish_reason": "length"}],
        }),
        _ => json!({
            "text": "", "output_ids": [r.tok],
            "meta_info": {"id": id, "finish_reason": {"type": "length", "length": 1},
                          "prompt_tokens": r.prompt, "completion_tokens": 1, "cached_tokens": r.cached},
        }),
    })
    .into_response()
}

async fn server_info(State(d): State<Arc<Door>>) -> Json<Value> {
    Json(d.server_info.clone())
}

async fn model_info(State(d): State<Arc<Door>>) -> Json<Value> {
    Json(d.model_info.clone())
}

async fn models(State(d): State<Arc<Door>>) -> Json<Value> {
    Json(json!({"object": "list", "data": [{"id": d.model_info["served_model_name"], "object": "model"}]}))
}
