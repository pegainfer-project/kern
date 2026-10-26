//! What the endpoint refuses before the engine sees it.
//!
//! A manifest's programs hand back token ids, never logits, so kern-serve
//! has no logprobs to give. The frontend admits a request that asks for
//! them, the forward runs, and its response builder fails on the missing
//! numbers: a 500 for a request that was never servable. A manifest
//! without sampling fills draws nothing: the request for a draw that
//! reaches the scheduler is served the most likely token, which is not
//! what it asked for. These routes answer first. A request for
//! `logprobs`, `top_logprobs` or `prompt_logprobs` gets a 400 naming the
//! parameter; one for a draw of a manifest that cannot make it gets a 400
//! saying so; any other request, and any body the frontend will parse for
//! itself, passes through untouched. A request that sets no temperature
//! passes too: the generation config's default is the frontend's to
//! apply, and the scheduler logs what it then serves greedily.

use axum::body::{to_bytes, Body};
use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use serde::Serialize;
use serde_json::Value;
use tower::ServiceExt as _;

/// What the frontend reads a body up to; a larger one is its to refuse.
const BODY_LIMIT: usize = 16 * 1024 * 1024;

const LOGPROBS: [&str; 3] = ["logprobs", "top_logprobs", "prompt_logprobs"];

/// Below this a temperature is greedy to the frontend.
const GREEDY_BELOW: f64 = 1e-5;

/// The parameter a request asks logprobs by, set to anything but `null` or
/// `false`: a completion's `logprobs: 0` asks for the sampled token's, as a
/// chat's `logprobs: true` does.
pub fn asks_logprobs(body: &Value) -> Option<&'static str> {
    LOGPROBS.into_iter().find(|k| body.get(k).is_some_and(|v| !v.is_null() && *v != Value::Bool(false)))
}

/// The temperature a request draws at, when it does: one the frontend
/// reads as non-greedy, with no `top_k` of 1 (one survivor is the most
/// likely token).
pub fn asks_draw(body: &Value) -> Option<f64> {
    let t = body.get("temperature")?.as_f64().filter(|&t| t >= GREEDY_BELOW)?;
    (body.get("top_k").and_then(Value::as_i64) != Some(1)).then_some(t)
}

/// The vLLM router behind the refusal of what kern-serve does not serve;
/// `samples` is whether the manifest draws.
pub fn routes(vllm: Router, samples: bool) -> Router {
    Router::new()
        .route("/v1/completions", post(refuse))
        .route("/v1/chat/completions", post(refuse))
        .with_state(Gate { vllm: vllm.clone(), samples })
        .fallback_service(vllm)
}

#[derive(Clone)]
struct Gate {
    vllm: Router,
    samples: bool,
}

/// Why `body` is not servable, if it is not.
fn refused(body: &Value, samples: bool) -> Option<String> {
    if let Some(k) = asks_logprobs(body) {
        return Some(format!("{k} is not supported by this engine"));
    }
    match asks_draw(body) {
        Some(t) if !samples => Some(format!(
            "a temperature of {t} asks for a draw, but this engine's manifest takes no sampling fills; send temperature 0 for the most likely token"
        )),
        _ => None,
    }
}

async fn refuse(State(g): State<Gate>, request: Request) -> Response {
    let (parts, body) = request.into_parts();
    let body = match to_bytes(body, BODY_LIMIT).await {
        Ok(b) => b,
        Err(e) => return invalid(format!("reading the request body: {e}")),
    };
    let why = serde_json::from_slice::<Value>(&body).ok().and_then(|v| refused(&v, g.samples));
    match why {
        Some(why) => invalid(why),
        None => match g.vllm.oneshot(Request::from_parts(parts, Body::from(body))).await {
            Ok(r) => r,
            Err(never) => match never {},
        },
    }
}

/// The OpenAI error envelope, as the frontend's own refusals are shaped.
#[derive(Serialize)]
struct Refusal {
    error: Detail,
}

#[derive(Serialize)]
struct Detail {
    message: String,
    r#type: &'static str,
    code: &'static str,
}

fn invalid(message: String) -> Response {
    let error = Detail { message, r#type: "invalid_request_error", code: "invalid_request_error" };
    (StatusCode::BAD_REQUEST, Json(Refusal { error })).into_response()
}
