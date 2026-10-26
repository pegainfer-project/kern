//! What the endpoint refuses before the engine sees it.
//!
//! A manifest's programs hand back token ids, never logits, so kern-serve
//! has no logprobs to give. The frontend admits a request that asks for
//! them, the forward runs, and its response builder fails on the missing
//! numbers: a 500 for a request that was never servable. These routes
//! answer first. A request for `logprobs`, `top_logprobs` or
//! `prompt_logprobs` gets a 400 naming the parameter; any other request,
//! and any body the frontend will parse for itself, passes through untouched.

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

/// The parameter a request asks logprobs by, set to anything but `null` or
/// `false`: a completion's `logprobs: 0` asks for the sampled token's, as a
/// chat's `logprobs: true` does.
pub fn asks_logprobs(body: &Value) -> Option<&'static str> {
    LOGPROBS.into_iter().find(|k| body.get(k).is_some_and(|v| !v.is_null() && *v != Value::Bool(false)))
}

/// The vLLM router behind the refusal of what kern-serve does not serve.
pub fn routes(vllm: Router) -> Router {
    Router::new()
        .route("/v1/completions", post(refuse))
        .route("/v1/chat/completions", post(refuse))
        .with_state(vllm.clone())
        .fallback_service(vllm)
}

async fn refuse(State(vllm): State<Router>, request: Request) -> Response {
    let (parts, body) = request.into_parts();
    let body = match to_bytes(body, BODY_LIMIT).await {
        Ok(b) => b,
        Err(e) => return invalid(format!("reading the request body: {e}")),
    };
    let asked = serde_json::from_slice::<Value>(&body).ok().and_then(|v| asks_logprobs(&v));
    match asked {
        Some(k) => invalid(format!("{k} is not supported by this engine")),
        None => match vllm.oneshot(Request::from_parts(parts, Body::from(body))).await {
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
