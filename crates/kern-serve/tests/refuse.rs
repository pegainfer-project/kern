//! A request for logprobs, or for a draw of a manifest that takes no
//! sampling fills, is refused with a 400 before the engine sees it; every
//! other request reaches the vLLM router as it was sent.

use axum::body::{to_bytes, Body};
use axum::extract::Request;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::Router;
use kern_serve::refuse;
use serde_json::{json, Value};
use tower::ServiceExt as _;

const COMPLETIONS: &str = "/v1/completions";
const CHAT: &str = "/v1/chat/completions";

/// The router that would run the request: it echoes the body it received.
fn engine() -> Router {
    let echo = |body: String| async move { body };
    Router::new().route(COMPLETIONS, post(echo)).route(CHAT, post(echo)).route("/v1/models", get(|| async { "models" }))
}

async fn send(samples: bool, path: &str, body: &str) -> (StatusCode, String) {
    let req = Request::post(path).header("content-type", "application/json").body(Body::from(body.to_owned())).unwrap();
    let res = refuse::routes(engine(), samples).oneshot(req).await.unwrap();
    let status = res.status();
    let bytes = to_bytes(res.into_body(), usize::MAX).await.unwrap();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

#[test]
fn a_set_logprobs_parameter_is_named_and_an_unset_one_is_not() {
    let cases = [
        (json!({"logprobs": 2}), Some("logprobs")),
        (json!({"logprobs": 0}), Some("logprobs")),
        (json!({"logprobs": true}), Some("logprobs")),
        (json!({"logprobs": true, "top_logprobs": 3}), Some("logprobs")),
        (json!({"top_logprobs": 3}), Some("top_logprobs")),
        (json!({"prompt_logprobs": 0}), Some("prompt_logprobs")),
        (json!({"logprobs": null}), None),
        (json!({"logprobs": false}), None),
        (json!({"logprobs": false, "top_logprobs": null}), None),
        (json!({"prompt": "Hi", "max_tokens": 2}), None),
        (json!([1, 2]), None),
    ];
    for (body, asks) in cases {
        assert_eq!((refuse::asks_logprobs(&body), &body), (asks, &body));
    }
}

#[tokio::test]
async fn a_request_for_logprobs_is_a_400_naming_the_parameter() {
    let cases = [
        (COMPLETIONS, r#"{"prompt": "Hi", "max_tokens": 2, "logprobs": 2}"#, "logprobs"),
        (COMPLETIONS, r#"{"prompt": "Hi", "prompt_logprobs": 0}"#, "prompt_logprobs"),
        (CHAT, r#"{"messages": [], "logprobs": true}"#, "logprobs"),
        (CHAT, r#"{"messages": [], "top_logprobs": 2}"#, "top_logprobs"),
    ];
    for (path, body, name) in cases {
        let (status, text) = send(true, path, body).await;
        let error: Value = serde_json::from_str(&text).unwrap();
        let message = error["error"]["message"].as_str().unwrap();
        assert_eq!((path, body, status), (path, body, StatusCode::BAD_REQUEST));
        assert_eq!((path, body, &error["error"]["type"]), (path, body, &json!("invalid_request_error")));
        assert!(message.contains(name), "{path} {body}: {message}");
    }
}

#[tokio::test]
async fn any_other_request_reaches_the_engine_as_sent() {
    let cases = [
        (COMPLETIONS, r#"{"prompt": "Hi", "max_tokens": 2}"#),
        (COMPLETIONS, r#"{"prompt": "Hi", "logprobs": null}"#),
        (CHAT, r#"{"messages": [], "logprobs": false, "top_logprobs": null}"#),
        (CHAT, "not json: the frontend's to refuse"),
    ];
    for (path, body) in cases {
        assert_eq!(send(true, path, body).await, (StatusCode::OK, body.to_owned()));
    }
    let models =
        refuse::routes(engine(), false).oneshot(Request::get("/v1/models").body(Body::empty()).unwrap()).await.unwrap();
    assert_eq!(models.status(), StatusCode::OK);
}

#[test]
fn a_temperature_asks_for_a_draw_unless_it_is_greedy_or_one_token_survives() {
    let cases = [
        (json!({"temperature": 0.8}), Some(0.8)),
        (json!({"temperature": 1}), Some(1.0)),
        (json!({"temperature": 0.8, "top_k": 50}), Some(0.8)),
        (json!({"temperature": 0.8, "top_k": 1}), None),
        (json!({"temperature": 0}), None),
        (json!({"temperature": 0.000001}), None),
        (json!({"temperature": null}), None),
        (json!({"top_p": 0.9, "seed": 1}), None),
        (json!({"prompt": "Hi"}), None),
    ];
    for (body, asks) in cases {
        assert_eq!((refuse::asks_draw(&body), &body), (asks, &body));
    }
}

#[tokio::test]
async fn a_draw_is_a_400_where_the_manifest_takes_no_sampling_fills_and_passes_where_it_does() {
    let drawn = [
        (COMPLETIONS, r#"{"prompt": "Hi", "max_tokens": 2, "temperature": 0.8}"#),
        (CHAT, r#"{"messages": [], "temperature": 0.8, "top_p": 0.9}"#),
    ];
    for (path, body) in drawn {
        let (status, text) = send(false, path, body).await;
        let error: Value = serde_json::from_str(&text).unwrap();
        let message = error["error"]["message"].as_str().unwrap();
        assert_eq!((path, status), (path, StatusCode::BAD_REQUEST));
        assert!(message.contains("temperature of 0.8") && message.contains("temperature 0"), "{path}: {message}");
        assert_eq!(send(true, path, body).await, (StatusCode::OK, body.to_owned()));
    }
    let greedy = [
        (COMPLETIONS, r#"{"prompt": "Hi", "temperature": 0}"#),
        (COMPLETIONS, r#"{"prompt": "Hi", "temperature": 0.8, "top_k": 1}"#),
        (CHAT, r#"{"messages": []}"#),
    ];
    for (path, body) in greedy {
        assert_eq!(send(false, path, body).await, (StatusCode::OK, body.to_owned()));
    }
}
