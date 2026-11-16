//! Pure-CPU cost repricing, memory budgeting, and discrete-event cluster
//! simulation over [`kern_manifest`] and [`kern_pool`].
//!
//! Everything here runs on the host without touching a GPU:
//! - [`cost`] builds a content-addressed operator cost book from `kern bench`
//!   reports or traced Parquet step/kernel pairs, matching active launches at
//!   concrete call variables across composed manifests and pricing program
//!   critical paths over `lane`/`after` DAGs;
//! - [`memory`] mirrors `kern-runtime`'s static allocation accounting (`vars`
//!   upper bounds, op scratch, `derive` staging windows, per-sequence slots,
//!   1 GiB headroom, and optional calibrated CUDA driver overhead) to predict
//!   exact KV page capacity;
//! - [`workload`] reads `workload.toml` sweeps or `steps.parquet` traces and
//!   prices candidate manifests side by side;
//! - [`trace`] loads Mooncake/JSONL request traces or synthesizes multi-turn
//!   session workloads, supporting both open-loop timestamps and closed-loop
//!   think-time session chaining;
//! - [`router`] and [`cluster`] drive N simulated endpoints — each backed by a
//!   real [`kern_manifest::Protocol`], [`kern_pool::Pool`], and
//!   [`kern_pool::Prefix`] — through a deterministic discrete-event timeline.

#![forbid(unsafe_code)]

pub mod cluster;
pub mod cost;
pub mod memory;
pub mod router;
pub mod trace;
pub mod workload;
