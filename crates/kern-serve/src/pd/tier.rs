//! The decode tier's head as a process: its ranks loaded and joined
//! (`executor`), a [`Decoder`] over them, and one prefill head connected.
//!
//! It opens with [`Pools`], then reads the head's asks on a thread of its
//! own and, on the scheduler's thread, answers each in arrival order and
//! runs a step whenever there is work. With nothing running and nothing
//! landing it blocks on the next ask. The prefill head going away ends the
//! process, as a closed connection does everywhere else.

use std::collections::BTreeMap;
use std::net::TcpListener;
use std::sync::mpsc;
use std::time::Instant;

use anyhow::{Context, Result};
use tracing::{error, info};

use super::{Ask, Decoder, Run, Shape, Tell};
use crate::logline;
use crate::pp::{recv, send};
use crate::ranks::Ranks;

/// Serve the prefill head that connects at `listen` from `ranks` run by
/// `exec`, whose pools `handles` name, until it goes.
pub fn serve<E: Run>(
    ranks: Ranks,
    exec: E,
    handles: Vec<String>,
    states: BTreeMap<String, Shape>,
    listen: &str,
    max_seqs: usize,
) -> Result<()> {
    let mut d = Decoder::new(ranks, exec, max_seqs)?;
    let listener = TcpListener::bind(listen).with_context(|| format!("binding {listen}"))?;
    info!(listen, seats = d.cap(), "decode tier waiting for its prefill head");
    let (mut s, addr) = listener.accept().context("accepting the prefill head")?;
    let _ = s.set_nodelay(true);
    send(&mut s, &Tell::Pools(d.pools(handles, states)))?;
    info!(%addr, "prefill head connected");
    let (tx, asks) = mpsc::channel::<Ask>();
    std::thread::spawn({
        let mut r = s.try_clone()?;
        move || loop {
            match recv::<Ask>(&mut r) {
                Ok(Some(a)) => {
                    if tx.send(a).is_err() {
                        return;
                    }
                }
                Ok(None) => {
                    info!("the prefill head closed its connection");
                    std::process::exit(0);
                }
                Err(e) => {
                    error!("prefill head connection: {e:#}");
                    std::process::exit(1);
                }
            }
        }
    });
    let mut window = (Instant::now(), 0u64, 0u64, 0u128);
    loop {
        if !d.busy() {
            let a = asks.recv().context("the prefill head's reader is gone")?;
            for t in d.ask(a)? {
                send(&mut s, &t)?;
            }
        }
        while let Ok(a) = asks.try_recv() {
            for t in d.ask(a)? {
                send(&mut s, &t)?;
            }
        }
        let t0 = Instant::now();
        for t in d.step()? {
            if let Tell::Tokens { step, .. } = &t {
                window.1 += 1;
                window.2 += step.len() as u64;
                window.3 += t0.elapsed().as_nanos();
            }
            send(&mut s, &t)?;
        }
        if window.0.elapsed().as_secs() >= 5 {
            let (since, steps, tokens, ns) = std::mem::replace(&mut window, (Instant::now(), 0, 0, 0));
            if steps > 0 {
                let load = d.load();
                info!(
                    seated = load.seats,
                    running = d.running(),
                    kv_pct = (load.pages as f64 * 1000.0 / load.pages_total.max(1) as f64).round() / 10.0,
                    steps,
                    step_ms = ns as f64 / 1e6 / steps as f64,
                    tok_s = tokens as f64 / since.elapsed().as_secs_f64(),
                    window_s = logline::secs(since.elapsed()),
                    "decode stats"
                );
            }
        }
    }
}
