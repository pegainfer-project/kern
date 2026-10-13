//! A group fails fast: when a member dies, the others exit rather than
//! wait forever on a collective, a stream or a device that will never
//! finish (docs/protocol.md, L1).
//!
//! Every member watches its connections to the members it talks to (the
//! head every rank and stage, each of those the head) on a thread of its
//! own. A connection the other end closed ends the process at once,
//! without unwinding: the thread that would unwind may be blocked on its
//! device, and dropping a runtime waits on the device too. A member that
//! exits for any reason closes its connections, so death travels through
//! the head to the whole group in two hops.

use std::io::ErrorKind;
use std::net::TcpStream;
use std::time::Duration;

use anyhow::{Context, Result};
use tracing::error;

/// End the process once the other end of `s`, `peer`, closes it.
pub fn exit_with(s: &TcpStream, peer: String) -> Result<()> {
    let s = s.try_clone().with_context(|| format!("watching {peer}"))?;
    std::thread::Builder::new().name(format!("watch {peer}")).spawn(move || {
        let mut b = [0u8; 1];
        loop {
            match s.peek(&mut b) {
                Ok(0) => break,
                // Bytes its reader has not taken yet: look again once it may have.
                Ok(_) => std::thread::sleep(Duration::from_millis(10)),
                Err(e) if matches!(e.kind(), ErrorKind::Interrupted | ErrorKind::WouldBlock | ErrorKind::TimedOut) => {}
                Err(_) => break,
            }
        }
        error!(%peer, "the connection closed: the group is gone, exiting");
        std::process::exit(1);
    })?;
    Ok(())
}
