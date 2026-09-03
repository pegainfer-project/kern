//! One sequence driven over a manifest's serving protocol. The runtime is
//! model-agnostic and executes blindly; [`Protocol`] is what the manifest
//! declares about calling it (which buffer carries which role, what shape
//! of call each program takes); [`Caller`] is the one sequence, holding
//! one [`Lease`] of the runtime's token slots for its whole life and
//! staging every call through the protocol's fills. Nothing here names a
//! buffer, a program or a var. `kern run` (generation) and `kern test`
//! (A/B evidence) both drive the runtime through it.

#![forbid(unsafe_code)]

pub mod attest;
pub mod config;
pub mod run;

use std::collections::BTreeMap;

use anyhow::{ensure, Result};
use kern_manifest::protocol::{Axis, Forward, Rows};
use kern_manifest::types::Fill;
use kern_manifest::Protocol;
use kern_runtime::{Lease, Runtime};

/// Default stop tokens (Qwen3 <|endoftext|>, <|im_end|>) for raw
/// (template-free) completion; `kern-run --stop-tokens` overrides.
pub const STOP_TOKENS: [i64; 2] = [151643, 151645];

pub fn le_bytes_i32(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// The var env of one call.
pub type Env = BTreeMap<String, u64>;

/// What one call handed back for the sequence: the tokens it takes, in
/// order (one for a decode step or a prefill chunk, `count` of `rows` for
/// a speculative round).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Emitted(pub Vec<i64>);

/// A runtime plus the single sequence: its token slots and position cursor.
pub struct Caller {
    pub rt: Runtime,
    pub protocol: Protocol,
    /// The sequence's slots: as many as one page-table row (or the whole
    /// state) holds, leased once for the caller's life.
    lease: Lease,
    /// Tokens already in the state (next slot to fill).
    pub pos: i64,
}

impl Caller {
    /// Leases the sequence's slots and writes its row into every page table
    /// once. A table has a row per sequence the manifest allows; this
    /// caller is sequence 0, but every row must hold valid page ids. Line
    /// tables of a per-sequence state likewise get this sequence's lines
    /// in every column, in entry 0 of a wide cell.
    pub fn new(mut rt: Runtime) -> Result<Caller> {
        let protocol = Protocol::check(&rt.manifest)?;
        let lease = rt.lease(rt.max_seq_tokens().min(rt.capacity() as usize))?;
        for t in &protocol.page_tables {
            let mut table = Vec::new();
            for _ in 0..protocol.groups.max {
                lease.extend_row(&t.name, &mut table)?;
            }
            rt.write_input(&t.name, &le_bytes_i32(&table))?;
        }
        for t in &protocol.line_tables {
            let cols = match t.axis {
                Axis::Tray => protocol.tray.as_ref().map_or(1, |b| b.max),
                _ => protocol.groups.max,
            };
            let mut table = Vec::new();
            for r in 0..t.lines {
                let line = lease.seq_line(&t.name, r)?;
                for _ in 0..cols {
                    table.push(line);
                    table.extend(std::iter::repeat_n(0, t.width - 1));
                }
            }
            rt.write_input(&t.name, &le_bytes_i32(&table))?;
        }
        Ok(Caller { rt, protocol, lease, pos: 0 })
    }

    /// Token slots the sequence can hold.
    pub fn limit(&self) -> usize {
        self.lease.tokens()
    }

    /// Stage one call's rows at the cursor: `ids` as consecutive positions
    /// of this one sequence, in every fill the manifest declares. Does not
    /// advance. Returns the call's var env.
    pub fn stage(&mut self, ids: &[i64]) -> Result<Env> {
        let c = ids.len();
        let pos = self.pos as usize;
        let e = self.protocol.env(1, c as u64, 1);
        let p = self.protocol.clone();
        let mut put =
            |f: &kern_manifest::protocol::Filled, v: &[i64]| self.rt.write_input_at(&f.name, &f.encode(v), &e);
        put(p.token_rows(), ids)?;
        put(p.slots(), &self.lease.slots(pos..pos + c))?;
        put(p.seq_lens(), &[(pos + c) as i64])?;
        if let Some(f) = p.filled(Fill::Token, Axis::Groups) {
            put(f, &ids[..1])?;
        }
        if let Some(f) = p.filled(Fill::Position, Axis::Rows) {
            put(f, &(self.pos..self.pos + c as i64).collect::<Vec<_>>())?;
        }
        if let Some(f) = p.any(Fill::CuSeqlens) {
            put(f, &[0, c as i64])?;
        }
        Ok(e)
    }

    /// Stage a fixed-rows call at the cursor: `tok` is the sequence's next
    /// token, in row 0 and again in every other row. A program that
    /// drafts its own rows (a speculative round) overwrites rows 1.. on
    /// the device; a one-row step reads only row 0.
    pub fn stage_rows(&mut self, tok: i64, rows: u64) -> Result<Env> {
        self.stage(&vec![tok; rows as usize])
    }

    /// Reset the cursor (a new prompt reuses the slots from position 0).
    pub fn reset(&mut self) {
        self.pos = 0;
    }

    pub fn advance(&mut self, n: u64) {
        self.pos += n as i64;
    }

    /// The forward for one sequence of `rows` rows per call.
    pub fn forward(&self, rows: Rows) -> Result<Forward> {
        self.protocol.forward(1, rows).cloned().ok_or_else(|| {
            anyhow::anyhow!(
                "no program takes one sequence of {} rows; the manifest declares rows {:?}{}",
                match rows {
                    Rows::Const(r) => r.to_string(),
                    Rows::Var => "as many as fed".into(),
                },
                self.protocol.row_shapes(),
                if self.protocol.chunk().is_some() { " and a chunk" } else { "" }
            )
        })
    }

    /// The program a chunk of the prompt goes through.
    pub fn chunk_forward(&self) -> Result<Forward> {
        self.forward(Rows::Var)
    }

    /// Chunked prefill of `ids` (eager or graph-captured full chunks),
    /// advancing the cursor past them. Returns the token the last chunk
    /// handed back, if the chunk program emits one, and whether a graph
    /// was captured.
    pub fn prefill(&mut self, ids: &[i64], chunk: u64, eager: bool) -> Result<(Option<i64>, bool)> {
        let f = self.chunk_forward()?;
        let chunk = chunk.min(self.protocol.rows.max).max(1);
        let mut captured = false;
        let mut last = None;
        let mut i = 0usize;
        while i < ids.len() {
            let c = ((ids.len() - i) as u64).min(chunk) as usize;
            let e = self.stage(&ids[i..i + c])?;
            if !eager && c as u64 == chunk {
                if !captured {
                    self.rt.capture(&f.name, &e)?;
                    captured = true;
                }
                self.rt.run_captured(&f.name, &e)?;
            } else {
                self.rt.run(&f.name, &e)?;
            }
            self.advance(c as u64);
            last = self.emitted(&f)?.0.first().copied();
            i += c;
        }
        Ok((last, captured))
    }

    /// Vocabulary size as declared by the token fill's domain (1000 if none).
    pub fn vocab(&self) -> u64 {
        let m = &self.rt.manifest;
        m.buffers[&self.protocol.token_rows().name]
            .domain
            .as_ref()
            .and_then(|d| d.resolve(m, &self.protocol.env(1, 1, 1), &self.rt.provision()).ok())
            .and_then(|r| r.hi)
            .map_or(1000, |hi| hi as u64 + 1)
    }

    /// What the last run of `f` handed back for this sequence: its `tokens`
    /// output's first cell, cut to its `count` (one without a count).
    pub fn emitted(&self, f: &Forward) -> Result<Emitted> {
        let Some(i) = f.emits else { return Ok(Emitted(Vec::new())) };
        let t = &self.protocol.fills[i];
        let mut v = t.decode(&self.rt.read_output(&t.name)?);
        let n = match f.count {
            Some(c) => {
                let c = &self.protocol.fills[c];
                let n = c.decode(&self.rt.read_output(&c.name)?)[0];
                ensure!(n >= 1 && n <= t.width as i64, "`{}` says {n} of the {} rows are taken", c.name, t.width);
                n as usize
            }
            None => 1,
        };
        v.truncate(n);
        Ok(Emitted(v))
    }
}
