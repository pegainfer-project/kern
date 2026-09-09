//! One sequence driven over a manifest's serving protocol. The runtime is
//! model-agnostic and executes blindly; [`Protocol`] is what the manifest
//! declares about calling it (which buffer carries which role, what shape
//! of call each program takes); [`Caller`] is the one sequence, holding
//! one [`Lease`] of the runtime's token slots for its whole life and
//! staging every call through the protocol's fills. Nothing here names a
//! buffer, a program or a var. `kern run` (generation) and `kern test`
//! (A/B evidence) both drive the runtime through it.

#![deny(unsafe_code)]

pub mod bench;
pub mod config;
pub mod run;
pub mod test;

use std::collections::BTreeMap;
use std::sync::LazyLock;

use anyhow::{ensure, Context, Result};
use kern_manifest::protocol::{Axis, Forward, Rows};
use kern_manifest::types::Fill;
use kern_manifest::Protocol;
use kern_runtime::{Lease, Runtime};

/// What `kern --version` prints: the crate version, the commit it was built
/// from, and the CUDA API the runtime binds; the three facts a bug report
/// needs. The commit comes from `build.rs`.
pub static VERSION: LazyLock<String> = LazyLock::new(|| {
    let (major, minor) = (kern_runtime::CUDA_API / 1000, kern_runtime::CUDA_API % 1000 / 10);
    format!("{} ({}, cuda {major}.{minor})", env!("CARGO_PKG_VERSION"), env!("KERN_COMMIT"))
});

/// What a checkpoint directory says besides its tensors: `tokenizer.json`
/// and the ids that end generation (`generation_config.json`'s
/// `eos_token_id`, else `config.json`'s; one id or a list). Over several
/// `--weights` entries the first tokenizer wins and the eos ids are the
/// union in order; a bare .safetensors file carries neither.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Checkpoint {
    pub tokenizer: Option<std::path::PathBuf>,
    pub stop_tokens: Vec<i64>,
}

pub fn checkpoint(paths: &[std::path::PathBuf]) -> Checkpoint {
    let mut c = Checkpoint::default();
    for d in paths.iter().filter(|p| p.is_dir()) {
        let t = d.join("tokenizer.json");
        if c.tokenizer.is_none() && t.is_file() {
            c.tokenizer = Some(t);
        }
        for id in eos_ids(d) {
            if !c.stop_tokens.contains(&id) {
                c.stop_tokens.push(id);
            }
        }
    }
    c
}

/// The eos ids one HF directory declares: `generation_config.json`'s,
/// else `config.json`'s, else none.
pub fn eos_ids(dir: &std::path::Path) -> Vec<i64> {
    let read = |f: &str| -> Option<serde_json::Value> {
        serde_json::from_str(&std::fs::read_to_string(dir.join(f)).ok()?).ok()
    };
    ["generation_config.json", "config.json"]
        .iter()
        .filter_map(|f| read(f))
        .map(|v| eos_of(&v))
        .find(|ids| !ids.is_empty())
        .unwrap_or_default()
}

/// `eos_token_id` as a list, whether it was written as one id or several.
fn eos_of(v: &serde_json::Value) -> Vec<i64> {
    match v.get("eos_token_id") {
        Some(serde_json::Value::Number(n)) => n.as_i64().into_iter().collect(),
        Some(serde_json::Value::Array(a)) => a.iter().filter_map(|x| x.as_i64()).collect(),
        _ => Vec::new(),
    }
}

pub fn le_bytes_i32(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// The var vars of one call.
pub type Vars = BTreeMap<String, u64>;

/// The safetensors a `--weights` entry stands for: the file itself, or
/// every `*.safetensors` under a directory (a checkpoint's shards) in
/// name order, mapped read-only. Nothing is read up front: the runtime
/// parses headers and copies each bound segment straight out of the map.
pub fn map_weights(paths: &[std::path::PathBuf]) -> Result<Vec<memmap2::Mmap>> {
    let mut files = Vec::new();
    for p in paths {
        if p.is_dir() {
            let mut shards: Vec<_> = std::fs::read_dir(p)
                .with_context(|| format!("weights dir {}", p.display()))?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|f| f.extension().is_some_and(|x| x == "safetensors"))
                .collect();
            ensure!(!shards.is_empty(), "weights dir {}: no .safetensors in it", p.display());
            shards.sort();
            files.extend(shards);
        } else {
            files.push(p.clone());
        }
    }
    files.iter().map(|f| map_file(f)).collect()
}

#[allow(unsafe_code)]
fn map_file(f: &std::path::Path) -> Result<memmap2::Mmap> {
    let file = std::fs::File::open(f).with_context(|| format!("weights {}", f.display()))?;
    // Mapped, not read: a 50 GB checkpoint costs no DRAM of its own and
    // two runtimes on one host share the page cache.
    unsafe { memmap2::Mmap::map(&file) }.with_context(|| format!("mapping weights {}", f.display()))
}

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
    /// advance. Returns the call's var vars.
    pub fn stage(&mut self, ids: &[i64]) -> Result<Vars> {
        let c = ids.len();
        let pos = self.pos as usize;
        let e = self.protocol.vars(1, c as u64, c as u64);
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
    pub fn stage_rows(&mut self, tok: i64, rows: u64) -> Result<Vars> {
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
    pub fn prefill(&mut self, ids: &[i64], chunk: u64) -> Result<Option<i64>> {
        let f = self.chunk_forward()?;
        let mut last = None;
        let mut i = 0usize;
        while i < ids.len() {
            let c = ((ids.len() - i) as u64).min(chunk) as usize;
            let e = self.stage(&ids[i..i + c])?;
            self.rt.issue(&f.name, &e)?;
            self.rt.synchronize()?;
            self.advance(c as u64);
            last = self.emitted(&f)?.0.first().copied();
            i += c;
        }
        Ok(last)
    }

    /// Vocabulary size as declared by the token fill's domain (1000 if none).
    pub fn vocab(&self) -> u64 {
        let m = &self.rt.manifest;
        m.buffers[&self.protocol.token_rows().name]
            .domain
            .as_ref()
            .and_then(|d| d.resolve(m, &self.protocol.vars(1, 1, 1), &self.rt.provision()).ok())
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn eos_is_one_id_or_a_list() {
        assert_eq!(eos_of(&json!({"eos_token_id": 7})), vec![7]);
        assert_eq!(eos_of(&json!({"eos_token_id": [3, 5]})), vec![3, 5]);
        assert_eq!(eos_of(&json!({"eos_token_id": [3, "x"]})), vec![3]);
        assert_eq!(eos_of(&json!({"bos_token_id": 1})), Vec::<i64>::new());
    }

    #[test]
    fn a_checkpoint_dir_names_its_tokenizer_and_eos() {
        let root = std::env::temp_dir().join(format!("kern-checkpoint-{}", std::process::id()));
        let (a, b) = (root.join("a"), root.join("b"));
        std::fs::create_dir_all(&a).unwrap();
        std::fs::create_dir_all(&b).unwrap();
        std::fs::write(a.join("tokenizer.json"), "{}").unwrap();
        std::fs::write(a.join("config.json"), r#"{"eos_token_id": 1}"#).unwrap();
        std::fs::write(a.join("generation_config.json"), r#"{"eos_token_id": [2, 3]}"#).unwrap();
        std::fs::write(b.join("tokenizer.json"), "{}").unwrap();
        std::fs::write(b.join("config.json"), r#"{"eos_token_id": [3, 4]}"#).unwrap();
        let file = root.join("w.safetensors");
        std::fs::write(&file, "").unwrap();
        // generation_config beats config; first tokenizer wins; ids union in order; a file adds nothing
        let ck = checkpoint(&[file, a.clone(), b]);
        assert_eq!((ck.tokenizer, ck.stop_tokens), (Some(a.join("tokenizer.json")), vec![2, 3, 4]));
        assert_eq!(checkpoint(&[root.join("missing")]), Checkpoint::default());
        std::fs::remove_dir_all(&root).unwrap();
    }
}
