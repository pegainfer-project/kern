//! A completed dump is its JSON index, the original manifest, and hashed
//! byte blobs. The index is published last, so interrupted captures remain
//! visibly incomplete. Metadata describes storage and argument views; this
//! layer preserves it without interpreting model or tensor semantics.

use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

const VERSION: u32 = 1;
const CHUNK: usize = 4 << 20;

/// One raw byte payload, addressed independently of model names and labels.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Blob {
    pub id: String,
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
}

/// The completion index of a dump. Completion describes successful file
/// writes; metadata separately declares any data still needed for replay.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Artifact {
    pub version: u32,
    pub manifest_sha256: String,
    pub metadata: Value,
    pub blobs: Vec<Blob>,
}

impl Artifact {
    /// Read a completed artifact and verify every file before returning it.
    /// Blob paths must use the generated namespace and cannot follow symlinks.
    pub fn read(dir: &Path) -> Result<Self> {
        let index = dir.join("dump.json");
        regular_file(&index).context("dump is incomplete or its completion index is invalid")?;
        let artifact: Self = serde_json::from_slice(&fs::read(&index)?)
            .with_context(|| format!("reading dump index {}", index.display()))?;
        ensure!(artifact.version == VERSION, "unsupported dump version {}; expected {VERSION}", artifact.version);

        let manifest = dir.join("manifest.json");
        regular_file(&manifest)?;
        let bytes = fs::read(&manifest).with_context(|| format!("reading {}", manifest.display()))?;
        ensure!(hex::encode(Sha256::digest(&bytes)) == artifact.manifest_sha256, "manifest.json SHA-256 mismatch");
        serde_json::from_slice::<Value>(&bytes).context("manifest.json is not valid JSON")?;
        ensure!(fs::symlink_metadata(dir.join("blobs"))?.is_dir(), "blobs must be a directory, not a symlink");

        let mut ids = BTreeSet::new();
        let mut block = vec![0; CHUNK];
        for blob in &artifact.blobs {
            let number = blob.id.parse::<u64>().context("blob id must be a positive decimal number")?;
            ensure!(number > 0 && blob.id == format!("{number:06}"), "invalid blob id `{}`", blob.id);
            ensure!(ids.insert(&blob.id), "duplicate blob id `{}`", blob.id);
            ensure!(blob.path == format!("blobs/{}.bin", blob.id), "invalid blob path `{}`", blob.path);
            let path = dir.join(&blob.path);
            regular_file(&path)?;
            let mut file = File::open(&path).with_context(|| format!("opening blob `{}`", blob.id))?;
            ensure!(
                file.metadata()?.len() == blob.bytes,
                "blob `{}` length mismatch: expected {} bytes",
                blob.id,
                blob.bytes
            );
            let mut hash = Sha256::new();
            let mut count = 0u64;
            loop {
                let n = file.read(&mut block).with_context(|| format!("reading blob `{}`", blob.id))?;
                if n == 0 {
                    break;
                }
                count = count.checked_add(n as u64).context("blob length overflow")?;
                hash.update(&block[..n]);
            }
            ensure!(count == blob.bytes, "blob `{}` length changed while reading", blob.id);
            ensure!(hex::encode(hash.finalize()) == blob.sha256, "blob `{}` SHA-256 mismatch", blob.id);
        }
        Ok(artifact)
    }
}

/// A capture in progress. The budget counts the sum of raw blob payloads,
/// excluding the manifest and index. A failed blob prevents completion.
pub struct Writer {
    dir: PathBuf,
    manifest_sha256: String,
    max_bytes: u64,
    bytes: u64,
    blobs: Vec<Blob>,
    failed: bool,
}

impl Writer {
    /// Create a capture in an absent or empty directory. Existing files are
    /// never replaced, and the original manifest bytes are preserved.
    pub fn create(out: &Path, manifest_json: &str, max_bytes: u64) -> Result<Self> {
        serde_json::from_str::<Value>(manifest_json).context("dump manifest is not valid JSON")?;
        fs::create_dir_all(out).with_context(|| format!("creating dump directory {}", out.display()))?;
        ensure!(fs::read_dir(out)?.next().transpose()?.is_none(), "dump directory {} is not empty", out.display());
        fs::create_dir(out.join("blobs"))?;
        let mut manifest = OpenOptions::new().write(true).create_new(true).open(out.join("manifest.json"))?;
        manifest.write_all(manifest_json.as_bytes())?;
        manifest.sync_all()?;
        Ok(Self {
            dir: out.to_owned(),
            manifest_sha256: hex::encode(Sha256::digest(manifest_json.as_bytes())),
            max_bytes,
            bytes: 0,
            blobs: Vec::new(),
            failed: false,
        })
    }

    /// Append one payload through reads of at most 4 MiB. The callback's
    /// offsets are relative to this blob and each read must return exactly
    /// the requested length; failure leaves the artifact incomplete.
    pub fn blob(&mut self, bytes: u64, mut read: impl FnMut(u64, usize) -> Result<Vec<u8>>) -> Result<Blob> {
        let budget = self.check_budget(bytes);
        self.failed = true;
        let total = budget?;
        let id = format!("{:06}", self.blobs.len() + 1);
        let path = format!("blobs/{id}.bin");
        let mut file = OpenOptions::new().write(true).create_new(true).open(self.dir.join(&path))?;
        let mut hash = Sha256::new();
        let mut offset = 0u64;
        while offset < bytes {
            let len = (bytes - offset).min(CHUNK as u64) as usize;
            let block = read(offset, len).with_context(|| format!("reading blob `{id}` at byte {offset}"))?;
            ensure!(block.len() == len, "blob `{id}` at byte {offset}: expected {len} bytes, got {}", block.len());
            file.write_all(&block)?;
            hash.update(&block);
            offset += len as u64;
        }
        file.sync_all()?;
        let blob = Blob { id, path, bytes, sha256: hex::encode(hash.finalize()) };
        self.bytes = total;
        self.blobs.push(blob.clone());
        self.failed = false;
        Ok(blob)
    }

    /// Check the complete planned payload before running a selected prefix.
    pub fn check_budget(&self, additional: u64) -> Result<u64> {
        ensure!(!self.failed, "dump writer cannot continue after a failed blob");
        let total = self.bytes.checked_add(additional).context("dump payload size overflow")?;
        ensure!(total <= self.max_bytes, "dump payload needs {total} bytes; budget is {} bytes", self.max_bytes);
        Ok(total)
    }

    /// Publish the completion index only after all blobs succeeded.
    /// Metadata must already encode special scalar values losslessly.
    pub fn finish(self, metadata: Value) -> Result<Artifact> {
        ensure!(!self.failed, "dump writer cannot finish after a failed blob");
        File::open(self.dir.join("blobs"))?.sync_all()?;
        let artifact =
            Artifact { version: VERSION, manifest_sha256: self.manifest_sha256, metadata, blobs: self.blobs };
        let pending = self.dir.join("dump.json.partial");
        let mut index = OpenOptions::new().write(true).create_new(true).open(&pending)?;
        serde_json::to_writer_pretty(&mut index, &artifact)?;
        index.write_all(b"\n")?;
        index.sync_all()?;
        let complete = self.dir.join("dump.json");
        // Linking publishes the complete file atomically and refuses every
        // existing directory entry, including a dangling symlink.
        fs::hard_link(&pending, &complete)
            .context("publishing dump completion index without replacing existing data")?;
        fs::remove_file(&pending).context("removing the published dump's partial index")?;
        File::open(&self.dir)?.sync_all()?;
        Ok(artifact)
    }
}

fn regular_file(path: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(path).with_context(|| format!("reading {}", path.display()))?;
    ensure!(metadata.is_file(), "{} must be a regular file, not a symlink", path.display());
    Ok(())
}
