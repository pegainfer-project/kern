//! Where a target's weights are. A `--weights` entry is a checkpoint
//! directory (every `*.safetensors` in it, in name order) or one
//! `.safetensors` file; the tensors of every entry bind by name. A rank's
//! shard of a tensor is the manifest's business (a bind selects rows,
//! columns or tensors by rank), never a per-rank file.

use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use kern_runtime::{Runtime, Safetensors};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Weights(Vec<PathBuf>);

impl Weights {
    pub fn parse(entries: &[String]) -> Result<Weights> {
        if entries.is_empty() {
            bail!("no weights");
        }
        Ok(Weights(entries.iter().map(PathBuf::from).collect()))
    }

    /// The checkpoint directories, where the tokenizer and the model's
    /// config live: each entry's (the directory itself, or the one holding
    /// the file).
    pub fn dirs(&self) -> Vec<PathBuf> {
        self.0.iter().map(|p| checkpoint_dir(p)).collect()
    }

    /// What identifies these weights to a pooled KV tier: a digest of
    /// every file's size and safetensors header (every tensor's dtype,
    /// shape and offset), in file order. Reads a few MB, never the
    /// tensors; two checkpoints with the same tensors laid out the same
    /// way but different values are told apart by nothing cheaper.
    pub fn digest(&self) -> Result<[u8; 32]> {
        use sha2::{Digest, Sha256};
        use std::io::Read;
        let mut h = Sha256::new();
        for path in shard_files(&self.0)? {
            let mut f = std::fs::File::open(&path).with_context(|| path.display().to_string())?;
            let len = f.metadata()?.len();
            let mut n = [0u8; 8];
            f.read_exact(&mut n).with_context(|| format!("{}: no safetensors header", path.display()))?;
            let mut header = vec![0u8; u64::from_le_bytes(n) as usize];
            f.read_exact(&mut header).with_context(|| format!("{}: short safetensors header", path.display()))?;
            h.update(len.to_le_bytes());
            h.update(n);
            h.update(&header);
        }
        Ok(h.finalize().into())
    }

    pub fn bind(&self, rt: &mut Runtime) -> Result<()> {
        Ok(rt.load_weights(&Safetensors::open(&shard_files(&self.0)?)?)?)
    }
}

impl std::fmt::Display for Weights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(" + "))
    }
}

/// The safetensors files the entries name: a file itself, or every
/// `*.safetensors` under a directory, in name order.
fn shard_files(paths: &[PathBuf]) -> Result<Vec<PathBuf>> {
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
    Ok(files)
}

fn checkpoint_dir(w: &Path) -> PathBuf {
    let dir = if w.extension().is_some_and(|x| x == "safetensors") { w.parent().unwrap_or(Path::new(".")) } else { w };
    if dir.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        dir.to_path_buf()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entries(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn entries_are_files() {
        assert_eq!(
            Weights::parse(&entries(&["a", "b/c.safetensors"])).unwrap(),
            Weights(vec!["a".into(), "b/c.safetensors".into()])
        );
        assert!(Weights::parse(&[]).unwrap_err().to_string().contains("no weights"));
        assert_eq!(Weights::parse(&entries(&["a", "b"])).unwrap().to_string(), "a + b");
    }

    #[test]
    fn checkpoint_dir_is_the_entry_or_the_file_s_directory() {
        let d = |s: &str| checkpoint_dir(Path::new(s));
        assert_eq!(
            (d("weights/Qwen3-4B"), d("weights/Qwen3-4B/model-00001.safetensors"), d("model.safetensors")),
            ("weights/Qwen3-4B".into(), "weights/Qwen3-4B".into(), PathBuf::from("."))
        );
    }
}
