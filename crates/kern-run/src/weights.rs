//! Where a target's weights are. A `--weights` entry is a checkpoint
//! directory (every `*.safetensors` in it, in name order) or one
//! `.safetensors` file; the tensors of every entry bind by name. A rank's
//! shard of a tensor is the manifest's business (a bind selects rows,
//! columns or tensors by rank), never a per-rank file.

use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use kern_runtime::{Runtime, Safetensors};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Weights {
    Files(Vec<PathBuf>),
}

impl Weights {
    pub fn parse(entries: &[String]) -> Result<Weights> {
        if entries.is_empty() {
            bail!("no weights");
        }
        Ok(Weights::Files(entries.iter().map(PathBuf::from).collect()))
    }

    /// The checkpoint directories, where the tokenizer and the model's
    /// config live: each entry's (the directory itself, or the one holding
    /// the file).
    pub fn dirs(&self) -> Vec<PathBuf> {
        match self {
            Weights::Files(paths) => paths.iter().map(|p| checkpoint_dir(p)).collect(),
        }
    }

    pub fn bind(&self, rt: &mut Runtime) -> Result<()> {
        match self {
            Weights::Files(paths) => Ok(rt.load_weights(&Safetensors::open(&crate::shard_files(paths)?)?)?),
        }
    }
}

impl std::fmt::Display for Weights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Weights::Files(paths) => {
                write!(f, "{}", paths.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(" + "))
            }
        }
    }
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
            Weights::Files(vec!["a".into(), "b/c.safetensors".into()])
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
