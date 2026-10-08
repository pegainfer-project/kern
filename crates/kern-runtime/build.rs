//! The identity of the runtime a binary links: a digest of the sources of
//! `kern-manifest`, `kern-pool`, `kern-runtime` and `kern-run` — the four
//! crates `kern` and `kern-serve` both link — plus the workspace's dependency
//! table, recorded in every binary as [`kern_runtime::ID`]. A launcher hands
//! its own over to the binary it starts, which refuses when the two disagree:
//! the failure this catches is a server left holding the runtime from before
//! a change while the launcher tests the one after it.
//!
//! Everything under a crate's `src/`, not only its `.rs` files: `compare.ptx`
//! and `profile.ptx` are `include_str!`d into the runtime, and a rule that
//! picked extensions would leave the next such input out.
//!
//! Relative paths and bytes, so one source tree gives one id on every machine.
//! The toolchain is not part of it: the id answers "the same sources", and a
//! release binary and a source build of one commit have to agree.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// The crates `kern` and `kern-serve` both link, in one fixed order.
const CRATES: [&str; 4] = ["kern-manifest", "kern-pool", "kern-runtime", "kern-run"];

fn sources(dir: &Path, files: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    for entry in entries {
        let path = entry.expect("a directory entry").path();
        if path.is_dir() {
            sources(&path, files);
        } else {
            files.push(path);
        }
    }
}

fn main() {
    let crates = Path::new(env!("CARGO_MANIFEST_DIR")).parent().expect("crates/").to_path_buf();
    let repo = crates.parent().expect("the repository root").to_path_buf();
    let mut files = Vec::new();
    for krate in CRATES {
        let dir = crates.join(krate);
        for path in [dir.join("Cargo.toml"), dir.join("build.rs")] {
            println!("cargo:rerun-if-changed={}", path.display());
            if path.is_file() {
                files.push(path);
            }
        }
        println!("cargo:rerun-if-changed={}", dir.join("src").display());
        sources(&dir.join("src"), &mut files);
    }
    // The dependency versions both binaries resolve (`cudarc`, `half`, the
    // pinned-exact ones) are decisions this table holds.
    println!("cargo:rerun-if-changed={}", repo.join("Cargo.toml").display());
    files.push(repo.join("Cargo.toml"));
    files.sort();
    let mut digest = Sha256::new();
    for file in &files {
        let name = file.strip_prefix(&repo).expect("under the repository root");
        digest.update(name.to_string_lossy().replace('\\', "/").as_bytes());
        digest.update([0]);
        digest.update(std::fs::read(file).unwrap_or_else(|e| panic!("{}: {e}", file.display())));
        digest.update([0]);
    }
    println!("cargo:rustc-env=KERN_RUNTIME_ID={}", &hex::encode(digest.finalize())[..12]);
}
