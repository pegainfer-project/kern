//! A hosted runtime's kernels on disk: the files kern-serve's `--trace-dir`
//! writes (kern-trace), from inside another engine's process.
//!
//! The process subscribed CUPTI before its first CUDA context
//! (`kern.trace_start()`). [`start`] is the thread that collects every
//! [`EVERY`] and appends what arrived under a directory: the steps of the
//! programs the host issued or launched as its own graphs, and the kernels
//! no step claims, which are the host's own.
//!
//! A pair of files closes at the quarter hour, or when the file `rotate`
//! in the directory is touched. A host ends its processes with a signal
//! nobody handles, which leaves the pair being written without its
//! footer, so whoever wants the last minutes touches the file first.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use kern_runtime::trace;
use kern_trace::{Now, Recorder, Sites};

const EVERY: Duration = Duration::from_millis(200);

fn touched(path: &Path) -> Option<SystemTime> {
    std::fs::metadata(path).and_then(|m| m.modified()).ok()
}

/// Starts the thread that records under `dir` as `rank`.
pub fn start(dir: PathBuf, rank: usize, sites: Sites, meta: Vec<(String, String)>) -> std::io::Result<()> {
    let record = move || {
        let trigger = dir.join("rotate");
        let mut recorder = Recorder::new(dir.clone(), rank, meta);
        let (mut asked, mut failing) = (touched(&trigger), false);
        loop {
            std::thread::sleep(EVERY);
            let written = trace::collect()
                .and_then(|c| Ok((c, Now::read()?)))
                .map_err(anyhow::Error::from)
                .and_then(|(c, now)| recorder.write(&c, &sites, now));
            let again = touched(&trigger);
            let closed = match again != asked {
                true => recorder.rotate().map(drop),
                false => Ok(()),
            };
            asked = again;
            match written.and(closed) {
                Ok(()) => failing = false,
                Err(e) if !failing => {
                    eprintln!("kern: recording kernels under {}: {e:#}", dir.display());
                    failing = true;
                }
                Err(_) => {}
            }
        }
    };
    std::thread::Builder::new().name("kern-trace".into()).spawn(record).map(drop)
}
