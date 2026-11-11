//! A hosted runtime's kernels on disk: the files kern-serve's `--trace-dir`
//! writes (kern-trace), from inside another engine's process.
//!
//! The process subscribed CUPTI before its first CUDA context
//! (`kern.trace.start()`). [`start`] is the thread that collects every
//! [`EVERY`] and appends what arrived under a directory: the steps of the
//! programs the host issued or launched as its own graphs, and the kernels
//! no step claims, which are the host's own.
//!
//! A pair of files closes at the quarter hour, when the file `rotate` in
//! the directory is touched, and when the recording ends ([`finish`], which
//! `kern.trace` asks for as the interpreter exits). A host that ends its
//! processes with a signal nobody handles never gets there, which leaves
//! the pair being written without its footer, so whoever wants the last
//! minutes of such a host touches the file first.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::Mutex;
use std::time::{Duration, SystemTime};

use anyhow::{bail, Context, Result};
use kern_runtime::trace;
use kern_trace::{Now, Recorder, Sites};

const EVERY: Duration = Duration::from_millis(200);
/// How long [`finish`] waits for the last collect: a device that never
/// answers does not keep the process from ending.
const FINISH: Duration = Duration::from_secs(3);

/// The process's recording thread: what ends it, and what it says once it
/// has ended.
static RECORDING: Mutex<Option<(mpsc::Sender<()>, mpsc::Receiver<()>)>> = Mutex::new(None);

fn touched(path: &Path) -> Option<SystemTime> {
    std::fs::metadata(path).and_then(|m| m.modified()).ok()
}

/// Starts the thread that records under `dir` as `rank`; one per process.
pub fn start(dir: PathBuf, rank: usize, sites: Sites, meta: Vec<(String, String)>) -> Result<()> {
    let mut recording = RECORDING.lock().unwrap_or_else(|e| e.into_inner());
    if recording.is_some() {
        bail!("this process already records its kernels");
    }
    let (stop, stopped) = mpsc::channel();
    let (ended, end) = mpsc::channel();
    let record = move || {
        let trigger = dir.join("rotate");
        let mut recorder = Recorder::new(dir.clone(), rank, meta);
        let (mut asked, mut failing, mut last) = (touched(&trigger), false, false);
        while !last {
            last = !matches!(stopped.recv_timeout(EVERY), Err(RecvTimeoutError::Timeout));
            let collected = if last { trace::finish() } else { trace::collect() };
            let written = collected
                .and_then(|c| Ok((c, Now::read()?)))
                .map_err(anyhow::Error::from)
                .and_then(|(c, now)| recorder.write(&c, &sites, now));
            let again = touched(&trigger);
            let closed = match last || again != asked {
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
        let _ = ended.send(());
    };
    std::thread::Builder::new().name("kern-trace".into()).spawn(record).context("starting the recording thread")?;
    *recording = Some((stop, end));
    Ok(())
}

/// Ends the recording: the thread collects once more, with everything
/// CUPTI still holds, and closes its files. Nothing when nothing records.
pub fn finish() {
    let recording = RECORDING.lock().unwrap_or_else(|e| e.into_inner()).take();
    if let Some((stop, end)) = recording {
        let _ = stop.send(());
        let _ = end.recv_timeout(FINISH);
    }
}
