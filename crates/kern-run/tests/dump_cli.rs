//! Invalid dump requests fail before device loading or artifact creation.
//! The command uses a tiny serving contract and no CUDA-visible devices.

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

use kern_manifest::{verify, Manifest};
use serde_json::{json, Value};

struct Temp(PathBuf);

impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        loop {
            let root = std::env::temp_dir().join(format!(
                "kern-dump-cli-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&root) {
                Ok(()) => {
                    let temp = Self(root);
                    fs::create_dir(temp.0.join("weights")).unwrap();
                    fs::write(temp.0.join("tokenizer.json"), "{}").unwrap();
                    fs::write(
                        temp.0.join("kern.toml"),
                        "[targets.fixture]\nmanifest = \"manifest.json\"\nweights = [\"weights\"]\ntokenizer = \"tokenizer.json\"\n",
                    )
                    .unwrap();
                    let manifest = serde_json::to_string(&fixture()).unwrap();
                    verify(Manifest::from_json(&manifest).unwrap())
                        .unwrap_or_else(|e| panic!("fixture does not verify: {e}"));
                    fs::write(temp.0.join("manifest.json"), manifest).unwrap();
                    return temp;
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => panic!("creating test directory: {e}"),
            }
        }
    }

    fn command(&self) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_kern"));
        command
            .current_dir(&self.0)
            .env("CUDA_VISIBLE_DEVICES", "")
            .arg("--config")
            .arg(self.0.join("kern.toml"))
            .arg("dump");
        command
    }

    fn out(&self) -> PathBuf {
        self.0.join("out")
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn fixture() -> Value {
    json!({
        "schema_version": 7, "model": "fixture",
        "vars": {"rows": {"max": 2, "axis": "rows"}, "groups": {"max": 1, "axis": "groups"}},
        "states": {"cache": {"bytes": 16}},
        "buffers": {
            "input": {"kind": "input", "dtype": "i64", "shape": ["rows"], "fill": "token"},
            "output": {"kind": "output", "dtype": "i64", "shape": ["groups"], "fill": "tokens"}
        },
        "modules": {},
        "ops": {"emit": {
            "params": ["in buffer<i64>", "out buffer<i64>", "inout state"],
            "impl": {"launches": [{"entry": "extern:fixture"}]}
        }},
        "programs": {"step": {"batch": {"groups": 1, "rows": 1}, "calls": [
            {"label": "prepare", "op": "emit", "args": [{"buf": "input"}, {"buf": "output"}, {"state": "cache"}]},
            {"label": "repeat", "op": "emit", "args": [{"buf": "input"}, {"buf": "output"}, {"state": "cache"}]}
        ]}}
    })
}

fn rejected(output: Output, code: i32, expected: &str) {
    let message = String::from_utf8_lossy(&output.stdout).into_owned() + &String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(code), "{message}");
    assert!(message.contains(expected), "missing `{expected}` in {message}");
}

#[test]
fn valid_phase_and_range_syntax_reaches_static_selection_checks() {
    for phase in ["before", "after", "both"] {
        let temp = Temp::new();
        let output = temp
            .command()
            .args([
                "--program",
                "step",
                "--call-label",
                "absent",
                "--out",
                "out",
                "--phase",
                phase,
                "--range",
                "buffer:input=0:8",
                "--range",
                "state:cache=0:16",
                "--stop-tokens",
                "7",
            ])
            .output()
            .unwrap();
        rejected(output, 1, "no call has label `absent`");
        assert!(!temp.out().exists());
    }
}

#[test]
fn static_program_labels_and_indices_fail_before_gpu_loading() {
    let cases: &[(&[&str], &str)] = &[
        (&["--program", "absent", "--call-index", "0"], "absent"),
        (&["--program", "step", "--call-label", "par"], "no call has label `par`"),
        (&["--program", "step", "--call-label", "prepare,absent"], "no call has label `absent`"),
        (&["--program", "step", "--call-index", "0,9"], "call index 9 outside 0..2"),
    ];
    for (args, expected) in cases {
        let temp = Temp::new();
        let output = temp.command().args(*args).args(["--out", "out", "--stop-tokens", "7"]).output().unwrap();
        rejected(output, 1, expected);
        assert!(!temp.out().exists());
    }
}

#[test]
fn occupied_output_is_preserved_before_manifest_or_gpu_loading() {
    let temp = Temp::new();
    let out = temp.out();
    fs::create_dir(&out).unwrap();
    fs::write(out.join("existing.txt"), b"keep this").unwrap();
    // A later static error must not hide the refusal to overwrite this folder.
    let output = temp.command().args(["--program", "absent", "--call-index", "0", "--out", "out"]).output().unwrap();
    rejected(output, 1, "not empty");
    assert_eq!(
        (fs::read(out.join("existing.txt")).unwrap(), fs::read_dir(&out).unwrap().count()),
        (b"keep this".to_vec(), 1)
    );
}
