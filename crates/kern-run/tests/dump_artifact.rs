//! Dump files as a caller writes and reads them: exact bytes and metadata,
//! bounded capture reads, interrupted writes, and corrupt files.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::bail;
use kern_run::dump::artifact::{Artifact, Writer};
use serde_json::{json, Value};

struct Temp(PathBuf);

impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        loop {
            let path = std::env::temp_dir().join(format!(
                "kern-dump-artifact-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => panic!("creating test directory: {e}"),
            }
        }
    }

    fn dump(&self) -> PathBuf {
        self.0.join("dump")
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn completed(dir: &Path) -> Artifact {
    let mut writer = Writer::create(dir, "{\"model\":\"fixture\"}\n", 3).unwrap();
    let blob = writer.blob(3, |at, len| Ok(b"abc"[at as usize..at as usize + len].to_vec())).unwrap();
    assert_eq!(blob.sha256, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    writer.finish(json!({"blob": blob.id})).unwrap()
}

fn edit_index(dir: &Path, edit: impl FnOnce(&mut Value)) {
    let path = dir.join("dump.json");
    let mut value: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    edit(&mut value);
    fs::write(path, serde_json::to_vec(&value).unwrap()).unwrap();
}

#[test]
fn an_artifact_preserves_manifest_bytes_payload_bits_and_metadata() {
    let temp = Temp::new();
    let dir = temp.dump();
    fs::create_dir(&dir).unwrap();
    let manifest = "{\n  \"model\": \"typed-and-opaque\"\n}\n";
    let bits = [0x7fc12345u32, 0xffc54321, 0x7f800000, 0xff800000, 0x80000000];
    let payload: Vec<u8> = bits.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut writer = Writer::create(&dir, manifest, payload.len() as u64).unwrap();
    let blob =
        writer.blob(payload.len() as u64, |at, len| Ok(payload[at as usize..at as usize + len].to_vec())).unwrap();
    let metadata = json!({"view": {"blob": blob.id, "offset": 8}, "opaque": null});
    assert!(!dir.join("dump.json").exists());
    writer.finish(metadata.clone()).unwrap();
    let restored = Artifact::read(&dir).unwrap();
    assert_eq!(
        (restored.metadata, fs::read(dir.join("manifest.json")).unwrap(), fs::read(dir.join(&blob.path)).unwrap()),
        (metadata, manifest.as_bytes().to_vec(), payload)
    );
    assert!(!dir.join("dump.json.partial").exists());
}

#[test]
fn streamed_blobs_cover_every_byte_once_without_reads_over_four_mib() {
    const CHUNK: usize = 4 << 20;
    let temp = Temp::new();
    let dir = temp.dump();
    let bytes = CHUNK as u64 + 17;
    let mut writer = Writer::create(&dir, "{}", bytes).unwrap();
    let mut reads = Vec::new();
    let blob = writer
        .blob(bytes, |at, len| {
            reads.push((at, len));
            Ok((at..at + len as u64).map(|i| (i % 251) as u8).collect())
        })
        .unwrap();
    let empty = writer.blob(0, |_, _| panic!("an empty blob has no reads")).unwrap();
    writer.finish(json!({})).unwrap();
    let mut cursor = 0;
    for (at, len) in reads {
        assert_eq!(at, cursor);
        assert!((1..=CHUNK).contains(&len));
        cursor += len as u64;
    }
    assert_eq!(cursor, bytes);
    let data = fs::read(dir.join(&blob.path)).unwrap();
    assert_eq!((data.len() as u64, data.iter().enumerate().all(|(i, &v)| v == (i % 251) as u8)), (bytes, true));
    assert_eq!(
        (fs::read(dir.join(&empty.path)).unwrap(), empty.sha256.as_str()),
        (vec![], "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
    );
    Artifact::read(&dir).unwrap();
}

#[test]
fn failed_reads_sizes_and_budgets_never_publish_completion() {
    for kind in ["read", "short", "budget", "overflow"] {
        let temp = Temp::new();
        let dir = temp.dump();
        let mut writer = Writer::create(&dir, "{}", if kind == "overflow" { u64::MAX } else { 3 }).unwrap();
        writer.blob(1, |_, _| Ok(vec![7])).unwrap();
        let result = match kind {
            "read" => writer.blob(1, |_, _| bail!("source read failed")),
            "short" => writer.blob(1, |_, _| Ok(vec![])),
            "budget" => writer.blob(3, |_, _| panic!("budget must be checked before reading")),
            "overflow" => writer.blob(u64::MAX, |_, _| panic!("size overflow must be checked before reading")),
            _ => unreachable!(),
        };
        assert!(result.is_err(), "{kind}");
        assert!(writer.blob(0, |_, _| Ok(vec![])).is_err(), "{kind}");
        assert!(writer.finish(json!({})).is_err(), "{kind}");
        assert!(!dir.join("dump.json").exists(), "{kind}");
    }
}

#[test]
fn readers_refuse_incomplete_missing_or_corrupt_artifacts() {
    let cases: &[(&str, fn(&Path))] = &[
        ("incomplete", |d| fs::rename(d.join("dump.json"), d.join("dump.json.partial")).unwrap()),
        ("version", |d| edit_index(d, |v| v["version"] = json!(999))),
        ("manifest", |d| fs::write(d.join("manifest.json"), b"{}").unwrap()),
        ("missing-manifest", |d| fs::remove_file(d.join("manifest.json")).unwrap()),
        ("missing-blob", |d| fs::remove_file(d.join("blobs/000001.bin")).unwrap()),
        ("length", |d| edit_index(d, |v| v["blobs"][0]["bytes"] = json!(2))),
        ("digest", |d| fs::write(d.join("blobs/000001.bin"), b"abd").unwrap()),
        ("duplicate", |d| {
            edit_index(d, |v| {
                let blob = v["blobs"][0].clone();
                v["blobs"].as_array_mut().unwrap().push(blob);
            });
        }),
    ];
    for (kind, corrupt) in cases {
        let temp = Temp::new();
        let dir = temp.dump();
        completed(&dir);
        corrupt(&dir);
        assert!(Artifact::read(&dir).is_err(), "{kind}");
    }
}
