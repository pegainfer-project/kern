//! Replay a four-rank fixture with the real Runtime compiler, peer mapper and graphs.
use kern_runtime::{Capacity, PeerHandle, Runtime, Topology};
use sha2::Digest;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::{Arc, Barrier, Mutex},
};
fn stage(source: &Path, dest: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(dest)?;
    for e in std::fs::read_dir(source)? {
        let p = e?.path();
        if p.extension().is_some_and(|e| e == "cubin") {
            let b = std::fs::read(&p)?;
            let sha = format!("{:x}", sha2::Sha256::digest(&b));
            let stem = p.file_stem().unwrap().to_string_lossy();
            std::fs::write(dest.join(format!("{stem}-{}.cubin", &sha[..12])), b)?;
        }
    }
    Ok(())
}
fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    anyhow::ensure!(args.len() >= 3, "usage: dsv41-runtime-replay CASE CUBINS [--graph]");
    let case = PathBuf::from(&args[1]);
    let kernels = std::env::temp_dir().join(format!("dsv41-replay-{}", std::process::id()));
    stage(Path::new(&args[2]), &kernels)?;
    let graph = args.iter().any(|x| x == "--graph");
    let trace = args.iter().any(|x| x == "--trace");
    let handles: Arc<Mutex<Vec<Option<BTreeMap<String, PeerHandle>>>>> = Arc::new(Mutex::new(vec![None; 4]));
    let gate = Arc::new(Barrier::new(4));
    let mut threads = vec![];
    for rank in 0..4 {
        let (case, kernels, handles, gate) = (case.clone(), kernels.clone(), handles.clone(), gate.clone());
        threads.push(std::thread::spawn(move || {
            let run = || -> anyhow::Result<()> {
                let dir = case.join(format!("rank{rank}"));
                let manifest =
                    kern_manifest::Verified::from_json(&std::fs::read_to_string(dir.join("manifest.json"))?)?;
                let io: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("io.json"))?)?;
                let mut rt = Runtime::load(
                    &manifest,
                    &kernels,
                    rank,
                    io.get("capacity_tokens")
                        .and_then(|v| v.as_u64())
                        .map(|tokens| Capacity { tokens: Some(tokens), seqs: 1 }),
                    Some(&Topology::one("ep", rank as u64, 4)),
                )?;
                let vars: BTreeMap<String, u64> =
                    io.get("vars").map(|v| serde_json::from_value(v.clone())).transpose()?.unwrap_or_default();
                if let Some(paths) = io.get("weights").and_then(|v| v.as_array()) {
                    let paths: Vec<PathBuf> = paths.iter().map(|p| PathBuf::from(p.as_str().unwrap())).collect();
                    let maps = kern_run::map_weights(&paths)?;
                    rt.load_weights(&maps.iter().map(|m| &m[..]).collect::<Vec<_>>())?;
                    println!("rank{rank}: checkpoint weights loaded");
                }
                handles.lock().unwrap()[rank] = Some(rt.export_handles()?);
                gate.wait();
                let members: Vec<_> = handles.lock().unwrap().iter().map(|h| h.clone().unwrap()).collect();
                rt.import_peers("ep", &members)?;
                if io.get("weights").is_some() {
                    rt.run("load", &vars)?;
                    println!("rank{rank}: once complete");
                }
                for (name, file) in io["inputs"].as_object().unwrap() {
                    rt.write_buffer(name, &std::fs::read(dir.join(file.as_str().unwrap()))?)?;
                }
                gate.wait();
                let program = io.get("program").and_then(|v| v.as_str()).unwrap_or("probe");
                if trace {
                    for i in 0..rt.call_count(program)? {
                        if rank == 0 {
                            println!("call {i}: starting");
                        }
                        rt.run_range(program, &vars, i, i + 1)?;
                        gate.wait();
                    }
                } else {
                    rt.run(program, &vars)?;
                }
                gate.wait();
                if graph {
                    rt.capture(program, &vars)?;
                    gate.wait();
                    rt.run_captured(program, &vars)?;
                    gate.wait();
                }
                for (name, file) in io["outputs"].as_object().unwrap() {
                    let expected = std::fs::read(dir.join(file.as_str().unwrap()))?;
                    let actual = rt.read_buffer_prefix(name, expected.len())?;
                    std::fs::write(dir.join(format!("{name}.kern.bin")), &actual)?;
                    if let Some(limit) = io.get("bf16_relative_squared_limit").and_then(|v| v.as_f64()) {
                        let width = if io.get("output_dtypes").and_then(|v| v.get(name)).and_then(|v| v.as_str())
                            == Some("f32")
                        {
                            4
                        } else {
                            2
                        };
                        let decode = |b: &[u8]| {
                            if width == 4 {
                                f32::from_le_bytes(b.try_into().unwrap()) as f64
                            } else {
                                f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16) as f64
                            }
                        };
                        let (mut error, mut norm) = (0.0, 0.0);
                        for (a, e) in actual.chunks_exact(width).zip(expected.chunks_exact(width)) {
                            let (a, e) = (decode(a), decode(e));
                            error += (a - e) * (a - e);
                            norm += e * e;
                        }
                        let rel = error / (norm + 1e-30);
                        println!("rank{rank} {name}: relative_squared_error={rel}");
                        anyhow::ensure!(
                            rel.is_finite() && rel < limit,
                            "rank{rank} {name} relative error {rel} exceeds {limit}"
                        );
                    } else {
                        anyhow::ensure!(actual == expected, "rank{rank} {name} differs");
                    }
                }
                println!("rank{rank}: Runtime peer replay passed graph={graph}");
                gate.wait();
                Ok(())
            };
            if let Err(e) = run() {
                eprintln!("rank{rank}: {e:#}");
                std::process::exit(1);
            }
        }));
    }
    for thread in threads {
        thread.join().unwrap();
    }
    Ok(())
}
