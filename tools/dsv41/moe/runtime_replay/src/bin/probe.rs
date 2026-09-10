//! Single-GPU Runtime integration fixture with state snapshots and host weights.
use kern_runtime::{Capacity, Runtime, Topology};
use sha2::Digest;
use std::{collections::BTreeMap, path::PathBuf};
fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    anyhow::ensure!(args.len() >= 3, "usage: probe CASE CUBINS [--graph]");
    let case = PathBuf::from(&args[1]);
    let io: serde_json::Value = serde_json::from_slice(&std::fs::read(case.join("io.json"))?)?;
    let manifest = kern_manifest::Verified::from_json(&std::fs::read_to_string(case.join("manifest.json"))?)?;
    let kernels = std::env::temp_dir().join(format!("dsv41-probe-{}", std::process::id()));
    std::fs::create_dir_all(&kernels)?;
    for entry in std::fs::read_dir(&args[2])? {
        let p = entry?.path();
        if p.extension().is_some_and(|e| e == "cubin") {
            let b = std::fs::read(&p)?;
            let sha = format!("{:x}", sha2::Sha256::digest(&b));
            std::fs::write(
                kernels.join(format!("{}-{}.cubin", p.file_stem().unwrap().to_string_lossy(), &sha[..12])),
                b,
            )?;
        }
    }
    let vars: BTreeMap<String, u64> = serde_json::from_value(io["vars"].clone())?;
    let mut rt = Runtime::load(
        &manifest,
        &kernels,
        1,
        Some(Capacity { tokens: Some(32768), seqs: vars.get("seqs").copied().unwrap_or(1) }),
        Some(&Topology::default()),
    )?;
    let paths: Vec<_> = io["weights"].as_array().unwrap().iter().map(|v| PathBuf::from(v.as_str().unwrap())).collect();
    let maps = kern_run::map_weights(&paths)?;
    rt.load_weights(&maps.iter().map(|v| &v[..]).collect::<Vec<_>>())?;
    println!("original checkpoint loaded");
    if manifest.programs.contains_key("load") {
        rt.run("load", &vars)?;
    }
    for (name, file) in io["inputs"].as_object().unwrap() {
        rt.write_buffer(name, &std::fs::read(case.join(file.as_str().unwrap()))?)?;
    }
    let program = io.get("program").and_then(|v| v.as_str()).unwrap_or("probe");
    for i in 0..rt.call_count(program)? {
        println!("call {i}");
        rt.run_range(program, &vars, i, i + 1)?;
    }
    if args.iter().any(|a| a == "--graph") {
        rt.capture(program, &vars)?;
        rt.run_captured(program, &vars)?;
    }
    for category in ["outputs", "state_outputs"] {
        if let Some(outputs) = io.get(category).and_then(|v| v.as_object()) {
            for (name, file) in outputs {
                let len = std::fs::metadata(case.join(file.as_str().unwrap()))?.len() as usize;
                let bytes = if category == "outputs" {
                    rt.read_buffer_prefix(name, len)?
                } else {
                    rt.read_state_at(name, 0, len)?
                };
                std::fs::write(case.join(format!("{name}.kern.bin")), bytes)?;
            }
        }
    }
    println!("probe complete");
    Ok(())
}
