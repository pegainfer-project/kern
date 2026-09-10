//! A single canonical greedy history on all EP ranks, with target taps.
//! Uses the serving manifest unchanged and the Runtime's actual page leases.
use anyhow::{Context, ensure};
use kern_manifest::{Protocol, Verified, protocol::Axis, types::Fill};
use kern_runtime::{Capacity, HostWeights, PeerHandle, Runtime, Topology};
use std::{
    collections::BTreeMap,
    io::Write,
    path::PathBuf,
    sync::{Arc, Barrier, Mutex},
};

fn bytes_i32(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn main() -> anyhow::Result<()> {
    let file = std::env::args().nth(1).context("usage: dsv41-ar-trace CONFIG.json")?;
    let cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(file)?)?;
    let manifest =
        Arc::new(Verified::from_json(&std::fs::read_to_string(cfg["manifest"].as_str().context("manifest")?)?)?);
    let kernels = PathBuf::from(cfg["kernels"].as_str().context("kernels")?);
    let output = PathBuf::from(cfg["output"].as_str().context("output")?);
    std::fs::create_dir_all(&output)?;
    let prompt: Vec<i64> = serde_json::from_value(cfg["prompt_ids"].clone())?;
    let count = cfg["generation_tokens"].as_u64().unwrap_or(64) as usize;
    let teacher = Arc::new(serde_json::from_value::<Option<Vec<i64>>>(cfg["teacher_tokens"].clone())?);
    ensure!(teacher.as_ref().as_ref().is_none_or(|v| v.len() == count), "teacher_tokens length must equal generation_tokens");
    ensure!(prompt.len() >= 2 && count >= 6, "need at least 2 prompt and 6 generated tokens");
    let capacity = cfg["capacity_tokens"].as_u64().unwrap_or(32768);
    let draft = cfg["draft_proposals"].as_bool().unwrap_or(false);
    let save_logits = cfg["target_logits"].as_bool().unwrap_or(false);
    let draft_end = if draft {
        Some(manifest.programs.get("round").context("round program")?.calls.iter()
            .position(|c| c.label.as_deref() == Some("splice_verify"))
            .context("round must label its target boundary splice_verify")?)
    } else { None };
    let weights: Vec<Vec<PathBuf>> = serde_json::from_value(cfg["weights"].clone())?;
    ensure!(weights.len() == 4, "weights must contain one file/directory list per EP rank");
    let host = Arc::new(HostWeights::new());
    let handles = Arc::new(Mutex::new(vec![None::<BTreeMap<String, PeerHandle>>; 4]));
    let next = Arc::new(Mutex::new([0i64; 4]));
    let gate = Arc::new(Barrier::new(4));
    let mut threads = Vec::new();
    for (rank, paths) in weights.into_iter().enumerate() {
        let teacher = teacher.clone();
        let (m, kernels, output, prompt, host, handles, gate, next) = (
            manifest.clone(),
            kernels.clone(),
            output.clone(),
            prompt.clone(),
            host.clone(),
            handles.clone(),
            gate.clone(),
            next.clone(),
        );
        threads.push(std::thread::spawn(move || {
            let work = || -> anyhow::Result<()> {
                let protocol = Protocol::check(&m)?;
                ensure!(protocol.tray.is_none(), "this trace harness expects DP4 EP4 local rows");
                let mut rt = Runtime::load_with_host_weights(&m, &kernels, rank,
                    Some(Capacity { tokens: Some(capacity), seqs: 1 }),
                    Some(&Topology::one("ep", rank as u64, 4)), &host)?;
                let maps = kern_run::map_weights(&paths)?;
                rt.load_weights(&maps.iter().map(|x| &x[..]).collect::<Vec<_>>())?;
                drop(maps);
                handles.lock().unwrap()[rank] = Some(rt.export_handles()?);
                gate.wait();
                rt.import_peers("ep", &handles.lock().unwrap().iter().map(|x| x.clone().unwrap()).collect::<Vec<_>>())?;
                for once in &protocol.once { rt.run(once, &protocol.vars(1, 1, 1))?; }
                let pad = rt.lease(1)?;
                let lease = rt.lease(prompt.len() + count + 6)?;
                let mut ids = prompt.clone();
                let mut actual_argmax = Vec::new();
                let mut proposals = Vec::new();
                let mut taps = if rank == 0 { Some(std::fs::File::create(output.join("taps.bf16"))?) } else { None };
                let mut logits_file = if rank == 0 && save_logits { Some(std::fs::File::create(output.join("target_logits.f32"))?) } else { None };
                let mut processed = 0;
                while ids.len() < prompt.len() + count {
                    let rows = if processed < prompt.len() { (prompt.len() - processed).min(protocol.rows.max as usize) } else { 1 };
                    let prefill = processed < prompt.len();
                    let program = if prefill { "prefill" } else { "decode_batch" };
                    let vars = protocol.vars(1, rows as u64, rows as u64);
                    let chunk = &ids[processed..processed + rows];
                    for f in &protocol.fills {
                        let values: Vec<i64> = match f.fill {
                            Fill::Token => match f.axis { Axis::Groups => vec![chunk[0]], _ => chunk.to_vec() },
                            Fill::Position => (processed..processed + rows).map(|x| x as i64).collect(),
                            Fill::Valid => vec![1; rows],
                            Fill::Slot => lease.slots(processed..processed + rows),
                            Fill::SeqLen => vec![(processed + rows) as i64],
                            Fill::CuSeqlens => vec![0, rows as i64],
                            Fill::SpanAt => vec![0],
                            Fill::Blocks => anyhow::bail!("unexpected collective row blocks in DP manifest"),
                            Fill::Tokens | Fill::Count | Fill::Error => continue,
                        };
                        rt.write_input_at(&f.name, &f.encode(&values), &vars)?;
                    }
                    for t in &protocol.page_tables {
                        let mut values = Vec::new(); lease.extend_row(&t.name, &mut values)?;
                        rt.write_input_at(&t.name, &bytes_i32(&values), &vars)?;
                    }
                    for t in &protocol.line_tables {
                        ensure!(t.axis == Axis::Groups, "unexpected cross-rank line table");
                        let columns = protocol.groups.max as usize;
                        let mut values = vec![0; t.lines * columns * t.width];
                        for r in 0..t.lines { for c in 0..columns {
                            values[(r * columns + c) * t.width] = if c == 0 { lease.seq_line(&t.name, r)? } else { pad.seq_line(&t.name, r)? };
                        }}
                        rt.write_input(&t.name, &bytes_i32(&values))?;
                    }
                    gate.wait();
                    rt.run(program, &vars).with_context(|| format!("{program} at {processed}"))?;
                    let token = i64::from_le_bytes(rt.read_buffer_prefix("next_token", 8)?.try_into().unwrap());
                    next.lock().unwrap()[rank] = token;
                    gate.wait();
                    ensure!(next.lock().unwrap().iter().all(|x| *x == token), "ranks disagree at {processed}");
                    if let Some(file) = &mut taps {
                        let name = if prefill { "prefill.draft_context.taps" } else { "decode.draft_context.taps" };
                        file.write_all(&rt.read_buffer_prefix(name, rows * 15360 * 2)?)?;
                    }
                    processed += rows;
                    if processed >= prompt.len() {
                        actual_argmax.push(token);
                        if let Some(file) = &mut logits_file {
                            let name = if prefill { "prefill.target_head.logits" } else { "decode.target_head.logits" };
                            file.write_all(&rt.read_buffer_prefix(name, 129280 * 4)?)?;
                        }
                    }
                    let token = if processed >= prompt.len() {
                        teacher.as_ref().as_ref().map_or(token, |v| v[ids.len()-prompt.len()])
                    } else { token };
                    if processed >= prompt.len() { ids.push(token); }
                    if let Some(end) = draft_end.filter(|_| processed >= prompt.len()) {
                        // Stage exactly the serving six-row round metadata, run
                        // only the draft prefix, then continue canonical AR.
                        // Draft writes future draft-cache slots only; the next
                        // target call publishes its true context at those slots.
                        let dv = protocol.vars(1, 6, 6);
                        for f in &protocol.fills {
                            let values: Vec<i64> = match f.fill {
                                Fill::Token => match f.axis { Axis::Groups => vec![token], _ => vec![token; 6] },
                                Fill::Position => (processed..processed+6).map(|x| x as i64).collect(),
                                Fill::Valid => vec![1; 6],
                                Fill::Slot => lease.slots(processed..processed+6),
                                Fill::SeqLen => vec![(processed+6) as i64],
                                Fill::CuSeqlens => vec![0, 6],
                                Fill::SpanAt => vec![0],
                                Fill::Blocks => anyhow::bail!("unexpected blocks"),
                                Fill::Tokens | Fill::Count | Fill::Error => continue,
                            };
                            rt.write_input_at(&f.name, &f.encode(&values), &dv)?;
                        }
                        gate.wait();
                        rt.run_range("round", &dv, 0, end)?;
                        if rank == 0 {
                            let raw = rt.read_buffer_prefix("draft_tokens", 40)?;
                            let pred: Vec<i64> = raw.chunks_exact(8).map(|v| i64::from_le_bytes(v.try_into().unwrap())).collect();
                            proposals.push(serde_json::json!({"anchor_position": processed, "anchor_id": token, "draft_ids": pred}));
                        }
                        gate.wait();
                    }
                    if rank == 0 { eprintln!("processed={processed} generated={} next={token}", ids.len() - prompt.len()); }
                    gate.wait();
                }
                if rank == 0 {
                    std::fs::write(output.join("tokens.json"), serde_json::to_vec_pretty(&ids)?)?;
                    std::fs::write(output.join("actual_argmax.json"), serde_json::to_vec_pretty(&actual_argmax)?)?;
                    if draft { std::fs::write(output.join("kern_proposals.json"), serde_json::to_vec_pretty(&proposals)?)?; }
                    std::fs::write(output.join("trace.json"), serde_json::to_vec_pretty(&serde_json::json!({
                        "format": 1, "prompt_length": prompt.len(), "tap_rows": processed,
                        "tap_width": 15360, "tap_dtype": "bf16", "tokens": "tokens.json", "taps": "taps.bf16",
                        "generation_tokens": count, "target": "kern target; same prompt on four DP/EP ranks",
                        "teacher_forced": teacher.is_some(), "actual_argmax": "actual_argmax.json",
                        "logits": if save_logits { Some("target_logits.f32") } else { None },
                        "logits_rows": if save_logits { count } else { 0 }, "logits_width": 129280,
                        "logits_first_input_position": prompt.len()-1,
                        "stop_at_eos": false, "note": "Fixed-length diagnostic trace; evaluator must exclude EOS boundaries."
                    }))?)?;
                }
                gate.wait();
                Ok(())
            };
            if let Err(e) = work() { eprintln!("rank{rank}: {e:#}"); std::process::exit(1); }
        }));
    }
    for t in threads {
        t.join().unwrap();
    }
    Ok(())
}
