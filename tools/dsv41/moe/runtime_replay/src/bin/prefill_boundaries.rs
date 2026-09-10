//! Independent prompt prefill with per-layer numerical boundary snapshots.
use anyhow::{Context, Result, ensure};
use kern_runtime::{Capacity, HostWeights, Lease, PeerHandle, Runtime, Topology};
use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::{Arc, Barrier, Mutex},
};
fn bytes32(v: impl IntoIterator<Item = i32>) -> Vec<u8> {
    v.into_iter().flat_map(i32::to_le_bytes).collect()
}
fn bytes64(v: impl IntoIterator<Item = i64>) -> Vec<u8> {
    v.into_iter().flat_map(i64::to_le_bytes).collect()
}
fn stage(rt: &mut Runtime, leases: &[&Lease], positions: &[usize], ids: &[Vec<i64>]) -> Result<BTreeMap<String, u64>> {
    let seqs = leases.len();
    let width = ids[0].len();
    ensure!(ids.len() == seqs && positions.len() == seqs && ids.iter().all(|x| x.len() == width));
    let vars = BTreeMap::from([("tokens".into(), (seqs * width) as u64), ("seqs".into(), seqs as u64)]);
    for (name, data) in [
        ("input_ids", bytes64(ids.iter().flatten().copied())),
        ("anchor_token", bytes64(ids.iter().map(|v| v[0]))),
        ("positions", bytes32(positions.iter().flat_map(|p| (0..width).map(move |j| (p + j) as i32)))),
        ("valid", bytes32(vec![1; seqs * width])),
        ("slot_mapping", bytes64(leases.iter().zip(positions).flat_map(|(l, p)| l.slots(*p..p + width)))),
        ("seq_lens", bytes32(positions.iter().map(|p| (p + width) as i32))),
        ("cu_seqlens", bytes32((0..=seqs).map(|i| (i * width) as i32))),
    ] {
        rt.write_input_at(name, &data, &vars)?;
    }
    let tables: Vec<_> = rt.page_tables().map(str::to_owned).collect();
    for name in tables {
        let mut table = vec![];
        for l in leases {
            l.extend_row(&name, &mut table)?;
        }
        rt.write_input_at(&name, &bytes32(table), &vars)?;
    }
    // DSV4 ratio-2 short compressor state: one line per request.
    for source in [2, 8, 14] {
        let name = format!("compressor.{source}.lines");
        let lines = leases.iter().map(|l| l.seq_line(&name, 0)).collect::<std::result::Result<Vec<_>, _>>()?;
        rt.write_input_at(&name, &bytes32(lines), &vars)?;
    }
    Ok(vars)
}
fn main()->Result<()> {
 let cfg:serde_json::Value=serde_json::from_slice(&std::fs::read(std::env::args().nth(1).context("usage: prefill_boundaries CONFIG")?)?)?;
 let raw:serde_json::Value=serde_json::from_slice(&std::fs::read(cfg["manifest"].as_str().context("manifest")?)?)?;
 let ids:Vec<i64>=serde_json::from_value(cfg["prompt_ids"].clone())?;
 let out=PathBuf::from(cfg["output"].as_str().context("output")?);std::fs::create_dir_all(&out)?;
 let host=Arc::new(HostWeights::new());let gate=Arc::new(Barrier::new(4));
 let handles:Arc<Mutex<Vec<Option<BTreeMap<String,PeerHandle>>>>>=Arc::new(Mutex::new(vec![None;4]));let mut jobs=vec![];
 for rank in 0..4 {
  let(cfg,raw,ids,out,host,gate,handles)=(cfg.clone(),raw.clone(),ids.clone(),out.clone(),host.clone(),gate.clone(),handles.clone());
  jobs.push(std::thread::spawn(move|| {
   let run=||->Result<()> {
    let m=kern_manifest::Verified::from_json(&serde_json::to_string(&raw)?)?;
    let mut rt=Runtime::load_with_host_weights(&m,&PathBuf::from(cfg["kernels"].as_str().unwrap()),rank,Some(Capacity{tokens:Some(cfg["capacity_tokens"].as_u64().unwrap_or(32768)),seqs:1}),Some(&Topology::one("ep",rank as u64,4)),&host)?;
    let paths:Vec<PathBuf>=serde_json::from_value(cfg["weights"][rank].clone())?;let maps=kern_run::map_weights(&paths)?;
    rt.load_weights(&maps.iter().map(|m|&m[..]).collect::<Vec<_>>())?;drop(maps);
    handles.lock().unwrap()[rank]=Some(rt.export_handles()?);gate.wait();
    let members=handles.lock().unwrap().iter().map(|h|h.clone().unwrap()).collect::<Vec<_>>();rt.import_peers("ep",&members)?;
    rt.run("load",&BTreeMap::from([("tokens".into(),ids.len() as u64),("seqs".into(),1)]))?;gate.wait();
    let lease=rt.lease(ids.len()+1)?;let vars=stage(&mut rt,&[&lease],&[0],std::slice::from_ref(&ids))?;
    let calls=raw["programs"]["prefill"]["calls"].as_array().unwrap();let mut cursor=0;let mut index=vec![];
    for (i,call) in calls.iter().enumerate() {
     let label=call["label"].as_str().unwrap();let words=label.split('.').collect::<Vec<_>>();
     let layer=words.iter().position(|x|*x=="layers").and_then(|i|words.get(i+1)).and_then(|s|s.parse::<usize>().ok());
     let mut outputs:Vec<(String,String)>=vec![];
     if label=="prefill.target.embed" {outputs.push(("embedding".into(),"prefill.target.embedding".into()));}
     if let Some(l)=layer {
      if label.ends_with(".attn.mhc") {outputs.push((format!("layer.{l:02}.attn_norm"),"prefill.target.normalized".into()));}
      if label.ends_with(".ffn.mhc") {outputs.push((format!("layer.{l:02}.ffn_norm"),"prefill.target.normalized".into()));}
      if label.ends_with(".wo_b.gemm") {outputs.push((format!("layer.{l:02}.attn"),"prefill.target.attention_result".into()));}
      if label.ends_with(".experts") {
       outputs.push((format!("layer.{l:02}.ffn"),"prefill.target.ffn_result".into()));
       for field in ["residual2","post2","comb2","pre2"] {outputs.push((format!("layer.{l:02}.{field}"),format!("prefill.target.{field}")));}
      }
     }
     if outputs.is_empty(){continue;}
     gate.wait();rt.run_range("prefill",&vars,cursor,i+1)?;gate.wait();cursor=i+1;
     if rank==0 {
      for (name,buffer) in outputs {
       let spec=&raw["buffers"][&buffer];let shape=spec["shape"].as_array().unwrap();let cols=shape[1..].iter().map(|v|v.as_u64().unwrap() as usize).product::<usize>();
       let dtype=spec["dtype"].as_str().unwrap();let bytes=match dtype{"bf16"=>2,"f32"=>4,_=>anyhow::bail!("unsupported boundary dtype")};
       let file=format!("{name}.{dtype}");std::fs::write(out.join(&file),rt.read_buffer_prefix(&buffer,ids.len()*cols*bytes)?)?;
       index.push(serde_json::json!({"name":name,"buffer":buffer,"dtype":dtype,"shape":[ids.len(),cols],"file":file,"call":i,"label":label}));
      }
     }
    }
    gate.wait();if cursor<calls.len(){rt.run_range("prefill",&vars,cursor,calls.len())?;}gate.wait();
    if rank==0 {
     let vocab=raw["buffers"]["head.weight"]["shape"][0].as_u64().unwrap() as usize;
     std::fs::write(out.join("logits.f32"),rt.read_buffer_prefix("prefill.target_head.logits",vocab*4)?)?;
     std::fs::write(out.join("index.json"),serde_json::to_vec_pretty(&index)?)?;
     std::fs::write(out.join("input_ids.json"),serde_json::to_vec_pretty(&ids)?)?;
     println!("prefill {} rows, {} boundaries complete",ids.len(),index.len());
    }
    gate.wait();Ok(())
   };if let Err(e)=run(){eprintln!("rank{rank}: {e:#}");std::process::exit(1);}
  }));
 }
 for j in jobs {j.join().unwrap();}Ok(())
}
