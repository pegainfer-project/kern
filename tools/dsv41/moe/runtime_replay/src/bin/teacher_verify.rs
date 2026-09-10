//! Same-history plain vs six-row verification, using independent forked leases.
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
fn boundary_replay(rt:&mut Runtime, raw:&serde_json::Value, program:&str, vars:&BTreeMap<String,u64>, begin:usize,end:usize, mode:&str,row:usize,dir:&std::path::Path,rank:usize,barrier:&Barrier,cachepage:usize,prefix:usize,patch:Option<&str>)->Result<()> {
    let calls=raw["programs"][program]["calls"].as_array().unwrap();let mut cursor=begin;let mut records=vec![];let mut layer20=false;
    if rank==0 {std::fs::create_dir_all(dir)?;}
    for i in begin..end {
        let label=calls[i]["label"].as_str().unwrap();
        if label.ends_with("layers.20.attn.mhc") {layer20=true;}
        if label.ends_with("layers.20.ffn.mhc") {layer20=false;}
        let suffix=if layer20 && label.ends_with(".attention") {Some("target.attention.attention_raw")}
            else if layer20 && label.ends_with("layers.20.wq_a.gemm") {Some("target.attention.qr_raw")}
            else if layer20 && label.ends_with("layers.20.wq_b.gemm") {Some("target.attention.q")}
            else if layer20 && label.ends_with("layers.20.wkv.gemm") {Some("target.attention.kv_raw")}
            else if layer20 && label.ends_with("layers.20.q_rope") {Some("target.attention.q_rotated")}
            else if layer20 && label.ends_with("layers.20.kv_rope") {Some("target.attention.kv_rotated")}
            else if layer20 && label.ends_with("compress20.wkv") {Some("compress20.wkv")}
            else if layer20 && label.ends_with("compress20.norm") {Some("compress20.latent")}
            else if layer20 && label.ends_with("compress20.index_k") {Some("compress20.key_raw")}
            else if layer20 && label.ends_with("compress20.key_quant") {Some("compress20.key_dequant")}
            else if layer20 && label.ends_with("compressed20.write") {Some("compress20.kv_rope")}
            else if layer20 && label.ends_with("index20.metadata") {Some("c1_end")}
            else if layer20 && label.ends_with("index20.query.gemm") {Some("index20.query")}
            else if layer20 && label.ends_with("index20.quant") {Some("index20.dequant")}
            else if layer20 && label.ends_with("index20.scale_weights") {Some("index20.weights_f32")}
            else if layer20 && label.ends_with("index20.score") {Some("index20.scores")}
            else if layer20 && label.ends_with("index20.select_tokens") {Some("index20.logical")}
            else if layer20 && label.ends_with("index20.block_scores") {Some("index20.block_scores")}
            else if layer20 && label.ends_with("index20.filter_blocks") {Some("index20.candidates")}
            else if layer20 && label.ends_with("compressed.mask") {Some("index20.physical")}
            else if label.ends_with(".mhc") {Some("target.normalized")}
            else if label.ends_with(".wo_b.gemm") {Some("target.attention_result")}
            else if label.ends_with(".experts") {Some("target.ffn_result")}
            else if label.ends_with(".inject") {Some("target.engram_residual")}
            else if label.ends_with(".lookup") {Some("engram.embedding")}
            else if label.ends_with(".hash") {Some("hashes")} else {None};
        if let Some(suffix)=suffix {
            barrier.wait();rt.run_range(program,vars,cursor,i+1)?;barrier.wait();cursor=i+1;
            if rank==0 {
                if label.ends_with("compressed20.write") {
                    for (state,bpt) in [("compressed.20",288),("index_k.20",68)] {
                        let unit=rt.page() as usize;let packed=if bpt==288 {256}else{64};
                        let mut data=rt.read_state_at(state,cachepage*unit*bpt,prefix*packed)?;
                        data.extend(rt.read_state_at(state,cachepage*unit*bpt+unit*packed,prefix*(bpt-packed))?);
                        std::fs::write(dir.join(format!("{state}.cache.bin")),data)?;
                    }
                }
                let name=format!("{mode}.{suffix}");let spec=&raw["buffers"][&name];
                let width=spec["shape"].as_array().context("boundary shape")?[1..].iter().map(|v|v.as_u64().unwrap() as usize).product::<usize>();
                let dtype=spec["dtype"].as_str().unwrap();let bytes=match dtype {"bf16"=>2,"f32"|"i32"=>4,"i64"=>8,_=>anyhow::bail!("boundary dtype {dtype}")};
                let data=rt.read_buffer_prefix(&name,(row+1)*width*bytes)?;
                let key=label.trim_start_matches("verify.").trim_start_matches("decode.");
                let file=format!("{key}.bin");std::fs::write(dir.join(&file),&data[row*width*bytes..])?;
                records.push(serde_json::json!({"label":key,"buffer":suffix,"dtype":dtype,"width":width,"file":file,"call":i}));
            }
            if mode=="verify" && label.ends_with("compress20.wkv") {
                if let Some(file)=patch {
                    let source=std::fs::read(file)?;ensure!(source.len()==1024);
                    let name="verify.compress20.wkv";let mut data=rt.read_buffer_prefix(name,(row+1)*1024)?;
                    let at=row*1024+296*2;data[at..at+2].copy_from_slice(&source[296*2..297*2]);
                    rt.write_buffer(name,&data)?;
                }
            }
        }
    }
    if cursor<end {barrier.wait();rt.run_range(program,vars,cursor,end)?;barrier.wait();}
    if rank==0 {std::fs::write(dir.join("index.json"),serde_json::to_vec_pretty(&records)?)?;}
    Ok(())
}
fn main() -> Result<()> {
    let cfg: serde_json::Value =
        serde_json::from_slice(&std::fs::read(std::env::args().nth(1).context("usage: teacher_verify CONFIG.json")?)?)?;
    let out = PathBuf::from(cfg["output"].as_str().context("output")?);
    std::fs::create_dir_all(&out)?;
    let barrier = Arc::new(Barrier::new(4));
    let handles: Arc<Mutex<Vec<Option<BTreeMap<String, PeerHandle>>>>> = Arc::new(Mutex::new(vec![None; 4]));
    let host = Arc::new(HostWeights::new());
    let mut jobs = vec![];
    for rank in 0..4 {
        let (cfg, out, barrier, handles, host) =
            (cfg.clone(), out.clone(), barrier.clone(), handles.clone(), host.clone());
        jobs.push(std::thread::spawn(move || {
      let run=||->Result<()> {
        let source=std::fs::read_to_string(cfg["manifest"].as_str().context("manifest")?)?;
        let raw:serde_json::Value=serde_json::from_str(&source)?;
        let manifest=kern_manifest::Verified::from_json(&source)?;
        let histories:Vec<Vec<i64>>=serde_json::from_value(cfg["histories"][rank].clone())?;
        let teacher:Vec<Vec<i64>>=serde_json::from_value(cfg["teacher_ids"][rank].clone())?;
        let b=histories.len();ensure!(b>0 && teacher.len()==b && teacher.iter().all(|t|t.len()==6));
        ensure!(histories.iter().all(|h|!h.is_empty()));
        let mut rt=Runtime::load_with_host_weights(&manifest,&PathBuf::from(cfg["kernels"].as_str().context("kernels")?),rank,
            Some(Capacity{tokens:Some(cfg["capacity_tokens"].as_u64().unwrap_or(32768)),seqs:(2*b+1) as u64}),Some(&Topology::one("ep",rank as u64,4)),&host)?;
        let paths:Vec<PathBuf>=serde_json::from_value(cfg["weights"][rank].clone())?;
        let maps=kern_run::map_weights(&paths)?;rt.load_weights(&maps.iter().map(|m|&m[..]).collect::<Vec<_>>())?;drop(maps);
        handles.lock().unwrap()[rank]=Some(rt.export_handles()?);barrier.wait();
        let members=handles.lock().unwrap().iter().map(|h|h.clone().unwrap()).collect::<Vec<_>>();rt.import_peers("ep",&members)?;
        rt.run("load",&BTreeMap::from([("tokens".into(),1),("seqs".into(),1)]))?;barrier.wait();
        let mut plain=vec![];let mut verify=vec![];
        // Each history is prefilled separately on all four ranks. All ranks must
        // have equal history lengths at a given index to keep EP geometry equal.
        for (i,history) in histories.iter().enumerate() {
            let mut lease=rt.lease(history.len()+7)?;
            let vars=stage(&mut rt,&[&lease],&[0],std::slice::from_ref(history))?;
            barrier.wait();rt.run("prefill",&vars)?;barrier.wait();
            let twin=rt.fork(&mut lease,history.len(),history.len()+7)?;
            plain.push(lease);verify.push(twin);
            if rank==0 {println!("prefilled and forked sequence {i}");}
        }
        let dir=out.join(format!("rank{rank}"));std::fs::create_dir_all(&dir)?;
        let vocab=raw["buffers"]["head.weight"]["shape"][0].as_u64().context("vocab")? as usize;
        let boundary=cfg["boundaries"].as_bool().unwrap_or(false);
        if boundary && rank==0 {
            let (p,v)=(&plain[6],&verify[6]);let unit=rt.page() as usize;let mut states=vec![];
            for (name,st) in raw["states"].as_object().unwrap() {
                let (a,z)=if let Some(bytes)=st["bytes_per_token"].as_u64() {
                    let size=bytes as usize*unit;
                    (p.page_ids().iter().map(|i|rt.read_state_at(name,*i as usize*size,size)).collect::<std::result::Result<Vec<_>,_>>()?.concat(),
                     v.page_ids().iter().map(|i|rt.read_state_at(name,*i as usize*size,size)).collect::<std::result::Result<Vec<_>,_>>()?.concat())
                } else if let Some(bytes)=st["bytes_per_seq"].as_u64() {
                    (rt.read_state_at(name,p.seq_slot().unwrap() as usize*bytes as usize,bytes as usize)?,rt.read_state_at(name,v.seq_slot().unwrap() as usize*bytes as usize,bytes as usize)?)
                } else {continue};
                states.push(serde_json::json!({"state":name,"bytes":a.len(),"equal":a==z}));
            }
            std::fs::write(dir.join("fork_states.json"),serde_json::to_vec_pretty(&serde_json::json!({"plain_pages":p.page_ids(),"verify_pages":v.page_ids(),"plain_slot":p.seq_slot(),"verify_slot":v.seq_slot(),"states":states}))?)?;
        }
        let mut reference=vec![0u8;b*6*vocab*4];
        for step in 0..6 {
            let ids=teacher.iter().map(|t|vec![t[step]]).collect::<Vec<_>>();let pos=histories.iter().map(|h|h.len()+step).collect::<Vec<_>>();
            let vars=stage(&mut rt,&plain.iter().collect::<Vec<_>>(),&pos,&ids)?;
            if boundary && step==0 {boundary_replay(&mut rt,&raw,"decode_batch",&vars,0,raw["programs"]["decode_batch"]["calls"].as_array().unwrap().len(),"decode",6,&dir.join("plain-boundary"),rank,&barrier,plain[6].page_ids()[0] as usize,histories[6].len()+1,cfg["patch_compressor_from"].as_str())?;}
            else {barrier.wait();rt.run("decode_batch",&vars)?;barrier.wait();}
            let logits=rt.read_buffer_prefix("decode.target_head.logits",b*vocab*4)?;
            for row in 0..b {reference[(row*6+step)*vocab*4..(row*6+step+1)*vocab*4].copy_from_slice(&logits[row*vocab*4..(row+1)*vocab*4]);}
            if rank==0 {println!("teacher plain step {step}");}
        }
        std::fs::write(dir.join("plain.f32"),&reference)?;
        let pos=histories.iter().map(Vec::len).collect::<Vec<_>>();
        let vars=stage(&mut rt,&verify.iter().collect::<Vec<_>>(),&pos,&teacher)?;
        rt.write_buffer("verify_ids",&bytes64(teacher.iter().flatten().copied()))?;
        let calls=raw["programs"]["round"]["calls"].as_array().context("round calls")?;
        let begin=calls.iter().position(|c|c["label"].as_str().is_some_and(|l|l.starts_with("verify."))).context("verify start")?;
        let end=calls.iter().position(|c|c["label"]=="count").context("count boundary")?;
        if boundary {boundary_replay(&mut rt,&raw,"round",&vars,begin,end,"verify",36,&dir.join("verify-boundary"),rank,&barrier,verify[6].page_ids()[0] as usize,histories[6].len()+1,cfg["patch_compressor_from"].as_str())?;}
        else {barrier.wait();rt.run_range("round",&vars,begin,end)?;barrier.wait();}
        std::fs::write(dir.join("verify.f32"),rt.read_buffer_prefix("verify.target_head.logits",b*6*vocab*4)?)?;
        // Commit only an accepted prefix, then replace the rejected suffix with
        // the next teacher token. Compare with the already saved plain row.
        let accepted=cfg["accepted"].as_u64().unwrap_or(3) as usize;
        ensure!((1..6).contains(&accepted), "accepted must be 1..5 for suffix replacement");
        rt.write_buffer("nacc",&bytes32(vec![accepted as i32;b]))?;
        barrier.wait();rt.run_range("round",&vars,end+1,calls.len())?;barrier.wait();
        let next_ids=teacher.iter().map(|t|vec![t[accepted]]).collect::<Vec<_>>();
        let next_pos=histories.iter().map(|h|h.len()+accepted).collect::<Vec<_>>();
        let next_vars=stage(&mut rt,&verify.iter().collect::<Vec<_>>(),&next_pos,&next_ids)?;
        barrier.wait();rt.run("decode_batch",&next_vars)?;barrier.wait();
        std::fs::write(dir.join("after_commit.f32"),rt.read_buffer_prefix("decode.target_head.logits",b*vocab*4)?)?;
        std::fs::write(dir.join("metadata.json"),serde_json::to_vec_pretty(&serde_json::json!({"rank":rank,"seqs":b,"width":6,"vocab":vocab,"positions":pos,"histories":histories,"teacher_ids":teacher,"verify_call_range":[begin,end],"accepted":accepted}))?)?;
        println!("rank {rank}: teacher-forced outputs dumped");barrier.wait();Ok(())
      };if let Err(e)=run(){eprintln!("rank {rank}: {e:#}");std::process::exit(1);}
    }));
    }
    for j in jobs {
        j.join().unwrap();
    }
    Ok(())
}
