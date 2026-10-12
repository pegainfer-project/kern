//! The push gate: one runtime pushes pages and slots of its pooled states into
//! other runtimes' states through mirrors of their chunks, and every byte
//! lands where the destination's lease says. Three runtimes in one
//! process on one GPU (a chunk's handle maps within a process as across
//! processes): A, the source; B, a whole destination; C, a destination
//! that holds one tensor-parallel member's shard of every slot.
//!
//!   cargo run --release -p kern-runtime --example push_states -- \
//!       --manifest target/toy/toy-stateful.json --kernels target/toy/kernels [--gpu 0] [--of 4]
//!
//! Pages and the slot go A to B, slot parts A to C (each member's part
//! into its own slot of C), then B remaps a page chunk into a slot, A
//! re-places its mirror of B and pushes into the new slot, and B pushes
//! that slot back into A, which reads it: a push into a remapped object
//! lands. No weights are loaded: the states are all this touches.
//! `KERN_NO_FABRIC=1` runs it over local handles instead of fabric ones.
//!
//! `--cross` runs the destination in a second process instead: it leases,
//! writes its pool handle, placement and ids to a directory in their wire
//! forms, and checks the bytes once the source says it has pushed.
//!
//! `--rows` is the gate for rows dealt over a group: see `rows.rs`.

mod rows;

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use kern_manifest::types::Manifest;
use kern_pool::{runs, Denied, Lease, Placement};
use kern_runtime::{Capacity, Error, MappedPool, PoolHandle, Runtime, Split};

/// The byte at `i` of a region whose pattern base is `base`.
fn pattern(base: u64, len: u64) -> Vec<u8> {
    (0..len).map(|i| ((base + i) ^ ((base + i) >> 12) ^ ((base + i) >> 24) ^ (base >> 40)) as u8).collect()
}

/// The same manifest with every per-sequence state `of` times narrower:
/// what a member of a tensor-parallel group that shards it loads.
fn shard(m: &Manifest, of: u64) -> Manifest {
    let mut m = m.clone();
    let per_seq: Vec<String> = m.states.iter().filter(|(_, s)| s.is_per_seq()).map(|(n, _)| n.clone()).collect();
    for n in &per_seq {
        m.states.get_mut(n).unwrap().bytes_per_seq /= of;
    }
    for b in m.buffers.values_mut() {
        if let Some(d) = b.domain.as_mut().filter(|d| d.index_into.as_ref().is_some_and(|t| per_seq.contains(t))) {
            d.stride /= of;
        }
    }
    m
}

fn lease_slot(rt: &mut Runtime) -> Result<Lease, Error> {
    loop {
        match rt.lease_slot() {
            Err(Error::Denied(Denied::Remapping)) => rt.settle()?,
            r => return r,
        }
    }
}

fn mirror(from: &Runtime, of: &Runtime, what: &str) -> Result<MappedPool, Error> {
    let h = of.export_pool()?;
    println!("{what}: {} handles", if h.to_bytes().is_some() { "fabric" } else { "local" });
    let mut m = from.map_pool(&h, what)?;
    from.place(&mut m, &of.placement())?;
    Ok(m)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let all: Vec<String> = std::env::args().skip(1).collect();
    if all.iter().any(|a| a == "--rows") {
        return rows::run(all);
    }
    let mut manifest = PathBuf::from("target/toy/toy-stateful.json");
    let mut kernels = PathBuf::from("target/toy/kernels");
    let mut gpu = 0usize;
    let mut of = 4u64;
    let mut cross = false;
    let mut destination: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut v = || args.next().expect("value");
        match a.as_str() {
            "--manifest" => manifest = PathBuf::from(v()),
            "--kernels" => kernels = PathBuf::from(v()),
            "--gpu" => gpu = v().parse()?,
            "--of" => of = v().parse()?,
            "--cross" => cross = true,
            "--destination" => destination = Some(PathBuf::from(v())),
            _ => return Err(format!("unknown arg {a}").into()),
        }
    }
    let whole = Manifest::from_json(&std::fs::read_to_string(&manifest)?)?;
    let m = kern_manifest::verify(whole.clone()).map_err(|e| e.to_string())?;
    let m4 = kern_manifest::verify(shard(&whole, of)).map_err(|e| e.to_string())?;
    let unit = kern_pool::page_unit(&m);
    let paged: Vec<(String, u64)> = m
        .states
        .iter()
        .filter(|(_, s)| s.bytes_per_token > 0)
        .map(|(n, s)| (n.clone(), s.bytes_per_token * unit))
        .collect();
    let (slotted, slot_bytes) = m
        .states
        .iter()
        .find(|(_, s)| s.is_per_seq())
        .map(|(n, s)| (n.clone(), s.bytes_per_seq))
        .ok_or("the manifest has no per-sequence state")?;
    let cap = Capacity { tokens: Some(1 << 14), seqs: 4, landings: 0 };
    if let Some(dir) = destination {
        let mut b = Runtime::load(&m, Some(&kernels), gpu, Some(cap), None)?;
        return be_destination(&mut b, &paged, &slotted, slot_bytes, unit, &dir);
    }
    if cross {
        let mut a = Runtime::load(&m, Some(&kernels), gpu, Some(cap), None)?;
        let peer = [("--manifest", &manifest), ("--kernels", &kernels)];
        return push_across(&mut a, &paged, &slotted, slot_bytes, unit, &peer, gpu);
    }
    let mut a = Runtime::load(&m, Some(&kernels), gpu, Some(cap), None)?;
    let mut b = Runtime::load(&m, Some(&kernels), gpu, Some(cap), None)?;
    let mut c = Runtime::load(&m4, Some(&kernels), gpu, Some(cap), None)?;

    // A: twelve pages and a slot of known bytes, keyed by state and page.
    let la = a.lease(12 * unit as usize)?;
    let (ap, a_slot) = (la.page_ids().to_vec(), la.seq_slot().unwrap());
    for (si, (name, bytes)) in paged.iter().enumerate() {
        for &p in &ap {
            a.write_state_at(
                name,
                p as usize * *bytes as usize,
                &pattern(((si as u64) << 48) + p as u64 * bytes, *bytes),
            )?;
        }
    }
    let slot = pattern(7 << 48, slot_bytes);
    a.write_state_at(&slotted, a_slot as usize * slot_bytes as usize, &slot)?;

    // B leases eight pages and a slot; its stream passes the lease before A pushes.
    let lb = b.lease(8 * unit as usize)?;
    let (bp, b_slot) = (lb.page_ids().to_vec(), lb.seq_slot().unwrap());
    b.synchronize()?;
    let mut mb = mirror(&a, &b, "B")?;
    let pairs: Vec<(i32, i32)> =
        [(3, 1), (5, 2), (8, 3), (9, 4), (10, 5), (11, 6)].map(|(s, d)| (ap[s], bp[d])).to_vec();
    let folded = runs(&pairs.iter().map(|&(s, d)| (s, d as u64)).collect::<Vec<_>>());
    println!(
        "pages {:?}: {} copies, runs of {:?} pages",
        pairs,
        folded.len(),
        folded.iter().map(|r| r.2).collect::<Vec<_>>()
    );
    assert_eq!(
        folded.iter().map(|r| r.2).collect::<Vec<_>>(),
        [1, 1, 4],
        "the four consecutive pages fold into one copy"
    );
    for (name, _) in &paged {
        a.push_pages(name, &pairs, &mb)?;
    }
    a.push_slot(&slotted, a_slot, b_slot, &mb)?;

    // C: one slot per member, each taking that member's part of A's slot.
    let lc: Vec<Lease> = (0..of).map(|_| lease_slot(&mut c)).collect::<Result<_, _>>()?;
    c.synchronize()?;
    let mc = mirror(&a, &c, "C")?;
    let segments = [(1, slot_bytes / 4), (4, slot_bytes / 16)];
    let split = Split::new(&segments, of)?;
    let lines = slot_bytes / split.line();
    for (k, l) in lc.iter().enumerate() {
        a.push_slot_part(&slotted, a_slot, l.seq_slot().unwrap(), &split, k as u64, &mc)?;
    }
    a.mark()?.wait()?;
    a.fault()?;

    let mut compared = 0u64;
    for (si, (name, bytes)) in paged.iter().enumerate() {
        for &(s, d) in &pairs {
            let got = b.read_state_at(name, d as usize * *bytes as usize, *bytes as usize)?;
            assert!(got == pattern(((si as u64) << 48) + s as u64 * bytes, *bytes), "state `{name}` page {s} -> {d}");
            compared += bytes;
        }
        for d in [bp[0], bp[7]] {
            let got = b.read_state_at(name, d as usize * *bytes as usize, *bytes as usize)?;
            assert!(got.iter().all(|&x| x == 0), "state `{name}` page {d} was not pushed into");
        }
    }
    println!(
        "pages: {} pairs of {} states, {compared} bytes identical; B's other pages untouched",
        pairs.len(),
        paged.len()
    );
    assert!(b.read_state_at(&slotted, b_slot as usize * slot_bytes as usize, slot_bytes as usize)? == slot);
    println!("slot: {slot_bytes} bytes identical");
    // A member's part, by the definition: the k-th of `of` pieces of every row of every segment, line by line.
    let part = slot_bytes / of;
    let rows: Vec<u64> = segments.iter().flat_map(|&(r, w)| (0..r).map(move |_| w)).collect();
    for (k, l) in lc.iter().enumerate() {
        let got = c.read_state_at(&slotted, l.seq_slot().unwrap() as usize * part as usize, part as usize)?;
        let mut want = vec![0u8; part as usize];
        let (mut src, mut dst) = (0u64, 0u64);
        for _ in 0..lines {
            for &w in &rows {
                let piece = w / of;
                let from = (src + k as u64 * piece) as usize;
                want[dst as usize..(dst + piece) as usize].copy_from_slice(&slot[from..from + piece as usize]);
                (src, dst) = (src + w, dst + piece);
            }
        }
        assert!(got == want, "part {k} of {of}");
    }
    println!(
        "slot parts: {of} × {part} bytes identical ({} strided copies each over {lines} lines)",
        split.copies(lines, 0).len()
    );

    // B trades a page chunk for slots; A follows by placement and pushes into the new slot.
    drop(lb);
    let before = b.placement();
    let mut held = Vec::new();
    while b.placement() == before {
        held.push(lease_slot(&mut b)?);
        if held.len() > 4096 {
            return Err("B never remapped a chunk".into());
        }
    }
    let fresh = held.last().unwrap().seq_slot().unwrap();
    b.synchronize()?;
    println!("B: {} slots leased, {} remaps; slot {fresh} sits on a chunk that held pages", held.len(), b.remaps());
    a.place(&mut mb, &b.placement())?;
    a.push_slot(&slotted, a_slot, fresh, &mb)?;
    a.mark()?.wait()?;
    let la2 = lease_slot(&mut a)?;
    let back = la2.seq_slot().unwrap();
    a.synchronize()?;
    let ma = mirror(&b, &a, "A")?;
    b.push_slot(&slotted, fresh, back, &ma)?;
    b.mark()?.wait()?;
    b.fault()?;
    assert!(a.read_state_at(&slotted, back as usize * slot_bytes as usize, slot_bytes as usize)? == slot);
    println!("remapped slot: {slot_bytes} bytes there and back identical");
    println!("PASS");
    Ok(())
}

/// Wait for `path` to appear, written whole by the other process.
fn appear(path: &Path) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let t = Instant::now();
    while !path.exists() {
        if t.elapsed() > Duration::from_secs(120) {
            return Err(format!("{} never appeared", path.display()).into());
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    Ok(std::fs::read(path)?)
}

/// Publish `bytes` at `path` in one rename, so the reader never sees half.
fn publish(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(tmp, path)
}

/// The destination process: lease four pages and a slot, hand out how to
/// reach them, and check them once the source has pushed.
fn be_destination(
    b: &mut Runtime,
    paged: &[(String, u64)],
    slotted: &str,
    slot_bytes: u64,
    unit: u64,
    dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let l = b.lease(4 * unit as usize)?;
    b.synchronize()?;
    let h = b.export_pool()?.to_bytes().ok_or("local handles do not cross a process")?;
    publish(&dir.join("pool"), &h)?;
    publish(&dir.join("placement"), &b.placement().to_bytes())?;
    let ids: Vec<i32> = l.page_ids().iter().copied().chain(l.seq_slot()).collect();
    publish(&dir.join("ids"), &ids.iter().flat_map(|i| i.to_le_bytes()).collect::<Vec<_>>())?;
    appear(&dir.join("pushed"))?;
    let mut compared = 0u64;
    for (si, (name, bytes)) in paged.iter().enumerate() {
        for (i, &d) in l.page_ids().iter().enumerate() {
            let got = b.read_state_at(name, d as usize * *bytes as usize, *bytes as usize)?;
            assert!(got == pattern(((si as u64) << 48) + i as u64 * bytes, *bytes), "state `{name}` page {d}");
            compared += bytes;
        }
    }
    let s = l.seq_slot().unwrap() as usize;
    assert!(b.read_state_at(slotted, s * slot_bytes as usize, slot_bytes as usize)? == pattern(7 << 48, slot_bytes));
    println!("destination process: {compared} page bytes and {slot_bytes} slot bytes identical");
    Ok(())
}

/// The source process: start the destination, mirror its pool from the
/// wire forms it wrote, and push four consecutive pages and the slot.
fn push_across(
    a: &mut Runtime,
    paged: &[(String, u64)],
    slotted: &str,
    slot_bytes: u64,
    unit: u64,
    peer: &[(&str, &PathBuf)],
    gpu: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::temp_dir().join(format!("kern-push-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let mut child = std::process::Command::new(std::env::current_exe()?)
        .args(peer.iter().flat_map(|(k, v)| [k.to_string(), v.display().to_string()]))
        .args(["--gpu".to_string(), gpu.to_string(), "--destination".to_string(), dir.display().to_string()])
        .spawn()?;
    let la = a.lease(8 * unit as usize)?;
    let ap = la.page_ids()[2..6].to_vec();
    for (si, (name, bytes)) in paged.iter().enumerate() {
        for (i, &p) in ap.iter().enumerate() {
            a.write_state_at(
                name,
                p as usize * *bytes as usize,
                &pattern(((si as u64) << 48) + i as u64 * bytes, *bytes),
            )?;
        }
    }
    let a_slot = la.seq_slot().unwrap();
    a.write_state_at(slotted, a_slot as usize * slot_bytes as usize, &pattern(7 << 48, slot_bytes))?;
    let ids: Vec<i32> = appear(&dir.join("ids"))?.as_chunks::<4>().0.iter().map(|&w| i32::from_le_bytes(w)).collect();
    let h = PoolHandle::from_bytes(&appear(&dir.join("pool"))?).ok_or("not a pool handle")?;
    let p = Placement::from_bytes(&appear(&dir.join("placement"))?).ok_or("not a placement")?;
    let t = Instant::now();
    let mut mp = a.map_pool(&h, "destination process")?;
    a.place(&mut mp, &p)?;
    let mapped = t.elapsed();
    let pairs: Vec<(i32, i32)> = ap.iter().copied().zip(ids[..4].iter().copied()).collect();
    for (name, _) in paged {
        a.push_pages(name, &pairs, &mp)?;
    }
    a.push_slot(slotted, a_slot, ids[4], &mp)?;
    a.mark()?.wait()?;
    a.fault()?;
    println!(
        "source process: pool mapped and placed in {mapped:?}, pages {pairs:?} and slot {a_slot} -> {} pushed",
        ids[4]
    );
    publish(&dir.join("pushed"), b"")?;
    let status = child.wait()?;
    std::fs::remove_dir_all(&dir)?;
    if !status.success() {
        return Err(format!("destination process: {status}").into());
    }
    println!("PASS");
    Ok(())
}
