//! Real call-boundary capture on a tiny integer CUDA program: state written
//! by its prefix, aliased packed storage beyond the live prefix, multiple
//! outputs, and a state-only suffix. Every artifact and compiled fixture is
//! removed on success, an error return, or a failed assertion.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{ensure, Context, Result};
use kern_manifest::Verified;
use kern_run::dump::artifact::{Artifact, Writer};
use kern_run::dump::capture::record;
use kern_run::dump::plan::{Phase, Plan, Selection};
use kern_runtime::{Capacity, Runtime};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const SOURCE: &str = r#"
extern "C" __global__ void dump_prefix(
    const long long* ids, long long* work, unsigned long long* memory) {
    if (blockIdx.x || threadIdx.x) return;
    memory[0] += 1;
    for (int i = 0; i < 16; ++i)
        work[i] = 100 * i + ids[0] + memory[0];
}

extern "C" __global__ void dump_target(
    const long long* left, long long* right,
    const unsigned long long* memory, long long* visits,
    long long* out, long long* other) {
    if (blockIdx.x || threadIdx.x) return;
    out[0] = left[0] + right[0] + memory[0];
    other[0] = left[0] - right[0] + memory[2];
    right[0] += 1000;
    visits[0] += 1;
}

extern "C" __global__ void dump_suffix(unsigned long long* memory) {
    if (blockIdx.x || threadIdx.x) return;
    memory[2] += 1;
}
"#;

struct Temp(PathBuf);

impl Temp {
    fn new() -> Result<Self> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        loop {
            let path = std::env::temp_dir().join(format!(
                "kern-dump-capture-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Ok(Self(path)),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e).context("creating CUDA dump fixture directory"),
            }
        }
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

struct Fixture {
    temp: Temp,
    manifest: String,
    verified: Verified,
}

impl Fixture {
    fn new() -> Result<Self> {
        let temp = Temp::new()?;
        let source = temp.0.join("fixture.cu");
        let cubin = temp.0.join("fixture.cubin");
        fs::write(&source, SOURCE)?;
        let cuda = PathBuf::from(std::env::var_os("CUDA_PATH").unwrap_or_else(|| "/usr/local/cuda-13.2".into()));
        let compiled = Command::new(cuda.join("bin/nvcc"))
            .args(["-cubin", "-arch=sm_86", "-O2"])
            .arg(&source)
            .arg("-o")
            .arg(&cubin)
            .output()
            .context("compiling CUDA dump fixture with nvcc")?;
        ensure!(compiled.status.success(), "nvcc failed: {}", String::from_utf8_lossy(&compiled.stderr));
        let sha = hex::encode(Sha256::digest(fs::read(&cubin)?));
        let launch = |entry: &str| json!({"module":"fixture", "entry":entry, "grid":[1,1,1], "block":[1,1,1]});
        let manifest = json!({
            "schema_version":7, "model":"dump-capture-fixture",
            "vars":{
                "rows":{"max":4,"axis":"rows"},
                "groups":{"max":1,"axis":"groups"}
            },
            "states":{"memory":{"bytes":32}},
            "buffers":{
                "ids":{"kind":"input","dtype":"i64","shape":["rows"],"fill":"token"},
                "work":{"kind":"workspace","dtype":"i64","shape":["rows",4]},
                "visits":{"kind":"carry","dtype":"i64","shape":[1]},
                "out":{"kind":"output","dtype":"i64","shape":["groups"],"fill":"tokens"},
                "other":{"kind":"output","dtype":"i64","shape":["groups"]}
            },
            "modules":{"fixture":{"source":"fixture.cubin","sha256":sha}},
            "ops":{
                "prefix":{
                    "params":["in buffer<i64>","out buffer<i64>","inout state"],
                    "impl":{"launches":[launch("dump_prefix")]}
                },
                "target":{
                    "params":["in buffer<i64>","inout buffer<i64>","in state","inout buffer<i64>","out buffer<i64>","out buffer<i64>"],
                    "impl":{"launches":[launch("dump_target")]}
                },
                "suffix":{
                    "params":["inout state"],
                    "impl":{"launches":[launch("dump_suffix")]}
                }
            },
            "programs":{"step":{"batch":{"groups":1,"rows":1},"graph":true,"calls":[
                {"label":"prefix","op":"prefix","args":[{"buf":"ids"},{"buf":"work"},{"state":"memory"}]},
                {"label":"target","op":"target","args":[
                    {"buf":"work","offset":64},{"buf":"work","offset":96},{"state":"memory"},
                    {"buf":"visits"},{"buf":"out"},{"buf":"other"}
                ]},
                {"label":"suffix","op":"suffix","args":[{"state":"memory"}]}
            ]}}
        })
        .to_string();
        let verified = Verified::from_json(&manifest)?;
        fs::write(temp.0.join("manifest.json"), &manifest)?;
        Ok(Self { temp, manifest, verified })
    }

    fn runtime(&self) -> Result<Runtime> {
        let gpu = std::env::var("KERN_TEST_GPU").ok().map(|g| g.parse()).transpose()?.unwrap_or(0);
        let mut rt = Runtime::load(
            &self.verified,
            Some(&self.temp.0),
            gpu,
            Some(Capacity { tokens: Some(1), slots: 1, landings: 0 }),
            None,
        )?;
        rt.write_input("ids", &7i64.to_le_bytes())?;
        rt.write_buffer("visits", &0i64.to_le_bytes())?;
        rt.write_buffer("out", &(-91i64).to_le_bytes())?;
        rt.write_buffer("other", &(-92i64).to_le_bytes())?;
        rt.write_state_at("memory", 0, &words(&[0, 41, 0, 43]))?;
        Ok(rt)
    }

    fn plan(&self, invocation: u64, indices: Vec<usize>) -> Result<Plan> {
        Plan::check(
            &self.verified,
            Selection { program: "step".into(), invocation, labels: vec![], indices, phase: Phase::Both },
        )
    }
}

fn words(values: &[i64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn integers(bytes: &[u8]) -> Vec<i64> {
    assert_eq!(bytes.len() % 8, 0);
    bytes.as_chunks::<8>().0.iter().copied().map(i64::from_le_bytes).collect()
}

fn payload(dir: &Path, call: &Value, phase: &str, storage: &str) -> Result<Vec<u8>> {
    let path = call[phase][storage]["path"].as_str().context("observation has no payload path")?;
    Ok(fs::read(dir.join(path))?)
}

fn vars() -> BTreeMap<String, u64> {
    BTreeMap::from([("rows".into(), 1), ("groups".into(), 1)])
}

#[test]
#[ignore = "requires a CUDA GPU and nvcc"]
fn real_boundaries_preserve_read_only_state_aliases_and_every_output() -> Result<()> {
    let fixture = Fixture::new()?;
    let rt = fixture.runtime()?;
    for invocation in 0..2 {
        let dir = fixture.temp.0.join(format!("capture-{invocation}"));
        let plan = fixture.plan(invocation, vec![1, 2])?;
        let mut writer = Writer::create(&dir, &fixture.manifest, 4096)?;
        let metadata = record(&rt, &plan, &vars(), invocation as i64, &BTreeMap::new(), &mut writer)?;
        writer.finish(metadata)?;
        let artifact = Artifact::read(&dir)?;
        let metadata = &artifact.metadata;
        let target = &metadata["calls"][0];
        let suffix = &metadata["calls"][1];
        let count = invocation as i64 + 1;
        let before_work: Vec<i64> = (0..16).map(|i| 100 * i + 7 + count).collect();
        let mut after_work = before_work.clone();
        after_work[12] += 1000;
        let before_state = vec![count, 41, count - 1, 43];
        let final_state = vec![count, 41, count, 43];

        assert_eq!(
            (&metadata["program"], &metadata["invocation"], &metadata["cursor"], &metadata["execution"]),
            (&json!("step"), &json!(invocation), &json!(invocation), &json!("eager_serial"))
        );
        assert_eq!(
            (metadata["calls"].as_array().unwrap().len(), target["index"].as_u64(), suffix["index"].as_u64()),
            (2, Some(1), Some(2))
        );
        assert_eq!(
            (
                integers(&payload(&dir, target, "before", "buffer:work")?),
                integers(&payload(&dir, target, "after", "buffer:work")?)
            ),
            (before_work, after_work)
        );
        assert_eq!(
            (
                integers(&payload(&dir, target, "before", "state:memory")?),
                integers(&payload(&dir, target, "after", "state:memory")?)
            ),
            (before_state.clone(), before_state.clone())
        );
        assert_eq!(
            (
                integers(&payload(&dir, suffix, "before", "state:memory")?),
                integers(&payload(&dir, suffix, "after", "state:memory")?),
                integers(&rt.read_state_at("memory", 0, 32)?)
            ),
            (before_state, final_state.clone(), final_state)
        );
        assert_eq!(
            (integers(&payload(&dir, target, "after", "buffer:visits")?), integers(&rt.read_buffer("visits")?)),
            (vec![count], vec![count])
        );
        assert_eq!(
            (
                integers(&payload(&dir, target, "after", "buffer:out")?),
                integers(&payload(&dir, target, "after", "buffer:other")?)
            ),
            (vec![2014 + 3 * count], vec![-400 + count - 1])
        );
        if invocation == 0 {
            assert_eq!(
                (
                    integers(&payload(&dir, target, "before", "buffer:out")?),
                    integers(&payload(&dir, target, "before", "buffer:other")?)
                ),
                (vec![-91], vec![-92])
            );
        }

        let stores = target["storages"].as_array().unwrap();
        let work = stores.iter().find(|s| s["id"] == "buffer:work").context("no packed storage metadata")?;
        let state = stores.iter().find(|s| s["id"] == "state:memory").context("no state metadata")?;
        assert_eq!(
            (&work["range"], &work["storage"]["shape"], &work["storage"]["bytes"], &work["storage"]["arguments"]),
            (
                &json!([0, 128]),
                &json!([1, 4]),
                &json!(128),
                &json!([
                    {"index":0,"param":"in buffer<i64>","offset":64},
                    {"index":1,"param":"inout buffer<i64>","offset":96}
                ])
            )
        );
        assert_eq!(
            (stores.len(), state["storage"]["dtype"].is_null(), state["storage"]["shape"].is_null()),
            (5, true, true)
        );
        assert_eq!(metadata["replay_requirements"], json!([]));
    }
    Ok(())
}
