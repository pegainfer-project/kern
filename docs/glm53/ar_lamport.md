# P1: GLM-5.3-Flash TP8 Lamport allreduce

Status: implemented, cubin built, CPU verification PASS; device validation pending.
Scope: H100/H800 sm_90a, decode rows 1..16, hidden 4096. No GPU work
before the decode bench log contains `EXIT=`. Never overwrite
the baseline manifest. No P2 consumer fusion here.

## 1. Architecture gate (before kernel implementation)

**The peer ABI already works with replicated rows. No Rust patch is needed.**

* `crates/kern-runtime/src/load.rs:149-181` allocates exported buffers with
  VMM and records peer arrays, independent of row topology.
* `crates/kern-runtime/src/peers.rs:48-145` exports allocations, imports each
  group in rank order (including the local pointer), fills u64 pointer arrays,
  and rejects execution with missing peers. No replicated-row condition exists.
* `crates/kern-run/src/lib.rs:204-248` constructs both eight-rank groups and
  calls `connect_peers`; `bench/mod.rs` invokes this before execution.
* `crates/kern-serve/src/tray.rs:457-545` reads the topology, constructs
  TP-local rank indices, and imports each TP group before execution. The only
  replicated-row branch in this load path exempts it from requiring `blocks`.
* `crates/kern-manifest/src/protocol.rs:597-606` forbids tray/blocks in this
  mode, not export/peer buffers. `docs/glm53/tp_replicated.md` specifies the same
  logical row order on all members. Physical KV page IDs need not be equal.
* VMM is zero-filled by `device.rs::alloc_vmm`. On a driver without fabric
  handles, this same-process deployment imports allocation handles instead.
  Installed H100 peer access still requires a device smoke test.

The old `tools/kernels-src/peer_allreduce.cu` documents `-DNATURAL` to remove
rotation. Our one-shot path is always natural: vector index i refers to the
same row/column on all ranks. No tray axis, blocks input, gather, or rotation
is required. Retain the existing runtime rather than patching Rust.

## 2. Exact integration map

All payloads are contiguous bf16 `[seqs,4096]`: 8192*S bytes, maximum 131072.
Only the live S rows are read. DeepGEMM can write junk in rows S..15.

| op | producer | reduction buffer | next consumer | calls/45 layers |
|---|---|---|---|---:|
| kda_o_proj_ar | cuBLASLt: kda_o[S,1024] times o_proj[4096,1024] | sub_out | hc_fma | 34 |
| dsa_ar | DeepGEMM dsa_o_proj: av[S,2048] times o_proj[4096,2048] | sub_out | hc_fma | 11 |
| moe_ar | moe_sum_reduce of 9 expert partials/row | sub_out | hc_post | 42 |
| mlp_ar | dense down projection in layers 0..2 | sub_out | hc_post | 3 |

Source has **90 hidden allreduces**, not 91. Head logits still use an NCCL
allgather. `_ar()` in ops_moe.py serves both dense and MoE calls. KDA keeps
op param 2 as **BF16O**, since cuBLASLt produces it within that op. The
Lamport reduction must be a separate manifest call (see the verifier fork below).
Separate DSA/MoE ops retain inout sub_out. No producer/consumer buffer moves.

### Verifier fork found at integration

`crates/kern-manifest/src/verify.rs:862` rejects ANY peer argument to an op containing an extern launch,
even if the peer argument goes only to that op's cubin launch. Therefore the
KDA cuBLASLt producer and Lamport reducer become two calls in Lamport mode.
`kda_o_proj_ar` retains its existing name and BF16O producer interface;
`kda_ar` receives the same sub_out inout, after the producer. This avoids a
Rust/schema change. GPU launches do not increase: the old call had two
launches already. Decode call count grows from 1150 to 1184; launch count is
unchanged. NCCL mode retains the original combined call.

## 3. Wire protocol and numerics

Source oracle is under
`<sglang checkout>/python/sglang/`:

* `kernels/jit/csrc/distributed/custom_all_reduce.cuh:136-217`:
  16-byte vectors, LamportTrait with **kAtom=4**, push/poll, reduce, clear.
* `kernels/jit/include/sgl_kernel/distributed/communicator.cuh:99-147`:
  positive zero marker; clear_pos_zero changes a zero packed **u32** to
  **0x00008000**. ONLY the low bf16 of an all-zero pair changes to -0.
  Rewriting every bf16 +0 independently is NOT bit-identical for signed zeros.
* `kernels/jit/include/sgl_kernel/vec.cuh:144-173` and `type.cuh`:
  convert both halves exactly to fp32; initialize from rank 0 (not +0), then
  add ranks 1..7 in order, and convert once with round-to-nearest-even.

Retain the kern port's peer-pointer push/local-poll design, but replace its
f32 -0 poison and three-stage global phase publication with sglang's two
halves and per-CTA phase. Each rank pushes to all eight rank slots including
itself with `st.relaxed.sys.global.v4.b32`. Poll with matching system-scope
loads until all four u32 atoms in all eight vectors are nonzero. A vector
need not be atomic as a whole: each 32-bit atom is data plus its own marker.
After reduction, clear the eight local vectors back to zero.

Use a fixed grid of 16 CTAs x 128 threads and sglang's warp-striped vector
mapping. Slot stride is ALWAYS 128 KiB, not current payload bytes. Each CTA
flips its local phase even when idle. All calls share one exported carry:
2 halves x 8 slots x 128 KiB plus 16 u32 phases. Kern zeroes it at allocation,
not per call. The phase lives inside the export so `profile.rs::program`
does not restore it independently of peer writes (non-exported carries are
restored for bench samples). Graph capture records work, not a host phase.

A fast rank can start round n+1 but cannot finish it until every peer has
started n+1, after finishing and clearing n. Thus it cannot reuse half n for
n+2 while a peer consumes n. Fixed mapping, positive row count, identical
collective order/shapes on all ranks, and no concurrent workspace use are
required. Shape changes 1..16 retain these properties. No grid barrier or
co-residency assumption is used.

Numeric equivalence is for identical bf16 rank inputs and the specified
sglang push algorithm, not a promise that upstream model intermediates match.
Use explicit FP32 round-nearest adds, no reassociation/fast math, then the
same CUDA bf16 RN intrinsic. Signed-zero handling matches the packed oracle.
NaN/infinity and subnormal cases remain mandatory bit-level GPU tests.

## 4. Integration and failure handling

A shared helper in ops_common.py selects NCCL or Lamport launches. gen.py
adds carry/peer/error buffers and lifts them into the four affected op
interfaces. `--allreduce lamport` emits `examples/glm53-flash-ar.json` by
default; `--allreduce nccl` remains available for A/B. Explicit `--out` and
`--bundle kernels-glm53` remain supported. Never delete bundle artifacts.

The baseline launch has no PDL. An opt-in PDL entry may prefetch the input
cache line before `griddepcontrol.wait`, but MUST NOT read its value before
that wait: the producer may still be writing it. Phase and input loads occur
after wait. Trigger the secondary only after output, clears, and phase writes.

A bounded device poll uses globaltimer. On timeout, atomically set sticky
`ar_error[0]` to a missing source rank + 1, and write bf16 quiet NaNs, not a
false successful sum. No trap and no host synchronization in the kernel.
`ar_error` is an output so `Runtime::read_output("ar_error")` can read it
between completed steps on each rank. Stop the group and recreate runtimes
on any nonzero flag; timeout is not recovery. Existing kern-serve only reads
token outputs and does NOT automatically monitor arbitrary status outputs.
Adding production host error policy is outside this file scope and remains
a rollout gate. A later NCCL logits gather cannot recover a dead rank either.

## 5. Expected cost and risks

Per rank, each call sends 7*8192*S bytes over NVLink and stores one local copy;
local polling reads 8 payloads and clearing writes 8. At S=1/2/4/8 the remote
traffic is 56/112/224/448 KiB. A launch/arrival floor of about 3-4 us plus
traffic and reduction supports a **5-6 us target**, not a measured result.
At 90 calls, moving 25 us to 6 us saves about 1.71 ms. P1 alone would move
10.1 ms to about 8.4 ms, NOT below 5.68 ms. Later roadmap work is required.

Risks: installed peer mapping support; scheduling/arrival skew; polling and
clear traffic; fixed-grid tuning; atomic u32 protocol; nonfinite parity;
host status monitoring; batch/grid changes; workspace reuse across graphs;
upstream FA3 bs16 capacity risk; PDL dependencies. No claim of GPU correctness,
<=6 us, or end-to-end parity is made before device tests.

## 6. Build and CPU evidence

All commands below are CPU compilation, source inspection, and
CPU tests were used. The benchmark later reached `EXIT=0`; this work still did
not launch a GPU kernel, start a server, or change any tmux session.

```sh
cd <kern-repo>
/usr/local/cuda-13.0/bin/nvcc -cubin -arch=sm_90a -std=c++17 -O3 \
  -Xptxas=-v -ccbin /usr/bin/g++-14 \
  -o kernels-glm53-handwritten/glm53_ar_lamport.cubin \
  tools/glm53/kernels/glm53_ar_lamport.cu
PYTHONDONTWRITEBYTECODE=1 python3 tools/glm53/gen.py \
  --allreduce lamport --bundle kernels-glm53
cargo build --release -p kern-run
# Exports copied from the production launcher; do not source it
# (it execs the server). Keep its libcuda forwarding/compat directories first.
export LD_LIBRARY_PATH=<nccl + cuda-compat libs for your node>
target/release/kern verify examples/glm53-flash-ar.json
```

This checkout takes the manifest as a **positional** verify argument.
`kern verify --manifest ...` is rejected by its CLI. The positional command
passes using the requested release binary. No Rust source or kern-serve
binary was changed. The existing `build_kernels.sh` glob already includes
the new source; only this kernel was compiled, not the other agents' kernels.

Results:

* Both entries: sm_90a, 96 registers/thread, 4 static shared bytes, no stack
  or spill loads/stores. ELF ABI: five 8-byte pointers, rank at byte 40,
  rows at byte 44, timeout i64 at byte 48, total 56 bytes. All pointers remain.
* PTX across both entries: 112 explicit `add.rn.f32` (56 per eight-lane
  reduction), eight `cvt.rn.bf16x2.f32`, no `.ftz`; system-scope vector
  loads/stores. Only the PDL entry has wait/launch_dependents.
* Verify PASS: full Lamport baseline, full Lamport PDL, 4-layer probe Lamport,
  full NCCL fallback. Variants were verified through stdin; only the requested
  main Lamport manifest was saved.
* Every row count 1..16: index enumeration covers each live vector exactly
  once, with no read past the live rows. All 29 module hashes match files.
* The actual kernel encode/reduce helper bodies were compiled as CPU C++ with
  CUDA RN intrinsics emulated, against the extracted sglang LamportTrait and
  reduce_vec bodies: 100,000 vectors / 800,000 output bf16 lanes matched.
  Cases included all 256 pairs of 16 special values (zeros, subnormals,
  signs, maxima, infinities, NaNs), random words, and mixed special values.
  This tests the source algorithm, not GPU floating-point execution.
* An asynchronous CPU protocol model used individual 32-bit stores, fixed
  thread ownership, and changing row counts. Five seeds x 48 rounds passed
  with no stale reads or overwrite of unconsumed data; completed-round skew
  was at most one. This is a model, not a CUDA memory-model proof.
* Baseline-vs-new canonical comparison: all original buffers, weights,
  states, and non-AR ops unchanged; existing FA3 PDL unchanged. Only the four
  AR definitions and KDA call split change; only three buffers are added.
* Baseline SHA256 remains
  `047e0d07064e931c25e74cbed8b8f48dd7e1a57431d89b6aa7df8bb71d35d14d`.
* New cubin SHA256:
  `adbace143b6f6e5e1198b7f3227058c2f562672cccab77e45d847633cfe713dd`.
  Bundle: `kernels-glm53/glm53_ar_lamport-adbace143b6f.cubin`.
* Manifest: 1160 buffers, 29 modules, 59 ops, 1184 decode calls, no load calls.
  No NCCL allreduce remains. The logits NCCL allgather remains unchanged.

## 7. GPU validation plan (operator runs afterward)

### 7.1 Release gate and operator test

First check `grep 'EXIT=' <bench_decode.log>` and confirm
that no other job has since taken the GPUs. Do not kill another tmux session.
Use the LD_LIBRARY_PATH above for every binary. Never run the two manifests
or sglang at the same time on these GPUs.

Before accepting performance, test the operator with all eight ranks:

1. Exercise kern's actual exported VMM allocation/import path; verify all
   eight u64 peer addresses are populated and the local entry is local sym.
   Layout: payload bytes [0,2097152), 16 phase words at byte 2097152.
2. Use rank-distinct bf16 inputs for rows 1,2,4,8,16. Compare output **u16
   bits**, not an allclose tolerance, against sglang's
   `custom_all_reduce(comm, input, AllReduceAlgo.ONE_SHOT_PUSH)`. Compare
   every rank with rank 0. Include cancellation, large/small mixed magnitude,
   bf16 ties, all signed-zero pairs, subnormals, infinities, and NaNs. For a
   hand-written oracle, apply the packed-u32 zero rewrite FIRST; a plain
   torch sum can use a different order and is not the oracle.
3. Test input==output, overwrite unused rows with junk, and guard bytes
   beyond the live output. Verify tail rows are untouched by AR.
4. Run eager then captured replay; alternate buckets
   `1,16,1,2,8,4,1,16`, then at least 10,000 replays. Include an odd count of
   calls between bucket switches so both phases are used. All ranks must
   follow the same call/bucket sequence. Read ar_error on EVERY rank only
   after a completed step; require zero.
5. In an isolated AR-only test with fresh allocations, skip one rank's
   launch but keep its mapped allocation alive. Remaining launches must
   finish after the timeout with nonzero ar_error and qNaN output. This
   must NOT run through the full model's later NCCL gather. Recreate all
   ranks afterward; do not resume timed-out workspaces.
6. Only after the baseline passes, generate a separate PDL manifest with
   `--ar-pdl --out examples/glm53-flash-ar-pdl.json --bundle kernels-glm53`.
   Repeat correctness, shape-switch, and timeout tests before measuring it.

The operator harness and automatic host error polling are not supplied by
this patch. `Runtime::read_output("ar_error")` is the supported readback hook;
stock kern bench/serve do not automatically turn this status into a failure.
This check is a release requirement, not something that verify can establish.

### 7.2 A/B performance

The existing workload already uses groups 1/2/4/8, rows 1, contexts
0/2048/8192, samples 30, seed 1. Retain it for apples-to-apples evidence.
Run A/B/B/A, with no concurrent server. Example single A/B pass:

```sh
cd <kern-repo>
K=target/release/kern
W=weights/GLM-5.3-Flash
for variant in old lamport; do
  if [ "$variant" = old ]; then
    M=examples/glm53-flash.json
  else
    M=examples/glm53-flash-ar.json
  fi
  "$K" bench --manifest "$M" --kernels kernels-glm53 --weights "$W" \
    --tokenizer "$W/tokenizer.json" --gpu 0 \
    --workload bench_decode.toml \
    --out "bench_ar_${variant}.json" \
    > "bench_ar_${variant}.log" 2>&1 || break
done
```

Use fresh result names for repeat passes. Compare the slowest-rank graph
p50/p90 at each identical shape, and all rank tails. Do NOT use `--isolate`
for TP8. For new per-AR attribution inspect kda_ar/dsa_ar/moe_ar/mlp_ar;
old kda_o_proj_ar includes the GEMM. Summed attribution is instrumented and
has extra events for the KDA call split; whole-graph timings are the primary
end-to-end evidence. Target AR <=6 us at bs1/2/4/8. Also time an AR-only
captured burst to distinguish tracing overhead from actual kernel time.
Reject results with any nonzero ar_error, peer-map failure, or parity failure.

### 7.3 Full-stack parity

After benchmarks, start the existing server binary with the new manifest,
using the same weights/GPU/capacity flags as the production launcher. Do not
edit that launcher or the baseline JSON. Ensure port 8000 and the GPUs are
free before starting. Add `NO_PROXY=127.0.0.1,localhost` for local HTTP:

```sh
cd <kern-repo>
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
export NCCL_NVLS_ENABLE=0
target/release/kern-serve \
  --manifest examples/glm53-flash-ar.json --kernels kernels-glm53 \
  --weights weights/GLM-5.3-Flash \
  --gpus 0,1,2,3,4,5,6,7 --port 8000 --capacity 2000000 --max-seqs 64
# In a second shell, with the same no_proxy setting:
python3 parity_check.py  # our parity checker
```

The parity checker is a standalone script, not part of the probe tooling.
Require code **40/40**, agentic **33/33**. Source inspection shows these are
**word** prefix counts, not characters; the oracle texts contain 201 and
220 characters respectively. The script prints rather than asserts, and
uses the smaller word count as denominator. Also require the full generated
strings to equal `oracle/{code,agentic}.json` response.text, not just the
printed ratio. Keep France/Chinese output for diagnosis. On a failure, use
our `teacher_forced.py` and `compare.py` probe scripts to find
the first intermediate divergence. Do not use NCCL-ring output as the
bit-level AR oracle: the numeric change is intentional (arch_decode R4).

After bs1 parity, test concurrent bs2/4/8 requests, a five-request batch
padded to eight, and prefix retire/restore across bucket changes. bs16 is
an AR capacity test; the existing full-stack FA3 capacity risk is separate.

## 8. Files and remaining limits

Repository root: `the kern repo`.

* `tools/glm53/kernels/glm53_ar_lamport.cu` (new)
* `tools/glm53/ops_common.py`
* `tools/glm53/ops_kda.py`
* `tools/glm53/ops_moe.py`
* `tools/glm53/ops_dsa.py`
* `tools/glm53/gen.py`
* `docs/glm53/ar_lamport.md` (new)
* `kernels-glm53-handwritten/glm53_ar_lamport.cubin` (new build output)
* `kernels-glm53/glm53_ar_lamport-adbace143b6f.cubin` (new additive artifact)
* `examples/glm53-flash-ar.json` (new)

The release kern target was built at `<cargo target dir>`.
No other source files were changed. No GPU validation, measured latency,
P2 fusion, runtime patch, host status policy, server restart, or baseline
manifest rewrite was done. GPU bit equality and <=6 us remain open gates.
