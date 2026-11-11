# Trace an SGLang server

A kern manifest hosted in SGLang can record every GPU kernel its process runs:
when it started and how long it took, stamped by the GPU itself, and which
call of the manifest launched it. SGLang is not patched and the model does not
change; recording is one environment variable on the server.

The files are the Parquet pairs of the [trace format](/reference/traces):
one row per kernel, one row per program step. This page traces two served
models and shows what to ask of the files.

## Requirements

| | |
| --- | --- |
| GPU | B300 / GB300 (`sm_103a`), what the published kernel bundles are built for |
| kern | `pip install "kern-llm>=0.5.0"` inside the SGLang image |
| libcupti | 13.4, as the first `libcupti.so` on the loader path |
| SGLang | unmodified; `lmsysorg/sglang:dev-kimi-k31-pro-cu134` runs both models below |

That image already has libcupti 13.4 first on its path. An image built on an
older CUDA has an older one there, and a traced server then stops at model
load and says so. Put a newer one in front:

```sh
pip install --no-deps --target /opt/cupti 'nvidia-cuda-cupti==13.4.*'
ln -s libcupti.so.13 /opt/cupti/nvidia/cu13/lib/libcupti.so
export LD_LIBRARY_PATH=/opt/cupti/nvidia/cu13/lib:$LD_LIBRARY_PATH
```

## Qwen3.8-27B on one GPU

The bundle is the one [Run a model](/guides/run-a-model) uses; its `-sglang` manifests
run on SGLang's memory. The server is started as usual, with `KERN_TRACE_DIR`
added:

```sh
pip install "kern-llm>=0.5.0"
hf download Pegainfer/kern-qwen38-sm103 --local-dir kern-qwen38

KERN_TRACE_DIR=/var/kern-trace \
KERN_MANIFEST=kern-qwen38/manifests/qwen3.8-27b-rsi-sglang.json KERN_KERNELS=kern-qwen38/cubins \
sglang serve --model-path Qwen/Qwen3.8-27B \
  --page-size 64 --attention-backend trtllm_mha --mamba-radix-cache-strategy extra_buffer \
  --disable-prefill-cuda-graph --max-running-requests 128 --cuda-graph-max-bs-decode 128 \
  --chunked-prefill-size 8192
```

Serve traffic as usual, then close the pair of files being written:

```sh
touch /var/kern-trace/rotate
ls /var/kern-trace/date=*/rank=0/
# 092934.kernels.parquet  092934.steps.parquet
```

A pair also closes by itself at every quarter hour. SGLang ends its scheduler
with a signal, so touch `rotate` before stopping the server.

What the files hold:

- **A prefill** is one `prefill` step per request, each kernel under the
  manifest call that launched it.
- **A decode step** is SGLang's CUDA graph, which SGLang replays without
  calling the model. Each replay is one `decode_batch` step and one `head`
  step with `graph` true, holding the kernels and call names the same
  programs have when they run eagerly.
- **SGLang's own kernels** (sampling, the metadata it updates per step) are
  the rows whose `seq` is null.

## Kimi-K3.1 Pro decode on two nodes

kern runs Kimi-K3.1 Pro's decode at TP8 / DCP8 over two nodes of four GPUs as
the decode instance of an SGLang prefill/decode deployment; the deployment is
[kern-k31's](https://github.com/pegainfer-project/kern-k31/blob/main/docs/try-in-sglang.md).
Tracing it is the same variable in the environment of both decode servers,
with `kern-llm` 0.5.0 or newer installed on those nodes:

```sh
export KERN_TRACE_DIR=/var/kern-trace   # then the decode command of that page, unchanged
```

Every TP rank is a process of its own and records for itself, so each node's
directory fills with four of `rank=0` … `rank=7`. Touch `rotate` on both
nodes, then read the eight directories as one trace: the ranks share a clock
(`start_ns` is Unix time), and `rank` is a partition column.

## Read a trace

The examples run on a sample trace of the Qwen3.8-27B server above, published
as [`Pegainfer/kern-traces`](https://huggingface.co/datasets/Pegainfer/kern-traces):
384 requests of 2048 prompt tokens and 256 output tokens at 32 concurrent, 2.6
million kernels in 11.3 MB. DuckDB reads it where it is; for a trace of your
own, the path is `<dir>/date=*/rank=*/*.steps.parquet`.

```sql
CREATE VIEW steps   AS SELECT * FROM read_parquet('hf://datasets/Pegainfer/kern-traces/qwen3.8-27b-sglang/date=*/rank=*/*.steps.parquet');
CREATE VIEW kernels AS SELECT * FROM read_parquet('hf://datasets/Pegainfer/kern-traces/qwen3.8-27b-sglang/date=*/rank=*/*.kernels.parquet');
```

Where the GPU time goes, by program and operator:

```sql
SELECT s.program, k.op, count(*) AS kernels, round(sum(k.dur_ns) / 1e6, 1) AS gpu_ms
FROM steps s JOIN kernels k USING (rank, seq)
GROUP BY ALL ORDER BY gpu_ms DESC LIMIT 6;
```

| program | op | kernels | gpu_ms |
| --- | --- | ---: | ---: |
| prefill | gemm | 117696 | 22259.1 |
| decode_batch | gemm_gate_up_silu | 396544 | 13464.9 |
| decode_batch | gemm_n5120_norm | 1187328 | 12925.9 |
| decode_batch | gdn_step | 149184 | 5306.5 |
| decode_batch | gemm_in_proj | 149184 | 4476.9 |
| decode_batch | attn_batch | 49728 | 3438.9 |

What a decode step costs at each batch SGLang ran:

```sql
SELECT shape, count(*) AS steps, round(avg(end_ns - start_ns) / 1e6, 2) AS ms
FROM steps WHERE program = 'decode_batch' AND graph
GROUP BY shape ORDER BY steps DESC LIMIT 5;
```

| shape | steps | ms |
| --- | ---: | ---: |
| seqs=32,tokens=32 | 3060 | 12.16 |
| seqs=1,tokens=1 | 7 | 8.08 |

How much of the kernel time is the model's, and how much SGLang's own:

```sql
SELECT seq IS NULL AS sglang_own, count(*) AS kernels, round(sum(dur_ns) / 1e9, 2) AS gpu_s
FROM kernels GROUP BY ALL ORDER BY 1;
```

| sglang_own | kernels | gpu_s |
| --- | ---: | ---: |
| false | 2484851 | 71.79 |
| true | 125835 | 0.43 |

Every column, the files' metadata (`kern.hes` says the GPU stamped the
kernels, `kern.dropped_records` how many records were lost) and more uses are
in the [trace format](/reference/traces) reference.

## What recording costs

Measured with a fresh server per arm, tracer off and on in turn.

Qwen3.8-27B, one GB300, SGLang nightly `f70e8c68`, 384 requests of 2048 prompt
tokens and 256 output tokens at 32 concurrent:

| | output tokens / s | end-to-end p50 |
| --- | --- | --- |
| tracer off | 1432.1, 1430.2 | 5594–5601 ms |
| tracer on | 1425.3, 1424.0 | 5619–5623 ms |

A traced server kept 26.7 thousand kernels a second in 4.4 bytes each, none
dropped, with 99.0% of the kernel time in a step.

Kimi-K3.1 Pro decode, eight GB300 on two nodes, milliseconds per decode step:

| running requests | 1 | 32 | 64 | 128 |
| --- | --- | --- | --- | --- |
| tracer off | 9.60, 9.59 | 20.98, 21.01 | 27.55, 27.60 | 36.15, 36.14 |
| tracer on | 9.58, 9.58 | 20.94, 20.99 | 27.60, 27.60 | 36.13, 36.16 |

Behind SGLang's prefill instance for fifteen minutes of agent traffic, the
eight ranks kept 792 million kernels in 2.83 GB, none dropped, with 99.2% of
the kernel time in a step.

## Notes

- With `--dp-size` above one, every replica is a group of its own: each
  writes under `<dir>/dp=<i>` and watches its own `rotate`.
- A hosted step has no `kv`: the host writes a step's inputs itself, so the
  runtime does not see the context the step ran over. The `shape` of a graph
  step is the batch SGLang captured the graph for.
- CUPTI admits one such subscriber per process: a server that also runs
  kineto or `nsys` cannot be traced at the same time.
