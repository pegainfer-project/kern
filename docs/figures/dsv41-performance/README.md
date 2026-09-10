# DeepSeek V4.1 Flash performance

`performance.png` is the sharing image; `performance.svg` is the vector version.
Regenerate both with `python plot.py` (Python, seaborn and matplotlib).

Configuration: 4 NVIDIA GB300 GPUs, one process, DP4 attention / EP4 MoE,
BF16 O-A, original checkpoint, complete Engram tables in 512 MiB transparent
huge pages, chunk size 128. Measured on 2026-09-10 through the HTTP completions API.

- Prefill: four simultaneous requests with distinct prefixes, exactly 10,000
  input tokens and one output token each. Aggregate throughput is 40,000 /
  (last first-text arrival minus earliest request start). This includes HTTP
  and scheduler overhead; it is not a kernel-only throughput measurement.
- Decode: one request at a time, exactly 10,000 input and 256 output tokens.
  Throughput is 255 / (last-text arrival minus first-text arrival).
  DSpark may deliver several tokens per SSE event. This measures average output
  delivery, not constant inter-token latency. Plain uses rows=1; DSpark rows=6.
- Bars show the maximum observed throughput per metric. All trial values are
  in `data.json`: three prefill batches, three plain-decode requests, and six
  DSpark requests across two runs. The slower trials are retained in the data.
  Decode measurements were made in separate runs of the same THP runtime build.
  Prefix reuse was zero in the measured requests. Performance does not establish
  identical text, numerical equivalence or stable official-reference acceptance.
