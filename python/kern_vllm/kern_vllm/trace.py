"""Every step's shape as vLLM schedules it, one JSON line per step.

Off unless `KERN_TRACE` names a directory. Lines go to
`<dir>/<manifest sha12>.jsonl`, since a new manifest changes how vLLM
batches and the two distributions must not mix:

    {"t": <unix ns>, "q": [query len per request], "s": [seq len per request]}

`t` is when vLLM built the step's metadata. With async scheduling that
happens while the previous step runs, so a step's duration is the interval
between the next two lines, not between its own and the next.

The builder calls `step`, not the model: under full CUDA graphs vLLM
replays decode without calling the model, but builds metadata every step.
The calling thread only enqueues; encoding and writing happen on a listener
thread. Files rotate at `KERN_TRACE_MAX_BYTES` (256 MiB), rotated files are
gzipped, and `KERN_TRACE_FILES` (20) of them are kept.
"""
import atexit
import functools
import gzip
import hashlib
import json
import logging
import logging.handlers
import os
import queue
import shutil
import time


class _Enqueue(logging.handlers.QueueHandler):
    def prepare(self, record):
        return record


class _Line(logging.Formatter):
    def format(self, record):
        return json.dumps(record.msg, separators=(",", ":"))


def _gzip(src: str, dst: str):
    with open(src, "rb") as f, gzip.open(dst, "wb") as g:
        shutil.copyfileobj(f, g)
    os.remove(src)


@functools.cache
def _log() -> logging.Logger | None:
    root = os.environ.get("KERN_TRACE")
    if not root:
        return None
    sha = hashlib.sha256(open(os.environ["KERN_MANIFEST"], "rb").read()).hexdigest()[:12]
    os.makedirs(root, exist_ok=True)
    file = logging.handlers.RotatingFileHandler(
        os.path.join(root, f"{sha}.jsonl"),
        maxBytes=int(os.environ.get("KERN_TRACE_MAX_BYTES", 256 << 20)),
        backupCount=int(os.environ.get("KERN_TRACE_FILES", 20)))
    file.namer = lambda name: name + ".gz"
    file.rotator = _gzip
    file.setFormatter(_Line())
    q = queue.SimpleQueue()
    listener = logging.handlers.QueueListener(q, file)
    listener.start()
    atexit.register(listener.stop)
    log = logging.getLogger("kern_vllm.trace")
    log.propagate = False
    log.setLevel(logging.INFO)
    log.addHandler(_Enqueue(q))
    return log


def step(meta):
    """Records one step; padded rows (seq len 0) are left out."""
    log = _log()
    if log is None or meta.seq_lens_cpu_upper_bound is None:
        return
    cu = meta.query_start_loc_cpu.tolist()
    rows = [(b - a, s) for a, b, s in zip(cu, cu[1:], meta.seq_lens_cpu_upper_bound.tolist()) if s > 0]
    log.info({"t": time.time_ns(), "q": [q for q, _ in rows], "s": [s for _, s in rows]})
