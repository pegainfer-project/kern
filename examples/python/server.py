"""Minimal continuous-batching OpenAI server over `manifest` and `executor`.

Uses only Python's stdlib `http.server` plus `Executor`. Every model detail —
buffer names, KV page stride, per-sequence state line tables, prefill vs
decode/speculative shapes — comes from `Protocol`.

    python3 examples/python/server.py --config /path/to/kern.toml --port 8000
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from executor import Executor
from manifest import Sampling
from run import resolve_target


# ── Memory pool ────────────────────────────────────────────────────────────────


class Pool:
    """Free-list allocator for paged KV blocks and per-sequence state slots."""

    def __init__(self, n_pages: int, max_seqs: int, page_unit: int):
        self.page_unit = page_unit
        self.free_pages = list(range(n_pages - 1, -1, -1))
        self.free_slots = list(range(max_seqs, 0, -1))  # slot 0 is null

    def _need(self, n_tokens: int) -> int:
        return (n_tokens + self.page_unit - 1) // self.page_unit

    def alloc(self, n_tokens: int) -> tuple[list[int], int] | None:
        need = self._need(n_tokens)
        if not self.free_slots or len(self.free_pages) < need:
            return None
        return [self.free_pages.pop() for _ in range(need)], self.free_slots.pop()

    def grow(self, pages: list[int], n_tokens: int) -> bool:
        need = self._need(n_tokens) - len(pages)
        if need <= 0:
            return True
        if len(self.free_pages) < need:
            return False
        pages.extend(self.free_pages.pop() for _ in range(need))
        return True

    def free(self, pages: list[int], slot: int) -> None:
        self.free_pages.extend(pages)
        self.free_slots.append(slot)

    def slots_for(self, pages: list[int], start: int, count: int) -> list[int]:
        u = self.page_unit
        return [pages[(start + i) // u] * u + (start + i) % u for i in range(count)]


# ── Continuous-batching engine ─────────────────────────────────────────────────


@dataclass
class Request:
    req_id: str
    prompt_ids: list[int]
    max_tokens: int
    sampling: Sampling
    out: queue.Queue[tuple[str, str | None]] = field(default_factory=queue.Queue)
    pages: list[int] = field(default_factory=list)
    slot: int = 0
    pos: int = 0
    generated: list[int] = field(default_factory=list)
    decoded_len: int = 0


class Engine:
    """Single-GPU continuous-batching scheduler driving an `Executor`."""

    def __init__(
        self,
        rt: Executor,
        tokenizer: Tokenizer,
        stop_tokens: set[int],
        max_seqs: int,
        rows: int | None = None,
        use_graph: bool = True,
    ):
        self.rt = rt
        self.tok = tokenizer
        self.stop_tokens = stop_tokens
        self.p = rt.protocol
        self.max_seqs = min(max_seqs, self.p.groups_max)
        self.chunk_fwd = self.p.chunk_forward()
        self.step_rows = int(self.p.step_forward(rows=rows, groups=1).rows)
        self.use_graph = use_graph
        self.pool = Pool(rt.capacity_tokens // self.p.page_unit, self.max_seqs, self.p.page_unit)

        self._cv = threading.Condition()
        self._pending: deque[Request] = deque()
        self._active: list[Request] = []
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, prompt_ids: list[int], max_tokens: int, sampling: Sampling) -> Request:
        req = Request(f"cmpl-{uuid.uuid4().hex[:12]}", prompt_ids, max_tokens, sampling)
        with self._cv:
            self._pending.append(req)
            self._cv.notify()
        return req

    def _emit(self, req: Request, tokens: list[int]) -> str | None:
        finish: str | None = None
        for t in tokens:
            if t in self.stop_tokens:
                finish = "stop"
                break
            req.generated.append(t)
            if len(req.generated) >= req.max_tokens:
                finish = "length"
                break
        text = self.tok.decode(req.generated, skip_special_tokens=False)
        delta, req.decoded_len = text[req.decoded_len :], len(text)
        if delta or finish:
            req.out.put((delta, finish))
        return finish

    def _prefill(self, req: Request) -> str | None:
        self.rt.zero_seq_slot(req.slot)
        self.rt.write_tables([req.pages], [req.slot])
        pos, ids, emitted = 0, req.prompt_ids, []
        while pos < len(ids):
            chunk = ids[pos : pos + self.p.rows_max]
            slots = self.pool.slots_for(req.pages, pos, len(chunk))
            vals = self.rt.stage(self.chunk_fwd, [pos], [len(chunk)], chunk, slots, req.sampling)
            self.rt.run(self.chunk_fwd.name, vals, graph=self.use_graph)
            torch.cuda.synchronize()
            pos += len(chunk)
            if pos == len(ids):
                emitted = self.rt.emitted(self.chunk_fwd)
        req.pos = pos
        return self._emit(req, emitted)

    def _step_batch(self) -> None:
        active: list[Request] = []
        for req in self._active:
            if self.pool.grow(req.pages, req.pos + self.step_rows):
                active.append(req)
            else:
                req.out.put(("", "length"))
                self.pool.free(req.pages, req.slot)
        self._active = active
        if not active:
            return

        fwd = self.p.step_forward(rows=self.step_rows, groups=len(active))
        self.rt.write_tables([r.pages for r in active], [r.slot for r in active])

        token_ids: list[int] = []
        slots: list[int] = []
        for r in active:
            anchor = r.generated[-1] if r.generated else r.prompt_ids[-1]
            token_ids.extend([anchor] + [0] * (self.step_rows - 1))
            slots.extend(self.pool.slots_for(r.pages, r.pos, self.step_rows))

        vals = self.rt.stage(
            fwd,
            positions=[r.pos for r in active],
            lens=[self.step_rows] * len(active),
            token_ids=token_ids,
            slots=slots,
            sampling=[r.sampling for r in active],
        )
        self.rt.run(fwd.name, vals, graph=self.use_graph)
        torch.cuda.synchronize()

        kept: list[Request] = []
        for r, out in zip(active, self.rt.emitted_batch(fwd, len(active))):
            r.pos += len(out)
            if self._emit(r, out) is None:
                kept.append(r)
            else:
                self.pool.free(r.pages, r.slot)
        self._active = kept

    def _loop(self) -> None:
        torch.cuda.set_device(self.rt.gpu)
        while True:
            admitted: list[Request] = []
            with self._cv:
                while not self._pending and not self._active:
                    self._cv.wait()
                while self._pending and len(self._active) + len(admitted) < self.max_seqs:
                    req = self._pending[0]
                    alloc = self.pool.alloc(len(req.prompt_ids) + self.step_rows)
                    if alloc is None:
                        break
                    self._pending.popleft()
                    req.pages, req.slot = alloc
                    admitted.append(req)

            for req in admitted:
                if self._prefill(req) is None:
                    self._active.append(req)
                else:
                    self.pool.free(req.pages, req.slot)
            if self._active:
                self._step_batch()


# ── OpenAI HTTP / SSE frontend ─────────────────────────────────────────────────


def _stop_tokens(weight_dir: Path) -> set[int]:
    stops: set[int] = set()
    for name in ("generation_config.json", "config.json"):
        p = weight_dir / name
        if p.is_file():
            eos = json.loads(p.read_text()).get("eos_token_id")
            if isinstance(eos, int):
                stops.add(eos)
            elif isinstance(eos, list):
                stops.update(int(x) for x in eos)
    return stops


def _format_chat(messages: list[dict[str, Any]]) -> str:
    parts = [f"<|im_start|>{m['role']}\n{m.get('content', '')}<|im_end|>\n" for m in messages]
    return "".join(parts) + "<|im_start|>assistant\n"


def make_handler(engine: Engine, model_name: str):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[http] {self.address_string()} - {fmt % args}", file=sys.stderr)

        def _send_json(self, status: int, body: dict[str, Any]) -> None:
            raw = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _stream(self, req: Request, is_chat: bool, created: int) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            obj = "chat.completion.chunk" if is_chat else "text_completion"
            while True:
                delta, finish = req.out.get()
                choice = (
                    {"index": 0, "delta": {"content": delta}, "finish_reason": finish}
                    if is_chat
                    else {"index": 0, "text": delta, "finish_reason": finish}
                )
                chunk = {"id": req.req_id, "object": obj, "created": created, "model": model_name, "choices": [choice]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
                if finish is not None:
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    return

        def _complete(self, req: Request, is_chat: bool, created: int) -> None:
            parts, finish = [], None
            while finish is None:
                delta, finish = req.out.get()
                parts.append(delta)
            text = "".join(parts)
            choice = (
                {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}
                if is_chat
                else {"index": 0, "text": text, "finish_reason": finish}
            )
            self._send_json(200, {
                "id": req.req_id,
                "object": "chat.completion" if is_chat else "text_completion",
                "created": created,
                "model": model_name,
                "choices": [choice],
                "usage": {
                    "prompt_tokens": len(req.prompt_ids),
                    "completion_tokens": len(req.generated),
                    "total_tokens": len(req.prompt_ids) + len(req.generated),
                },
            })

        def do_GET(self) -> None:
            if self.path == "/v1/models":
                self._send_json(200, {"object": "list", "data": [{"id": model_name, "object": "model"}]})
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path not in ("/v1/completions", "/v1/chat/completions"):
                self._send_json(404, {"error": "not found"})
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                is_chat = self.path == "/v1/chat/completions"
                prompt = _format_chat(body["messages"]) if is_chat else body["prompt"]
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                self._send_json(400, {"error": f"bad request: {e}"})
                return

            prompt_ids = engine.tok.encode(prompt, add_special_tokens=False).ids
            sampling = Sampling(
                temperature=float(body.get("temperature", 0.0)),
                top_k=int(body.get("top_k", 0)),
                top_p=float(body.get("top_p", 1.0)),
                min_p=float(body.get("min_p", 0.0)),
                seed=int(body.get("seed", 0)),
            )
            req = engine.submit(prompt_ids, int(body.get("max_tokens", 128)), sampling)
            if body.get("stream", False):
                self._stream(req, is_chat, int(time.time()))
            else:
                self._complete(req, is_chat, int(time.time()))

    return Handler


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Minimal continuous-batching OpenAI server.")
    ap.add_argument("--config", type=Path)
    ap.add_argument("--target")
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--kernels", type=Path)
    ap.add_argument("--weights", nargs="+", type=Path)
    ap.add_argument("--tokenizer", type=Path)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-seqs", type=int, default=16)
    ap.add_argument("--rows", type=int)
    ap.add_argument("--capacity", type=int, default=16384)
    ap.add_argument("--served-model-name")
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args(argv)

    manifest, kernels, weights, tok_path = resolve_target(args)
    t0 = time.perf_counter()
    rt = Executor(
        manifest, kernels, weights,
        capacity_tokens=args.capacity,
        seq_slots=args.max_seqs + 1,
        gpu=args.gpu,
    )
    tok = Tokenizer.from_file(str(tok_path))
    model_name = args.served_model_name or rt.manifest.model
    engine = Engine(rt, tok, _stop_tokens(weights[0]), args.max_seqs, args.rows, not args.eager)
    print(
        f"serving `{model_name}` on http://{args.host}:{args.port} "
        f"(max_seqs={engine.max_seqs}, step_rows={engine.step_rows}, "
        f"loaded in {time.perf_counter() - t0:.2f}s)",
        file=sys.stderr,
    )
    ThreadingHTTPServer((args.host, args.port), make_handler(engine, model_name)).serve_forever()


if __name__ == "__main__":
    main()
