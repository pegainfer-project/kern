"""Weights straight from the checkpoint: a Qwen manifest's weight buffers
`bind` the Hugging Face tensors they are made of, and whatever is not in
the checkpoint (Gemma norms' `weight + 1`, f32 copies, rope tables, the
constant index tables the captured Triton kernels take as pointers) is a
`carry` buffer the `load` program computes once after load
(tools/once.py, tools/kernels-src/weight_prep.cu). No exported artifact, no torch: `kern run
--weights <HF snapshot dir>`.

Two families, both with their draft:

- `qwen38`: Qwen3.8-27B (`model.language_model.*` in the checkpoint) and
  its DFlash2 draft (`layers.*`, `fc.weight`, `candidate_selector.*`).
- `qwen3`: Qwen3-4B (`model.*`, tied lm_head) and its DSpark draft.

A pass is a pure function of the resolved manifest; the generators call it
last, and the CLI reapplies it to a checked-in example:

    python tools/qwen_weights.py qwen38 --input examples/qwen3.8-27b.json
"""
import argparse
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from kern_manifest import normalize, resolve_constants  # noqa: E402
from once import Once, bind, seg  # noqa: E402
from qwen_constants import name_constants  # noqa: E402


def weights(m):
    return [n for n, b in m["buffers"].items() if b["kind"] == "weight"]


def draft_layers(m):
    """The draft's layer count, read off its per-layer buffers: the fused KV
    buffer is every layer's k_proj then v_proj, in layer order."""
    return sum(1 for n in m["buffers"] if re.fullmatch(r"draft\.layers\.\d+\.self_attn\.o_proj\.weight", n))


def qwen38(m):
    """Qwen3.8-27B: mrope theta 1e7 over 64 of 256 dims (32-wide cos / sin
    tables), Gemma norms, GDN layers with f32 A_log; the DFlash2 draft's
    5-tap combiner is column blocks of one `fc.weight`, its KV projections
    one fused buffer, its rope a full 128-wide cos | sin cache."""
    m = resolve_constants(m)
    once = Once(m)
    P = "model.language_model."
    for name in weights(m):
        b = m["buffers"][name]
        if name == "model.embed_tokens.weight":
            b["bind"] = bind(P + "embed_tokens.weight")
        elif name == "lm_head.weight":
            b["bind"] = bind("lm_head.weight")
        elif name == "model.norm.weight_p1":
            once.p1(name, P + "norm.weight")
        elif name == "rope.cos":
            rows, half = b["shape"]
            once.rope(("rope.cos", 0), ("rope.sin", 0), rows, half, half, 1e7)
        elif name in ("rope.sin", "kv_scales", "draft.kv_scales"):
            if name != "rope.sin":
                once.fill(name, 1.0)
        elif name == "fla.chunk_indices":
            nt = b["shape"][0]
            once.fill(name, 0)
            once.iota(name, nt, stride=2, off=4)   # (0, i) pairs: one sequence, chunk i
        elif name in ("fla.chunk_offsets", "conv.batch_ptr"):
            once.fill(name, 0)
        elif name == "conv.token_chunk_offset":
            once.fill(name, 0)
            once.iota(name, b["shape"][0])
        elif name == "gdn.has_initial":
            once.fill(name, 1)
        elif (lm := re.fullmatch(r"model\.layers\.(\d+)\.(.+)", name)):
            i, rest = lm.groups()
            q = f"{P}layers.{i}."
            if rest.endswith(".weight_p1"):
                once.p1(name, q + rest[: -len("_p1")])
            elif rest == "mlp.gate_up_proj.weight":
                b["bind"] = bind(q + "mlp.gate_proj.weight", q + "mlp.up_proj.weight")
            elif rest == "self_attn.qkv_proj.weight":
                b["bind"] = bind(*(q + f"self_attn.{x}_proj.weight" for x in "qkv"))
            elif rest == "linear_attn.in_proj_qkvz.weight":
                b["bind"] = bind(q + "linear_attn.in_proj_qkv.weight", q + "linear_attn.in_proj_z.weight")
            elif rest == "linear_attn.in_proj_ba.weight":
                b["bind"] = bind(q + "linear_attn.in_proj_b.weight", q + "linear_attn.in_proj_a.weight")
            elif rest == "linear_attn.A_log":
                once.cast(name, q + rest)
            else:
                b["bind"] = bind(q + rest)
        elif name == "draft.rope.cos_sin_cache":
            rows, width = b["shape"]
            once.rope((name, 0), (name, width), rows, width // 2, width, 1e7)
        elif (dm := re.fullmatch(r"draft\.fc\.(\d)\.weight", name)):
            j = int(dm.group(1))
            h = b["shape"][1]
            b["bind"] = [seg("fc.weight", cols=(j * h, (j + 1) * h))]
        elif name == "draft.fused_kv.weight":
            b["bind"] = bind(*(f"layers.{l}.self_attn.{x}_proj.weight" for l in range(draft_layers(m)) for x in "kv"))
        elif name.startswith("draft.selector."):
            leaf = name[len("draft.selector."):]
            b["bind"] = bind("candidate_selector." + {"predecessor": "predecessor_codebook",
                                                      "successor": "successor_codebook"}.get(leaf, leaf))
        elif name in ("draft.hidden_norm.weight", "draft.norm.weight"):
            b["bind"] = bind(name[len("draft."):])
        elif (dl := re.fullmatch(r"draft\.layers\.(\d+)\.(.+)", name)):
            l, rest = dl.groups()
            q = f"layers.{l}."
            if rest == "mlp.gate_up_proj.weight":
                b["bind"] = bind(q + "mlp.gate_proj.weight", q + "mlp.up_proj.weight")
            elif rest == "self_attn.qkv_proj.weight":
                b["bind"] = bind(*(q + f"self_attn.{x}_proj.weight" for x in "qkv"))
            else:
                b["bind"] = bind(q + rest)
        else:
            raise ValueError(f"no binding for weight buffer `{name}`")
    return name_constants(normalize(once.finish()))


def qwen3(m):
    """Qwen3-4B: rope theta 1e6 over the whole 128-dim head (one 128-wide
    cos | sin cache), plain norms, tied lm_head; the DSpark draft has its
    own embedding and lm_head, a 5-tap `fc.weight` and a Markov head."""
    m = resolve_constants(m)
    once = Once(m)
    for name in weights(m):
        b = m["buffers"][name]
        if name == "lm_head.weight":
            b["bind"] = bind("model.embed_tokens.weight")
        elif name == "rope.cos_sin_cache":
            rows, width = b["shape"]
            once.rope((name, 0), (name, width), rows, width // 2, width, 1e6)
        elif name in ("kv_scales", "draft.kv_scales"):
            once.fill(name, 1.0)
        elif name.startswith("model."):
            lm = re.fullmatch(r"model\.layers\.(\d+)\.(.+)", name)
            rest = lm.group(2) if lm else None
            q = f"model.layers.{lm.group(1)}." if lm else ""
            if rest == "mlp.gate_up_proj.weight":
                b["bind"] = bind(q + "mlp.gate_proj.weight", q + "mlp.up_proj.weight")
            elif rest == "self_attn.qkv_proj.weight":
                b["bind"] = bind(*(q + f"self_attn.{x}_proj.weight" for x in "qkv"))
            else:
                b["bind"] = bind(name)
        elif (dm := re.fullmatch(r"draft\.fc\.(\d)\.weight", name)):
            j = int(dm.group(1))
            h = b["shape"][1]
            b["bind"] = [seg("fc.weight", cols=(j * h, (j + 1) * h))]
        elif name == "draft.fused_kv.weight":
            b["bind"] = bind(*(f"layers.{l}.self_attn.{x}_proj.weight" for l in range(draft_layers(m)) for x in "kv"))
        elif name == "draft.markov_w1":
            b["bind"] = bind("markov_head.markov_w1.weight")
        elif name == "draft.markov_w2.weight":
            b["bind"] = bind("markov_head.markov_w2.weight")
        elif (dl := re.fullmatch(r"draft\.layers\.(\d+)\.(.+)", name)):
            l, rest = dl.groups()
            q = f"layers.{l}."
            if rest == "mlp.gate_up_proj.weight":
                b["bind"] = bind(q + "mlp.gate_proj.weight", q + "mlp.up_proj.weight")
            elif rest == "self_attn.qkv_proj.weight":
                b["bind"] = bind(*(q + f"self_attn.{x}_proj.weight" for x in "qkv"))
            else:
                b["bind"] = bind(q + rest)
        elif name.startswith("draft."):
            b["bind"] = bind(name[len("draft."):])
        else:
            raise ValueError(f"no binding for weight buffer `{name}`")
    return name_constants(normalize(once.finish()))


FAMILIES = {"qwen38": qwen38, "qwen3": qwen3}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("family", choices=FAMILIES)
    p.add_argument("--input", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, help="default: rewrite --input")
    args = p.parse_args()
    m = FAMILIES[args.family](json.loads(args.input.read_text()))
    (args.output or args.input).write_text(json.dumps(m, indent=1) + "\n")


if __name__ == "__main__":
    main()
