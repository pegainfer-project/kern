"""Lower checkpoint storage transformations into the manifest once program."""
from copy import deepcopy
import hashlib
import json

from .moe import dense
from .programs import call, buf, integer


class Pieces:
    """Specialized operators, named by what specializes them.

    An op body is a pure function of its geometry (row capacity, matrix shape,
    strides), never of the layer or phase that lowers it, so equal bodies share
    one name: `<entry>.<digest>` where the digest is over the body. Forty layers
    of the same projection become one op; distinct geometry stays distinct
    without any caller choosing a namespace. `fixed` is for the few ops that
    programs address by a literal name.
    """

    def __init__(self):
        self.modules = {}
        self.ops = {}

    def add(self, pieces):
        modules, ops = pieces
        self._modules(modules)
        return {name: self._op(f"{name}.{digest(op)}", op) for name, op in ops.items()}

    def fixed(self, pieces):
        modules, ops = pieces
        self._modules(modules)
        for name, op in ops.items():
            self._op(name, op)

    def _modules(self, modules):
        for name, module in modules.items():
            if name in self.modules and self.modules[name] != module:
                raise ValueError(f"module identity changed while generating: {name}")
            self.modules[name] = deepcopy(module)

    def _op(self, key, op):
        if key in self.ops and self.ops[key] != op:
            raise ValueError(f"specialized op collision: {key}")
        self.ops[key] = deepcopy(op)
        return key


def digest(op):
    return hashlib.sha256(json.dumps(op, sort_keys=True).encode()).hexdigest()[:8]


def dense_scales(raw_buffers, pieces, *, cubin_dir=None, fused_attention=False, bf16_oa=False):
    """Return packed scale carry buffers and once calls for dense projections.

    Routed/shared experts use the MegaMoE packing provider separately.
    O-A is eight grouped matrices whose scales retain the same group order.
    All input scales remain original E8M0 checkpoint bindings.
    """
    if bf16_oa and fused_attention:
        raise ValueError("BF16 O-A requires ordinary BF16 attention output")
    buffers, calls, layouts = {}, [], {}
    for name, weight in sorted(raw_buffers.items()):
        if weight["dtype"] != "fp8e4m3" or not name.endswith(".weight"):
            continue
        if ".experts." in name or ".shared_experts." in name:
            continue
        if len(weight["shape"]) != 2:
            raise ValueError(f"dense projection must be a matrix: {name}")
        prefix = name.removesuffix(".weight")
        scale = prefix + ".scale"
        if scale not in raw_buffers:
            raise ValueError(f"missing checkpoint scale: {scale}")
        total_n, k = weight["shape"]
        groups = 8 if prefix.endswith(".wo_a") else 1
        if total_n % groups or k % 128:
            raise ValueError(f"unsupported grouped dense geometry: {name}")
        n = total_n // groups
        expected = [total_n // 32, k // 32]
        if raw_buffers[scale]["shape"] != expected or raw_buffers[scale]["dtype"] != "fp8e8m0":
            raise ValueError(f"incorrect dense scale storage: {scale}")
        if bf16_oa and groups == 8:
            from pathlib import Path
            from .attention.woa import load
            carry, once, layout = load(pieces, prefix, cubin=Path(cubin_dir)/"dsv41_woa.cubin")
            buffers.update(carry)
            calls.extend(once)
            layouts[prefix] = layout
            continue
        packed = prefix + ".sf"
        buffers[packed] = {"dtype": "i32", "shape": [groups, k // 128, n], "kind": "carry"}
        permutation = ("query" if prefix.endswith(".attn.wq_b") else
                       "output" if prefix.endswith(".attn.wo_a") else None)
        if fused_attention and permutation is not None:
            transformed = prefix + ".fused_weight"
            buffers[transformed] = {"dtype": "fp8e4m3", "shape": weight["shape"], "kind": "carry"}
            names = pieces.add(dense.layout_pieces(permutation, cubin_dir=cubin_dir))
            calls += [
                call(prefix + ".weight_layout", names[f"dsv41_{permutation}_weight_layout"],
                     buf(transformed), buf(name)),
                call(prefix + ".scale_layout", names[f"dsv41_{permutation}_scale_layout"],
                     buf(packed), buf(scale)),
            ]
            layouts[prefix] = {"n": n, "k": k, "groups": groups, "weight": transformed,
                               "scale": packed, "permutation": permutation}
            continue
        # Only the scale-pack op is used in load. Dynamic quant ops get their
        # own token geometry when lowering each forward program.
        modules, ops = dense.prep_pieces(1, n, k, groups=groups, cubin_dir=cubin_dir)
        pack = ops["dsv41_dense_sf_pack"]
        used = {launch["module"] for launch in pack["impl"]["launches"]}
        names = pieces.add(({key: modules[key] for key in used}, {"dsv41_dense_sf_pack": pack}))
        calls.append(call(prefix + ".pack_scale", names["dsv41_dense_sf_pack"],
                          buf(packed), buf(scale), integer(n), integer(k), integer(groups), integer(32)))
        layouts[prefix] = {"n": n, "k": k, "groups": groups, "weight": name, "scale": packed}
    return buffers, calls, layouts


def expert_weights(raw_buffers, pieces, *, cubin_dir=None):
    """Interleave gate/up and pack scales, retaining W2 checkpoint storage.

    Each rank's raw expert buffers are already contiguous after ranked binding.
    The transforms operate on per-expert byte ranges, never crossing ranks.
    """
    from math import prod
    from .moe import prep
    from .forward import selected
    from .head import offset

    buffers, calls, layouts = {}, [], {}
    prefixes = sorted(n.removesuffix(".w1.weight") for n in raw_buffers
                      if n.endswith(".w1.weight") and (".experts." in n or ".shared_experts." in n))
    for prefix in prefixes:
        shared = ".shared_experts" in prefix
        row_group, packed_per_byte = (32, 1) if shared else (1, 2)
        w1, w3, w2 = (prefix + f".w{i}.weight" for i in (1, 3, 2))
        s1, s3, s2 = (prefix + f".w{i}.scale" for i in (1, 3, 2))
        count = 1 if shared else raw_buffers[w1]["shape"][0]
        n, k = 2304, 5120
        expected = {
            w1: [n, k//packed_per_byte], w3: [n, k//packed_per_byte],
            w2: [k, n//packed_per_byte],
            s1: [n//row_group,k//32], s3: [n//row_group,k//32],
            s2: [k//row_group,n//32],
        }
        for name, shape in expected.items():
            if raw_buffers[name]["shape"] != (shape if shared else [count,*shape]):
                raise ValueError(f"incorrect expert checkpoint shape: {name}")
        out = prefix + ".packed"
        packed = {
            "w13": ("u8", [count,2*n,k//packed_per_byte]),
            "w13_sf": ("i32", [count,k//128,2*n]),
            "w2_sf": ("i32", [count,n//128,k]),
        }
        for suffix, (dtype, shape) in packed.items():
            buffers[out+"."+suffix] = {"dtype":dtype,"shape":shape,"kind":"carry"}
        mode = "shared" if shared else "routed"
        up = prep.pieces(n,k,row_group,True,cubin_dir)
        down = prep.pieces(k,n,row_group,False,cubin_dir)
        interleave_name = f"dsv41_interleave_gate_up_{mode}"
        sfup_name = f"dsv41_pack_expert_sf_{mode}_gate_up"
        sfdown_name = f"dsv41_pack_expert_sf_{mode}_down"
        interleave = pieces.add(selected(up,interleave_name))[interleave_name]
        sfup = pieces.add(selected(up,sfup_name))[sfup_name]
        sfdown = pieces.add(selected(down,sfdown_name))[sfdown_name]
        for expert in range(count):
            weight = lambda name: offset(name,expert*prod(expected[name]))
            target = lambda suffix: offset(out+"."+suffix,
                expert*prod(packed[suffix][1][1:])*(4 if suffix.endswith("_sf") else 1))
            tag = f"{prefix}.{expert}.prepare"
            calls += [
                call(tag+".gate_up",interleave,target("w13"),weight(w1),weight(w3),
                     integer(n),integer(k//packed_per_byte)),
                call(tag+".gate_up_sf",sfup,target("w13_sf"),weight(s1),weight(s3),
                     integer(2*n),integer(k),integer(row_group),integer(1)),
                call(tag+".down_sf",sfdown,target("w2_sf"),weight(s2),weight(s2),
                     integer(k),integer(n),integer(row_group),integer(0)),
            ]
        layouts[prefix] = {"w1":out+".w13","w1_sf":out+".w13_sf",
                           "w2":w2,"w2_sf":out+".w2_sf","count":count}
    return buffers,calls,layouts
