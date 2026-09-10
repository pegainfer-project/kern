"""Generate a full layer-0 integration manifest using original checkpoint binds.

This diagnostic joins metadata, attention, mHC and EP4 MoE in one program.
It is not the final serving model. Input embedding expands to the initial four
HC copies; outputs are the block residual and next pre mix for reference checks.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kern_manifest import SCHEMA_VERSION
from dsv41.loading import Pieces,dense_scales,expert_weights
from dsv41.blocks import Blocks
from dsv41.attention_forward import forward as attention
from dsv41.moe_forward import forward as moe
from dsv41.auxiliary.serving import build,Layout


def generate(raw, *, cubin_dir, auxiliary_cubin, attention_cubin, capacity=128, context=32768, fused_cubin=None):
    layer, mode = "layers.0", "prefill"
    weights = {n:b for n,b in raw.items() if n.startswith(layer+".")}
    pieces = Pieces()
    dense_buffers,dense_load,dense_layout = dense_scales(weights,pieces,cubin_dir=cubin_dir,fused_attention=fused_cubin is not None)
    expert_buffers,expert_load,expert_layout = expert_weights(weights,pieces,cubin_dir=cubin_dir)
    # One sequence per rank for this integration probe; rows remain dynamic.
    serving = build(auxiliary_cubin,Layout(max_seqs=1,max_tokens=capacity,max_context=context))
    blocks = Blocks(pieces,prefix=mode,rows="tokens",capacity=capacity,cubin_dir=cubin_dir)
    state,init = blocks.initialize("embedding")
    def attn(src,out):
        return attention(pieces,serving,layer,dense_layout,src,out,mode=mode,
                         prefix=mode+".attn",capacity=capacity,pool_tokens=context,
                         cos_sin="cos_sin",cubin_dir=cubin_dir,auxiliary_cubin=auxiliary_cubin,
                         attention_cubin=attention_cubin,fused_cubin=fused_cubin,
                         fused_cos_sin="cos_sin_split" if fused_cubin else None)
    def ffn(src,out):
        return moe(pieces,layer,expert_layout,src,out,rows="tokens",capacity=capacity,
                   workspace=mode+".moe",cubin_dir=cubin_dir)
    state,lowered = blocks.layer(state,layer,attn,ffn)
    state,finish = blocks.materialize(state,mode+".finish")
    programs = {
        "load":{"once":True,"calls":dense_load+expert_load},
        "prefill":{"batch":{"groups":1,"rows":"tokens"},
                   "calls":serving.prepare(mode,"input_ids")+init+lowered.calls+finish},
    }
    aux = serving.pieces(programs)
    buffers = {**dense_buffers,**expert_buffers,**lowered.buffers,**aux["buffers"],
               "embedding":{"dtype":"bf16","shape":[capacity,5120],"kind":"input"},
               "cos_sin":{"dtype":"f32","shape":[context,64],"kind":"input"}}
    if fused_cubin:
        buffers["cos_sin_split"]={"dtype":"f32","shape":[context,64],"kind":"input"}
    buffers[state.residual]["kind"]="output"
    buffers[state.pre]["kind"]="output"
    used = {a["buf"] for p in programs.values() for c in p["calls"] for a in c["args"] if "buf" in a}
    buffers.update({n:weights[n] for n in used-buffers.keys()})
    # Final pruning keeps only values used by generated ops and peer/domain contracts.
    for b in list(buffers.values()):
        if b.get("of"): used.add(b["of"])
    buffers={n:b for n,b in buffers.items() if n in used}
    # A layer without Engram uses its window pool as the canonical slot domain.
    for spec in buffers.values():
        if spec.get("domain",{}).get("index_into") == "engram_history":
            spec["domain"]["index_into"] = "target.window.0"
    aux["states"].pop("engram_history",None)
    used_ops = {c["op"] for p in programs.values() for c in p["calls"]}
    ops = {n:o for n,o in {**pieces.ops,**aux["ops"]}.items() if n in used_ops}
    used_modules = {l["module"] for o in ops.values() for l in o["impl"]["launches"] if "module" in l}
    modules = dict(pieces.modules)
    for name,module in aux["modules"].items():
        if name in modules and modules[name]!=module:
            raise ValueError(f"module changed during generation: {name}")
        modules[name]=module
    return {"schema_version":SCHEMA_VERSION,"model":"dsv41-layer0-integration",
            "topology":{"groups":{"ep":4}},"vars":aux["vars"],"states":aux["states"],
            "buffers":buffers,"modules":{n:m for n,m in modules.items() if n in used_modules},"ops":ops,"programs":programs}


def main():
    p=argparse.ArgumentParser()
    for n in ("bindings","cubin-dir","auxiliary-cubin","attention-cubin","out"):
        p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--fused-cubin",type=Path)
    a=p.parse_args()
    m=generate(json.loads(a.bindings.read_text())["gpu"],cubin_dir=a.cubin_dir,
               auxiliary_cubin=a.auxiliary_cubin,attention_cubin=a.attention_cubin,fused_cubin=a.fused_cubin)
    a.out.write_text(json.dumps(m,indent=2))
    print("full layer0 calls:",len(m["programs"]["prefill"]["calls"]),flush=True)


if __name__=="__main__":
    main()
