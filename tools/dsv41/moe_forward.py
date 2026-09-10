"""Lower routing, quantization and fused EP4 MoE without intermediate copies."""
from .forward import Lowered, scalar
from .head import offset
from .moe import gate, moe
from .programs import buf, call, integer


def forward(pieces, layer, layouts, source, output, *, rows, capacity,
            workspace, experts=384, cubin_dir=None):
    if capacity > 8192 or capacity < 1:
        raise ValueError("MegaMoE slab supports 1..8192 local rows")
    modules, ops, geometry = moe.pieces(experts, cubin_dir, rows)
    names = pieces.add((modules,ops))
    gate_name = f"dsv41_gate_e{experts}"
    gates = pieces.add(gate.pieces(
        rows,experts,cubin_dir,max_tokens=capacity,raw_outputs=True))
    slab, peers, stats = (f"{workspace}.e{experts}.{n}" for n in ("slab","peers","stats"))
    buffers = {
        slab: {"kind":"carry","dtype":"u8","shape":[geometry["slab_bytes"]],"export":True},
        peers: {"kind":"peer","dtype":"u64","shape":[4],"of":slab,"group":"ep"},
        stats: {"kind":"carry","dtype":"i32","shape":[experts//4]},
        output: {"kind":"workspace","dtype":"bf16","shape":[capacity,5120]},
    }
    region = lambda name: offset(slab,geometry["offsets"][name])
    routed, shared = (layouts[layer+".ffn."+n] for n in ("experts","shared_experts"))
    tag = workspace+"."+layer
    calls = [
        call(tag+".gate",gates[gate_name],buf(source),buf(layer+".ffn.gate.weight"),
             buf(layer+".ffn.gate.bias"),region("idx"),region("weights"),scalar(rows)),
        call(tag+".quant",names["dsv41_moe_quant_x"],buf(source),region("x"),region("x_sf"),
             scalar(rows),integer(5120),integer(5120),integer(40),
             region("shared_x_sf"),integer(geometry["shared_sf_rows"])),
        call(tag+".experts",names[f"dsv41_mega_moe_e{experts}"],
             buf(output),buf(stats),scalar(rows),buf(peers),{"rank":"ep"},
             region("l1"),region("l1_sf"),buf(routed["w1"]),buf(routed["w1_sf"]),
             region("l2"),region("l2_sf"),buf(routed["w2"]),buf(routed["w2_sf"]),
             region("x"),region("shared_x_sf"),buf(shared["w1"]),buf(shared["w1_sf"]),
             region("shared_l2"),region("shared_l2_sf"),buf(shared["w2"]),buf(shared["w2_sf"])),
    ]
    return Lowered(buffers,calls)
