"""Generate prefill, batch decode and five-draft/six-verify serving programs."""
import argparse
import json
from pathlib import Path
import sys
import hashlib
import shutil

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kern_manifest import SCHEMA_VERSION
from dsv41.loading import Pieces, dense_scales, expert_weights
from dsv41.auxiliary.serving import Layout, build
from dsv41.draft_context import capture, publish
from dsv41.target import forward, head
from dsv41.draft import forward as draft_forward
from dsv41.programs import Stages, assemble, spec_ops


def bundle(manifest, destination, roots):
    """Copy the pinned cubins as `<module>-<sha12>.cubin`, the registry layout.

    The manifest's module sources become those names, so nothing in a
    published manifest points at a build tree; the runtime resolves by sha256.
    """
    destination.mkdir(parents=True,exist_ok=True)
    for name, module in manifest["modules"].items():
        source = Path(module["source"])
        candidates = [source,*(root/source.name for root in roots)]
        match = next((path for path in candidates if path.is_file()
                      and hashlib.sha256(path.read_bytes()).hexdigest() == module["sha256"]),None)
        if match is None:
            raise ValueError(f"cannot find pinned cubin for module {name}")
        output = destination/f"{name}-{module['sha256'][:12]}.cubin"
        if match.resolve() != output.resolve():
            shutil.copyfile(match,output)
        if hashlib.sha256(output.read_bytes()).hexdigest() != module["sha256"]:
            raise ValueError(f"cubin changed while bundling module {name}")
        module["source"] = output.name


def generate(raw, constants, *, cubin_dir, auxiliary_cubin, attention_dir,
             head_cubin, copy_cubin, spec_cubin, capacity=128, max_seqs=16, context=32768):
    pieces = Pieces()
    device_weights = {name:b for name,b in raw.items() if b.get("placement") != "host"}
    dense_buffers, dense_load, dense_layouts = dense_scales(device_weights,pieces,cubin_dir=cubin_dir,fused_attention=True)
    expert_buffers, expert_load, expert_layouts = expert_weights(device_weights,pieces,cubin_dir=cubin_dir)
    serving = build(auxiliary_cubin,Layout(max_tokens=capacity,max_seqs=max_seqs,max_context=context))
    buffers = {**dense_buffers,**expert_buffers,**constants["buffers"]}
    pieces.fixed((constants["modules"],constants["ops"]))
    programs = {"load":{"once":True,"calls":constants["calls"]+dense_load+expert_load}}
    vocab = raw["embed.weight"]["shape"][0]
    buffers["next_token"] = {"kind":"output","dtype":"i64","shape":["seqs"],
                              "fill":"tokens","domain":{"index_into":"embed.weight"}}
    forwards, contexts, commits = {}, {}, []
    for mode in ("prefill","decode","verify"):
        target = forward(pieces,serving,dense_layouts,expert_layouts,mode=mode,ids="verify_ids" if mode=="verify" else "input_ids",
                         capacity=capacity,pool_tokens=context,constants=constants,cubin_dir=cubin_dir,
                         auxiliary_cubin=auxiliary_cubin,attention_cubin=attention_dir/"libdsv41_paged_decode.2.sm_103a.cubin",
                         fused_cubin=attention_dir/"libdsv41_fused_decode.2.sm_103a.cubin",
                         head_cubin=head_cubin,dense_cubin=attention_dir/"paged_indexer.cubin",
                         sparse_cubin=attention_dir/"sparse_indexer.cubin",
                         select_cubin=attention_dir/"libdsv41_select.2.sm_103a.cubin",
                         candidate_cubin=attention_dir/"candidate.cubin",
                         capture_tap=lambda layer,hc: capture(pieces,hc,layer,mode=mode,rows="tokens",
                                                              capacity=capacity,auxiliary_cubin=auxiliary_cubin))
        output = head(pieces,target.normalized,"verify_tokens" if mode=="verify" else "next_token",mode=mode,capacity=capacity,vocab=vocab,
                      head_cubin=head_cubin,copy_cubin=copy_cubin)
        context_stage = publish(pieces,serving,dense_layouts,mode=mode,capacity=capacity,
                                auxiliary_cubin=auxiliary_cubin,cubin_dir=cubin_dir,
                                cos_sin=constants["rope"]["window"]["interleaved"])
        for stage in (target.lowered,output,context_stage):
            buffers.update(stage.buffers)
        forwards[mode] = target.lowered.calls+output.calls
        contexts[mode] = context_stage.calls
        commits.extend(target.commit)
    draft = draft_forward(pieces,serving,dense_layouts,expert_layouts,max_seqs=max_seqs,pool_tokens=context,
                          constants=constants,cubin_dir=cubin_dir,auxiliary_cubin=auxiliary_cubin,
                          attention_cubin=attention_dir/"libdsv41_paged_decode.2.sm_103a.cubin",
                          fused_cubin=attention_dir/"libdsv41_fused_decode.2.sm_103a.cubin",
                          head_cubin=head_cubin,vocab=vocab)
    buffers.update(draft.buffers)
    for name,kind,width in (("draft_ids","workspace",5),("verify_ids","workspace",6),
                            ("draft_tokens","workspace",5),("verify_tokens","output",6)):
        buffers[name] = {"kind":kind,"dtype":"i64","shape":["seqs",width],
                         "domain":{"index_into":"embed.weight"}}
    buffers["verify_tokens"]["fill"] = "tokens"
    buffers["nacc"] = {"kind":"output","dtype":"i32","shape":["seqs"],"fill":"count",
                        "domain":{"min":1,"max":6}}
    pieces.fixed(({"dsv41_spec_round":{"source":spec_cubin.name,
                                       "sha256":hashlib.sha256(spec_cubin.read_bytes()).hexdigest()}},
                  spec_ops("dsv41_spec_round")))
    programs = assemble(Stages(programs["load"]["calls"],forwards["prefill"],forwards["decode"],draft.calls,
                               forwards["verify"],contexts["prefill"],contexts["decode"],contexts["verify"],commits),
                        max_seqs=max_seqs)
    aux = serving.pieces(programs)
    buffers.update(aux["buffers"])
    used = {a["buf"] for p in programs.values() for c in p["calls"] for a in c["args"] if "buf" in a}
    for name in sorted(used - buffers.keys()):
        buffers[name] = raw[name]
    for b in list(buffers.values()):
        if b.get("of"):
            used.add(b["of"])
        target = b.get("domain",{}).get("index_into")
        if target in raw:
            used.add(target)
            buffers.setdefault(target,raw[target])
    used_ops = {c["op"] for p in programs.values() for c in p["calls"]}
    ops = {name:op for name,op in {**pieces.ops,**aux["ops"]}.items() if name in used_ops}
    used_modules = {launch["module"] for op in ops.values() for launch in op["impl"]["launches"] if "module" in launch}
    modules = dict(pieces.modules)
    for name,module in aux["modules"].items():
        if name in modules and modules[name] != module:
            raise ValueError(f"module changed during generation: {name}")
        modules[name] = module
    return {"schema_version":SCHEMA_VERSION,"model":"DeepSeek-V4.1-Flash",
            "topology":{"groups":{"ep":4}},"vars":aux["vars"],"states":aux["states"],
            "buffers":{name:b for name,b in buffers.items() if name in used},
            "modules":{name:m for name,m in modules.items() if name in used_modules},
            "ops":ops,"programs":programs}


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bindings",type=Path)
    source.add_argument("--checkpoint",type=Path,help="read original shard headers directly; no weight export")
    for name in ("constants","cubin-dir","auxiliary-cubin","attention-dir","head-cubin","copy-cubin","spec-cubin","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--capacity",type=int,default=128)
    parser.add_argument("--max-seqs",type=int,default=16)
    parser.add_argument("--context",type=int,default=32768)
    parser.add_argument("--bundle",type=Path,help="copy pinned cubins to a serving artifact directory")
    args = vars(parser.parse_args())
    binding_file, checkpoint = args.pop("bindings"), args.pop("checkpoint")
    if checkpoint is not None:
        from dsv41_weights import inventory
        from dsv41.weights import bindings as bind
        config = json.loads((checkpoint/"config.json").read_text())
        if config.get("model_type") != "deepseek_v41":
            parser.error("expected a deepseek_v41 checkpoint")
        tensors, _ = inventory(checkpoint)
        gpu, host = bind(tensors,config["text_config"],ep=4)
        bindings = {"gpu":gpu,"host":host}
    else:
        bindings = json.loads(binding_file.read_text())
    constants_file = args.pop("constants")
    constants = json.loads(constants_file.read_text())
    output = args.pop("out")
    destination = args.pop("bundle")
    host = {name:dict(spec,placement="host") for name,spec in bindings["host"].items()}
    manifest = generate({**bindings["gpu"],**host},constants,**args)
    if destination is not None:
        bundle(manifest,destination,[args["cubin_dir"],args["attention_dir"],constants_file.parent,
                                     *(args[name].parent for name in ("auxiliary_cubin","head_cubin","copy_cubin","spec_cubin"))])
    output.write_text(json.dumps(manifest,indent=2)+"\n")
    print({name:len(p["calls"]) for name,p in manifest["programs"].items()})


if __name__ == "__main__":
    main()
