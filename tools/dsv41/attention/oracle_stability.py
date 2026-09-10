#!/usr/bin/env python3
"""Diagnose one original DSpark anchor without changing its kernel cache."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'trace', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--variant', choices=('plain', 'hooks', 'zero-cache'), default='hooks')
    a = p.parse_args()
    import torch
    torch.set_num_threads(4)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from dsv41.moe.dspark_oracle import DraftOracle
    a.output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    raw_get = DraftOracle.get

    def digest(tensor):
        return hashlib.sha256(tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()).hexdigest()

    def get(self, name):
        value = raw_get(self, name)
        hashes[name] = {'dtype': str(value.dtype), 'shape': list(value.shape), 'sha256': digest(value)}
        return value

    if a.variant == 'hooks':
        DraftOracle.get = get
    meta = json.loads((a.trace / 'trace.json').read_text())
    ids = json.loads((a.trace / 'tokens.json').read_text())
    prompt = meta['prompt_length']
    assert prompt >= 2 and prompt + 5 < 256
    taps = torch.frombuffer(bytearray((a.trace / 'taps.bf16').read_bytes()), dtype=torch.bfloat16).reshape(-1, 15360)
    # Match conditional_acceptance's setup, including tokenizer initialization.
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(a.checkpoint, trust_remote_code=True)
    oracle = DraftOracle(a.checkpoint, max_seq_len=256)
    if a.variant == 'zero-cache':
        for block in oracle.context.mtp:
            block.attn.window_kv_cache.zero_()
    seen = []
    phase = ['seed']

    def dump(prefix, value):
        if isinstance(value, torch.Tensor):
            file = prefix.replace('/', '_') + '.bin'
            data = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
            (a.output / file).write_bytes(data)
            seen.append({'name': prefix, 'file': file, 'dtype': str(value.dtype), 'shape': list(value.shape),
                         'sha256': hashlib.sha256(data).hexdigest()})
        elif isinstance(value, (tuple, list)):
            for i, child in enumerate(value):
                dump(f'{prefix}.{i}', child)

    for i, block in enumerate(oracle.context.mtp):
        if a.variant != 'hooks':
            continue
        block.register_forward_hook(lambda module, inputs, output, i=i: dump(f'{phase[0]}.block{i}', output))
        for name, module in (('attn', block.attn), ('ffn', block.ffn)):
            module.register_forward_hook(lambda module, inputs, output, i=i, name=name: dump(f'{phase[0]}.block{i}.{name}', output))
    def buffers(stage):
        if a.variant != 'hooks':
            return
        for i, block in enumerate(oracle.context.mtp):
            for name, value in block.named_buffers():
                dump(f'{stage}.block{i}.buffer.{name}', value)
    buffers('constructed')
    with torch.inference_mode():
        anchor = torch.tensor([ids[prompt]], device='cuda')
        oracle.draft(anchor, taps[:prompt].unsqueeze(0), 0)
        buffers('seeded')
        phase[0] = 'first'
        first = oracle.draft(anchor, taps[prompt-1:prompt].unsqueeze(0), prompt-1)
        if a.variant == 'hooks':
            dump('first.final', first)
        buffers('first')
        # Repeating the same anchor overwrites the same tentative slots. It
        # must not depend on stale values left by the preceding invocation.
        phase[0] = 'repeat'
        repeat = oracle.draft(anchor, taps[prompt-1:prompt].unsqueeze(0), prompt-1)
        # Plain and zero-cache do no diagnostic CPU copies until both forwards
        # have completed; hashing cannot synchronize or alter their allocation.
        dump('first.final', first)
        dump('repeat.final', repeat)
    loaded_parameters = {f'mtp.{i}.{name}': {'dtype': str(value.dtype), 'shape': list(value.shape), 'sha256': digest(value)}
                         for i, block in enumerate(oracle.context.mtp) for name, value in block.named_parameters()}
    for i, block in enumerate(oracle.context.mtp):
        for eid, fn in block.ffn.expert_cache.items():
            context = inspect.getclosurevars(fn).nonlocals['context']
            for name in ('w1', 'w2', 'w3'):
                weight = getattr(context, name).__defaults__[0]
                prefix = f'mtp.{i}.ffn.{"shared" if eid is None else eid}.{name}'
                for suffix, value in (('weight', weight), ('scale', weight.scale)):
                    loaded_parameters[prefix + '.' + suffix] = {'dtype': str(value.dtype), 'shape': list(value.shape), 'sha256': digest(value)}
    sources = [Path(inspect.getfile(DraftOracle)), a.checkpoint / 'inference/model.py', a.checkpoint / 'inference/kernel.py']
    report = {'variant': a.variant, 'anchor': prompt, 'first_ids': first[0].tolist(), 'repeat_ids': repeat[0].tolist(),
              'repeat_logits_exact': torch.equal(first[1], repeat[1]), 'checkpoint_weights': hashes,
              'loaded_parameters': loaded_parameters,
              'sources': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              'torch': torch.__version__, 'cuda': torch.version.cuda, 'tensors': seen}
    (a.output / 'probe.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ('checkpoint_weights', 'loaded_parameters', 'tensors')}, indent=2))
    oracle.close()


if __name__ == '__main__':
    main()
