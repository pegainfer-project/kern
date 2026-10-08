#!/usr/bin/env python3
"""Make a diagnostic MTP manifest with global target top-2 margins.

This does not alter the production MTP manifest. It replaces only verify.head
with a copy that keeps the normal token output and writes one FP32 margin per
verify row to the `spec_margin` output buffer.
"""
from __future__ import annotations
import os
import copy
import hashlib
import json
from pathlib import Path

ART = Path(os.environ.get("GLM53_SPEC_ARTIFACTS", "glm53-artifacts/mtp"))
BASE = ART / 'mtp-v2a.json'
OUT = ART / 'mtp-v2a-margin.json'
CUBIN = ART / 'bundle/spec_head_margin.cubin'

def main():
    m = json.loads(BASE.read_text())
    digest = hashlib.sha256(CUBIN.read_bytes()).hexdigest()
    named = ART / f'bundle/spec_head_margin-{digest[:12]}.cubin'
    if not named.exists():
        named.write_bytes(CUBIN.read_bytes())
    m['modules']['spec_head_margin'] = {
        'source': f'bundle/{named.name}', 'sha256': digest,
    }
    m['buffers']['spec_margin'] = {
        'dtype': 'f32', 'shape': [32], 'kind': 'output',
    }
    m['ops']['spec_head32_margin'] = {
        'params': [
            'in buffer<bf16>', 'in buffer<bf16>', 'out buffer<i64>',
            'out buffer<f32>', 'i32', 'i64',
        ],
        'impl': {
            'scratch': {
                'logits': {'dtype': 'bf16', 'shape': [32, 19360]},
                'choice': {'dtype': 'bf16', 'shape': [32, 8]},
                'gather': {'dtype': 'bf16', 'shape': [8, 32, 8]},
            },
            'launches': [
                {
                    'entry': 'extern:cublaslt_bf16_tn',
                    'params': [
                        'in buffer<bf16>', 'in buffer<bf16>', 'out buffer<bf16>',
                        'i32', 'i32', 'i32',
                    ],
                    'args': [
                        {'param': 0}, {'param': 1}, {'scratch': 'logits'},
                        {'param': 4}, {'i32': 19360}, {'i32': 4096},
                    ],
                },
                {
                    'module': 'spec_head_margin',
                    'entry': 'spec_head_margin_local',
                    'params': ['in buffer<bf16>', 'out buffer<bf16>', 'i32'],
                    'block': [1024, 1, 1], 'grid': ['tokens', 1, 1],
                    'args': [
                        {'scratch': 'logits'}, {'scratch': 'choice'}, {'rank': 'tp'},
                    ],
                },
                {
                    'entry': 'extern:nccl_allgather_bf16',
                    'params': [
                        'in buffer<bf16>', 'out buffer<bf16>', 'i64', 'i32',
                    ],
                    'args': [
                        {'scratch': 'choice'}, {'scratch': 'gather'},
                        {'param': 5}, {'rank': 'tp'},
                    ],
                },
                {
                    'module': 'spec_head_margin',
                    'entry': 'spec_head_margin_global',
                    'params': [
                        'in buffer<bf16>', 'out buffer<i64>', 'out buffer<f32>', 'i32',
                    ],
                    'block': [32, 1, 1], 'grid': ['tokens', 1, 1],
                    'args': [
                        {'scratch': 'gather'}, {'param': 2}, {'param': 3}, {'param': 4},
                    ],
                },
            ],
        },
    }
    m["ops"].pop("spec_head32", None)
    for c in m['programs']['round_k2']['calls']:
        if c.get('label') == 'verify.head':
            c['op'] = 'spec_head32_margin'
            c['args'] = [
                {'buf': 'spec_target_h'}, {'buf': 'lm_head'},
                {'buf': 'verify_tokens'}, {'buf': 'spec_margin'},
                {'var': 'tokens'}, {'expr': {'mul': ['tokens', 8]}},
            ]
            break
    else:
        raise SystemExit('verify.head call not found')
    OUT.write_text(json.dumps(m, indent=2) + '\n')
    print(OUT)
    print('module', named, digest)

if __name__ == '__main__':
    main()