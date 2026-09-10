"""Original BF16 grouped WO_A, with checkpoint dequantization in once.

Groups=8, N=1024 and K=4096. Input is contiguous [rows,8,4096],
output [rows,8,1024]. No activation quantization is introduced here.
"""
import hashlib
from pathlib import Path

from ..forward import Lowered, scalar
from ..programs import buf, call, integer


def definitions(cubin):
    cubin = Path(cubin)
    modules = {'dsv41_woa': {'source': cubin.name, 'sha256': hashlib.sha256(cubin.read_bytes()).hexdigest()}}
    params = ['in buffer<fp8e4m3>', 'in buffer<fp8e8m0>', 'out buffer<bf16>', 'i32', 'i32']
    ops = {'dequant': {'params': params, 'impl': {'launches': [{
        'module': 'dsv41_woa', 'entry': 'dsv41_woa_dequant', 'params': params,
        'args': [{'param': i} for i in range(5)],
        'grid': [131072, 1, 1], 'block': [256, 1, 1],
    }]}}}
    return modules, ops


def load(pieces, prefix, *, cubin):
    """Return carry buffers, once calls and the projection layout.

The caller retains original prefix.weight FP8[8192,4096] and
prefix.scale E8M0[256,128] checkpoint bindings. No transformed files.
"""
    weight = prefix + '.bf16'
    op = pieces.add('woa.load', definitions(cubin))['dequant']
    buffers = {weight: {'dtype': 'bf16', 'kind': 'carry', 'shape': [8192, 4096]}}
    calls = [call(prefix + '.dequant', op, buf(prefix + '.weight'), buf(prefix + '.scale'),
                  buf(weight), integer(8192), integer(4096))]
    layout = {'dtype': 'bf16', 'weight': weight, 'n': 1024, 'k': 4096, 'groups': 8}
    return buffers, calls, layout


def forward(pieces, label, layout, source, output, *, rows, capacity):
    """Eight strided BF16 GEMMs using the Runtime's existing cuBLASLt path.

The 8th extern argument is input row stride; 7th is output row stride.
Each group reads the original 4096-wide head slice and writes its 1024
columns into the row-major concatenation consumed by WO_B.
"""
    if (layout.get('dtype'), layout['groups'], layout['n'], layout['k']) != ('bf16', 8, 1024, 4096):
        raise ValueError('WO_A needs original BF16 [8,1024,4096] weights')
    if layout.get('permutation'):
        raise ValueError('BF16 WO_A expects ordinary attention head order')
    params = ['in buffer<bf16>', 'in buffer<bf16>', 'out buffer<bf16>'] + ['i32'] * 5
    op = {'params': params, 'impl': {'launches': [{'entry': 'extern:cublaslt_bf16_tn'}]}}
    name = pieces.add('woa', ({}, {'group': op}))['group']
    calls = []
    for g in range(8):
        calls.append(call(label + f'.group{g}', name,
                          {'buf': source, 'offset': g * 4096 * 2},
                          {'buf': layout['weight'], 'offset': g * 1024 * 4096 * 2},
                          {'buf': output, 'offset': g * 1024 * 2},
                          scalar(rows), integer(1024), integer(4096), integer(8192), integer(32768)))
    return Lowered({output: {'dtype': 'bf16', 'kind': 'workspace', 'shape': [capacity, 8192]}}, calls)
