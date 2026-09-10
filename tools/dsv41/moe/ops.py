"""Typed boundary ops; `tokens` is the program's flattened token count."""
from pathlib import Path
import hashlib

def pieces(rows="tokens", cubin_dir=None):
    cubin = (Path(cubin_dir) if cubin_dir else Path(__file__).resolve().parents[3] / 'target/cubins/dsv41') / 'boundary.cubin'
    modules = {'dsv41_moe_boundary': {'source': str((Path(cubin_dir) if cubin_dir else Path('target/cubins/dsv41'))/'boundary.cubin'), 'sha256': hashlib.sha256(cubin.read_bytes()).hexdigest()}}
    signatures = {
        'dsv41_hc_mean': (['out buffer<bf16>', 'in buffer<bf16>', 'i32'], 5120),
        'dsv41_hc_identity': (['out buffer<f32>', 'out buffer<f32>', 'out buffer<f32>', 'i32'], 16),
        'dsv41_swiglu': (['out buffer<bf16>', 'in buffer<bf16>', 'in buffer<bf16>', 'i32'], 2304),
        'dsv41_hc_init': (['out buffer<bf16>', 'in buffer<bf16>', 'i32'], 20480),
        'dsv41_hc_pre': (['out buffer<bf16>', 'in buffer<bf16>', 'in buffer<f32>', 'i32'], 5120),
        'dsv41_hc_post': (['out buffer<bf16>', 'in buffer<bf16>', 'in buffer<bf16>', 'in buffer<f32>', 'in buffer<f32>', 'i32'], 20480),
    }
    ops = {name: {'params': params, 'impl': {'launches': [{'module': 'dsv41_moe_boundary', 'entry': name, 'block': [256, 1, 1], 'grid': [{'ceil_div': [{'mul': [rows, width]}, 256]}, 1, 1]}]}} for name, (params, width) in signatures.items()}
    return modules, ops


def zero(count, cubin_dir=None):
    """`dsv41_zero_u64` over a fixed count of u64 words, for once-at-load clears."""
    modules, _ = pieces(cubin_dir=cubin_dir)
    op = {'params': ['out buffer<u64>', 'i32'], 'impl': {'launches': [{'module': 'dsv41_moe_boundary', 'entry': 'dsv41_zero_u64', 'block': [256, 1, 1], 'grid': [(count + 255) // 256, 1, 1]}]}}
    return modules, {'dsv41_zero_u64': op}
