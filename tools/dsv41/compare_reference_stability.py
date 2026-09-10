"""Measure original-reference repeated-call variation without imposing bitwise acceptance."""
import argparse
import json
from pathlib import Path

import numpy as np

from compare_target import compare


def tensor(directory, entry):
    path = directory / entry['file']
    if entry['dtype'] == 'bfloat16':
        return (np.fromfile(path, '<u2').astype(np.uint32) << 16).view(np.float32)
    if entry['dtype'] == 'int32':
        return np.fromfile(path, '<i4')
    if entry['dtype'] != 'float32':
        raise ValueError(f"unsupported capture dtype {entry['dtype']}")
    return np.fromfile(path, '<f4')


def boundaries(reference, actual):
    left = json.loads((reference / 'tensors.json').read_text())
    right = json.loads((actual / 'tensors.json').read_text())
    if left.keys() != right.keys():
        raise ValueError('capture boundaries differ')
    result = []
    for name, entry in left.items():
        if entry['shape'] != right[name]['shape']:
            raise ValueError(f'{name}: shapes differ')
        x = tensor(actual, right[name]).astype(np.float64)
        y = tensor(reference, entry).astype(np.float64)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f'{name}: nonfinite capture')
        delta = x - y
        result.append({'name': name, 'changed': int(np.count_nonzero(delta)),
                       'max_abs': float(abs(delta).max()),
                       'relative_squared': float(np.sum(delta * delta) / max(np.sum(y * y), 1e-30))})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--rank', type=int, default=0)
    args = parser.parse_args()
    output = {}
    for phase in ('pure2', 'with_decode', 'after_decode', 'after_reset'):
        if not (args.directory / phase).exists():
            continue
        suffix = Path('prefill-hooks') if args.rank == 0 else Path('prefill-hooks') / f'rank{args.rank}'
        rows = boundaries(args.directory / 'pure1' / suffix, args.directory / phase / suffix)
        report = compare(args.directory / phase, args.directory / 'pure1')
        output[phase] = {'logits': report, 'boundaries': rows}
        print(json.dumps({'phase': phase, 'logits_max_abs': report['max_abs'],
                          'logits_relative_squared': report['max_relative_squared'],
                          'first_changed': next((r for r in rows if r['changed']), None),
                          'first_relative_squared_above_1e-5': next((r for r in rows if r['relative_squared'] > 1e-5), None)}), flush=True)
    (args.directory / ('stability.json' if args.rank == 0 else f'stability-rank{args.rank}.json')).write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()
