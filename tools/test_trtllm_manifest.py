"""CPU regression gates for converting the two supported model contracts."""
import copy
import json
from pathlib import Path
import unittest

from kern_manifest import resolve_constants
from trtllm_attention import convert, op

ROOT = Path(__file__).resolve().parent.parent
CACHE = dict(source='cache-p64.cubin', sha256='a' * 64,
             entry='reshape_and_cache_kernel_flash')


class TrtllmManifestTests(unittest.TestCase):
    def test_target_page_layout_and_draft_isolation(self):
        # The checked-in qwen3.8-27b.json is already converted (29dbc9c); the
        # draft example still carries the captured attention and page 784.
        for name in ['qwen3.8-27b-dflash2']:
            with self.subTest(model=name):
                source = json.loads((ROOT / 'examples' / f'{name}.json').read_text())
                untouched = copy.deepcopy(source)
                old = resolve_constants(source)
                new = convert(source, CACHE)
                self.assertEqual(source, untouched)
                self.assertEqual(new['buffers']['block_table']['domain']['stride'], 64)
                self.assertEqual(new['buffers']['block_table']['shape'][-1], 4096)
                self.assertEqual(new['states'], old['states'])
                replaced = {'attn', 'attn_batch', 'attn_prefill', 'attn_verify', 'reshape_and_cache'}
                for name, value in old['ops'].items():
                    if name not in replaced:
                        self.assertEqual(new['ops'][name], value)
                cache = new['ops']['reshape_and_cache']['impl']['launches'][0]
                self.assertEqual(cache['args'][9], {'i64': 2097152})
                self.assertEqual(new['modules'][cache['module']]['sha256'], CACHE['sha256'])
                seen_layers = set()
                for program, value in old['programs'].items():
                    for before, after in zip(value['calls'], new['programs'][program]['calls'], strict=True):
                        old_kv = [a for a in before['args'] if a.get('state') == 'kv']
                        new_kv = [a for a in after['args'] if a.get('state') == 'kv']
                        for a, b in zip(old_kv, new_kv, strict=True):
                            layer, within = divmod(a.get('offset', 0), 784 * 4096)
                            seen_layers.add(layer)
                            self.assertEqual(b['offset'], layer * 64 * 4096 + within)
                        if before['op'] in replaced - {'reshape_and_cache'}:
                            launch = new['ops'][after['op']]['impl']['launches'][0]
                            self.assertTrue(launch['entry'].startswith('fmhaSm'))
                            self.assertEqual(after['args'][-2:], [{'var': 'seqs'}, {'var': 'tokens'}])
                            self.assertEqual(after['args'][6], {'buf': 'cu_seqlens_q'})
                self.assertEqual(seen_layers, set(range(16)))
                with self.assertRaises(ValueError):
                    convert(new, CACHE)

    def test_invalid_profiles_fail_before_launch(self):
        for bounds in [dict(splits=0), dict(splits=129), dict(max_context=0)]:
            kwargs = dict(max_rows=64, max_seqs=32, max_context=262144)
            kwargs.update(bounds)
            with self.assertRaises(ValueError):
                op('decode', **kwargs)


if __name__ == '__main__':
    unittest.main()
