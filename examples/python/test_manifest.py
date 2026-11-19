"""Pure-CPU tests for `manifest` (manifest parsing and protocol projection)."""

import json
from pathlib import Path
import unittest

from manifest import Manifest, eval_expr

ROOT = Path(__file__).resolve().parents[2]


class ManifestTest(unittest.TestCase):
    def test_minimal_example_parses(self):
        m = Manifest.load(ROOT / "examples" / "minimal.json")
        self.assertEqual(m.model, "toy")
        self.assertEqual(m.vars_max, {"tokens": 1024})
        self.assertEqual(m.buffer_bytes("x"), 1024 * 64 * 2)
        self.assertEqual(m.state_bytes("kv", capacity_tokens=512, seq_slots=2), 512 * 256)

    def test_constants_expansion_and_expressions(self):
        m = Manifest({
            "schema_version": 7,
            "model": "const-test",
            "constants": {"H": 64, "MAX_T": 128},
            "vars": {"tokens": {"max": "MAX_T", "axis": "rows"}},
            "buffers": {"x": {"dtype": "bf16", "shape": ["tokens", "H"], "kind": "input"}},
            "ops": {},
            "programs": {},
        })
        self.assertEqual(m.vars_max["tokens"], 128)
        self.assertEqual(m.buffer_bytes("x"), 128 * 64 * 2)
        self.assertEqual(eval_expr({"ceil_div": ["tokens", 16]}, {"tokens": 17}), 2)
        self.assertEqual(eval_expr({"mul": [{"add": ["tokens", 1]}, 4]}, {"tokens": 3}), 16)
        self.assertEqual(eval_expr({"clamp": ["tokens", 10, 8]}, {"tokens": 15}), 5)
        with self.assertRaises(ValueError):
            eval_expr({"sub": ["tokens", 1]}, {"tokens": 3})

    def test_constants_do_not_substitute_reference_names(self):
        raw = json.loads((ROOT / "examples" / "minimal.json").read_text())
        raw["constants"] = {"x": 64, "tokens": 32}
        raw["buffers"]["w"]["shape"] = ["x"]
        m = Manifest(raw)
        self.assertEqual(m.buffers["w"]["shape"], [64])
        self.assertEqual(m.programs["step"]["calls"][0]["args"][0], {"buf": "x"})
        self.assertEqual(m.programs["step"]["calls"][0]["args"][-1], {"var": "tokens"})

    def test_unsupported_schema_version_raises(self):
        with self.assertRaises(ValueError):
            Manifest({"schema_version": 6, "model": "old", "buffers": {}, "ops": {}, "programs": {}})

    def test_protocol_projection_for_prefill_decode_and_speculative_round(self):
        m = Manifest({
            "schema_version": 7,
            "model": "spec-test",
            "vars": {
                "tokens": {"max": 1024, "axis": "rows"},
                "seqs": {"max": 16, "axis": "groups"},
                "ctx": {"max": 8192},
            },
            "states": {
                "kv": {"bytes_per_token": 32768},
                "gdn": {"bytes_per_seq": 48 * 1024},
            },
            "buffers": {
                "token_ids": {"dtype": "i64", "shape": ["tokens"], "kind": "input", "fill": "token"},
                "positions": {"dtype": "i64", "shape": ["tokens"], "kind": "input", "fill": "position"},
                "pages": {"dtype": "i32", "shape": ["seqs", 64], "kind": "input", "domain": {"index_into": "kv", "stride": 64}},
                "gdn_lines": {"dtype": "i32", "shape": [48, "seqs"], "kind": "input", "domain": {"index_into": "gdn", "stride": 1024}},
                "next_token": {"dtype": "i64", "shape": ["seqs"], "kind": "output", "fill": "tokens"},
                "verify_tokens": {"dtype": "i64", "shape": ["seqs", 16], "kind": "output", "fill": "tokens"},
                "accepted": {"dtype": "i32", "shape": ["seqs"], "kind": "output", "fill": "count"},
            },
            "ops": {
                "emit_one": {"params": ["out buffer<i64>"], "impl": {"launches": []}},
                "emit_round": {"params": ["out buffer<i64>", "out buffer<i32>"], "impl": {"launches": []}},
            },
            "programs": {
                "load_b": {"once": True, "calls": []},
                "load_a": {"once": True, "calls": []},
                "prefill": {"batch": {"groups": 1, "rows": "tokens", "context": "ctx"}, "calls": [{"op": "emit_one", "args": [{"buf": "next_token"}]}]},
                "decode": {"batch": {"groups": 1, "rows": 1}, "graph": True, "calls": [{"op": "emit_one", "args": [{"buf": "next_token"}]}]},
                "round": {"batch": {"groups": 16, "rows": 16}, "graph": True, "calls": [{"op": "emit_round", "args": [{"buf": "verify_tokens"}, {"buf": "accepted"}]}]},
            },
        })
        self.assertEqual(m.once_programs(), ["load_a", "load_b"])
        p = m.protocol()
        self.assertEqual((p.rows_var, p.groups_var, p.page_unit), ("tokens", "seqs", 64))
        self.assertEqual(p.line_tables[0].lines_per_slot, 48)
        self.assertEqual(p.chunk_forward().name, "prefill")
        self.assertEqual(p.chunk_forward().emits, "next_token")
        self.assertEqual(p.vars_for(p.chunk_forward(), groups=1, rows=128, context=512), {"seqs": 1, "tokens": 128, "ctx": 512})
        self.assertEqual(p.step_forward().name, "round")
        self.assertEqual((p.step_forward().emits, p.step_forward().count), ("verify_tokens", "accepted"))
        self.assertEqual(p.step_forward(rows=1).name, "decode")
        self.assertEqual(p.vars_for(p.step_forward(), groups=1, rows=16), {"seqs": 1, "tokens": 16})
        with self.assertRaises(ValueError):
            p.step_forward(rows=1, groups=2)


if __name__ == "__main__":
    unittest.main()
