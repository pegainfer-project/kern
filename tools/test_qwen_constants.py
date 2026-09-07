"""Guard the numeric execution contracts and semantic names of the Qwen examples."""
import hashlib
import json
import pathlib
import unittest

from kern_manifest import resolve_constants
from qwen_constants import name_constants

ROOT = pathlib.Path(__file__).resolve().parent.parent
# Canonical JSON hashes before the named-constant migration. Changes to kernel
# wiring, capacities, float literals or even artifact pins require review.
BASELINES = {
    "qwen3-4b-dspark.json": "5e30e2dcc5afbe6c04b6ad7ebdb2098ae947b53af424721ed110baaeee2ce276",
    "qwen3-4b-silu-mined.json": "cc10e2f95754c3e367dbce3a6d00a33a2694215c5a5218667254a1c4de7a0f51",
    "qwen3-4b.json": "d180ee3072bd901f1ca538b27079c97796502e622079386945d52e82619baf40",
    "qwen3.8-27b-dflash2.json": "42bc3f2b23cc470babf1e69014811d01e79e7e195edd14e061888e608b01c968",
    "qwen3.8-27b.json": "43b99b1cf35d15737d288c729bfcde913627f2cebcc020d8206ffc81d28c6f74"
}


class QwenConstantsTests(unittest.TestCase):
    def test_execution_contracts_and_reproducible_generation(self):
        for name, digest in BASELINES.items():
            with self.subTest(name=name):
                named = json.loads((ROOT / "examples" / name).read_text())
                literal = resolve_constants(named)
                canonical = json.dumps(literal, sort_keys=True, separators=(",", ":")).encode()
                self.assertEqual(hashlib.sha256(canonical).hexdigest(), digest)
                self.assertEqual(name_constants(literal), named)
                self.assertEqual(name_constants(named), named)

    def test_equal_numbers_keep_distinct_meanings(self):
        qwen3 = json.loads((ROOT / "examples/qwen3-4b.json").read_text())
        self.assertEqual(qwen3["buffers"]["block_table"]["shape"][-1], "max_blocks_per_seq")
        self.assertEqual(qwen3["buffers"]["x"]["shape"][-1], "hidden_size")
        draft = json.loads((ROOT / "examples/qwen3.8-27b-dflash2.json").read_text())
        for name, expected in [("q_n", "q_dim"), ("gdn_v", "gdn_v_dim"), ("d_qkv", "draft_qkv_dim")]:
            self.assertEqual(draft["buffers"][name]["shape"][-1], expected)
        self.assertEqual(draft["buffers"]["hidden_r"]["shape"], ["max_verify_tokens", "selector_rank"])
        self.assertEqual(draft["buffers"]["model.layers.3.self_attn.q_norm.weight_p1"]["shape"], ["head_dim"])

    def test_reference_names_are_not_substituted(self):
        m = json.loads((ROOT / "examples/minimal.json").read_text())
        m["constants"] = {"x": 64, "scale": 3}
        m["buffers"]["w"]["shape"] = ["x"]
        resolved = resolve_constants(m)
        self.assertEqual(resolved["buffers"]["w"]["shape"], [64])
        self.assertEqual(resolved["programs"]["step"]["calls"][0]["args"][0], {"buf": "x"})
        self.assertEqual(resolved["programs"]["step"]["calls"][0]["args"][-1], {"var": "tokens"})
