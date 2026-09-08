"""Guard the numeric execution contracts and semantic names of the Qwen examples."""
import hashlib
import json
import pathlib
import unittest

from kern_manifest import resolve_constants
from qwen_constants import name_constants

ROOT = pathlib.Path(__file__).resolve().parent.parent
# Canonical JSON hashes of the checked-in examples (last: weights bound to the
# HF checkpoints, 2026-09-08). Changes to kernel wiring, capacities, float
# literals or even artifact pins require review.
BASELINES = {
    "qwen3-4b-dspark.json": "70d8b0988712159356b3dc787a1dc747b15328f68b2ce3a0b25ca8ff1f6e4941",
    "qwen3-4b-silu-mined.json": "e232fc7b61b77ff136c759846d559864accbf33edce4ce93e6fa24e2a1ffee25",
    "qwen3-4b.json": "8bf5550de3239be2db4f240df8e75361dc5352c1d36068a35eb7a3de63f90e7b",
    "qwen3.8-27b-dflash2.json": "e1394ee1dc58a6ac94adb76e62e012332e729e80bd7821f89dc15c3d28d30a76",
    "qwen3.8-27b.json": "1e3d176e6727b1419d5354c2359153e3916bc6fdb2eaae049c68d993a1dd5fb3"
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
        self.assertEqual(qwen3["buffers"]["block_table"]["shape"][-1], "MAX_BLOCKS_PER_SEQ")
        self.assertEqual(qwen3["buffers"]["x"]["shape"][-1], "HIDDEN_SIZE")
        draft = json.loads((ROOT / "examples/qwen3.8-27b-dflash2.json").read_text())
        for name, expected in [("q_n", "Q_DIM"), ("gdn_v", "GDN_V_DIM"), ("d_qkv", "DRAFT_QKV_DIM")]:
            self.assertEqual(draft["buffers"][name]["shape"][-1], expected)
        self.assertEqual(draft["buffers"]["hidden_r"]["shape"], ["MAX_VERIFY_TOKENS", "SELECTOR_RANK"])
        self.assertEqual(draft["buffers"]["model.layers.3.self_attn.q_norm.weight_p1"]["shape"], ["HEAD_DIM"])

    def test_reference_names_are_not_substituted(self):
        m = json.loads((ROOT / "examples/minimal.json").read_text())
        m["constants"] = {"x": 64, "scale": 3}
        m["buffers"]["w"]["shape"] = ["x"]
        resolved = resolve_constants(m)
        self.assertEqual(resolved["buffers"]["w"]["shape"], [64])
        self.assertEqual(resolved["programs"]["step"]["calls"][0]["args"][0], {"buf": "x"})
        self.assertEqual(resolved["programs"]["step"]["calls"][0]["args"][-1], {"var": "tokens"})
