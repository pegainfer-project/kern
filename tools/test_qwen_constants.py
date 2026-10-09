"""Guard the numeric execution contracts and semantic names of the Qwen examples."""
import hashlib
import json
import pathlib
import unittest

from kern_manifest import resolve_constants
from qwen_constants import name_constants

ROOT = pathlib.Path(__file__).resolve().parent.parent
# Canonical JSON hashes of the checked-in examples (last: unified attention
# reads the kv state, `inout` -> `in`, 2026-09-16; qwen3-4b: one kv state per
# layer and registry builds of argmax / embedding / weight_prep, 2026-10-07; all: vars
# declare their `axis`, 2026-10-08). Changes to kernel wiring,
# capacities, float literals or even artifact pins require review.
BASELINES = {
    "qwen3-4b-dspark.json": "40349cc8712c2efde92a04072ebe6444ec0718e90989e7d5702e21b07a5d0543",
    "qwen3-4b-silu-mined.json": "014b6caf3b7588ad28c19a8b439357e1b83cf5e42011ba7c363416192d815f5a",
    "qwen3-4b.json": "dac04a3a6f9793e10ee2d89e27d9252998c4b376d2d91cfb42eec1d10de73262",
    "qwen3.8-27b-dflash2.json": "7500f3047dd5bd68a21efccbe04ba83a3f100b727722947cbe84cdf4a7ee632f",
    "qwen3.8-27b.json": "0f404c48c06a310d894dfb6ab8e6600b4badaa65b0c55fdd14e4ef97d5603175"
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
