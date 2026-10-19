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
# declare their `axis`, 2026-10-08; schema 6, 2026-10-09; schema 7, 2026-10-09; qwen3.8-27b: decode_batch
# GEMMs on cuBLASLt, 2026-10-19). Changes to kernel wiring,
# capacities, float literals or even artifact pins require review.
BASELINES = {
    "qwen3-4b-dspark.json": "0c53d2d2c114c12584b38912491aaa2bb1e571b2c852c3c39b39ee0dd75bb229",
    "qwen3-4b-silu-mined.json": "512e71644455bc27f6e14af9f3a8cfc9be6d5d01a9e00dd2f6cc6de4a4a9fc3a",
    "qwen3-4b.json": "266ddf976a1e06dabca00aa4f41735240d745d02ce6ac084d5bf627928099c15",
    "qwen3.8-27b-dflash2.json": "487976495a64acb69f13aa4e4be2b7f66aef810e53dae99ab1b05a3dce045e2a",
    "qwen3.8-27b.json": "9c9ecad4ac461b6f439e9f2962e81ca4f7788305fd608630d328c20ef389b07b"
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
