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
# layer and registry builds of argmax / embedding / weight_prep, 2026-10-07). Changes to kernel wiring,
# capacities, float literals or even artifact pins require review.
BASELINES = {
    "qwen3-4b-dspark.json": "c72a03b5c0f817e742befcffa1e8241bc13f92c4f3458743a8ee27ccc6e4d30c",
    "qwen3-4b-silu-mined.json": "3cd79ecd0d0d54c830f842644138b5dafb92c88efcf283709535b096665fe23e",
    "qwen3-4b.json": "b54ac6fa2766667782626ec56b8d02ceb46b250eee7f3cf1a6d79561e5d58e84",
    "qwen3.8-27b-dflash2.json": "134839dd1b25fc8fb5a5cb97391a02576d2624f2f0ccc5431af6e10c9dd679dd",
    "qwen3.8-27b.json": "a9cc996a7548dabc756b30f9ef848459c2a5c2043075b79e11091054d1a30443"
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
