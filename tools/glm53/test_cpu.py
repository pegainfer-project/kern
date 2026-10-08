#!/usr/bin/env python3
"""CPU-only checks after gen.py emits both manifests. Never loads CUDA.

    python3 tools/glm53/test_cpu.py
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def scalar(arg, call, batch):
    if "param" in arg:
        return scalar(call["args"][arg["param"]], call, batch)
    if "i32" in arg:
        return arg["i32"]
    if "var" in arg:
        assert arg["var"] in ("tokens", "seqs")
        return batch
    if "expr" in arg:
        left, right = arg["expr"]["mul"]
        assert left in ("tokens", "seqs")
        return batch * right
    raise AssertionError(arg)


class CpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifests = [json.loads((ROOT / "examples" / name).read_text()) for name in
                         ("glm53-flash.json", "glm53-flash-probes.json")]

    def test_replicated_contract_and_artifacts(self):
        for m in self.manifests:
            self.assertEqual(m["topology"], {"groups": {"ep": 8, "tp": 8}, "replicated_rows": True})
            self.assertEqual(m["programs"]["decode"]["batch"], {"groups": 16, "rows": 1})
            self.assertFalse(any(b.get("fill") == "blocks" for b in m["buffers"].values()))
            self.assertEqual(m["buffers"]["lg_full"]["shape"], [8, "tokens", 19360])
            for module in m["modules"].values():
                path = ROOT / "examples" / module["source"]
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), module["sha256"])

    def test_dynamic_moe_pair_count(self):
        for m in self.manifests:
            calls = [c for c in m["programs"]["decode"]["calls"] if c["op"] in ("moe_w13", "moe_w2")]
            self.assertEqual(len(calls), 84)
            for call in calls:
                launch = m["ops"][call["op"]]["impl"]["launches"][0]
                self.assertIn("fused_moe_kernel", launch["entry"])
                for batch in (1, 2, 4, 5, 8, 16):
                    self.assertEqual(scalar(launch["args"][12], call, batch), 9 * batch)

    def test_fa3_pdl_off_and_head_cast_batch(self):
        # FA3 mainloop PDL is generation-time optional (GLM53_FA3_PDL=0 disables;
        # default on). The invariant that must hold for every fixture: the
        # prepare/combine launches never carry PDL, and at most one launch in
        # the whole manifest does (gen.py asserts the same).
        for m in self.manifests:
            launches = m["ops"]["dsa_attn"]["impl"]["launches"]
            for l in launches:
                if "device_kernel" not in l["entry"]:
                    self.assertFalse(l.get("pdl", False), l["entry"])
            total_pdl = sum(1 for op in m["ops"].values()
                            for l in op["impl"]["launches"] if l.get("pdl"))
            self.assertLessEqual(total_pdl, 1)
            call = next(c for c in m["programs"]["decode"]["calls"] if c["op"] == "head_cast")
            launch = m["ops"]["head_cast"]["impl"]["launches"][0]
            self.assertEqual(launch["entry"], "glm53_gathered_logits_f32")
            for batch in (1, 2, 4, 5, 8, 16):
                self.assertEqual([scalar(a, call, batch) for a in launch["args"][2:]], [batch, 19360, 8])

    def test_actual_head_indexing_on_cpu(self):
        # Run the actual kernel body as a single-thread grid-stride host
        # function. This checks every index, not CUDA execution or BF16 rounding.
        src = (HERE / "kernels/glm53_misc.cu").read_text()
        body = re.search(r"__global__ void glm53_gathered_logits_f32\(.*?\n}", src, re.S).group()
        body = body.replace("__global__ ", "")
        host = '''#include <cstddef>
#include <vector>
#include <cassert>
using bf16 = float;
float __bfloat162float(float x) { return x; }
struct Dim { unsigned int x; };
Dim blockIdx{0}, threadIdx{0}, blockDim{1}, gridDim{1};
''' + body + '''
int main() {
    const unsigned int shard = 19360, ranks = 8;
    for (unsigned int batch : {1, 2, 4, 5, 8, 16}) {
        std::vector<float> src(batch * ranks * shard), dst(src.size(), -1);
        for (size_t i = 0; i < src.size(); ++i) src[i] = float(i);
        glm53_gathered_logits_f32(src.data(), dst.data(), batch, shard, ranks);
        for (unsigned int b = 0; b < batch; ++b)
            for (unsigned int r = 0; r < ranks; ++r)
                for (unsigned int v = 0; v < shard; ++v)
                    assert(dst[(b * ranks + r) * shard + v] == src[(r * batch + b) * shard + v]);
    }
}
'''
        with tempfile.TemporaryDirectory(prefix="glm53-head-cpu-") as tmp:
            path = Path(tmp) / "head.cc"
            path.write_text(host)
            exe = Path(tmp) / "head"
            subprocess.run(["g++", "-std=c++17", "-O2", str(path), "-o", str(exe)], check=True)
            subprocess.run([str(exe)], check=True)

    def test_kpool_closure_and_page_pitch_source(self):
        src = (HERE / "kernels/glm53_dsa.cu").read_text()
        self.assertIn("int phys = pos & 7;", src)
        self.assertIn("int slot = pos & 3;", src)
        self.assertIn("if (pos_valid && slot == 3)", src)
        self.assertIn("IDX_PAGE_BYTES = 11 * 8448", src)
        self.assertEqual(src.count("page * IDX_PAGE_BYTES"), 2)
        self.assertNotIn("page * 8448", src)
        self.assertEqual([p for p in range(16) if p & 3 == 3], [3, 7, 11, 15])
        for page in (0, 1, 7):
            for layer in range(11):
                base = layer * 8448 + page * 92928
                self.assertTrue(page * 92928 <= base < (page + 1) * 92928)


if __name__ == "__main__":
    unittest.main(verbosity=2)
