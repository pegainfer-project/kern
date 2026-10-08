"""CPU-only checks of the generator's pinned-artifact resolution."""

import ast
from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.dump, self.hand = self.root / "dump", self.root / "hand"
        self.dump.mkdir()
        self.hand.mkdir()
        # Execute the real resolver without importing the capture-dependent
        # generator module. No capture, checkpoint, or CUDA installation is needed.
        source = Path(__file__).with_name("gen.py")
        tree = ast.parse(source.read_text(), filename=str(source))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == "resolve_modules")
        scope = {
            "hashlib": hashlib,
            "pathlib": __import__("pathlib"),
            "ops_common": SimpleNamespace(DUMP_DIR=self.dump, HAND_DIR=self.hand),
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), scope)
        self.resolve = scope["resolve_modules"]
        self.path = self.hand / "late.cubin"
        self.path.write_bytes(b"pinned test artifact")
        self.digest = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def ops(self, name):
        return {"late": {"impl": {"launches": [{
            "cubin": str(name), "sha256": self.digest, "entry": "kernel",
            "block": [32, 1, 1], "grid": ["seqs", 1, 1], "args": [{"param": 0}],
        }]}}}

    def test_late_launch_and_repeated_pass_preserve_abi(self):
        ops = self.ops(self.path.name)
        expected = deepcopy(ops)
        expected["late"]["impl"]["launches"][0]["cubin"] = str(self.path.resolve())
        self.resolve(ops)
        self.assertEqual(ops, expected)
        self.resolve(ops)
        self.assertEqual(ops, expected)

    def test_absolute_artifact_outside_search_roots(self):
        path = self.root / "compiled.cubin"
        path.write_bytes(self.path.read_bytes())
        ops = self.ops(path)
        self.resolve(ops)
        self.assertEqual(ops["late"]["impl"]["launches"][0]["cubin"], str(path))

    def test_symlink_alias_is_one_artifact(self):
        (self.dump / self.path.name).symlink_to(self.path)
        ops = self.ops(self.path.name)
        self.resolve(ops)
        self.assertEqual(ops["late"]["impl"]["launches"][0]["cubin"], str(self.path))

    def test_missing_wrong_hash_and_distinct_duplicates_fail(self):
        for name in ("missing.cubin", self.root / "missing.cubin"):
            with self.subTest(name=str(name)), self.assertRaises(AssertionError):
                self.resolve(self.ops(name))
        self.path.write_bytes(b"wrong bytes")
        with self.assertRaises(AssertionError):
            self.resolve(self.ops(self.path))
        self.path.write_bytes(b"pinned test artifact")
        (self.dump / self.path.name).write_bytes(self.path.read_bytes())
        with self.assertRaises(AssertionError):
            self.resolve(self.ops(self.path.name))


if __name__ == "__main__":
    unittest.main()
