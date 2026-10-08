"""CPU-only manifest-policy tests; no capture, checkpoint or CUDA required."""

import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.glm53 import fa3_schedule as policy


ROOT = Path(__file__).resolve().parents[2]


def source_fixture():
    # Reuse the public captured ABI. The clones below are structural test
    # inputs, not runnable MTP models or a substitute for the GPU control.
    full = json.loads((ROOT / "examples/glm53-flash-4l.json").read_text())
    op = copy.deepcopy(full["ops"]["dsa_attn"])
    for name in ("sem", "nmb", "nsd", "vbi"):
        op["impl"]["scratch"][name]["shape"] = [32]
    modules = {launch["module"] for launch in op["impl"]["launches"]}
    return {
        "model": full["model"],
        "vars": {"tokens": {"max": 32}, "seqs": {"max": 16}},
        "ops": {name: copy.deepcopy(op) for name in policy.OPS},
        "modules": {name: full["modules"][name] for name in modules},
        "programs": {}, "buffers": {}, "states": {},
    }


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.source = source_fixture()

    def test_public_fixture_pins_the_complete_preparation_abi(self):
        prep = self.source["ops"]["dsa_attn"]["impl"]["launches"][0]
        self.assertEqual(policy.prepare_abi_sha256(prep), policy.PREPARE_ABI_SHA256)
        self.assertEqual(prep["args"][12], {"i32": 132})
        self.assertEqual(prep["block"], [32, 1, 1])
        self.assertIn("kernelILi1ELb1E", prep["entry"])

    def test_only_three_preparation_fields_change_per_op(self):
        before = copy.deepcopy(self.source)
        result = policy.transform(self.source)
        self.assertEqual(self.source, before)
        for name in policy.OPS:
            old = self.source["ops"][name]["impl"]["launches"][0]
            new = result["ops"][name]["impl"]["launches"][0]
            self.assertEqual(new["block"], [64, 1, 1])
            self.assertEqual(new["args"][12], {"i32": 4224})
            self.assertEqual(new["entry"],
                             old["entry"].replace("kernelILi1ELb1E", "kernelILi2ELb1E"))
            new["block"], new["args"][12], new["entry"] = (
                old["block"], old["args"][12], old["entry"])
        self.assertEqual(result, before)

    def test_module_aliases_do_not_define_artifact_identity(self):
        names = list(self.source["modules"])
        for index, name in enumerate(names):
            alias = f"fa3_module_{index}"
            self.source["modules"][alias] = self.source["modules"].pop(name)
            for op in self.source["ops"].values():
                for launch in op["impl"]["launches"]:
                    if launch["module"] == name:
                        launch["module"] = alias
        self.assertEqual(policy.transform(self.source)["modules"], self.source["modules"])

    def test_rejects_wrong_model_and_row_bounds(self):
        mutations = [
            ("model", "another-model"), ("tokens", 16), ("tokens", 64),
            ("tokens", 32.0), ("seqs", 0), ("seqs", 33), ("seqs", True),
        ]
        for key, value in mutations:
            source = copy.deepcopy(self.source)
            if key == "model":
                source[key] = value
            else:
                source["vars"][key]["max"] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                policy.transform(source)

    def test_rejects_missing_or_additional_attention_ops(self):
        del self.source["ops"]["mtp16_dsa_attn"]
        with self.assertRaises(ValueError):
            policy.transform(self.source)
        self.source = source_fixture()
        self.source["ops"]["unexpected_dsa_attn"] = copy.deepcopy(self.source["ops"]["dsa_attn"])
        with self.assertRaises(ValueError):
            policy.transform(self.source)

    def test_rejects_hidden_preparation_clone(self):
        self.source["ops"]["different_name"] = copy.deepcopy(self.source["ops"]["dsa_attn"])
        with self.assertRaisesRegex(ValueError, "additional preparation"):
            policy.transform(self.source)

    def test_rejects_each_changed_module_hash(self):
        for name in self.source["modules"]:
            source = copy.deepcopy(self.source)
            source["modules"][name]["sha256"] = "0" * 64
            with self.subTest(module=name), self.assertRaises(ValueError):
                policy.transform(source)

    def test_rejects_each_narrow_scratch_and_wrong_dtype(self):
        for op in policy.OPS:
            for key in ("sem", "nmb", "nsd", "vbi"):
                for field, value in (("shape", [16]), ("dtype", "f32")):
                    source = copy.deepcopy(self.source)
                    source["ops"][op]["impl"]["scratch"][key][field] = value
                    with self.subTest(op=op, key=key, field=field), self.assertRaises(ValueError):
                        policy.transform(source)

    def test_rejects_changed_preparation_abi_and_partial_repairs(self):
        for field, value in (("grid", [2, 1, 1]), ("block", [64, 1, 1]),
                             ("params", []), ("args", []), ("shared_mem", 4)):
            source = copy.deepcopy(self.source)
            source["ops"]["mtp32_dsa_attn"]["impl"]["launches"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                policy.transform(source)
        for index in (0, 9, 12, 13, 24):
            source = copy.deepcopy(self.source)
            source["ops"]["dsa_attn"]["impl"]["launches"][0]["args"][index] = {"i32": 4224}
            with self.subTest(argument=index), self.assertRaises(ValueError):
                policy.transform(source)

    def test_rejects_reapplication(self):
        with self.assertRaises(ValueError):
            policy.transform(policy.transform(self.source))

    def test_rejects_incomplete_contract(self):
        for key in ("model", "ops", "vars", "modules"):
            source = copy.deepcopy(self.source)
            del source[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                policy.transform(source)
        for bad in (None, [], {"ops": None}):
            with self.subTest(source=bad), self.assertRaises(ValueError):
                policy.transform(bad)


class FileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = source_fixture()
        self.path = self.root / "source.json"
        self.path.write_text(json.dumps(self.source))
        self.out = self.root / "candidate.json"

    def test_actual_module_bytes_are_hashed(self):
        for index, module in enumerate(self.source["modules"].values()):
            data = f"synthetic artifact {index}".encode()
            path = self.root / f"{index}.cubin"
            path.write_bytes(data)
            module["source"] = path.name
            module["sha256"] = hashlib.sha256(data).hexdigest()
        policy.verify_modules(self.source, self.root)
        path.write_bytes(b"wrong artifact")
        with self.assertRaises(ValueError):
            policy.verify_modules(self.source, self.root)
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            policy.verify_modules(self.source, self.root)

    def test_cli_checks_artifacts_then_writes_separate_candidate(self):
        before = self.path.read_bytes()
        with patch.object(policy, "verify_modules") as verify, contextlib.redirect_stdout(io.StringIO()):
            policy.main([str(self.path), "--out", str(self.out)])
        verify.assert_called_once_with(self.source, self.root)
        self.assertEqual(json.loads(self.out.read_text()), policy.transform(self.source))
        self.assertEqual(self.path.read_bytes(), before)

    def test_rejects_invalid_module_source_paths(self):
        for value in ("", None, 7):
            source = copy.deepcopy(self.source)
            for module in source["modules"].values():
                module["source"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                policy.verify_modules(source, self.root)

    def test_cli_keeps_existing_files_and_rejects_relocation(self):
        self.out.write_text("previous experiment")
        for output in (self.path, self.out, self.root / "other" / "candidate.json"):
            with self.subTest(output=output), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    policy.main([str(self.path), "--out", str(output)])
                self.assertEqual(error.exception.code, 2)
        self.assertEqual(json.loads(self.path.read_text()), self.source)
        self.assertEqual(self.out.read_text(), "previous experiment")
        self.assertFalse((self.root / "other").exists())

    def test_cli_missing_or_bad_artifact_leaves_no_candidate(self):
        for error in (FileNotFoundError("missing cubin"), ValueError("wrong hash")):
            with patch.object(policy, "verify_modules", side_effect=error):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    policy.main([str(self.path), "--out", str(self.out)])
            self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
