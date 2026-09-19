import contextlib
import io
import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch

import check
import import_cubin
import index


class ExternalIndex(unittest.TestCase):
    def setUp(self):
        tmp = self.enterContext(tempfile.TemporaryDirectory())
        root = pathlib.Path(tmp)
        self.enterContext(patch.object(index, "REPO", root / "public"))
        self.enterContext(patch.object(index, "INDEX", root / "private" / "index"))
        self.doc = {
            "family": {"name": "sample", "kind": "cubin", "sm": "sm100f",
                       "upstream": "synthetic", "license": "MIT"},
            "variant": [{"name": "sample", "sha256": "a" * 64, "entries": ["sample"],
                         "launch": {"block": 128}}],
            "pick": [{"op": "op", "shape": "shape", "variant": "sample", "report": "b" * 64}],
        }

    def test_external_read_write_and_manifest_validation(self):
        index.save(self.doc)
        self.assertEqual(index.load("sample"), self.doc)
        module = index.pick("sample", "op", "shape").module
        examples = index.REPO / "examples"
        examples.mkdir(parents=True)
        example = examples / "sample.json"
        example.write_text(json.dumps({"modules": {"sample": {
            "source": module["cubin"], "sha256": module["sha256"]}}}))
        self.assertEqual(check.check(), ([], 1))
        example.write_text(json.dumps({"modules": {"sample": {
            "source": index.source("c" * 64), "sha256": "c" * 64}}}))
        self.assertIn("not in the index", check.check()[0][0])

    def test_missing_and_empty_index_fail(self):
        self.assertIn("KERN_INDEX_DIR", check.check()[0][0])
        with self.assertRaisesRegex(KeyError, "KERN_INDEX_DIR"):
            index.load("sample")
        index.INDEX.mkdir(parents=True)
        self.assertIn("empty", check.check()[0][0])

    def test_import_writes_outside_public_repo(self):
        args = import_cubin.argparse.Namespace(license_file=None)
        with patch.object(import_cubin, "entries", return_value=["sample"]), \
                patch.object(import_cubin.store, "put", return_value="a" * 64), \
                contextlib.redirect_stderr(io.StringIO()):
            import_cubin.import_cubin("sample", "unused.cubin", args=args)
        self.assertEqual(index.variant("sample").sha256, "a" * 64)
        self.assertFalse(index.REPO.exists())


if __name__ == "__main__":
    unittest.main()
