"""Guard the constant resolution every generator shares."""
import json
import pathlib
import unittest

from kern_manifest import resolve_constants

ROOT = pathlib.Path(__file__).resolve().parent.parent


class ResolveConstantsTests(unittest.TestCase):
    def test_reference_names_are_not_substituted(self):
        m = json.loads((ROOT / "examples/minimal.json").read_text())
        m["constants"] = {"x": 64, "scale": 3}
        m["buffers"]["w"]["shape"] = ["x"]
        resolved = resolve_constants(m)
        self.assertEqual(resolved["buffers"]["w"]["shape"], [64])
        self.assertEqual(resolved["programs"]["step"]["calls"][0]["args"][0], {"buf": "x"})
        self.assertEqual(resolved["programs"]["step"]["calls"][0]["args"][-1], {"var": "tokens"})


if __name__ == "__main__":
    unittest.main()
