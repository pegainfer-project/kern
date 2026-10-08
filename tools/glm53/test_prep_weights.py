"""CPU-only import checks; do not require PyTorch, CUDA, or model weights."""

import importlib.util
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


SOURCE = Path(__file__).with_name("prep_weights.py")


def load_prep(environment):
    torch = types.ModuleType("torch")
    safetensors = types.ModuleType("safetensors")
    safetensors.safe_open = Mock(side_effect=AssertionError("Import must not read weights"))
    safetensors_torch = types.ModuleType("safetensors.torch")
    safetensors_torch.save_file = Mock(
        side_effect=AssertionError("Import must not write weights")
    )
    spec = importlib.util.spec_from_file_location("glm53_prep_import_check", SOURCE)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(os.environ, environment, clear=True), patch.dict(
        "sys.modules",
        {
            "torch": torch,
            "safetensors": safetensors,
            "safetensors.torch": safetensors_torch,
        },
    ):
        spec.loader.exec_module(module)
    safetensors.safe_open.assert_not_called()
    safetensors_torch.save_file.assert_not_called()
    return module


class PrepImportTests(unittest.TestCase):
    def test_default_paths_without_loading_weights(self):
        module = load_prep({})
        self.assertEqual(module.CKPT, "weights/GLM-5.3-Flash")
        self.assertEqual(Path(module.OUT), Path("glm53-artifacts/derived.safetensors"))

    def test_explicit_paths_without_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory) / "artifact directory"
            checkpoint = Path(directory) / "checkpoint directory"
            module = load_prep({
                "GLM53_ARTIFACTS": str(artifacts),
                "GLM53_CHECKPOINT": str(checkpoint),
            })
            self.assertEqual(Path(module.CKPT), checkpoint)
            self.assertEqual(Path(module.OUT), artifacts / "derived.safetensors")
            self.assertFalse(artifacts.exists())


if __name__ == "__main__":
    unittest.main()
