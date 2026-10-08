"""Access to the mined decode recipe ($GLM53_ARTIFACTS/recipe.json).

The recipe is the ground truth for kernel identity + ABI: 1351 call sites of
one steady-state bs=1 decode forward (sglang TP8 eager, H100). Layer blocks:
layer L occupies sites BOUNDS[L] .. BOUNDS[L+1]-1. The tail (final norm,
lm_head, argmax) is 1345..1350.
"""

import os
import json
import pathlib
import struct

RECIPE_PATH = pathlib.Path(os.environ.get("GLM53_ARTIFACTS", "glm53-artifacts")) / "recipe.json"
_recipe = None


def recipe():
    global _recipe
    if _recipe is None:
        _recipe = json.loads(RECIPE_PATH.read_text())["bs1"]
    return _recipe


def layer_bounds():
    r = recipe()
    starts = [cs["i"] for cs in r if "hc_prenorm" in cs["symbol"]]
    assert len(starts) == 45
    return starts + [1345]


BOUNDS = None


def layer_sites(layer):
    global BOUNDS
    if BOUNDS is None:
        BOUNDS = layer_bounds()
    r = recipe()
    return [cs for cs in r if BOUNDS[layer] <= cs["i"] < BOUNDS[layer + 1]]


def tail_sites():
    return [cs for cs in recipe() if cs["i"] >= 1345]


def scalar(p):
    """decode a mined scalar param: (i32, f32) readings"""
    raw = bytes.fromhex(p["scalar"])
    i = int.from_bytes(raw, "little", signed=False)
    f = struct.unpack("<f", raw[:4])[0] if len(raw) >= 4 else None
    return i, f
