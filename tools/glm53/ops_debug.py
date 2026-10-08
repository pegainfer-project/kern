"""Debug value probes for the GLM-5.3-Flash manifest (value-parity bisect).

Included only when gen.py runs with --probes. Each probe is one CTA that
reduces sum / sumsq / min / max over a flat buffer, prints one line from
rank 0 (device printf -> kern-serve stdout), and writes the stats into the
dbg workspace row `tag` so a host-side readback stays possible. The launch
carries {"rank": "ep"} as the final i32 arg; verify allows rank only on
i32/i64 params.
"""

from .ops_common import a, handwritten  # noqa: F401

BF16X = "in buffer<bf16>"
F32X = "in buffer<f32>"
F32O = "out buffer<f32>"


def ops():
    out = {}
    for name, entry, first in (
        ("debug_probe_bf16", "glm53_probe_bf16", BF16X),
        ("debug_probe_f32", "glm53_probe_f32", F32X),
    ):
        out[name] = {
            "params": [first, F32O, "i32", "i32", "i32"],
            "impl": {"launches": [
                handwritten("glm53_misc", entry,
                            ["buffer", "buffer", "i32", "i32", "i32"],
                            [1024, 1, 1], [1, 1, 1],
                            [a(0), a(1), a(2), a(3), a(4)]),
            ]},
        }
    return out
