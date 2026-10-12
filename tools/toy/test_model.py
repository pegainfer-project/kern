"""The toy reference off the GPU: the carried shapes answer as they always
did, and the dealt model's sum is its twin's however it is dealt — every
position held by exactly one member, every head's carry by exactly one,
a member's share of a twin's line (cut as a push cuts it, by heads) being
the layout the kernels read at that member."""

import random
import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import model  # noqa: E402

TAILS = 255


def twin_line(cs: list[int], at: int) -> bytes:
    """A whole line as toy_heads_fold leaves it: each head's carry and next
    position, then tail rows of a word per head (toy.cu's heads_tail)."""
    n = len(cs)
    head = [w for c in cs for w in (c, at)]
    tail = [model.splitmix(cs[j] ^ r) for r in range(TAILS) for j in range(n)]
    return struct.pack(f"<{len(head) + len(tail)}Q", *head, *tail)


def share(line: bytes, segments: list[tuple[int, int]], of: int, part: int) -> bytes:
    """Member `part`'s cut of a line, as kern-runtime's `Split::copies`: every row's `part`-th 1/`of`."""
    out, at = b"", 0
    for rows, width in segments:
        w = width // of
        out += b"".join(line[at + r * width + part * w : at + r * width + (part + 1) * w] for r in range(rows))
        at += rows * width
    return out


def read(l: bytes, n: int) -> tuple[list[int], list[int], bool]:
    """A share of `n` heads as the kernels read it: carries, next positions, whether every tail word agrees."""
    words = struct.unpack(f"<{len(l) // 8}Q", l)
    cs, at = list(words[0 : 2 * n : 2]), list(words[1 : 2 * n : 2])
    ok = all(words[k] == model.splitmix(cs[(k - 2 * n) % n] ^ ((k - 2 * n) // n)) for k in range(2 * n, len(words)))
    return cs, at, ok


def held(n: int, w: int, m: int) -> int:
    """kern-pool's `Interleave::len`: positions of the first `n` member `m` of `w` holds."""
    return n // w + (n % w > m)


class ModelTest(unittest.TestCase):
    def test_the_carried_shapes_answer_as_before(self):
        paged = {"states": {"kv": {"bytes_per_token": 4096}}}
        lined = {"states": {"kv": {"bytes_per_token": 4096}, "mem": {"bytes_per_seq": 1 << 20}}}
        ids = list(b"A hash map trades memory for speed")
        self.assertEqual((model.heads(paged), model.heads(lined), model.heads({"model": model.DCP})), (0, 1, model.HEADS))
        self.assertEqual(model.generate(ids, 8, paged), [236, 129, 61, 189, 53, 218, 224, 143])
        self.assertEqual(model.generate(ids, 8, lined), [21, 139, 77, 189, 16, 9, 89, 124])

    def test_a_dealt_sum_is_the_twins_for_every_group(self):
        rng = random.Random(7)
        segments = [(1, model.HEADS * 16), (TAILS, model.HEADS * 8)]
        for _ in range(40):
            n = rng.choice([1, 2, 3, 7, 8, 9, 16, 17, 127, 128, 129, 300])
            ids = [rng.randrange(256) for _ in range(n)]
            s, cs = 0, [0] * model.HEADS
            for p, t in enumerate(ids):
                s = (s + model.slot_sum(t, p)) & model.M
                cs = [model.fold(c, t, p, h) for h, c in enumerate(cs)]
            line = twin_line(cs, n)
            for w in (1, 2, 4, 8, 16):
                k = model.HEADS // w
                dealt = sorted(q * w + m for m in range(w) for q in range(held(n, w, m)))
                parts = []
                for m in range(w):
                    got, at, ok = read(share(line, segments, w, m), k)
                    self.assertEqual((got, at, ok), (cs[m * k : (m + 1) * k], [n] * k, True))
                    pos = sum(model.slot_sum(ids[q * w + m], q * w + m) for q in range(held(n, w, m)))
                    parts.append(pos + sum(got))
                self.assertEqual((dealt, sum(parts) & model.M), (list(range(n)), (s + sum(cs)) & model.M))


if __name__ == "__main__":
    unittest.main()
