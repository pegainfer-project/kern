"""The toy model's arithmetic, the same as tools/kernels-src/toy.cu, so a
served answer has one right value: a sequence's next token is decided by
the sum of the marks its token slots hold (and, in a manifest with a
per-sequence state, the fold its line carries), which is what the states
under test hold. Byte-level tokens 0..255, eos 256."""

from __future__ import annotations

M = (1 << 64) - 1
WORDS = 64
EOS = 256
EOS_EVERY = 96
VOCAB_BYTES = 256


def splitmix(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & M
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & M
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & M
    return x ^ (x >> 31)


SALT = [splitmix(0x7071 + i) >> 1 for i in range(64)]


def mark(t: int, p: int, w: int) -> int:
    return splitmix(((t * 0x100000001B3) & M) ^ ((p * 0x9E3779B1) & M) ^ ((w * 0xC2B2AE35) & M) ^ SALT[w & 63])


def fold(c: int, t: int, p: int) -> int:
    return (((c << 7) | (c >> 57)) & M) ^ mark(t, p, 0)


def next_of(s: int) -> int:
    return EOS if (s >> 56) % EOS_EVERY == 0 else s % VOCAB_BYTES


def lined(manifest: dict) -> bool:
    """Whether the manifest carries a per-sequence state (its line joins the next token)."""
    return any(s.get("bytes_per_seq") not in (None, 0) for s in manifest.get("states", {}).values())


def generate(ids: list[int], max_tokens: int, manifest: dict) -> list[int]:
    """The greedy continuation of `ids`: up to `max_tokens` tokens, eos ending it (and not returned)."""
    s, c = 0, 0
    for p, t in enumerate(ids):
        s = (s + sum(mark(t, p, w) for w in range(WORDS))) & M
        c = fold(c, t, p)
    out: list[int] = []
    n = len(ids)
    for _ in range(max_tokens):
        t = next_of((s + c) & M if lined(manifest) else s)
        if t == EOS:
            break
        out.append(t)
        s = (s + sum(mark(t, n, w) for w in range(WORDS))) & M
        c = fold(c, t, n)
        n += 1
    return out
