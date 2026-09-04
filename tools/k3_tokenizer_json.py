#!/usr/bin/env python3
"""Build a `tokenizer.json` for Kimi-K3 from its tiktoken checkpoint.

The checkpoint ships `tiktoken.model` + `tokenization_kimi.py`; the serving
front end loads HF `tokenizers` files only. The conversion keeps the BPE
ranks, the pre-tokenization pattern and the special tokens (ids from
`tokenizer_config.json`'s `added_tokens_decoder`), and is checked against
the checkpoint's own tokenizer on a set of texts before anything is written.

    k3_tokenizer_json.py <checkpoint dir> <out dir>
"""
import json
import shutil
import sys
from pathlib import Path

from tokenizers import AddedToken
from transformers import AutoTokenizer
from transformers.convert_slow_tokenizer import TikTokenConverter

TEXTS = [
    "The capital of France is",
    "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)\n",
    "混合中文 and English, numbers 12345 and punctuation!!! 'don't'  \t\n\n",
    "<|im_user|>user<|im_middle|>hi<|im_end|><|im_assistant|>assistant<|im_middle|>",
    "  leading spaces and trailing   ",
    "Émilie naïve façade — “quotes” … 🙂 ok",
]


def main(src: Path, out: Path) -> None:
    slow = AutoTokenizer.from_pretrained(str(src), trust_remote_code=True)
    special = sorted(slow.special_tokens.items(), key=lambda kv: kv[1])
    base = len(slow.model._mergeable_ranks) if hasattr(slow, "model") else special[0][1]
    assert [i for _, i in special] == list(range(base, base + len(special))), "special ids are not contiguous"
    fast = TikTokenConverter(
        vocab_file=str(src / "tiktoken.model"),
        pattern=slow.pat_str,
        additional_special_tokens=[name for name, _ in special],
    ).converted()
    # The converter registers no special tokens; ids follow the base vocab.
    fast.add_special_tokens([AddedToken(name, special=True, normalized=False) for name, _ in special])
    for name, i in special:
        assert fast.token_to_id(name) == i, (name, i, fast.token_to_id(name))
    for t in TEXTS:
        want = slow.encode(t)
        got = fast.encode(t, add_special_tokens=False).ids
        assert got == want, (t, want, got)
        assert fast.decode(got, skip_special_tokens=False) == t, t
    for name, i in special[:16]:
        assert fast.encode(f"a {name} b", add_special_tokens=False).ids == slow.encode(f"a {name} b"), name
    out.mkdir(parents=True, exist_ok=True)
    fast.save(str(out / "tokenizer.json"))
    cfg = json.loads((src / "tokenizer_config.json").read_text())
    cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
    cfg.pop("auto_map", None)
    (out / "tokenizer_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    for f in ("config.json", "generation_config.json"):
        shutil.copy(src / f, out / f)
    print(f"ok: {len(special)} special tokens from id {base}, {len(TEXTS)} texts round-trip")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
