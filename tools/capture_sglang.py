"""capture_sglang.sh 的 payload：SGLang offline Engine 跑 4 条递增 prompt。

必须是真实文件而非 stdin：sglang 用 spawn 起 scheduler 子进程，spawn 要
重新 import __main__，stdin 脚本没有路径会直接挂。
"""

import sys

import sglang as sgl

if len(sys.argv) != 2:
    raise SystemExit("usage: capture_sglang.py CHECKPOINT")
model = sys.argv[1]

PROMPTS = [
    "hi",
    "The harbor master kept a ledger of every ship that wintered in the bay, "
    "noting cargo, crew, and the state of each hull in a cramped, looping hand.",
    "When the observatory finally reopened after the renovation, the docents "
    "discovered that the old refractor had been quietly recollimated by a "
    "retired machinist who lived nearby. He left no note, only a small brass "
    "shim on the pier and a chalked arrow pointing at Polaris. The director "
    "considered filing a complaint, then looked through the eyepiece at Saturn "
    "and decided some trespasses are better rewarded than reported.",
    "The floodplain census took three summers to complete. In the first "
    "summer the crews mapped oxbow lakes and counted heron rookeries from "
    "canoes, losing two clipboards and one outboard motor to the river. In "
    "the second they walked transects through willow thickets, recording "
    "beaver sign, sediment depth, and the stranded hulks of fence posts from "
    "farms abandoned in the forties. The third summer was all reconciliation: "
    "duplicate plots resolved, disputed species calls sent to the herbarium, "
    "and the great argument over whether the channel had truly migrated or "
    "merely braided settled by an afternoon with the 1962 aerial photographs. "
    "The final report ran to four hundred pages, and the appendix everyone "
    "actually read — the one with the flood marks painted on grain elevators "
    "— was written in a single evening by the youngest technician on the crew.",
]


def main():
    llm = sgl.Engine(model_path=model, disable_cuda_graph=True,
                     disable_piecewise_cuda_graph=True,
                     mem_fraction_static=0.45, context_length=4096)
    for p in PROMPTS:
        out = llm.generate(p, {"max_new_tokens": 4, "temperature": 0})
        print(f"output={out['text']!r} "
              f"prompt_tokens={out['meta_info']['prompt_tokens']}", flush=True)
    llm.shutdown()


if __name__ == "__main__":
    main()
