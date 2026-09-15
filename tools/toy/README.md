# tools/toy：不是模型的 manifest，给 serving 路径当门

kern 不认识模型，只认 manifest 的 `fill` / `batch` 声明——所以一份手写的 manifest、
一个几十行的整数 kernel，就能把 kern-serve 的 scheduler、kern-pool、runtime 整条路径
跑起来，runtime 分不出它和 qwen3.8 的区别。toy 的全部意义是**精确**：答案要么对
要么错，没有近平局可以争。

`tools/kernels-src/toy.cu` 是 kernel，`model.py` 是同一套算术的 Python 版（oracle），
`gen.py` 出全部 artifact：

```
python3 tools/toy/gen.py --out target/toy                     # nvcc 两个 cubin、10 份 manifest、tokenizer、权重、kern.toml
python3 tools/e2e/e2e.py --config target/toy/kern.toml --reference tools/toy/model.py --gpus 0,1 --out results/
```

## 算术

每个 token 槽开头是 64 个字 `mark(t, p, w)`（token t、位置 p、第 w 个字的 splitmix64），
后面到槽尾每个字都是头部的函数（`splitmix(head[w & 63] ^ w)`）。一条序列的下一个
token 由它**所有位置的 64 个头字按位置旋转后的和** S 决定：`(S >> 56) % 96 == 0`
出 eos，否则 `S % 256`。旋转让页的顺序进了答案（page table 把两页对调，和就变）；
任何一页被换错、复制旧了、slot 映射错了，和就变，下一个 token 就变；尾部哪个字和
头部对不上（拷贝拷短了、stride 错了）就把 S 毒掉——尾部只有 kernel 自己查，
Python 参考不算它，所以参考便宜。有 `bytes_per_seq` 的 manifest 多一条 line：折叠
C、已折叠的个数，然后整条 line 的尾部同样是 C 的函数；下一个 token 用 S + C；line
的位置不接上一步（旧拷贝、别人的）或尾部对不上就毒掉。投机轮的行 0 是 anchor，后面
每行是上一行预测出的 token，`count` = 1 + S % rows——接受多少不影响答案（答案永远是
greedy 那串），只影响走哪条路；S 均匀，所以接受率该在一半上下（`ACCEPT_PCT`，
e2e 把串行的接受率门在里面）。

词表是 256 个字节 token（id = 字节值）加 eos 256，tokenizer 是无 merge 的 byte-level
BPE，任何文本都能进。权重只有一个 64 项的 salt。

## 五种形状，每种两份

| target | 形状 | 对应的真 target |
|---|---|---|
| `toy-paged` | 4 KiB/token、页 16、prefill 不出 token | qwen3-4b |
| `toy-stateful` | + 1 MiB/seq 的 line、prefill 出 token | qwen3.8-27b |
| `toy-spec` | + 4 行的 `round` 与 `count` | qwen3-4b-dspark |
| `toy-stateful-spec` | line + round | qwen3.8-27b-dflash2 |
| `toy-big` | 64 KiB/token、页 64、64k 上下文：默认预算下 281 GiB 的 state | K3 的尺寸 |

每种再出一份 `-ref`：同样的 manifest，cubin 换成 128 线程块的构建，做 `kern test` 的
A——五种形状的每个 kernel 都过 A/B。尺寸在 `gen.py` 里改；kernel 写满、读满每个槽
和每条 line，runtime 的分页、park 拷贝、slot remap 按真尺寸做。

## 它测什么、不测什么

测 state 的路径：分页、每页 checkpoint、请求末快照、前缀命中、park 到 DRAM 再醒、slot
增长的 remap、投机轮的 count 与回滚、`--rows 1`、chunked prefill、graph 捕获。每个门
都是逐 token 精确的，包括真模型上只能"报不门"的 warm 对 cold 与醒来的对 cold。
不测 kernel 数值和性能——那是 `kern test` 和真模型 e2e 的事（docs/serve.md）。
