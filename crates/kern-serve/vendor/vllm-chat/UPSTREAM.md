# vllm-chat source

Vendored from https://github.com/vllm-project/vllm at
`d3e2888c7588fe3a7be93606c7c626dbfd304d2e`, directory `rust/src/chat`.
The upstream Apache-2.0 license is included as `LICENSE`.

Local changes: standalone dependency declarations preserve upstream versions;
the native `deepseek_v41` renderer and selection are added, based on the
released model encoding reference and validated against its text fixtures.
Existing renderer selections retain their implementations.

Reference validation: `tools/dsv41/test_renderer_reference.py` regenerates
`crates/kern-serve/tests/fixtures/deepseek_v41.json` from the released
`encoding.py` and `test_encoding.py`, including published text goldens1/2.
Internal classification/task roles and multimodal encoding are outside this
OpenAI text renderer. Output spaced-DSML tool parsing remains unadapted;
only tool input/history rendering is covered by these changes.
