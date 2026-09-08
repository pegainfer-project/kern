# Documentation layout

This directory has two different audiences, separated by the build boundary:

```text
docs/
├── site/          Public, maintained documentation built by VitePress
├── .vitepress/    Navigation, theme, and build configuration for site/
├── qwen38/        Checked-in evidence used by the Qwen3.8 bring-up record
└── *.md           Engineering contracts, experiments, and historical records
```

Only `site/` is published. A Markdown file elsewhere in this directory never
appears on the documentation site unless a public page deliberately links to
and explains it.

The files at the root are retained at their existing paths because source,
scripts, and other records refer to them. New public documentation belongs in
`site/`; new implementation notes should live next to the subsystem they
describe when possible.

## Engineering index

The stable, flat paths below are grouped here by purpose.

### Contracts and implementation

- [Manifest format and verifier](manifest.md)
- [Runtime](runtime.md)
- [Serving](serve.md)
- [`kern test` attestation](attest.md)
- [Speculative decode](spec-decode.md)
- [K3 kernel ABI](k3-kernel-abi.md)
- [Kernel mining](kernel-mining.md)
- [Single-GPU Performance Atlas](performance-atlas.md)

### Architecture and research

- [Original design exploration](design.md)
- [Schema v4 design](v4-design.md)
- [Multi-GPU runtime](multi-gpu.md)
- [Agent workload study](agent-workload.md)
- [DCP microbench](dcp-bench.md)
- [MoE communication survey](moe-comm-survey.md)

### Project records

- [Roadmap](roadmap.md)
- [Lessons](lessons.md)
- [Qwen3.8 bring-up](qwen38-bringup.md)
- [Qwen3.8 bring-up task](qwen38-bringup-prompt.md)
- [`qwen38/`](qwen38/) contains the raw inputs and results referenced by the
  bring-up record.

## Local development

Install the documentation dependencies and run its development server:

```sh
cd docs
npm ci
npm run dev
```

The website build includes the documentation after both dependency sets have
been installed:

```sh
cd website
npm ci
npm run build
```

VitePress checks public internal links during the build. The generated files
are written to `website/dist/docs/` and deployed with the rest of the website.
