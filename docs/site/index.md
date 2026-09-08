---
layout: home

hero:
  name: "KERN■"
  text: "Ship the model as a program"
  tagline: A small runtime verifies a typed manifest, loads hash-pinned GPU kernels, and executes the program the artifact declares.
  actions:
    - theme: brand
      text: Build and verify
      link: /getting-started/build-and-verify
    - theme: alt
      text: Understand the artifact
      link: /concepts/artifact

features:
  - title: One typed contract
    details: Buffers, state, calls, launch arguments, and serving protocol live in one verified manifest.
  - title: Kernels from anywhere
    details: Load cubins produced by existing frameworks or handwritten code. Every module is selected by SHA-256.
  - title: Model-agnostic runtime
    details: The executor knows how to verify and run a program. Model-specific structure remains in the artifact.
  - title: Evidence for a swap
    details: kern test compares a candidate manifest with a reference through structural, numerical, fuzz, and timing evidence.
---

## Start with the boundary

A deployable model has three parts:

```text
manifest.json     typed buffers, state, programs, and serving protocol
kernels/          compiled device modules pinned by digest
weights           the model's own checkpoint; the manifest says which tensor lands where
```

The runtime refuses an invalid manifest before using it. When it loads an
artifact, it also checks that the supplied modules and their launch ABI match
what the manifest declares.

[Read the overview →](/getting-started/)
