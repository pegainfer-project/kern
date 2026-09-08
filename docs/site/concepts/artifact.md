# The model artifact

Kern moves model-specific execution structure out of the runtime and into a
versioned artifact. The artifact is the unit that is verified and shipped.

## Its three parts

### Manifest

The manifest declares typed buffers, storage, variables, device modules,
programs, launch arguments, and the serving protocol. It is data: loading a
manifest does not import model-specific host code.

### Kernels

Programs point at compiled device modules. A module may come from an existing
framework, a compiler, or handwritten CUDA. Its digest is part of the manifest,
so changing the bytes changes the artifact identity.

### Weights

Weights are the model's own checkpoint: a Hugging Face snapshot directory or
a set of Safetensors files, unmodified. Each weight buffer's `bind` lists the
checkpoint tensors (or row / column rectangles of them) that fill it end to
end, so fused projections are declared, not exported. Tables no checkpoint
holds (rope caches, norm offsets) are `carry` buffers a `once` program
computes after load. Where the checkpoint lives belongs in local
configuration or command-line flags, not in the portable manifest.

## What belongs where

| Concern | Manifest | `kern.toml` |
| --- | --- | --- |
| Buffer dtype and shape | Yes | No |
| Program and launch order | Yes | No |
| Module SHA-256 | Yes | No |
| Local kernel directory | No | Yes |
| Local weight files | No | Yes |
| GPU ordinal | No | Yes |
| Reproducible test seed | No | Yes |

This boundary keeps the executable contract portable while allowing each host
to decide where large artifacts live.

## A program, not a plugin

The runtime does not call a model-specific extension point. It validates a
closed declaration and executes its programs. Adding a model therefore changes
the shipped artifact; it does not require teaching the runtime a new model
class.
