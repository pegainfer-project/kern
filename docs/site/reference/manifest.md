# Manifest schema

The manifest is Kern's portable wire contract. The current format is schema
version 4.

## Canonical schema

The JSON Schema is generated from the Rust manifest types and checked into the
repository:

- [Browse the rendered schema](https://kern-baa.pages.dev/schema/)
- [Download `manifest-v4.schema.json`](https://kern-baa.pages.dev/schema/manifest-v4.schema.json)
- [View the source on GitHub](https://github.com/pegainfer-project/kern/blob/master/schema/manifest-v4.schema.json)

Use it for editor completion and generic structural checks. Run `kern verify`
for the cross-field invariants and serving protocol validation implemented by
Kern.

## Top-level responsibilities

| Section | Declares |
| --- | --- |
| `constants` | Readable numeric dimensions expanded before typed validation |
| `vars` | Bounded values supplied for a particular call |
| `buffers` | Typed weights, inputs, outputs, scratch, and state |
| `modules` | Digest-pinned sources of compiled device code |
| `ops` | Typed launch interfaces and implementations |
| `programs` | Ordered calls and their serving shape |

The schema is exhaustive. Unknown variants or fields are errors, and a schema
version is interpreted only by a runtime that implements that version.
