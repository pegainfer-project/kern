# Dump artifact format

A dump records bytes at manifest call boundaries for later numerical experiments.
This document defines the file container; the capture metadata describes which
call, storage, and argument each payload represents. States remain opaque bytes.

The version 1 layout is:

```text
dump/
  manifest.json
  dump.json
  blobs/
    000001.bin
    000002.bin
```

`manifest.json` contains the exact original JSON bytes, including whitespace.
Blobs contain unconverted little-endian bytes. Their names are generated IDs,
independent of model names, buffer names, and call labels. A blob's byte count is
not a claim about its logical shape or meaningful elements.

## Completion index

`dump.json` is the completion marker and has these fields:

```json
{
  "version": 1,
  "manifest_sha256": "<SHA-256 of the exact manifest.json bytes>",
  "metadata": {},
  "blobs": [
    {
      "id": "000001",
      "path": "blobs/000001.bin",
      "bytes": 20,
      "sha256": "<SHA-256 of the raw payload>"
    }
  ]
}
```

Hashes use lowercase hexadecimal. Blob IDs are positive decimal numbers padded
to at least six digits; IDs within an artifact are unique. The only accepted
path for an ID is `blobs/<id>.bin`. The index, manifest, blobs, and blob directory
cannot be symlinks. Readers verify the version, paths, lengths, manifest digest,
and every blob digest before returning a completed artifact.

The writer accepts an absent or empty output directory and refuses existing
data. It writes the manifest and blobs first, then writes and synchronizes
`dump.json.partial` and atomically publishes `dump.json` with a same-directory
hard link that cannot replace an existing file, then removes the partial name. A failed payload
read, short read, size overflow, or exceeded budget prevents completion. A
directory with only partial files is not a completed dump. Partial output is
left available for diagnosis; reusing that directory requires explicit cleanup.

The output budget counts the cumulative raw blob payload bytes. The manifest,
JSON index, directory entries, and filesystem overhead are outside that budget.
Payload reads are at most 4 MiB each, including for large state allocations.
Empty blobs are permitted and have the SHA-256 digest of the empty byte string.

## Capture metadata

`metadata` is a JSON value owned by the capture layer. The file I/O layer
preserves it without interpreting tensor types or model semantics. Capture
metadata describes execution identity and actual variables, original call/op
declarations, storage identities and selected byte ranges, and arguments as
references to storage plus byte offsets. Multiple arguments may alias one
storage. Before and after images are different versions of that same storage;
immutable payloads may be referenced by several observations.

A buffer's declaration, its shape at the current variables, its allocation
size, and the logical view consumed by an argument are different facts. Full
allocation bytes can include padding and unwritten regions. Saving bytes does
not establish their validity as tensor elements. The capture layer records
state allocation sizes and original declarations without inventing a semantic
dtype, shape, or stride for opaque state.

Finite JSON numbers do not represent all floating-point bit patterns. A scalar
that needs exact preservation is represented by its declared type and original
bits, for example `{"type": "f32", "bits": "7fc12345"}` for one NaN payload.
Raw blobs preserve NaNs, infinities, signed zeros, and packed formats unchanged.
The file I/O layer does not convert special values or infer typed scalar fields.

Completion states that the files listed in the index were successfully written.
It does not establish that the artifact is self-contained for replay. The
capture metadata separately records missing external inputs, weights, state,
or unsupported pointer bindings in `replay_requirements`. Independently replaying
a call requires those dependencies and restoration of its before images.

## Capturing a real invocation

`kern dump` uses the same single-sequence input staging and generation driver as
`kern run`. Select a protocol program and one zero-based invocation of it, then
choose exact call labels or zero-based call indices:

```sh
kern dump target --program <program> --invocation 1 \
  --call-label <label1>,<label2> --out /tmp/operator-dump
```

The program determines a fixed-row step's shape automatically. A variable-row
program is reached through the prompt's chunks; `--chunk` controls their size.
Invocation counts program executions, not generated tokens. `--steps`, EOS,
and capacity retain the normal generation limits. A selection never reached
before those limits is an error and does not publish a completion index.

All selected calls run in one real invocation. Each call's prefix executes
once, its before images are read, the call executes once, and its after images
are read before later calls can overwrite them. After the last selected call,
the program's remaining calls finish and the dump command exits. Diagnostic
execution is serial and eager, including for programs declaring graphs. It
does not preserve the graph's performance or concurrency behavior. Hosted
states, peer buffers, and topology-bearing manifests are rejected.

By default every directly bound buffer and state is included, including actual
weights, read-only KV, and output pre-images. Each selected storage is saved as
its complete allocation, so static offsets and aliasing remain usable even when
the invocation's shape is smaller than its allocation bound. Immutable weights
are stored once. `--phase before|after|both` defaults to `both`; an after-only
observation declares its missing replay inputs.

To deliberately export a partial allocation, use `--range` with a half-open
byte interval, for example `--range 'state:cache=0:4096'`. Prefix storage names
with `buffer:` or `state:`. Partial observations list the incomplete storage in
their replay requirements; the runtime cannot infer a black-box kernel's full
access extent. `--max-bytes` sets the raw payload budget, default 1073741824
(1 GiB); an exceeded budget fails rather than silently dropping dependencies.

`metadata` records the actual prompt IDs, sampling parameters, capacity and
emitted tokens. Its `observation` holds the program, invocation, cursor, vars,
execution mode and `replay_requirements`. Each entry in `observation.calls`
contains the complete call/op declarations, scalar values, launch conditions,
storage declarations and ranges, and `before`/`after` maps keyed by
`buffer:<name>` or `state:<name>`. Each map value is a blob record. Within a
storage's argument list, `index`, `param` and `offset` preserve interface order,
type/direction and alias relationships. Float scalar values use their original
32-bit integer `bits` together with their declared parameter type.

Call labels are exact matches. Repeated labels select every matching index;
the index distinguishes their records. `--call-index` and `--call-label` are
mutually exclusive, and at least one is required. Existing `--probe-dir`
remains a separate activation-comparison workflow and cannot be combined with
`kern dump`.
