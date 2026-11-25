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
`dump.json.partial` and atomically renames it to `dump.json`. A failed payload
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
