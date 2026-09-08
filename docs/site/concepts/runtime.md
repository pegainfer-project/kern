# Runtime and verification

The runtime treats the manifest as untrusted input. It establishes the contract
before execution, then uses that verified structure to allocate and dispatch.

## Verification layers

1. **Wire format:** fields, tagged variants, values, and references must parse
   into the manifest types.
2. **Static semantics:** shapes, offsets, buffer use, program calls, and other
   cross-field invariants must be consistent.
3. **Serving protocol:** the manifest must expose a coherent set of fills,
   axes, state tables, and forward call shapes.
4. **Load-time module checks:** the runtime resolves digest-pinned device
   modules and validates the declared launch interface against the loaded code.

`kern verify` performs the first three layers without a GPU. `kern run` and
`kern test` continue through module loading and execution.

## Execution

The runtime allocates declared storage, binds weights, prepares arguments, and
executes calls in manifest order. Programs may be launched eagerly or captured
as CUDA graphs. Serving behavior is derived from the protocol rather than from
model names embedded in the runtime.

## Failure behavior

Invalid structure is an error. Missing inputs, a digest mismatch, an ABI
mismatch, or an impossible protocol stops the command instead of selecting a
fallback implementation.
