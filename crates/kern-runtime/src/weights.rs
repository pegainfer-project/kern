//! Assembling weight buffers out of checkpoint tensors. A weight buffer's
//! `bind` lists the tensors (or rectangles of them) that make it up, laid
//! end to end; the checkpoint is the model's own safetensors shards, read
//! through their headers only. [`plan`] is the pure part: it turns one
//! buffer's segments into byte copies and checks that they tile the buffer
//! exactly, so the shell that runs them has nothing left to decide.

use kern_manifest::types::{Buffer, DType, Rows, TensorSource};

use crate::error::{bail, Error, Result};

/// What a safetensors header says about one tensor: where its bytes are
/// (in which blob, at which offset) and how they are shaped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TensorInfo {
    pub blob: usize,
    pub offset: usize,
    pub dtype: DType,
    pub shape: Vec<u64>,
}

/// One host-to-device copy: `rows` rows of `width` bytes, `pitch` apart in
/// blob `blob` at byte `src`, landing contiguously at byte `dst` of the
/// buffer. `pitch == width` is a plain memcpy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Copy {
    pub dst: u64,
    pub blob: usize,
    pub src: usize,
    pub width: u64,
    pub rows: u64,
    pub pitch: u64,
}

/// Map safetensors' dtype names onto the manifest's; `None` for one the
/// manifest has no word for.
pub(crate) fn dtype_of(st: safetensors::Dtype) -> Option<DType> {
    use safetensors::Dtype as S;
    Some(match st {
        S::BF16 => DType::Bf16,
        S::F16 => DType::F16,
        S::F32 => DType::F32,
        S::F8_E4M3 => DType::Fp8E4m3,
        S::F8_E8M0 => DType::Fp8E8m0,
        S::I8 => DType::I8,
        S::I32 => DType::I32,
        S::U32 => DType::U32,
        S::I64 => DType::I64,
        S::U64 => DType::U64,
        S::U8 => DType::U8,
        _ => return None,
    })
}

/// The copies that assemble weight buffer `name` (`bytes` long) from its
/// segments, looked up by tensor name. Every segment's dtype is the
/// buffer's, its ranges lie inside the tensor read as `[rows, cols]`, and
/// the segments add up to the buffer exactly. A rank-selected tensor or
/// row range is this rank's entry of its table.
pub(crate) fn plan(
    name: &str,
    b: &Buffer,
    bytes: u64,
    lookup: impl Fn(&str) -> Result<TensorInfo>,
    rank: impl Fn(&str) -> Option<u64>,
) -> Result<Vec<Copy>> {
    let elt = b.dtype.bytes();
    let mut copies = Vec::with_capacity(b.bind.len());
    let mut dst = 0u64;
    for (i, s) in b.bind.iter().enumerate() {
        let ctx = || format!("weight `{name}` bind[{i}] (`{}`)", s.tensor);
        let tensor = match &s.tensor {
            TensorSource::Named(name) => name,
            TensorSource::Ranked { group, tensors } => select(&ctx, &rank, group, tensors, "tensor")?,
        };
        let t = lookup(tensor)?;
        if t.dtype != b.dtype {
            bail!(WeightArtifact, "{}: checkpoint tensor is {}, buffer declares {}", ctx(), t.dtype, b.dtype);
        }
        let (rows, cols) = matrix(&t.shape);
        let row_range = match &s.rows {
            None => None,
            Some(Rows::Range(r)) => Some(*r),
            Some(Rows::Ranked { group, ranges }) => Some(*select(&ctx, &rank, group, ranges, "row range")?),
        };
        let [r0, r1] = range(&ctx, "rows", row_range, rows)?;
        let [c0, c1] = range(&ctx, "cols", s.cols, cols)?;
        let (width, pitch) = ((c1 - c0) * elt, cols * elt);
        copies.push(Copy {
            dst,
            blob: t.blob,
            src: t.offset + (r0 * pitch + c0 * elt) as usize,
            width,
            rows: r1 - r0,
            pitch,
        });
        dst += width * (r1 - r0);
    }
    if dst != bytes {
        bail!(
            WeightArtifact,
            "weight `{name}`: its {} bound segment(s) total {dst} bytes, the buffer is {bytes}",
            b.bind.len()
        );
    }
    Ok(copies)
}

/// This rank's entry of a per-rank table (`what` names the entries).
fn select<'a, T>(
    ctx: &dyn Fn() -> String,
    rank: &dyn Fn(&str) -> Option<u64>,
    group: &str,
    table: &'a [T],
    what: &str,
) -> Result<&'a T> {
    let index = rank(group).ok_or_else(|| Error::WeightArtifact(format!("{}: no rank for `{group}`", ctx())))?;
    table
        .get(index as usize)
        .ok_or_else(|| Error::WeightArtifact(format!("{}: rank {index} outside {what} table", ctx())))
}

/// A tensor as a matrix: its first axis by the product of the rest (a
/// vector is one row).
fn matrix(shape: &[u64]) -> (u64, u64) {
    match shape {
        [] => (1, 1),
        [n] => (1, *n),
        [r, rest @ ..] => (*r, rest.iter().product()),
    }
}

fn range(ctx: &dyn Fn() -> String, axis: &str, r: Option<[u64; 2]>, extent: u64) -> Result<[u64; 2]> {
    match r {
        None => Ok([0, extent]),
        Some([from, to]) if to <= extent => Ok([from, to]),
        Some([from, to]) => {
            Err(Error::WeightArtifact(format!("{}: {axis} [{from}, {to}) outside the tensor's {extent}", ctx())))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kern_manifest::types::{BufferKind, Dim, Segment};

    fn weight(dtype: DType, shape: &[u64], bind: Vec<Segment>) -> Buffer {
        Buffer {
            dtype,
            shape: shape.iter().map(|&d| Dim::Const(d)).collect(),
            kind: BufferKind::Weight,
            placement: Default::default(),
            fill: None,
            domain: None,
            export: false,
            of: None,
            group: None,
            bind,
        }
    }

    fn seg(tensor: &str, rows: Option<[u64; 2]>, cols: Option<[u64; 2]>) -> Segment {
        Segment { tensor: tensor.to_string().into(), rows: rows.map(Into::into), cols }
    }

    fn lookup(name: &str) -> Result<TensorInfo> {
        let (blob, offset, dtype, shape) = match name {
            "q" => (0, 0, DType::Bf16, vec![8, 4]),
            "k" => (0, 64, DType::Bf16, vec![2, 4]),
            "fc" => (1, 100, DType::Bf16, vec![4, 8]),
            "conv" => (0, 200, DType::Bf16, vec![6, 1, 4]),
            "a_f32" => (0, 300, DType::F32, vec![4]),
            _ => return Err(Error::WeightArtifact(format!("no `{name}`"))),
        };
        Ok(TensorInfo { blob, offset, dtype, shape })
    }

    #[test]
    fn a_plan_is_one_copy_per_segment() {
        // Two whole tensors concatenated: contiguous copies end to end.
        let b = weight(DType::Bf16, &[10, 4], vec![seg("q", None, None), seg("k", None, None)]);
        assert_eq!(
            plan("qk", &b, 80, lookup, |_| None).unwrap(),
            [
                Copy { dst: 0, blob: 0, src: 0, width: 8, rows: 8, pitch: 8 },
                Copy { dst: 64, blob: 0, src: 64, width: 8, rows: 2, pitch: 8 },
            ]
        );
        // A column block is a strided copy.
        let b = weight(DType::Bf16, &[4, 4], vec![seg("fc", None, Some([4, 8]))]);
        assert_eq!(
            plan("fc.1", &b, 32, lookup, |_| None).unwrap(),
            [Copy { dst: 0, blob: 1, src: 108, width: 8, rows: 4, pitch: 16 }]
        );
        // A row range skips the leading rows.
        let b = weight(DType::Bf16, &[3, 4], vec![seg("q", Some([5, 8]), None)]);
        assert_eq!(
            plan("q.tail", &b, 24, lookup, |_| None).unwrap(),
            [Copy { dst: 0, blob: 0, src: 40, width: 8, rows: 3, pitch: 8 }]
        );
        // Trailing axes fold into columns.
        let b = weight(DType::Bf16, &[6, 4], vec![seg("conv", None, None)]);
        assert_eq!(
            plan("conv", &b, 48, lookup, |_| None).unwrap(),
            [Copy { dst: 0, blob: 0, src: 200, width: 8, rows: 6, pitch: 8 }]
        );
    }

    #[test]
    fn rank_selects_one_source_before_planning_copies() {
        let source = TensorSource::Ranked { group: "ep".into(), tensors: vec!["q".into(), "fc".into()] };
        let b = weight(
            DType::Bf16,
            &[4, 4],
            vec![Segment { tensor: source, rows: Some([0, 4].into()), cols: Some([0, 4]) }],
        );
        let first = plan("local", &b, 32, lookup, |_| Some(0)).unwrap();
        let second = plan("local", &b, 32, lookup, |_| Some(1)).unwrap();
        assert_eq!(first, [Copy { dst: 0, blob: 0, src: 0, width: 8, rows: 4, pitch: 8 }]);
        assert_eq!(second, [Copy { dst: 0, blob: 1, src: 100, width: 8, rows: 4, pitch: 16 }]);
        assert!(plan("local", &b, 32, lookup, |_| None).unwrap_err().to_string().contains("no rank"));
        assert!(plan("local", &b, 32, lookup, |_| Some(2)).unwrap_err().to_string().contains("outside tensor table"));
        // A rank-selected row range shards one tensor: each rank copies its
        // slice, the last one overlapping so every slice is the same size.
        let rows = Rows::Ranked { group: "ep".into(), ranges: vec![[0, 3], [3, 6], [5, 8]] };
        let b = weight(
            DType::Bf16,
            &[3, 4],
            vec![Segment { tensor: "q".to_string().into(), rows: Some(rows), cols: None }],
        );
        let slice = |r| plan("shard", &b, 24, lookup, move |_| r).map(|c| c[0].src);
        assert_eq!((slice(Some(0)).unwrap(), slice(Some(1)).unwrap(), slice(Some(2)).unwrap()), (0, 24, 40));
        assert!(slice(Some(3)).unwrap_err().to_string().contains("outside row range table"));
        assert_eq!(dtype_of(safetensors::Dtype::I8), Some(DType::I8));
        assert_eq!(dtype_of(safetensors::Dtype::F8_E8M0), Some(DType::Fp8E8m0));
    }

    #[test]
    fn mismatches_name_the_segment() {
        let err = |b: Buffer, bytes| plan("w", &b, bytes, lookup, |_| None).unwrap_err().to_string();
        assert!(
            err(weight(DType::F32, &[8, 4], vec![seg("q", None, None)]), 128).contains("is bf16, buffer declares f32")
        );
        assert!(err(weight(DType::Bf16, &[8, 4], vec![seg("q", Some([0, 9]), None)]), 64)
            .contains("rows [0, 9) outside the tensor's 8"));
        assert!(err(weight(DType::Bf16, &[8, 4], vec![seg("q", None, Some([2, 5]))]), 64)
            .contains("cols [2, 5) outside the tensor's 4"));
        assert!(err(weight(DType::Bf16, &[9, 4], vec![seg("q", None, None)]), 72)
            .contains("total 64 bytes, the buffer is 72"));
        assert!(err(weight(DType::Bf16, &[8, 4], vec![seg("ghost", None, None)]), 64).contains("no `ghost`"));
    }
}
