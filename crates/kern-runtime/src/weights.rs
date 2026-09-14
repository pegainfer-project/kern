//! Assembling weight buffers out of checkpoint tensors. A weight buffer's
//! `bind` lists the tensors (or rectangles of them) that make it up, laid
//! end to end. Where the tensors are is a [`Tensors`]: the model's own
//! safetensors shards in this process's memory ([`Safetensors`]), or
//! memory another process holds that this device can read (a weight
//! cache's buckets mapped into the context). [`plan`] is the pure part: it
//! turns one buffer's segments into byte copies and checks that they tile
//! the buffer exactly, so the shell that runs them has nothing left to
//! decide.

use std::collections::BTreeMap;

use kern_manifest::types::{Buffer, DType, Rows, TensorSource};

use crate::error::{bail, Error, Result};

/// Bytes a copy can read: a slice of this process's memory, or a span of
/// memory this context addresses on the device side (a mapped allocation
/// of another process, on this tray or across the fabric).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Blob<'a> {
    Host(&'a [u8]),
    Device { ptr: u64, bytes: u64 },
}

impl<'a> Blob<'a> {
    pub fn bytes(&self) -> u64 {
        match self {
            Self::Host(s) => s.len() as u64,
            Self::Device { bytes, .. } => *bytes,
        }
    }

    /// `len` bytes from `at`; `None` past the end.
    pub fn slice(&self, at: u64, len: u64) -> Option<Blob<'a>> {
        if at.checked_add(len)? > self.bytes() {
            return None;
        }
        Some(match *self {
            Self::Host(s) => Self::Host(&s[at as usize..(at + len) as usize]),
            Self::Device { ptr, .. } => Self::Device { ptr: ptr + at, bytes: len },
        })
    }

    pub(crate) fn ptr(&self) -> u64 {
        match self {
            Self::Host(s) => s.as_ptr() as u64,
            Self::Device { ptr, .. } => *ptr,
        }
    }
}

/// One checkpoint tensor: its dtype, its shape and its bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tensor<'a> {
    pub dtype: DType,
    pub shape: Vec<u64>,
    pub data: Blob<'a>,
}

/// A checkpoint: tensors by name. `find` fails, naming the tensor, for a
/// name the checkpoint does not have or a dtype the manifest has no word
/// for; the bytes it hands out are the whole tensor, row-major.
pub trait Tensors {
    fn find(&self, name: &str) -> Result<Tensor<'_>>;
}

/// Safetensors blobs (a model's shards, a draft's next to them) with their
/// headers parsed and every tensor name indexed. Only headers are read;
/// a tensor's bytes are a slice of the blob it is in. A name that appears
/// in more than one blob is ambiguous and refused.
pub struct Safetensors<'a> {
    blobs: Vec<safetensors::SafeTensors<'a>>,
    index: BTreeMap<String, usize>,
}

impl<'a> Safetensors<'a> {
    pub fn parse(blobs: &[&'a [u8]]) -> Result<Self> {
        let blobs = blobs
            .iter()
            .map(|b| safetensors::SafeTensors::deserialize(b))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| Error::WeightArtifact(format!("unparseable safetensors: {e}")))?;
        let mut index = BTreeMap::new();
        let mut names: Vec<(&str, usize)> =
            blobs.iter().enumerate().flat_map(|(i, st)| st.names().into_iter().map(move |n| (n, i))).collect();
        names.sort();
        for (name, i) in names {
            if index.insert(name.to_string(), i).is_some() {
                bail!(WeightArtifact, "tensor `{name}` is in more than one of the {} artifact(s)", blobs.len());
            }
        }
        Ok(Self { blobs, index })
    }
}

impl Tensors for Safetensors<'_> {
    fn find(&self, name: &str) -> Result<Tensor<'_>> {
        let Some(&i) = self.index.get(name) else {
            bail!(WeightArtifact, "tensor `{name}` is in none of the {} artifact(s)", self.blobs.len());
        };
        let t = self.blobs[i].tensor(name).map_err(|e| Error::WeightArtifact(format!("tensor `{name}`: {e}")))?;
        let Some(dtype) = dtype_of(t.dtype()) else {
            bail!(WeightArtifact, "tensor `{name}`: dtype {:?} has no manifest dtype", t.dtype());
        };
        Ok(Tensor { dtype, shape: t.shape().iter().map(|&d| d as u64).collect(), data: Blob::Host(t.data()) })
    }
}

/// One copy: `rows` rows of `width` bytes, `pitch` apart in `src`, landing
/// contiguously at byte `dst` of the buffer. `pitch == width` is a plain
/// memcpy. `src` starts at the first byte copied and ends at the last.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Copy<'a> {
    pub dst: u64,
    pub src: Blob<'a>,
    pub width: u64,
    pub rows: u64,
    pub pitch: u64,
}

/// The manifest dtype a safetensors dtype name (`BF16`, `F8_E4M3`, …)
/// stands for; `None` for one the manifest has no word for.
pub fn dtype_named(name: &str) -> Option<DType> {
    serde_json::from_value::<safetensors::Dtype>(serde_json::Value::String(name.into())).ok().and_then(dtype_of)
}

fn dtype_of(st: safetensors::Dtype) -> Option<DType> {
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
/// buffer's, its tensor's bytes are its shape's worth, its ranges lie
/// inside the tensor read as `[rows, cols]`, and the segments add up to
/// the buffer exactly. A rank-selected tensor or row range is this rank's
/// entry of its table.
pub(crate) fn plan<'a>(
    name: &str,
    b: &Buffer,
    bytes: u64,
    lookup: impl Fn(&str) -> Result<Tensor<'a>>,
    rank: impl Fn(&str) -> Option<u64>,
) -> Result<Vec<Copy<'a>>> {
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
        if t.data.bytes() != rows * cols * elt {
            bail!(WeightArtifact, "{}: {} bytes for shape {:?} of {}", ctx(), t.data.bytes(), t.shape, t.dtype);
        }
        let row_range = match &s.rows {
            None => None,
            Some(Rows::Range(r)) => Some(*r),
            Some(Rows::Ranked { group, ranges }) => Some(*select(&ctx, &rank, group, ranges, "row range")?),
        };
        let [r0, r1] = range(&ctx, "rows", row_range, rows)?;
        let [c0, c1] = range(&ctx, "cols", s.cols, cols)?;
        let (width, pitch, rows) = ((c1 - c0) * elt, cols * elt, r1 - r0);
        let span = if rows == 0 { 0 } else { pitch * (rows - 1) + width };
        // The ranges were checked against the shape, and the bytes are
        // the shape's worth: the rectangle is inside the tensor.
        let src = t.data.slice(r0 * pitch + c0 * elt, span).expect("rectangle inside the tensor");
        copies.push(Copy { dst, src, width, rows, pitch });
        dst += width * rows;
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

    // One byte pool; every tensor's bytes are its offsets into it, so a
    // copy's `src` says where it starts.
    static POOL: [u8; 512] = {
        let mut p = [0u8; 512];
        let mut i = 0;
        while i < 512 {
            p[i] = i as u8;
            i += 1;
        }
        p
    };

    fn lookup(name: &str) -> Result<Tensor<'static>> {
        let (offset, dtype, shape): (usize, _, Vec<u64>) = match name {
            "q" => (0, DType::Bf16, vec![8, 4]),
            "k" => (64, DType::Bf16, vec![2, 4]),
            "fc" => (100, DType::Bf16, vec![4, 8]),
            "conv" => (200, DType::Bf16, vec![6, 1, 4]),
            "a_f32" => (300, DType::F32, vec![4]),
            "short" => (400, DType::Bf16, vec![8, 4]),
            _ => return Err(Error::WeightArtifact(format!("no `{name}`"))),
        };
        let bytes = if name == "short" { 60 } else { shape.iter().product::<u64>() as usize * dtype.bytes() as usize };
        Ok(Tensor { dtype, shape, data: Blob::Host(&POOL[offset..offset + bytes]) })
    }

    fn at(offset: usize, len: usize) -> Blob<'static> {
        Blob::Host(&POOL[offset..offset + len])
    }

    #[test]
    fn a_plan_is_one_copy_per_segment() {
        // Two whole tensors concatenated: contiguous copies end to end.
        let b = weight(DType::Bf16, &[10, 4], vec![seg("q", None, None), seg("k", None, None)]);
        assert_eq!(
            plan("qk", &b, 80, lookup, |_| None).unwrap(),
            [
                Copy { dst: 0, src: at(0, 64), width: 8, rows: 8, pitch: 8 },
                Copy { dst: 64, src: at(64, 16), width: 8, rows: 2, pitch: 8 },
            ]
        );
        // A column block is a strided copy: from its first byte to its last.
        let b = weight(DType::Bf16, &[4, 4], vec![seg("fc", None, Some([4, 8]))]);
        assert_eq!(
            plan("fc.1", &b, 32, lookup, |_| None).unwrap(),
            [Copy { dst: 0, src: at(108, 56), width: 8, rows: 4, pitch: 16 }]
        );
        // A row range skips the leading rows.
        let b = weight(DType::Bf16, &[3, 4], vec![seg("q", Some([5, 8]), None)]);
        assert_eq!(
            plan("q.tail", &b, 24, lookup, |_| None).unwrap(),
            [Copy { dst: 0, src: at(40, 24), width: 8, rows: 3, pitch: 8 }]
        );
        // Trailing axes fold into columns.
        let b = weight(DType::Bf16, &[6, 4], vec![seg("conv", None, None)]);
        assert_eq!(
            plan("conv", &b, 48, lookup, |_| None).unwrap(),
            [Copy { dst: 0, src: at(200, 48), width: 8, rows: 6, pitch: 8 }]
        );
    }

    #[test]
    fn a_plan_refuses_what_does_not_tile_the_buffer() {
        let err = |b: &Buffer, bytes| plan("w", b, bytes, lookup, |_| None).unwrap_err().to_string();
        // Wrong dtype.
        let b = weight(DType::Bf16, &[4], vec![seg("a_f32", None, None)]);
        assert!(err(&b, 8).contains("checkpoint tensor is f32, buffer declares bf16"));
        // Bytes that are not the shape's worth.
        let b = weight(DType::Bf16, &[8, 4], vec![seg("short", None, None)]);
        assert!(err(&b, 64).contains("60 bytes for shape [8, 4]"));
        // Ranges outside the tensor.
        let b = weight(DType::Bf16, &[8, 4], vec![seg("q", Some([4, 9]), None)]);
        assert!(err(&b, 64).contains("rows [4, 9) outside the tensor's 8"));
        let b = weight(DType::Bf16, &[8, 4], vec![seg("q", None, Some([2, 6]))]);
        assert!(err(&b, 64).contains("cols [2, 6) outside the tensor's 4"));
        // Segments that do not add up.
        let b = weight(DType::Bf16, &[8, 4], vec![seg("q", Some([0, 4]), None)]);
        assert!(err(&b, 64).contains("total 32 bytes, the buffer is 64"));
        // A tensor the checkpoint does not have.
        let b = weight(DType::Bf16, &[8, 4], vec![seg("missing", None, None)]);
        assert!(err(&b, 64).contains("no `missing`"));
    }

    #[test]
    fn ranked_segments_take_the_ranks_entry() {
        let mut b = weight(DType::Bf16, &[2, 4], vec![seg("q", None, None)]);
        b.bind[0].tensor = TensorSource::Ranked { group: "ep".into(), tensors: vec!["q".into(), "k".into()] };
        let rank = |g: &str| (g == "ep").then_some(1);
        assert_eq!(
            plan("e", &b, 16, lookup, rank).unwrap(),
            [Copy { dst: 0, src: at(64, 16), width: 8, rows: 2, pitch: 8 }]
        );
        b.bind[0].tensor = TensorSource::Named("q".into());
        b.bind[0].rows = Some(Rows::Ranked { group: "ep".into(), ranges: vec![[0, 2], [2, 4]] });
        assert_eq!(
            plan("e", &b, 16, lookup, rank).unwrap(),
            [Copy { dst: 0, src: at(16, 16), width: 8, rows: 2, pitch: 8 }]
        );
        let e = plan("e", &b, 16, lookup, |_| None).unwrap_err().to_string();
        assert!(e.contains("no rank for `ep`"), "{e}");
        let e = plan("e", &b, 16, lookup, |_| Some(2)).unwrap_err().to_string();
        assert!(e.contains("rank 2 outside row range table"), "{e}");
        // A sharded table's last slice may overlap the one before it, so
        // every rank's slice is the same size.
        b.bind[0].rows = Some(Rows::Ranked { group: "ep".into(), ranges: vec![[0, 3], [3, 6], [5, 8]] });
        let b = weight(DType::Bf16, &[3, 4], b.bind);
        let slice = |r| plan("shard", &b, 24, lookup, move |_| r).map(|c| c[0].src);
        assert_eq!(
            (slice(Some(0)).unwrap(), slice(Some(1)).unwrap(), slice(Some(2)).unwrap()),
            (at(0, 24), at(24, 24), at(40, 24))
        );
    }

    #[test]
    fn checkpoint_dtypes_are_named_the_safetensors_way() {
        assert_eq!(dtype_named("BF16"), Some(DType::Bf16));
        assert_eq!(dtype_named("F8_E4M3"), Some(DType::Fp8E4m3));
        assert_eq!(dtype_named("F8_E8M0"), Some(DType::Fp8E8m0));
        assert_eq!((dtype_named("F64"), dtype_named("bf16")), (None, None));
    }
}
