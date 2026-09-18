//! Tray collectives as NCCL: the ops a manifest names `extern:nccl_*`
//! instead of shipping a kernel, one communicator per topology group.
//!
//! A tray batch's rows are reduced and gathered across a group by the
//! library built for the NVLink domain, not by a peer kernel of our own:
//! for the hundreds of megabytes a prefill chunk moves, NCCL's NVLS path
//! runs at the link's rate where the Lamport kernel of the decode path
//! (`peer_allreduce.cu`) sits at a third of it. The manifest's part is one
//! launch per collective, wired `[send, recv, count, {"rank": g}]`, the
//! rank arg naming the group whose communicator runs it. The driver's part
//! is [`connect_nccl`] when one process holds every rank of the group, or
//! [`Runtime::join_nccl`] with an [`NcclId`] carried over whatever channel
//! the fabric handles ride; either way before the first issue, which
//! refuses a group still unconnected.
//!
//! An all-gather takes `count` elements from every rank and lays them in
//! rank order, so rank `r`'s block lands at `r * count`: a manifest that
//! deals a chunk's rows in equal blocks with the last one short reads the
//! natural row order straight out of the receive buffer, the tail past the
//! chunk being rows nobody reads.

use std::collections::BTreeSet;

use cudarc::nccl::{result, sys};

use crate::compile::{LaunchKind, RVal};
use crate::error::{bail, Error, Result};
use crate::Runtime;

/// The identity of a communicator, minted once per group and handed to
/// every rank joining it.
#[derive(Clone, Copy)]
pub struct NcclId(sys::ncclUniqueId);

impl NcclId {
    pub fn new() -> Result<NcclId> {
        Ok(NcclId(result::get_uniqueid().map_err(nccl)?))
    }

    /// The id as bytes, for a channel that carries bytes.
    #[allow(clippy::unnecessary_cast)] // c_char is u8 on some targets, i8 on others
    pub fn to_bytes(self) -> [u8; 128] {
        self.0.internal.map(|c| c as u8)
    }

    #[allow(clippy::unnecessary_cast)]
    pub fn from_bytes(b: [u8; 128]) -> NcclId {
        NcclId(sys::ncclUniqueId { internal: b.map(|c| c as std::os::raw::c_char) })
    }
}

/// A connected communicator: this rank's end of a group.
pub(crate) struct Comm {
    comm: sys::ncclComm_t,
    world: u64,
}

impl Drop for Comm {
    fn drop(&mut self) {
        let _ = unsafe { result::comm_destroy(self.comm) };
    }
}

/// Which collective an `extern:nccl_*` launch is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Coll {
    AllReduce,
    AllGather,
}

/// The element type of an `extern:nccl_*` launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Elem {
    F32,
    Bf16,
}

impl Elem {
    fn bytes(self) -> u64 {
        match self {
            Elem::F32 => 4,
            Elem::Bf16 => 2,
        }
    }

    fn nccl(self) -> sys::ncclDataType_t {
        match self {
            Elem::F32 => sys::ncclDataType_t::ncclFloat32,
            Elem::Bf16 => sys::ncclDataType_t::ncclBfloat16,
        }
    }
}

/// `extern:nccl_<coll>_<elem>` parsed, or `None` for another extern.
pub(crate) fn extern_op(name: &str) -> Option<(Coll, Elem)> {
    let (coll, elem) = name.strip_prefix("nccl_")?.split_once('_')?;
    let coll = match coll {
        "allreduce" => Coll::AllReduce,
        "allgather" => Coll::AllGather,
        _ => return None,
    };
    let elem = match elem {
        "f32" => Elem::F32,
        "bf16" => Elem::Bf16,
        _ => return None,
    };
    Some((coll, elem))
}

fn nccl(e: result::NcclError) -> Error {
    Error::Nccl(format!("{:?}", e.0))
}

impl Runtime {
    /// The groups the manifest's `extern:nccl_*` launches name; each needs
    /// a communicator before the first issue.
    pub fn nccl_groups(&self) -> BTreeSet<String> {
        self.programs
            .values()
            .flat_map(|p| &p.launches)
            .filter_map(|l| match &l.kind {
                LaunchKind::Nccl { group, .. } => Some(group.clone()),
                _ => None,
            })
            .collect()
    }

    /// Join group `group`'s communicator as this rank. Blocks until every
    /// rank of the group has joined with the same id, unless called inside
    /// an NCCL group call ([`connect_nccl`]).
    pub fn join_nccl(&mut self, group: &str, id: &NcclId) -> Result<()> {
        let Some(&index) = self.ranks.get(group) else {
            bail!(Api, "no topology group `{group}` to join");
        };
        let world = self.manifest.topology.as_ref().and_then(|t| t.groups.get(group)).copied().expect("verified group");
        if self.nccl.contains_key(group) {
            bail!(Api, "group `{group}` already joined");
        }
        self.ctx.bind_to_thread()?;
        // Every channel connects at init: NCCL's default connects a peer the
        // first time a collective needs it, and that first time may be
        // inside a graph capture, where enabling peer access is refused.
        // The protocol is LL128: in a runtime's process the Simple protocol
        // returns wrong sums and gathers past ~100 MB while NCCL's own tests
        // pass at those sizes (docs/lessons.md, 2026-09-18); LL128 is exact
        // and within 20% of Simple's bandwidth. Either variable set by the
        // caller wins.
        for (k, v) in [("NCCL_RUNTIME_CONNECT", "0"), ("NCCL_PROTO", "LL128")] {
            if std::env::var_os(k).is_none() {
                std::env::set_var(k, v);
            }
        }
        let mut comm = std::ptr::null_mut();
        unsafe { result::comm_init_rank(&mut comm, world as i32, id.0, index as i32) }.map_err(nccl)?;
        self.nccl.insert(group.to_string(), Comm { comm, world });
        Ok(())
    }

    /// Nothing launches with a group unconnected.
    pub(crate) fn require_nccl(&self) -> Result<()> {
        let missing: Vec<String> = self.nccl_groups().into_iter().filter(|g| !self.nccl.contains_key(g)).collect();
        if missing.is_empty() {
            return Ok(());
        }
        bail!(Api, "nccl groups not joined yet: {}", missing.join(", "))
    }

    /// Issue one collective on the compute stream: `args` as the launch
    /// wired them, `[send, recv, count, rank]`, `count` in elements per
    /// rank.
    pub(crate) fn collective(&self, coll: Coll, elem: Elem, group: &str, args: &[RVal]) -> Result<()> {
        let [send, recv, count, _] = args else {
            bail!(Manifest, "nccl collective takes [send, recv, count, rank], got {} args", args.len());
        };
        let Some(c) = self.nccl.get(group) else {
            bail!(Api, "nccl group `{group}` not joined");
        };
        let n = count.val;
        let out = match coll {
            Coll::AllReduce => n,
            Coll::AllGather => n * c.world,
        };
        if send.bytes < n * elem.bytes() || recv.bytes < out * elem.bytes() {
            bail!(
                Manifest,
                "nccl {coll:?}: {n} elements per rank, send {} B, recv {} B (needs {} B)",
                send.bytes,
                recv.bytes,
                out * elem.bytes()
            );
        }
        if n == 0 {
            return Ok(());
        }
        let (s, r) = (send.val as *const std::ffi::c_void, recv.val as *mut std::ffi::c_void);
        let stream = self.stream.cu_stream() as sys::cudaStream_t;
        unsafe {
            match coll {
                Coll::AllReduce => {
                    result::all_reduce(s, r, n as usize, elem.nccl(), sys::ncclRedOp_t::ncclSum, c.comm, stream)
                }
                Coll::AllGather => result::all_gather(s, r, n as usize, elem.nccl(), c.comm, stream),
            }
        }
        .map_err(nccl)?;
        Ok(())
    }
}

/// Connect group `group` across `rts`, every rank of it in this process:
/// one id, every runtime joining inside one NCCL group call so the
/// blocking inits complete together.
pub fn connect_nccl(group: &str, rts: &mut [Runtime]) -> Result<()> {
    let id = NcclId::new()?;
    result::group_start().map_err(nccl)?;
    let joined = rts.iter_mut().try_for_each(|rt| rt.join_nccl(group, &id));
    result::group_end().map_err(nccl)?;
    joined
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_extern_name_is_a_collective_and_an_element_type_or_nothing() {
        assert_eq!(extern_op("nccl_allreduce_f32"), Some((Coll::AllReduce, Elem::F32)));
        assert_eq!(extern_op("nccl_allgather_bf16"), Some((Coll::AllGather, Elem::Bf16)));
        assert_eq!(
            (extern_op("nccl_reduce_f32"), extern_op("nccl_allreduce_f16"), extern_op("cublas_bf16_tn_f32")),
            (None, None, None)
        );
    }
}
