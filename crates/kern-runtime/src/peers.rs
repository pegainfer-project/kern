//! Peers: what this rank exports and what it maps from the others.
//!
//! A manifest with a `topology` is SPMD: every rank loads it with its own
//! [`Topology`](crate::Topology), its index in each group. `export`
//! buffers and every fixed state are virtual-memory allocations every
//! rank can map (a fabric handle where the device offers one, the
//! allocation handle itself otherwise: [`PeerHandle`]); a pooled state is
//! reached through its chunks instead (`push`'s [`Runtime::export_pool`]).
//! [`Runtime::export_handles`] hands this rank's out, and
//! [`Runtime::import_peers`] maps the other ranks' allocations into this
//! address space and writes the group's addresses into each `peer` buffer,
//! the array a kernel indexes by rank. A remote allocation is mapped once
//! however many peer buffers name it, and the mapping lives as long as the
//! addresses do.
//!
//! The handshake every caller runs is one exchange: each rank makes an
//! [`Offer`] (the manifest it loaded, its handles, and the communicator id
//! of every NCCL group it is member 0 of), the caller carries the offers
//! to the group over whatever channel it has (a TCP head, an engine's gloo
//! group; [`Offer::to_bytes`] is the wire form), and [`Runtime::connect`]
//! checks that every member loaded the same manifest, maps the peers and
//! joins the communicators. One process holding every rank does all of it
//! with [`connect_local`]. What runs once after load, and the barrier
//! behind it, stay the caller's: only it knows when every rank has run
//! them.
//!
//! Nothing launches with a peer array still holding zeros:
//! [`Runtime::require_peers`] guards every program issue.

use std::collections::BTreeMap;

use cudarc::nccl::result as nccl;
use kern_manifest::Verified;
use sha2::{Digest, Sha256};

use crate::compile;
use crate::device::{self, Mapped, PeerHandle};
use crate::error::bail;
use crate::{Error, NcclId, Result, Runtime};

/// What one rank brings to its groups: a digest of the manifest it loaded,
/// a handle to every allocation it exports, and the communicator id of
/// every NCCL group it is member 0 of.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Offer {
    pub manifest: u64,
    pub handles: BTreeMap<String, PeerHandle>,
    pub nccl: BTreeMap<String, [u8; 128]>,
}

const MAGIC: &[u8; 8] = b"kernoff1";

impl Offer {
    /// The wire form: a magic, the digest, then each map as a count and
    /// `(u16 name length, name, fixed-size value)` entries, little-endian.
    /// A handle only this process can map has no wire form; the error says
    /// why the device handed one out.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut out = MAGIC.to_vec();
        out.extend(self.manifest.to_le_bytes());
        out.extend((self.handles.len() as u32).to_le_bytes());
        for (name, h) in &self.handles {
            let b = h.to_bytes().ok_or_else(|| {
                Error::Cuda(format!(
                    "`{name}` has no fabric handle, so no other process can map it: the device allocated it \
                     without one (KERN_NO_FABRIC is set, the device has no fabric support, or cuMemCreate \
                     refused a fabric handle because no IMEX channel is visible; a container needs \
                     --device /dev/nvidia-caps-imex-channels/channel0)"
                ))
            })?;
            entry(&mut out, name, &b);
        }
        out.extend((self.nccl.len() as u32).to_le_bytes());
        for (g, id) in &self.nccl {
            entry(&mut out, g, id);
        }
        Ok(out)
    }

    pub fn from_bytes(b: &[u8]) -> Result<Offer> {
        let mut r = Reader(b);
        if r.take(MAGIC.len())? != MAGIC {
            bail!(Api, "an offer starts with {MAGIC:?}; these bytes are something else");
        }
        let manifest = u64::from_le_bytes(r.array()?);
        let handles = (0..r.count()?)
            .map(|_| Ok((r.name()?, PeerHandle::from_bytes(r.take(PeerHandle::BYTES)?).expect("72 bytes"))))
            .collect::<Result<_>>()?;
        let nccl = (0..r.count()?).map(|_| Ok((r.name()?, r.array()?))).collect::<Result<_>>()?;
        if !r.0.is_empty() {
            bail!(Api, "an offer has {} bytes past its end", r.0.len());
        }
        Ok(Offer { manifest, handles, nccl })
    }
}

fn entry(out: &mut Vec<u8>, name: &str, value: &[u8]) {
    out.extend((name.len() as u16).to_le_bytes());
    out.extend(name.as_bytes());
    out.extend(value);
}

struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.0.len() < n {
            bail!(Api, "an offer ends {} bytes short", n - self.0.len());
        }
        let (head, rest) = self.0.split_at(n);
        self.0 = rest;
        Ok(head)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        Ok(self.take(N)?.try_into().expect("took N bytes"))
    }

    fn count(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn name(&mut self) -> Result<String> {
        let n = u16::from_le_bytes(self.array()?) as usize;
        String::from_utf8(self.take(n)?.to_vec()).map_err(|_| Error::Api("an offer's name is not UTF-8".into()))
    }
}

/// The manifest as every member must have loaded it: the first eight
/// bytes of its JSON's SHA-256.
fn digest(m: &Verified) -> u64 {
    u64::from_le_bytes(Sha256::digest(m.to_json().as_bytes())[..8].try_into().expect("sha256 is 32 bytes"))
}

/// Connect ranks that all live in this process: every rank's offer in
/// every group, the joins inside one NCCL group call so their blocking
/// inits complete together.
pub fn connect_local(rts: &mut [Runtime]) -> Result<()> {
    if let Some((i, g, at)) = rts
        .iter()
        .enumerate()
        .flat_map(|(i, rt)| rt.ranks.iter().map(move |(g, &at)| (i, g, at)))
        .find(|&(i, _, at)| at != i as u64)
    {
        bail!(Api, "runtime {i} is member {at} of group `{g}`: connect_local takes every group's members in order");
    }
    let offers = rts.iter().map(Runtime::offer).collect::<Result<Vec<_>>>()?;
    let groups = rts.first().map(|rt| rt.ranks.keys().cloned().collect::<Vec<_>>()).unwrap_or_default();
    let table: BTreeMap<String, Vec<Offer>> = groups.into_iter().map(|g| (g, offers.clone())).collect();
    nccl::group_start().map_err(crate::nccl::err)?;
    let joined = rts.iter_mut().try_for_each(|rt| rt.connect(&table));
    nccl::group_end().map_err(crate::nccl::err)?;
    joined
}

/// A `peer` buffer waiting for (or holding) its group's addresses.
pub(crate) struct PeerSlot {
    pub(crate) of: String,
    pub(crate) group: String,
    pub(crate) filled: bool,
}

impl Runtime {
    /// This rank's index in a topology group.
    pub fn rank(&self, group: &str) -> Option<u64> {
        self.ranks.get(group).copied()
    }

    /// Map another process's allocation into this context: a weight
    /// cache's bucket, whose fabric handle the daemon published. The
    /// mapping lives as long as the [`Mapped`] does.
    pub fn map(&self, handle: &PeerHandle, what: &str) -> Result<Mapped> {
        self.ctx.bind_to_thread()?;
        Ok(Mapped::new(device::import(&self.stream, self.gpu as i32, handle, what)?))
    }

    /// Copy buffer `name`, as long as its shape is at `vars`, into `to`:
    /// another runtime's buffer of the same shape, mapped. On the compute
    /// stream, so it follows every launch issued before it and precedes
    /// every one issued after; a [`Mark`](crate::Mark) taken next says
    /// when it has landed.
    pub fn push(&self, name: &str, vars: &BTreeMap<String, u64>, to: &Mapped) -> Result<()> {
        self.ctx.bind_to_thread()?;
        let Some(b) = self.manifest.buffers.get(name) else {
            bail!(Api, "no buffer `{name}`");
        };
        let bytes = compile::shaped_bytes(&format!("buffer `{name}`"), &b.shape, b.dtype.bytes(), vars)?;
        if bytes > to.bytes() {
            bail!(Api, "buffer `{name}` is {bytes} bytes at these vars; the mapping it goes to holds {}", to.bytes());
        }
        device::copy_1d(self.stream.cu_stream(), to.ptr(), self.buffers[name].ptr, bytes)
    }

    /// Exported twin `slot` of buffer `name`, allocated at load
    /// ([`crate::Capacity::landings`]): a pipeline stage before this one pushes
    /// items into its twins while this runtime still reads `name` for an
    /// item before, until [`Runtime::land`] moves one over.
    pub fn landing(&self, name: &str, slot: usize) -> Result<PeerHandle> {
        self.ctx.bind_to_thread()?;
        let Some(twin) = self.landings.get(&(name.to_string(), slot)) else {
            bail!(Api, "buffer `{name}` has no landing {slot}: load takes them for exported input buffers only, as many as `Capacity::landings`");
        };
        twin.export()?
            .ok_or_else(|| Error::Cuda(format!("landing {slot} of `{name}`: not a virtual-memory allocation")))
    }

    /// Copy twin `slot` of buffer `name`, as long as its shape is at
    /// `vars`, into the buffer, on the compute stream: behind every launch
    /// issued before, so the item before is through with `name`.
    pub fn land(&self, name: &str, slot: usize, vars: &BTreeMap<String, u64>) -> Result<()> {
        self.ctx.bind_to_thread()?;
        let Some(twin) = self.landings.get(&(name.to_string(), slot)) else {
            bail!(Api, "buffer `{name}` has no landing {slot}");
        };
        let b = &self.manifest.buffers[name];
        let bytes = compile::shaped_bytes(&format!("buffer `{name}`"), &b.shape, b.dtype.bytes(), vars)?;
        device::copy_1d(self.stream.cu_stream(), self.buffers[name].ptr, twin.ptr, bytes)
    }

    /// A handle for every `export` buffer and every fixed state, by name:
    /// what the other ranks pass to [`Runtime::import_peers`].
    pub fn export_handles(&self) -> Result<BTreeMap<String, PeerHandle>> {
        self.ctx.bind_to_thread()?;
        let mut out = BTreeMap::new();
        for (name, b) in &self.buffers {
            if self.manifest.buffers[name].export {
                let h = b.export()?.ok_or_else(|| {
                    Error::Cuda(format!("buffer `{name}`: exported but not a virtual-memory allocation"))
                })?;
                out.insert(name.clone(), h);
            }
        }
        for (name, s) in &self.states {
            if let Some(h) = s.export()? {
                out.insert(name.clone(), h);
            }
        }
        Ok(out)
    }

    /// The `peer` buffers not yet filled, by name.
    pub fn pending_peers(&self) -> Vec<&str> {
        self.peers.iter().filter(|(_, p)| !p.filled).map(|(n, _)| n.as_str()).collect()
    }

    /// This rank's offer to its groups: what every other member passes to
    /// [`Runtime::connect`].
    pub fn offer(&self) -> Result<Offer> {
        let nccl = self
            .nccl_groups()
            .into_iter()
            .filter(|g| self.ranks.get(g) == Some(&0))
            .map(|g| Ok((g, NcclId::new()?.to_bytes())))
            .collect::<Result<_>>()?;
        Ok(Offer { manifest: digest(&self.manifest), handles: self.export_handles()?, nccl })
    }

    /// Map every peer and join every communicator, `members[g]` holding
    /// group `g`'s offers in member order (this rank's own among them).
    /// Every member must have loaded this manifest, and no peer buffer may
    /// be left unfilled. A join blocks until the whole group has joined.
    pub fn connect(&mut self, members: &BTreeMap<String, Vec<Offer>>) -> Result<()> {
        let own = digest(&self.manifest);
        let groups: Vec<String> = self.ranks.keys().cloned().collect();
        for g in &groups {
            let Some(offers) = members.get(g) else {
                bail!(Api, "no offers for group `{g}`");
            };
            if let Some((i, o)) = offers.iter().enumerate().find(|(_, o)| o.manifest != own) {
                bail!(
                    Api,
                    "group `{g}` member {i} loaded another manifest (digest {:016x}, this rank's {own:016x})",
                    o.manifest
                );
            }
            let handles: Vec<_> = offers.iter().map(|o| o.handles.clone()).collect();
            self.import_peers(g, &handles)?;
        }
        let pending = self.pending_peers();
        if !pending.is_empty() {
            bail!(Manifest, "peer buffers {} still unfilled after every group was imported", pending.join(", "));
        }
        for g in self.nccl_groups() {
            let Some(id) = members.get(&g).and_then(|o| o.first()).and_then(|o| o.nccl.get(&g)) else {
                bail!(Api, "group `{g}`: member 0 offered no nccl id");
            };
            self.join_nccl(&g, &NcclId::from_bytes(*id))?;
        }
        Ok(())
    }

    /// Map every group member's exported allocations and fill the group's
    /// `peer` buffers with their addresses. `members[i]` is what rank `i`'s
    /// [`Runtime::export_handles`] returned (this rank's own entry may be
    /// anything: its local addresses are used). Synchronous.
    pub fn import_peers(&mut self, group: &str, members: &[BTreeMap<String, PeerHandle>]) -> Result<()> {
        self.ctx.bind_to_thread()?;
        let Some(&me) = self.ranks.get(group) else {
            bail!(Api, "no topology group `{group}`");
        };
        let size = self.manifest.group_size(group).unwrap_or(0);
        if members.len() as u64 != size {
            bail!(Api, "group `{group}` has {size} members, got handles for {}", members.len());
        }
        let stream = self.stream.clone();
        let mut mapped: BTreeMap<(usize, String), u64> = BTreeMap::new();
        let names: Vec<String> = self.peers.iter().filter(|(_, p)| p.group == group).map(|(n, _)| n.clone()).collect();
        for name in names {
            let of = self.peers[&name].of.clone();
            let own = self.buffers.get(&of).or_else(|| self.states.get(&of)).ok_or_else(|| {
                Error::Manifest(format!("peer buffer `{name}`: `of` `{of}` is neither a buffer nor a state"))
            })?;
            let own_bytes = own.export()?.map(|h| h.bytes()).unwrap_or(own.bytes);
            let own_ptr = own.ptr;
            let mut addrs = Vec::with_capacity(size as usize);
            for (i, m) in members.iter().enumerate() {
                if i as u64 == me {
                    addrs.push(own_ptr);
                    continue;
                }
                let key = (i, of.clone());
                let ptr = match mapped.get(&key) {
                    Some(&p) => p,
                    None => {
                        let Some(h) = m.get(&of) else {
                            bail!(Api, "group `{group}` rank {i}: no handle for `{of}`");
                        };
                        if h.bytes() != own_bytes {
                            bail!(
                                Api,
                                "group `{group}` rank {i}: `{of}` is {} bytes there, {own_bytes} here",
                                h.bytes()
                            );
                        }
                        let buf =
                            device::import(&stream, self.gpu as i32, h, &format!("group `{group}` rank {i} `{of}`"))?;
                        let p = buf.ptr;
                        self.imports.push(buf);
                        mapped.insert(key, p);
                        p
                    }
                };
                addrs.push(ptr);
            }
            let bytes: Vec<u8> = addrs.iter().flat_map(|a| a.to_le_bytes()).collect();
            let dst = self.buffers.get_mut(&name).unwrap();
            if bytes.len() as u64 != dst.bytes {
                bail!(
                    Manifest,
                    "peer buffer `{name}`: {} bytes for {size} addresses, allocated {}",
                    bytes.len(),
                    dst.bytes
                );
            }
            stream.memcpy_htod(&bytes, dst)?;
            self.peers.get_mut(&name).unwrap().filled = true;
            tracing::info!("peer buffer `{name}`: {size} addresses of `{of}` over group `{group}`");
        }
        stream.synchronize()?;
        Ok(())
    }

    /// Nothing launches with a peer array still holding zeros.
    pub(crate) fn require_peers(&self) -> Result<()> {
        let pending = self.pending_peers();
        if pending.is_empty() {
            return Ok(());
        }
        bail!(Api, "peer buffers not imported yet: {}", pending.join(", "))
    }
}
