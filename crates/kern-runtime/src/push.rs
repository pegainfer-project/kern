//! Pushes: this runtime's pages and slots copied into another runtime's
//! pooled states, in this process or in another one on the same NVLink
//! domain. A prefill stage hands the decode rank that continues a prompt
//! the prompt's KV pages and recurrent state this way.
//!
//! A pooled state is not one allocation. It is an address range reserved
//! once over physical chunks that every pooled state shares and the remap
//! thread moves between pages and slots (`kern_pool`'s chunks), so the
//! destination exports its chunks, not its states:
//! [`Runtime::export_pool`] hands out a [`PoolHandle`] (every chunk's
//! handle, and each pooled state's name, bytes per page or slot and
//! positions reserved), and the pusher's [`Runtime::map_pool`] imports
//! the chunks and reserves a range of the same size per state with
//! nothing mapped yet: a [`MappedPool`]. Which chunk backs which position
//! is a [`Placement`], the destination's [`Runtime::placement`]; the
//! pusher's [`Runtime::place`] brings its mirror to one by running the
//! remap between the placement it holds and the new one. After that page
//! `p` of state `s` sits at the same offset in the mirror as at the
//! destination, `p` times the page's bytes, and slot `j` at `j` times the
//! slot's.
//!
//! What a push relies on: it writes only objects the destination holds
//! (pages and the slot of a lease it took), and the mirror was placed
//! with a placement taken after that lease. A remap takes chunks only
//! from objects nobody holds, so a held object stays where that placement
//! says for as long as it is held. A push into anything else writes
//! wherever the mirror maps that position: an unmapped one faults, a
//! stale one is some other object's bytes. That is undefined and the
//! caller rules it out: the destination sends a placement along with the
//! objects it leased (a fresh one only when its [`Runtime::remaps`]
//! moved), and only once its own stream has passed the lease, which
//! zeroes the slot there.
//!
//! Every push runs on the pusher's compute stream, after every launch
//! issued before it, and a [`Mark`](crate::Mark) taken next says when it
//! has landed. Pages fold into runs of consecutive pairs
//! ([`kern_pool::runs`]), one copy per run. A slot goes whole, or as one
//! member's part when the destination shards the state by heads over a
//! tensor-parallel group ([`Split`]): one strided copy per row of a line's
//! segments, every line at once.
//!
//! A destination that deals its paged states by position over a group
//! (`kern_pool::Interleave`) takes rows, not pages: every `w`-th row of
//! the source's pages, packed into its own. [`Runtime::push_rows`] moves
//! one member's [`Deal`] of one state in one launch of the runtime's own
//! gather kernel, whatever the number of runs: at 219k tokens over eight
//! members a stage moves some 80 thousand runs of eight rows, and a copy
//! call per run would spend a third of a second on the host. The runs are
//! cut into pieces of at most [`Rows::PIECE`] bytes ([`Rows::dealt`]), one
//! block each, so a deal over adjacent pages that folds into one long run
//! still spreads over the device. The table of pieces is read by the
//! kernel straight out of page-locked memory and goes back to the runtime
//! once a mark taken after the launch has passed: nothing is freed or
//! waited for on the issuing thread.

use std::ffi::CStr;
use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

use cudarc::driver::{sys, CudaContext};
use kern_pool::{runs, Deal, Kind, Placement};

use crate::device::{copy_1d, copy_2d, Arena, Mapper, PeerHandle, Physical, Pinned, Space};
use crate::error::{bail, cuda_check};
use crate::exec::Mark;
use crate::{Result, Runtime};

/// A pooled state as a peer sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Layout {
    state: String,
    kind: Kind,
    /// Bytes per page or per slot.
    object: u64,
    /// Chunk positions reserved.
    positions: usize,
}

/// What a peer maps to reach a runtime's pooled states: every physical
/// chunk behind them, in chunk order, and every pooled state's layout in
/// the pool's arena order. From [`Runtime::export_pool`], for
/// [`Runtime::map_pool`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoolHandle {
    chunk: u64,
    /// Rows per page of every paged state.
    unit: u64,
    chunks: Vec<PeerHandle>,
    states: Vec<Layout>,
}

impl PoolHandle {
    /// Wire form, what a caller carries to another process: the chunk
    /// size, the page's rows, the states (name, kind, object bytes,
    /// positions), then each chunk's [`PeerHandle::to_bytes`];
    /// little-endian. `None` when the chunks have local handles, which no
    /// other process can map.
    pub fn to_bytes(&self) -> Option<Vec<u8>> {
        let mut out = self.chunk.to_le_bytes().to_vec();
        out.extend(self.unit.to_le_bytes());
        out.extend((self.states.len() as u32).to_le_bytes());
        for s in &self.states {
            out.extend((s.state.len() as u32).to_le_bytes());
            out.extend(s.state.as_bytes());
            out.push(matches!(s.kind, Kind::Slot) as u8);
            out.extend(s.object.to_le_bytes());
            out.extend((s.positions as u64).to_le_bytes());
        }
        out.extend((self.chunks.len() as u32).to_le_bytes());
        for c in &self.chunks {
            out.extend(c.to_bytes()?);
        }
        Some(out)
    }

    /// The handle [`PoolHandle::to_bytes`] wrote; `None` for bytes that
    /// are not one.
    pub fn from_bytes(b: &[u8]) -> Option<PoolHandle> {
        let mut r = b;
        let mut take = |n: usize| -> Option<&[u8]> {
            let (head, tail) = r.split_at_checked(n)?;
            r = tail;
            Some(head)
        };
        let chunk = u64::from_le_bytes(take(8)?.try_into().ok()?);
        let unit = u64::from_le_bytes(take(8)?.try_into().ok()?);
        let n = u32::from_le_bytes(take(4)?.try_into().ok()?);
        let mut states = Vec::new();
        for _ in 0..n {
            let len = u32::from_le_bytes(take(4)?.try_into().ok()?) as usize;
            let state = String::from_utf8(take(len)?.to_vec()).ok()?;
            let kind = match take(1)?[0] {
                0 => Kind::Page,
                1 => Kind::Slot,
                _ => return None,
            };
            let object = u64::from_le_bytes(take(8)?.try_into().ok()?);
            let positions = u64::from_le_bytes(take(8)?.try_into().ok()?) as usize;
            states.push(Layout { state, kind, object, positions });
        }
        let n = u32::from_le_bytes(take(4)?.try_into().ok()?);
        let chunks = (0..n).map(|_| PeerHandle::from_bytes(take(PeerHandle::BYTES)?)).collect::<Option<Vec<_>>>()?;
        r.is_empty().then_some(PoolHandle { chunk, unit, chunks, states })
    }
}

/// Another runtime's pooled states mirrored into this context: its chunks
/// imported, a range per state reserved, mapped as the last
/// [`Runtime::place`] said. Unmapped and released on drop; the owner's
/// chunks outlive it.
pub struct MappedPool {
    mapper: Mapper,
    states: Vec<Layout>,
    chunk: u64,
    unit: u64,
    chunks: usize,
    placed: Placement,
}

impl MappedPool {
    /// State `name`'s arena: its index, layout, and how many objects its
    /// range holds.
    fn arena(&self, name: &str) -> Option<(usize, &Layout, u64)> {
        let (a, l) = self.states.iter().enumerate().find(|(_, l)| l.state == name)?;
        Some((a, l, l.positions as u64 * self.chunk / l.object))
    }
}

/// How every line of a per-sequence state divides over the `of` members
/// of a tensor-parallel group that shards it by heads. A line is
/// segments back to back, each `rows` rows of `width` bytes whose leading
/// dimension is the heads: a recurrent `[heads, dk, dv]` block is one row,
/// a conv window `[taps, heads × d]` is `taps` rows. Member `k` holds
/// bytes `[k·w, (k+1)·w)` of every row, `w = width / of`, and its own line
/// is the same segments `w` wide, back to back. A slot is whole lines, so
/// member `k`'s slot is its part of every line, in line order.
///
/// The manifest declares a slot's bytes and a line table's stride, never
/// what is inside a line, so a split is the generator's knowledge that
/// the caller passes in; [`Runtime::push_slot_part`] checks it against
/// the state's bytes on both sides.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Split {
    segments: Vec<(u64, u64)>,
    of: u64,
}

/// One strided copy: `rows` rows of `width` bytes from `src` (rows
/// `src_pitch` apart) to `dst` (rows `dst_pitch` apart), offsets into a
/// slot on either side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rows {
    pub src: u64,
    pub dst: u64,
    pub width: u64,
    pub src_pitch: u64,
    pub dst_pitch: u64,
    pub rows: u64,
}

impl Rows {
    /// Bytes a piece of a dealt run moves at most, unless one row is
    /// more: a block's worth of work.
    pub const PIECE: u64 = 1 << 16;

    /// Each row as (source offset, destination offset, bytes).
    pub fn ranges(&self) -> impl Iterator<Item = (u64, u64, u64)> + '_ {
        (0..self.rows).map(|i| (self.src + i * self.src_pitch, self.dst + i * self.dst_pitch, self.width))
    }

    /// A deal's runs as strided copies of `row`-byte rows, offsets from
    /// the start of either side's arena, in run order: each run cut into
    /// pieces of at most [`Rows::PIECE`] bytes, or one row when a row is
    /// more.
    pub fn dealt(deal: &Deal, row: u64) -> Vec<Rows> {
        let most = (Rows::PIECE / row.max(1)).max(1);
        deal.runs()
            .iter()
            .flat_map(|r| {
                (0..r.rows).step_by(most as usize).map(move |i| Rows {
                    src: (r.src_row + i * r.src_stride) * row,
                    dst: (r.dst_row + i) * row,
                    width: row,
                    src_pitch: r.src_stride * row,
                    dst_pitch: row,
                    rows: most.min(r.rows - i),
                })
            })
            .collect()
    }
}

impl Split {
    /// Segments of (rows, width bytes), in line order, over `of` members:
    /// every segment non-empty and its width a multiple of `of`.
    pub fn new(segments: &[(u64, u64)], of: u64) -> Result<Split> {
        if of == 0 || segments.is_empty() {
            bail!(Api, "a split needs a segment and a member; got {} segments over {of}", segments.len());
        }
        if let Some(&(r, w)) = segments.iter().find(|&&(r, w)| r == 0 || w == 0 || !w.is_multiple_of(of)) {
            bail!(Api, "split segment of {r} rows × {w} bytes: expected non-empty rows {of} members divide");
        }
        Ok(Split { segments: segments.to_vec(), of })
    }

    pub fn of(&self) -> u64 {
        self.of
    }

    /// Bytes of a whole line.
    pub fn line(&self) -> u64 {
        self.segments.iter().map(|(r, w)| r * w).sum()
    }

    /// Member `part`'s share of a slot of `lines` lines (`part < of`), as
    /// one strided copy per row of each segment covering every line:
    /// sources a whole line apart, destinations a member's line apart.
    pub fn copies(&self, lines: u64, part: u64) -> Vec<Rows> {
        assert!(part < self.of, "part {part} of {}", self.of);
        let (line, of) = (self.line(), self.of);
        let mut at = 0;
        let mut out = Vec::new();
        for &(rows, width) in &self.segments {
            let w = width / of;
            out.extend((0..rows).map(|r| Rows {
                src: at + r * width + part * w,
                dst: at / of + r * w,
                width: w,
                src_pitch: line,
                dst_pitch: line / of,
                rows: lines,
            }));
            at += rows * width;
        }
        out
    }

    /// Member `part`'s share of a slot of `lines` lines as (source offset,
    /// destination offset, bytes) ranges: over every member the sources
    /// tile the slot, and each member's destinations tile its own slot.
    pub fn ranges(&self, lines: u64, part: u64) -> Vec<(u64, u64, u64)> {
        self.copies(lines, part).iter().flat_map(Rows::ranges).collect()
    }
}

impl Runtime {
    /// Every pooled state's chunks and layout, for another runtime to
    /// mirror ([`Runtime::map_pool`]): fabric handles where the device
    /// gave the chunks one, local ones (this process only) otherwise.
    pub fn export_pool(&self) -> Result<PoolHandle> {
        self.ctx.bind_to_thread()?;
        let states = self
            .pool
            .pooled()
            .iter()
            .map(|a| Layout { state: a.state.clone(), kind: a.kind, object: a.object, positions: a.positions })
            .collect();
        Ok(PoolHandle { chunk: self.pool.chunk(), unit: self.pool.unit(), chunks: self.chunks.export()?, states })
    }

    /// Which chunk backs every position of every pooled state now: what a
    /// mirror is placed with ([`Runtime::place`]). An object held when it
    /// is taken stays where it says while it is held.
    pub fn placement(&self) -> Placement {
        self.pool.placement()
    }

    /// Import another runtime's chunks and reserve a range per pooled
    /// state, nothing mapped until [`Runtime::place`]. Synchronous.
    pub fn map_pool(&self, h: &PoolHandle, what: &str) -> Result<MappedPool> {
        self.ctx.bind_to_thread()?;
        let physical = Arc::new(Physical::import(h.chunk as usize, &h.chunks, what)?);
        let arenas = h
            .states
            .iter()
            .map(|s| Arena::reserve(self.gpu as i32, h.chunk as usize, s.positions))
            .collect::<Result<Vec<_>>>()?;
        let positions: Vec<usize> = h.states.iter().map(|s| s.positions).collect();
        Ok(MappedPool {
            mapper: Mapper::new(arenas, physical),
            states: h.states.clone(),
            chunk: h.chunk,
            unit: h.unit,
            chunks: h.chunks.len(),
            placed: Placement::unmapped(&positions),
        })
    }

    /// Map `to` as `p` says: every position whose chunk changed since the
    /// last placement comes off and gets its new one. Synchronous, and
    /// nothing may be pushing into `to` meanwhile (this runtime's pushes
    /// are on its stream, which this waits for).
    pub fn place(&self, to: &mut MappedPool, p: &Placement) -> Result<()> {
        self.ctx.bind_to_thread()?;
        if p.shape() != to.placed.shape() {
            bail!(Api, "placement of {:?} positions for a pool of {:?}", p.shape(), to.placed.shape());
        }
        if p.chunks() as usize > to.chunks {
            bail!(Api, "placement names chunk {} of a pool of {}", p.chunks() - 1, to.chunks);
        }
        self.stream.synchronize()?;
        to.mapper.run(&to.placed.toward(p))?;
        to.placed = p.clone();
        Ok(())
    }

    /// Copy pages of paged state `state` into `to`'s state of that name:
    /// `(source page, destination page)` pairs, consecutive ones folded
    /// into one copy. The destination pages are ones it holds (see the
    /// module doc); the source pages are this runtime's to read.
    pub fn push_pages(&self, state: &str, pairs: &[(i32, i32)], to: &MappedPool) -> Result<()> {
        self.ctx.bind_to_thread()?;
        let e = self.ends(state, Kind::Page, to)?;
        if e.object != e.theirs {
            bail!(Api, "state `{state}`: {} bytes per page here, {} there", e.object, e.theirs);
        }
        let (have, room) = (self.pool.pages_max() as i64, e.room as i64);
        if let Some(&(s, d)) = pairs.iter().find(|&&(s, d)| s < 0 || s as i64 >= have || d < 0 || d as i64 >= room) {
            bail!(Api, "state `{state}`: page pair ({s}, {d}) outside 0..{have} here, 0..{room} there");
        }
        let units: Vec<(i32, u64)> = pairs.iter().map(|&(s, d)| (s, d as u64)).collect();
        for (p, d, n) in runs(&units) {
            copy_1d(self.stream.cu_stream(), e.dst + d * e.object, e.src + p as u64 * e.object, n as u64 * e.object)?;
        }
        Ok(())
    }

    /// Copy slot `from` of per-sequence state `state` whole into slot `into`
    /// of `to`'s state of that name, the same bytes per slot on both sides.
    pub fn push_slot(&self, state: &str, from: i32, into: i32, to: &MappedPool) -> Result<()> {
        self.ctx.bind_to_thread()?;
        let e = self.ends(state, Kind::Slot, to)?;
        if e.object != e.theirs {
            bail!(Api, "state `{state}`: {} bytes per slot here, {} there", e.object, e.theirs);
        }
        self.slots_in_range(state, from, into, &e)?;
        copy_1d(self.stream.cu_stream(), e.dst + into as u64 * e.theirs, e.src + from as u64 * e.object, e.object)
    }

    /// Copy member `part`'s share (as `split` defines it) of slot `from`
    /// of per-sequence state `state` into slot `into` of `to`'s state of
    /// that name, which holds that member's shard: a slot here is whole
    /// lines of `split.line()` bytes, a slot there `1 / split.of()` of it.
    pub fn push_slot_part(
        &self,
        state: &str,
        from: i32,
        into: i32,
        split: &Split,
        part: u64,
        to: &MappedPool,
    ) -> Result<()> {
        self.ctx.bind_to_thread()?;
        let (of, line) = (split.of(), split.line());
        let e = self.ends(state, Kind::Slot, to)?;
        if part >= of || e.object != e.theirs * of {
            bail!(Api, "state `{state}`: part {part} of {of} of a {}-byte slot into a {}-byte one", e.object, e.theirs);
        }
        if !e.object.is_multiple_of(line) {
            bail!(Api, "state `{state}`: a slot of {} bytes is not whole lines of {line} bytes", e.object);
        }
        self.slots_in_range(state, from, into, &e)?;
        let (from, into) = (e.src + from as u64 * e.object, e.dst + into as u64 * e.theirs);
        for c in split.copies(e.object / line, part) {
            let d = (into + c.dst, c.dst_pitch, Space::Device);
            let s = (from + c.src, c.src_pitch, Space::Device);
            copy_2d(self.stream.cu_stream(), d, s, c.width, c.rows)?;
        }
        Ok(())
    }

    /// Copy member `m`'s rows of a sequence in this runtime's pages of
    /// paged state `state` into that member's pages in `to`, as `deal`
    /// says (`kern_pool::deal` over this runtime's pages and `to`'s): one
    /// launch, rows of the same bytes on both sides. Every row the deal
    /// names on `to` is in a page it holds (see the module doc).
    pub fn push_rows(&self, state: &str, deal: &Deal, to: &MappedPool) -> Result<()> {
        self.ctx.bind_to_thread()?;
        let e = self.ends(state, Kind::Page, to)?;
        let (row, theirs) = (e.object / self.pool.unit(), e.theirs / to.unit);
        if row != theirs {
            bail!(Api, "state `{state}`: {row} bytes per row here, {theirs} there");
        }
        if !row.is_multiple_of(16) {
            bail!(Api, "state `{state}`: rows of {row} bytes; a row push moves 16-byte words");
        }
        let (have, room) = (self.pool.pages_max() as u64 * self.pool.unit(), e.room * to.unit);
        let outside =
            |r: &&kern_pool::RowRun| r.src_row + (r.rows - 1) * r.src_stride >= have || r.dst_row + r.rows > room;
        if let Some(r) = deal.runs().iter().find(outside) {
            bail!(Api, "state `{state}`: rows {r:?} outside 0..{have} here, 0..{room} there");
        }
        let copies = Rows::dealt(deal, row);
        if copies.is_empty() {
            return Ok(());
        }
        let table = self.gather.stage(&copies, self.gpu)?;
        if let Err(err) = self.gather.launch(&table, copies.len() as u32, e.src, e.dst, self.stream.cu_stream()) {
            self.gather.unstage(table, None);
            return Err(err);
        }
        match self.mark() {
            Ok(m) => {
                self.gather.unstage(table, Some(m));
                Ok(())
            }
            Err(err) => {
                // The launch may still read the table: it is never reused.
                std::mem::forget(table);
                Err(err)
            }
        }
    }

    /// Both ends of a push of `state`, the same kind of object on either.
    fn ends(&self, state: &str, kind: Kind, to: &MappedPool) -> Result<Ends> {
        let Some(here) = self.pool.pooled().iter().find(|a| a.state == state) else {
            bail!(Api, "no pooled state `{state}` here");
        };
        let Some((a, l, room)) = to.arena(state) else {
            bail!(Api, "no pooled state `{state}` in the mapped pool");
        };
        if here.kind != kind || l.kind != kind {
            bail!(Api, "state `{state}`: a {kind:?} push, the state is {:?} here and {:?} there", here.kind, l.kind);
        }
        Ok(Ends { src: self.states[state].ptr, object: here.object, dst: to.mapper.ptr(a), theirs: l.object, room })
    }

    /// Slot 0 is the null line on either side, never a push's.
    fn slots_in_range(&self, state: &str, from: i32, into: i32, e: &Ends) -> Result<()> {
        let (have, room) = (self.pool.slots_max() as i64, e.room as i64);
        if from < 1 || from as i64 >= have || into < 1 || into as i64 >= room {
            bail!(Api, "state `{state}`: slots ({from}, {into}) outside 1..{have} here, 1..{room} there");
        }
        Ok(())
    }
}

/// A push's two arenas: where each starts, bytes per object on either
/// side, and how many objects the mirror's range holds.
struct Ends {
    src: u64,
    object: u64,
    dst: u64,
    theirs: u64,
    room: u64,
}

/// The runtime's row gather: one block per [`Rows`] of a table in
/// page-locked memory, each moving its rows in 16-byte words, then a
/// system-scope fence so a mark after it covers the peer's memory too.
pub(crate) struct Gather {
    module: sys::CUmodule,
    function: sys::CUfunction,
    /// Tables, each free again once its mark has passed.
    tables: Mutex<Vec<Table>>,
}

unsafe impl Send for Gather {}
unsafe impl Sync for Gather {}

/// A table of pieces: the host's view, the kernel's, and the mark that
/// says the launch reading it is done.
struct Table {
    host: Pinned,
    device: u64,
    landed: Option<Mark>,
}

/// Fields of a [`Rows`] the kernel reads, in [`Rows`] order.
const FIELDS: usize = 6;

const GATHER_PTX: &CStr = c"
.version 8.0
.target sm_90
.address_size 64
.visible .entry gather(.param .u64 table, .param .u64 src, .param .u64 dst)
{
    .reg .b32 %i, %n, %nt, %vpr, %r, %v, %rows, %x<4>;
    .reg .b64 %t, %s, %d, %e<6>, %a, %b, %w;
    .reg .pred %p;
    ld.param.u64 %t, [table];
    ld.param.u64 %s, [src];
    ld.param.u64 %d, [dst];
    cvta.to.global.u64 %t, %t;
    cvta.to.global.u64 %s, %s;
    cvta.to.global.u64 %d, %d;
    mov.u32 %i, %ctaid.x;
    mul.wide.u32 %w, %i, 48;
    add.u64 %t, %t, %w;
    ld.global.v2.u64 {%e0, %e1}, [%t];
    ld.global.v2.u64 {%e2, %e3}, [%t+16];
    ld.global.v2.u64 {%e4, %e5}, [%t+32];
    add.u64 %s, %s, %e0;
    add.u64 %d, %d, %e1;
    shr.u64 %w, %e2, 4;
    cvt.u32.u64 %vpr, %w;
    cvt.u32.u64 %rows, %e5;
    mul.lo.u32 %n, %rows, %vpr;
    mov.u32 %i, %tid.x;
    mov.u32 %nt, %ntid.x;
L:
    setp.ge.u32 %p, %i, %n;
    @%p bra D;
    div.u32 %r, %i, %vpr;
    mul.lo.u32 %v, %r, %vpr;
    sub.u32 %v, %i, %v;
    mul.wide.u32 %w, %v, 16;
    cvt.u64.u32 %a, %r;
    mad.lo.u64 %b, %a, %e4, %d;
    mad.lo.u64 %a, %a, %e3, %s;
    add.u64 %a, %a, %w;
    add.u64 %b, %b, %w;
    ld.global.v4.u32 {%x0, %x1, %x2, %x3}, [%a];
    st.global.v4.u32 [%b], {%x0, %x1, %x2, %x3};
    add.u32 %i, %i, %nt;
    bra L;
D:
    fence.acq_rel.sys;
    ret;
}
";

impl Gather {
    pub(crate) fn load(ctx: &Arc<CudaContext>) -> Result<Gather> {
        ctx.bind_to_thread()?;
        let mut module: sys::CUmodule = std::ptr::null_mut();
        cuda_check(
            unsafe { sys::cuModuleLoadData(&mut module, GATHER_PTX.as_ptr() as *const c_void) },
            "cuModuleLoadData(gather)",
        )?;
        let mut function: sys::CUfunction = std::ptr::null_mut();
        let r = unsafe { sys::cuModuleGetFunction(&mut function, module, c"gather".as_ptr()) };
        if let Err(e) = cuda_check(r, "cuModuleGetFunction(gather)") {
            unsafe { sys::cuModuleUnload(module) };
            return Err(e);
        }
        Ok(Gather { module, function, tables: Mutex::default() })
    }

    /// A table holding `copies`: a passed one big enough, else a new one.
    fn stage(&self, copies: &[Rows], gpu: usize) -> Result<Table> {
        let bytes = (copies.len() * FIELDS * 8) as u64;
        let spare = {
            let mut v = self.tables.lock().unwrap_or_else(|e| e.into_inner());
            let free = |t: &Table| t.host.bytes() >= bytes && t.landed.as_ref().is_none_or(Mark::passed);
            v.iter().position(free).map(|i| v.swap_remove(i))
        };
        let table = match spare {
            Some(t) => t,
            None => {
                let host = Pinned::alloc_mapped(bytes.next_power_of_two().max(Rows::PIECE), gpu as i32)?;
                let device = host.device()?;
                Table { host, device, landed: None }
            }
        };
        let entries = unsafe { std::slice::from_raw_parts_mut(table.host.ptr() as *mut [u64; FIELDS], copies.len()) };
        for (e, c) in entries.iter_mut().zip(copies) {
            *e = [c.src, c.dst, c.width, c.src_pitch, c.dst_pitch, c.rows];
        }
        Ok(table)
    }

    /// Back to the free list, reusable once `landed` has passed (at once
    /// without one: nothing was launched to read it).
    fn unstage(&self, mut t: Table, landed: Option<Mark>) {
        t.landed = landed;
        self.tables.lock().unwrap_or_else(|e| e.into_inner()).push(t);
    }

    fn launch(&self, t: &Table, pieces: u32, src: u64, dst: u64, stream: sys::CUstream) -> Result<()> {
        let mut params = [&t.device, &src, &dst].map(|p| p as *const u64 as *mut c_void);
        let r = unsafe {
            sys::cuLaunchKernel(
                self.function,
                pieces,
                1,
                1,
                256,
                1,
                1,
                0,
                stream,
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        cuda_check(r, "cuLaunchKernel(gather)")
    }
}

impl Drop for Gather {
    fn drop(&mut self) {
        unsafe { sys::cuModuleUnload(self.module) };
    }
}
