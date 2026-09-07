//! The byte permutation behind a buffer's `layout`: a row-major matrix from
//! the weights file becomes the tiled image the manifest declares, once, at
//! load. Pure host code; `Runtime::load_weights` is the only caller.
//!
//! Tile `(i, j)` of a `[rows, cols]` matrix with tile `[tr, tc]` is the
//! `tr x tc` block at rows `i*tr..`, cols `j*tc..`; tiles are stored
//! row-major inside, one after another in `(i, j)` order, so each is one
//! contiguous run of `tr * tc` elements. With swizzle 8 the 16-byte chunk
//! `c` of tile row `r` lands at chunk `c ^ (r % 8)`: the image a 1-D bulk
//! copy drops into shared memory then feeds `ldmatrix` without bank
//! conflicts, the same pattern TMA's 128-byte swizzle produces.

use kern_manifest::types::Layout;

/// The tiled image of `src`, a row-major `rows x cols` matrix of
/// `elem_bytes`-wide elements. The verifier has checked that the tile
/// divides the shape, that a tile row is whole 16-byte chunks and, for
/// swizzle 8, at least eight of them.
pub fn tile(src: &[u8], rows: usize, cols: usize, elem_bytes: usize, l: &Layout) -> Vec<u8> {
    let (tr, tc) = (l.tile[0] as usize, l.tile[1] as usize);
    let row_bytes = cols * elem_bytes;
    let trow_bytes = tc * elem_bytes;
    let chunks = trow_bytes / 16;
    let tiles_per_row = cols / tc;
    assert_eq!(src.len(), rows * row_bytes, "layout source is not rows x cols");
    let mut out = vec![0u8; src.len()];
    let tile_bytes = tr * trow_bytes;
    // Tiles are independent; hand each thread a band of tile rows.
    let bands = std::thread::available_parallelism().map_or(1, |n| n.get()).min(rows / tr).max(1);
    let tile_rows = rows / tr;
    std::thread::scope(|s| {
        for (band, out_band) in out.chunks_mut(tile_bytes * tiles_per_row * tile_rows.div_ceil(bands)).enumerate() {
            let i0 = band * tile_rows.div_ceil(bands);
            s.spawn(move || {
                for (k, dst) in out_band.chunks_mut(tile_bytes).enumerate() {
                    let (i, j) = (i0 + k / tiles_per_row, k % tiles_per_row);
                    for r in 0..tr {
                        let row = &src[(i * tr + r) * row_bytes + j * trow_bytes..][..trow_bytes];
                        let drow = &mut dst[r * trow_bytes..][..trow_bytes];
                        if l.swizzle == 8 {
                            for c in 0..chunks {
                                let d = c ^ (r & 7);
                                drow[d * 16..d * 16 + 16].copy_from_slice(&row[c * 16..c * 16 + 16]);
                            }
                        } else {
                            drow.copy_from_slice(row);
                        }
                    }
                }
            });
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(rows: usize, cols: usize) -> Vec<u8> {
        // element (r, c) = r * cols + c as a little-endian u16
        (0..rows * cols).flat_map(|i| (i as u16).to_le_bytes()).collect()
    }
    fn elem(img: &[u8], i: usize) -> u16 {
        u16::from_le_bytes([img[2 * i], img[2 * i + 1]])
    }

    #[test]
    fn plain_tiles_are_contiguous_blocks() {
        let (rows, cols) = (128, 192);
        let src = matrix(rows, cols);
        let l = Layout { tile: [64, 64], swizzle: 0 };
        let out = tile(&src, rows, cols, 2, &l);
        assert_eq!(out.len(), src.len());
        // tile (1, 2), row 3, col 5 -> element (67, 133)
        let t = (1 * 3 + 2) * 64 * 64;
        assert_eq!(elem(&out, t + 3 * 64 + 5), (67 * cols + 133) as u16);
        // every element appears exactly once
        let mut seen: Vec<u16> = (0..rows * cols).map(|i| elem(&out, i)).collect();
        seen.sort_unstable();
        assert!(seen.iter().enumerate().all(|(i, &v)| v == i as u16));
    }

    #[test]
    fn swizzle_xors_chunk_with_row() {
        let (rows, cols) = (64, 128);
        let src = matrix(rows, cols);
        let l = Layout { tile: [64, 64], swizzle: 8 };
        let out = tile(&src, rows, cols, 2, &l);
        // tile (0, 1), row 11 (11 % 8 = 3): source chunk 2 (cols 16..24 of the tile) lands at chunk 2 ^ 3 = 1
        let t = 64 * 64;
        assert_eq!(elem(&out, t + 11 * 64 + 1 * 8), (11 * cols + 64 + 16) as u16);
        // rows 0, 8, 16.. are not permuted
        assert_eq!(elem(&out, t + 16 * 64 + 2 * 8), (16 * cols + 64 + 16) as u16);
        let mut seen: Vec<u16> = (0..rows * cols).map(|i| elem(&out, i)).collect();
        seen.sort_unstable();
        assert!(seen.iter().enumerate().all(|(i, &v)| v == i as u16));
    }

    #[test]
    fn bands_cover_every_tile_row() {
        // more tile rows than threads and a count that does not divide evenly
        let (rows, cols) = (64 * 37, 64);
        let src = matrix(rows, cols);
        let out = tile(&src, rows, cols, 2, &Layout { tile: [64, 64], swizzle: 0 });
        // a single tile column: the tiled image is the row-major image
        assert_eq!(out, src);
    }
}
