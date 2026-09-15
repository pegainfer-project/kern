// A model of nothing: integer kernels whose whole job is to be exact, so
// that a served answer is right or wrong with no margin to argue. The
// states are the thing under test — every byte they hold is written by
// these kernels and read back into the next token, so a page served
// stale, a slot remapped wrong, a page listed out of order, a line
// restored from the wrong copy or copied short all change the output.
//
// A token slot holds WORDS marks, mark(t, p, w) for its token t at
// position p, then a tail every word of which is a function of the head
// (a slot copied short shows). A sequence's sum S over the head words of
// every position it holds, each rotated by its position (so the order
// of its pages matters), decides the next token: eos when
// (S >> 56) % EOS_EVERY == 0, else S % VOCAB_BYTES; a tail that does not
// match its head poisons S. A line (a per-sequence state) carries a fold
// C over its tokens in order, folded only for the tokens taken, plus how
// many, then a tail that is a function of C; the next token adds C to S.
// tools/toy/model.py is the same arithmetic in Python (the tails are the
// kernels' own check, the reference never sees them).
//
//   toy_write:    every word of every row's slot.                grid [rows]
//   toy_predict:  next token of every sequence from its pages   grid [seqs]
//                 (and its line, when `mem` is a line's state).
//   toy_fold:     a group's rows folded into its line, in order  grid [groups]
//   toy_round:    a speculative round: row 0 is the anchor, each
//                 next row the token predicted after the last;
//                 `count` says 1 + S % rows of them are taken.  grid [seqs]
//
//   nvcc -cubin -arch=sm_103a -o target/cubins/toy.cubin tools/kernels-src/toy.cu
#include <cstdint>

typedef unsigned long long u64;
typedef long long i64;

#define WORDS 64
#define EOS_EVERY 96ull
#define VOCAB_BYTES 256ull
#define POISON 0xBAD0BAD0BAD0BAD0ull
#ifndef BLOCK
#define BLOCK 256
#endif

__device__ static inline u64 splitmix(u64 x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

__device__ static inline u64 rotl(u64 x, int k) {
  k &= 63;
  return k ? (x << k) | (x >> (64 - k)) : x;
}

__device__ static inline u64 mark(i64 t, i64 p, int w, const i64* salt) {
  return splitmix((u64)t * 0x100000001B3ull ^ (u64)p * 0x9E3779B1ull ^ (u64)w * 0xC2B2AE35ull ^ (u64)salt[w & 63]);
}

__device__ static inline u64 fold(u64 c, i64 t, i64 p, const i64* salt) { return rotl(c, 7) ^ mark(t, p, 0, salt); }

__device__ static inline i64 next_of(u64 s) {
  return ((s >> 56) % EOS_EVERY == 0) ? (i64)VOCAB_BYTES : (i64)(s % VOCAB_BYTES);
}

__device__ static inline u64* slot_words(unsigned char* kv, i64 slot, i64 bytes_per_token) {
  return reinterpret_cast<u64*>(kv + slot * bytes_per_token);
}

// Word `w` of a slot's tail, from its head.
__device__ static inline u64 tail_of(const u64* head, i64 w) { return splitmix(head[w & (WORDS - 1)] ^ (u64)w); }

// Word `w` of a line's tail, from its carry.
__device__ static inline u64 line_tail(u64 c, i64 w) { return splitmix(c ^ (u64)w); }

__device__ static u64 block_sum(u64 v) {
  __shared__ u64 s[BLOCK];
  s[threadIdx.x] = v;
  __syncthreads();
  for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) s[threadIdx.x] += s[threadIdx.x + stride];
    __syncthreads();
  }
  u64 out = s[0];
  __syncthreads();
  return out;
}

// The whole slot of token `t` at position `p`, by the block; the
// thread's share of the head's rotated sum.
__device__ static u64 fill_slot(u64* s, i64 t, i64 p, i64 words, const i64* salt) {
  u64 acc = 0;
  for (int w = threadIdx.x; w < WORDS; w += BLOCK) {
    u64 m = mark(t, p, w, salt);
    s[w] = m;
    acc += rotl(m, (int)p);
  }
  __syncthreads();
  for (i64 w = WORDS + threadIdx.x; w < words; w += BLOCK) s[w] = tail_of(s, w);
  return acc;
}

// The sum over positions [0, n) of sequence `g`, through its page-table
// row, poisoned when any tail word disagrees with its head.
__device__ static u64 pages_sum(const int* table, int width, int page, const unsigned char* kv, i64 bytes_per_token,
                                i64 n) {
  const int* row = table + (i64)blockIdx.x * width;
  i64 limit = (i64)width * page;
  if (n > limit) n = limit;
  i64 words = bytes_per_token / 8;
  u64 acc = 0, bad = 0;
  for (i64 q = 0; q < n; ++q) {
    i64 slot = (i64)row[q / page] * page + q % page;
    const u64* s = reinterpret_cast<const u64*>(kv + slot * bytes_per_token);
    for (i64 w = threadIdx.x; w < words; w += BLOCK) {
      if (w < WORDS) acc += rotl(s[w], (int)q);
      else bad += s[w] != tail_of(s, w);
    }
  }
  u64 sum = block_sum(acc);
  return block_sum(bad) ? sum ^ POISON : sum;
}

// A line's carry, poisoned when its tail disagrees with it.
__device__ static u64 line_read(const u64* l, i64 line_bytes) {
  u64 bad = 0;
  for (i64 w = 2 + threadIdx.x; w < line_bytes / 8; w += BLOCK) bad += l[w] != line_tail(l[0], w);
  return block_sum(bad) ? l[0] ^ POISON : l[0];
}

__device__ static void line_write(u64* l, u64 c, i64 next, i64 line_bytes) {
  if (threadIdx.x == 0) {
    l[0] = c;
    l[1] = (u64)next;
  }
  for (i64 w = 2 + threadIdx.x; w < line_bytes / 8; w += BLOCK) l[w] = line_tail(c, w);
}

// The carry a group continues from: none at position 0, poisoned when
// the line does not continue at `p0` (a stale or foreign copy).
__device__ static inline u64 carry_at(const u64* l, u64 c, i64 p0) {
  return p0 == 0 ? 0 : ((i64)l[1] != p0 ? c ^ POISON : c);
}

extern "C" __global__ void toy_write(const i64* __restrict__ token_ids, const i64* __restrict__ positions,
                                     const i64* __restrict__ slot_mapping, unsigned char* __restrict__ kv,
                                     const i64* __restrict__ salt, i64 bytes_per_token) {
  i64 r = blockIdx.x;
  fill_slot(slot_words(kv, slot_mapping[r], bytes_per_token), token_ids[r], positions[r], bytes_per_token / 8, salt);
}

extern "C" __global__ void toy_predict(const int* __restrict__ seq_lens, const int* __restrict__ block_table,
                                       const unsigned char* __restrict__ kv, const i64* __restrict__ salt,
                                       i64* __restrict__ next_token, i64 bytes_per_token, int page, int width) {
  u64 s = pages_sum(block_table, width, page, kv, bytes_per_token, seq_lens[blockIdx.x]);
  if (threadIdx.x == 0) next_token[blockIdx.x] = next_of(s);
}

extern "C" __global__ void toy_predict_mem(const int* __restrict__ seq_lens, const int* __restrict__ block_table,
                                           const unsigned char* __restrict__ kv, const int* __restrict__ line_index,
                                           const unsigned char* __restrict__ mem, const i64* __restrict__ salt,
                                           i64* __restrict__ next_token, i64 bytes_per_token, int page, int width,
                                           i64 line_bytes) {
  u64 s = pages_sum(block_table, width, page, kv, bytes_per_token, seq_lens[blockIdx.x]);
  int line = line_index[blockIdx.x];
  u64 c = line > 0 ? line_read(reinterpret_cast<const u64*>(mem + (i64)line * line_bytes), line_bytes) : 0;
  if (threadIdx.x == 0) next_token[blockIdx.x] = next_of(s + c);
}

// Rows `g*rows .. (g+1)*rows` of group `g` folded into its line in order.
extern "C" __global__ void toy_fold(const i64* __restrict__ token_ids, const i64* __restrict__ positions,
                                    const int* __restrict__ line_index, unsigned char* __restrict__ mem,
                                    const i64* __restrict__ salt, int rows, i64 line_bytes) {
  int line = line_index[blockIdx.x];
  if (line <= 0) return;
  u64* l = reinterpret_cast<u64*>(mem + (i64)line * line_bytes);
  i64 r0 = (i64)blockIdx.x * rows;
  u64 c = carry_at(l, line_read(l, line_bytes), positions[r0]);
  __shared__ u64 folded;
  if (threadIdx.x == 0) {
    for (int i = 0; i < rows; ++i) c = fold(c, token_ids[r0 + i], positions[r0 + i], salt);
    folded = c;
  }
  __syncthreads();
  line_write(l, folded, positions[r0] + rows, line_bytes);
}

__device__ static void round_chain(const i64* token_ids, const i64* positions, const i64* slot_mapping,
                                   const int* block_table, unsigned char* kv, const i64* salt, i64* out, int* count,
                                   i64 bytes_per_token, int page, int width, int rows, u64 carry, u64* folded,
                                   int* taken) {
  i64 r0 = (i64)blockIdx.x * rows;
  u64 s = pages_sum(block_table, width, page, kv, bytes_per_token, positions[r0]);
  __shared__ i64 t;
  __shared__ u64 c;
  if (threadIdx.x == 0) {
    t = token_ids[r0];
    c = carry;
  }
  __syncthreads();
  for (int i = 0; i < rows; ++i) {
    i64 p = positions[r0 + i];
    u64 acc = fill_slot(slot_words(kv, slot_mapping[r0 + i], bytes_per_token), t, p, bytes_per_token / 8, salt);
    s += block_sum(acc);
    if (threadIdx.x == 0) {
      c = fold(c, t, p, salt);
      i64 n = next_of(s + (folded ? c : 0));
      out[r0 + i] = n;
      t = n;
      if (i == 0) *taken = 1 + (int)(s % (u64)rows);
      if (folded && i + 1 == *taken) *folded = c;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) count[blockIdx.x] = *taken;
}

extern "C" __global__ void toy_round(const i64* __restrict__ token_ids, const i64* __restrict__ positions,
                                     const i64* __restrict__ slot_mapping, const int* __restrict__ block_table,
                                     unsigned char* __restrict__ kv, const i64* __restrict__ salt,
                                     i64* __restrict__ out, int* __restrict__ count, i64 bytes_per_token, int page,
                                     int width, int rows) {
  __shared__ int taken;
  round_chain(token_ids, positions, slot_mapping, block_table, kv, salt, out, count, bytes_per_token, page, width,
              rows, 0, nullptr, &taken);
}

// The line folds the rows taken and no more, so it continues where the
// next step starts.
extern "C" __global__ void toy_round_mem(const i64* __restrict__ token_ids, const i64* __restrict__ positions,
                                         const i64* __restrict__ slot_mapping, const int* __restrict__ block_table,
                                         unsigned char* __restrict__ kv, const int* __restrict__ line_index,
                                         unsigned char* __restrict__ mem, const i64* __restrict__ salt,
                                         i64* __restrict__ out, int* __restrict__ count, i64 bytes_per_token,
                                         int page, int width, int rows, i64 line_bytes) {
  __shared__ int taken;
  __shared__ u64 folded;
  int line = line_index[blockIdx.x];
  u64* l = line > 0 ? reinterpret_cast<u64*>(mem + (i64)line * line_bytes) : nullptr;
  i64 p0 = positions[(i64)blockIdx.x * rows];
  u64 carry = l ? carry_at(l, line_read(l, line_bytes), p0) : 0;
  round_chain(token_ids, positions, slot_mapping, block_table, kv, salt, out, count, bytes_per_token, page, width,
              rows, carry, &folded, &taken);
  if (l) line_write(l, folded, p0 + taken, line_bytes);
}
