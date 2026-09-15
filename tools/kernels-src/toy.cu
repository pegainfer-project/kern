// A model of nothing: integer kernels whose whole job is to be exact, so
// that a served answer is right or wrong with no margin to argue. The
// states are the thing under test — every byte they hold is written by
// these kernels and read back into the next token, so a page served
// stale, a slot remapped wrong, a line restored from the wrong copy all
// change the output.
//
// A token slot holds `words` marks, mark(t, p, w) for its token t at
// position p (the rest of the slot is untouched payload); a sequence's
// sum S over every word of every position it holds decides the next
// token: eos when (S >> 56) % EOS_EVERY == 0, else S % VOCAB_BYTES. The
// sum is commutative so any order of reading is one answer. A line (a
// per-sequence state) carries a fold C over its tokens in order, folded
// only for the tokens taken, plus how many; the next token adds C to S.
// tools/toy/model.py is the same arithmetic in Python.
//
//   toy_write:    marks of every row into its slot.            grid [rows]
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

#define EOS_EVERY 96ull
#define VOCAB_BYTES 256ull
#ifndef BLOCK
#define BLOCK 256
#endif

__device__ static inline u64 splitmix(u64 x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

__device__ static inline u64 mark(i64 t, i64 p, int w, const i64* salt) {
  return splitmix((u64)t * 0x100000001B3ull ^ (u64)p * 0x9E3779B1ull ^ (u64)w * 0xC2B2AE35ull ^ (u64)salt[w & 63]);
}

__device__ static inline u64 fold(u64 c, i64 t, i64 p, const i64* salt) {
  return ((c << 7) | (c >> 57)) ^ mark(t, p, 0, salt);
}

__device__ static inline i64 next_of(u64 s) {
  return ((s >> 56) % EOS_EVERY == 0) ? (i64)VOCAB_BYTES : (i64)(s % VOCAB_BYTES);
}

__device__ static inline u64* slot_words(unsigned char* kv, i64 slot, i64 bytes_per_token) {
  return reinterpret_cast<u64*>(kv + slot * bytes_per_token);
}

__device__ static inline u64 block_sum(u64 v) {
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

// The sum over positions [0, n) of sequence `g`, through its page-table row.
__device__ static u64 pages_sum(const int* table, int width, int page, const unsigned char* kv, i64 bytes_per_token,
                                int words, i64 n) {
  const int* row = table + (i64)blockIdx.x * width;
  i64 limit = (i64)width * page;
  if (n > limit) n = limit;
  u64 acc = 0;
  for (i64 i = threadIdx.x; i < n * words; i += BLOCK) {
    i64 q = i / words;
    int w = (int)(i % words);
    i64 slot = (i64)row[q / page] * page + q % page;
    acc += reinterpret_cast<const u64*>(kv + slot * bytes_per_token)[w];
  }
  return block_sum(acc);
}

__device__ static inline u64 line_carry(const int* line_index, const unsigned char* mem, i64 line_bytes) {
  int line = line_index[blockIdx.x];
  return line > 0 ? reinterpret_cast<const u64*>(mem + (i64)line * line_bytes)[0] : 0;
}

extern "C" __global__ void toy_write(const i64* __restrict__ token_ids, const i64* __restrict__ positions,
                                     const i64* __restrict__ slot_mapping, unsigned char* __restrict__ kv,
                                     const i64* __restrict__ salt, int words, i64 bytes_per_token) {
  i64 r = blockIdx.x;
  u64* s = slot_words(kv, slot_mapping[r], bytes_per_token);
  for (int w = threadIdx.x; w < words; w += BLOCK) s[w] = mark(token_ids[r], positions[r], w, salt);
}

extern "C" __global__ void toy_predict(const int* __restrict__ seq_lens, const int* __restrict__ block_table,
                                       const unsigned char* __restrict__ kv, const i64* __restrict__ salt,
                                       i64* __restrict__ next_token, int words, i64 bytes_per_token, int page,
                                       int width) {
  u64 s = pages_sum(block_table, width, page, kv, bytes_per_token, words, seq_lens[blockIdx.x]);
  if (threadIdx.x == 0) next_token[blockIdx.x] = next_of(s);
}

extern "C" __global__ void toy_predict_mem(const int* __restrict__ seq_lens, const int* __restrict__ block_table,
                                           const unsigned char* __restrict__ kv, const int* __restrict__ line_index,
                                           const unsigned char* __restrict__ mem, const i64* __restrict__ salt,
                                           i64* __restrict__ next_token, int words, i64 bytes_per_token, int page,
                                           int width, i64 line_bytes) {
  u64 s = pages_sum(block_table, width, page, kv, bytes_per_token, words, seq_lens[blockIdx.x]);
  if (threadIdx.x == 0) next_token[blockIdx.x] = next_of(s + line_carry(line_index, mem, line_bytes));
}

// Rows `g*rows .. (g+1)*rows` of group `g` folded into its line in order.
// A line that does not continue at the first row's position (a stale or
// foreign copy) is marked so the next token shows it.
extern "C" __global__ void toy_fold(const i64* __restrict__ token_ids, const i64* __restrict__ positions,
                                    const int* __restrict__ line_index, unsigned char* __restrict__ mem,
                                    const i64* __restrict__ salt, int rows, i64 line_bytes) {
  if (threadIdx.x != 0) return;
  int line = line_index[blockIdx.x];
  if (line <= 0) return;
  u64* l = reinterpret_cast<u64*>(mem + (i64)line * line_bytes);
  i64 r0 = (i64)blockIdx.x * rows;
  u64 c = l[0];
  if (positions[r0] == 0) c = 0;
  else if ((i64)l[1] != positions[r0]) c ^= 0xBAD0BAD0BAD0BAD0ull;
  for (int i = 0; i < rows; ++i) c = fold(c, token_ids[r0 + i], positions[r0 + i], salt);
  l[0] = c;
  l[1] = (u64)(positions[r0] + rows);
}

__device__ static void round_chain(const i64* token_ids, const i64* positions, const i64* slot_mapping,
                                   const int* block_table, unsigned char* kv, const i64* salt, i64* out, int* count,
                                   int words, i64 bytes_per_token, int page, int width, int rows, u64 carry,
                                   u64* folded, int* taken) {
  i64 r0 = (i64)blockIdx.x * rows;
  u64 s = pages_sum(block_table, width, page, kv, bytes_per_token, words, positions[r0]);
  __shared__ i64 t;
  __shared__ u64 c;
  if (threadIdx.x == 0) { t = token_ids[r0]; c = carry; }
  __syncthreads();
  for (int i = 0; i < rows; ++i) {
    i64 p = positions[r0 + i];
    u64* w = slot_words(kv, slot_mapping[r0 + i], bytes_per_token);
    u64 acc = 0;
    for (int j = threadIdx.x; j < words; j += BLOCK) {
      u64 m = mark(t, p, j, salt);
      w[j] = m;
      acc += m;
    }
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
                                     i64* __restrict__ out, int* __restrict__ count, int words, i64 bytes_per_token,
                                     int page, int width, int rows) {
  __shared__ int taken;
  round_chain(token_ids, positions, slot_mapping, block_table, kv, salt, out, count, words, bytes_per_token, page,
              width, rows, 0, nullptr, &taken);
}

// The line folds the rows taken and no more, so it continues where the
// next step starts.
extern "C" __global__ void toy_round_mem(const i64* __restrict__ token_ids, const i64* __restrict__ positions,
                                         const i64* __restrict__ slot_mapping, const int* __restrict__ block_table,
                                         unsigned char* __restrict__ kv, const int* __restrict__ line_index,
                                         unsigned char* __restrict__ mem, const i64* __restrict__ salt,
                                         i64* __restrict__ out, int* __restrict__ count, int words,
                                         i64 bytes_per_token, int page, int width, int rows, i64 line_bytes) {
  __shared__ int taken;
  __shared__ u64 folded;
  int line = line_index[blockIdx.x];
  u64* l = line > 0 ? reinterpret_cast<u64*>(mem + (i64)line * line_bytes) : nullptr;
  i64 p0 = positions[(i64)blockIdx.x * rows];
  u64 carry = l ? (p0 == 0 ? 0 : ((i64)l[1] != p0 ? l[0] ^ 0xBAD0BAD0BAD0BAD0ull : l[0])) : 0;
  round_chain(token_ids, positions, slot_mapping, block_table, kv, salt, out, count, words, bytes_per_token, page,
              width, rows, carry, &folded, &taken);
  if (threadIdx.x == 0 && l) {
    l[0] = folded;
    l[1] = (u64)(p0 + taken);
  }
}
