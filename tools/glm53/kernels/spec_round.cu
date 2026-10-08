// GLM-5.3 MTP k>=2 data-flow glue. No host decisions depend on nacc.
// All rows are sequence-major. Never treat a batch-row carry as sequence state.
// The round width (tokens per sequence) and the draft count are runtime ABI
// (rows/per args), so k=2 stays bitwise identical: rows=3, per=2.
#include <cuda_bf16.h>
#include <stdint.h>
using bf16 = __nv_bfloat16;

extern "C" __global__ void spec_splice(
    const int64_t* anchor, const int64_t* drafts, int64_t* ids, int rows) {
    int s = blockIdx.x, j = threadIdx.x;
    if (j < rows) ids[s*rows+j] = j == 0 ? anchor[s] : drafts[s*(rows-1)+j-1];
}

// verify row j predicts the token AFTER input row j. Include one bonus.
// cu is compact: gather recurrent inputs with spec_pack BEFORE using this cu.
// Invalid sequences have nacc=0 (they must not update pooled state).
extern "C" __global__ void spec_accept(
    const int64_t* drafts, const int64_t* verify, const int* valid,
    int64_t* tokens, int* nacc, int* cu, int seqs, int rows) {
    if (threadIdx.x || blockIdx.x) return;
    int offset = 0;
    cu[0] = 0;
    for (int s=0; s<seqs; ++s) {
        int a = 0;
        if (valid[s*rows]) {
            while (a < rows-1 && drafts[s*(rows-1)+a] == verify[s*rows+a]) ++a;
            ++a;
        }
        nacc[s] = a;
        offset += a;
        cu[s+1] = offset;
        for (int j=0; j<rows; ++j) tokens[s*rows+j] = verify[s*rows+j];
    }
}

// Compact each accepted prefix. Pitches in elements; no read of rejected rows.
extern "C" __global__ void spec_pack(
    const bf16* src, bf16* dst, const int* nacc, const int* cu,
    int cols, int src_pitch, int rows) {
    int s=blockIdx.x, j=blockIdx.y;
    if (j >= nacc[s]) return;
    for (int c=threadIdx.x; c<cols; c+=blockDim.x)
        dst[(int64_t)(cu[s]+j)*cols+c] = src[(int64_t)(rows*s+j)*src_pitch+c];
}

// H is POST hc_contract AND final_norm. Each sequence has its own pool line.
extern "C" __global__ void spec_carry_store(
    const bf16* h, bf16* state, const int* lines, const int* nacc,
    const int* valid, int rows) {
    int s=blockIdx.x, a=nacc[s];
    if (!valid[s*rows] || a < 1 || lines[s] <= 0) return;
    for (int c=threadIdx.x; c<4096; c+=blockDim.x)
        state[(int64_t)lines[s]*4096+c] = h[(int64_t)(s*rows+a-1)*4096+c];
}
extern "C" __global__ void spec_carry_load(
    const bf16* state, const int* lines, bf16* h) {
    int s=blockIdx.x;
    for (int c=threadIdx.x; c<4096; c+=blockDim.x)
        h[(int64_t)s*4096+c] = lines[s] > 0 ? state[(int64_t)lines[s]*4096+c] : __float2bfloat16(0);
}

// Delayed first draft: token x[p] + target H[p-1], stored at draft pos p-1.
// Second autoregressive draft: d1 + draft hidden, stored at p.
// At bootstrap p=0, valid=0; all draft cache writers MUST use this validity.
extern "C" __global__ void spec_draft_meta(
    const int* positions, const int* valid, const int* bt,
    int* draft_pos, int* draft_len, int* draft_slot, int* draft_valid,
    int per, int step, int bt_cols) {
    int s=blockIdx.x;
    if (threadIdx.x) return;
    int p=positions[s*per]-1+step;
    int ok=valid[s*per] && p >= 0;
    p=max(p,0);
    draft_pos[s]=p; draft_len[s]=p+1; draft_valid[s]=ok;
    draft_slot[s]=ok ? bt[(int64_t)s*bt_cols+(p>>8)]*256+(p&255) : 0;
}

// Accepted draft cache repair uses TARGET h[j] with successor verify[j],
// at the same draft position p+j. The final accepted slot may be delayed
// until next round; this version repairs nacc-1 slots (at most two).
extern "C" __global__ void spec_repair_meta(
    const int64_t* verify, const int* positions, const int* nacc,
    int64_t* ids, int* valid, int rows) {
    int s=blockIdx.x, j=threadIdx.x;
    if (j<rows) { ids[rows*s+j]=verify[rows*s+j]; valid[rows*s+j]=(j < nacc[s]-1); }
}

extern "C" __global__ void spec_uniform(const int* valid, int* ones, int* cu, int rows) {
    int s=blockIdx.x;
    if (threadIdx.x) return;
    ones[s]=valid[s*rows] ? 1 : 0;
    cu[s]=s;
    if (s==gridDim.x-1) cu[s+1]=s+1;
}
extern "C" __global__ void spec_record_draft(const int64_t* next, int64_t* drafts, int step, int per) {
    if (!threadIdx.x) drafts[blockIdx.x*per+step]=next[blockIdx.x];
}
extern "C" __global__ void spec_pack_kda(
    const bf16* fused, const bf16* conv, const bf16* forget,
    bf16* pf, bf16* pc, bf16* pa, const int* nacc, const int* cu, int rows) {
    int s=blockIdx.x,j=blockIdx.y;
    if (j>=nacc[s]) return;
    int src=rows*s+j,dst=cu[s]+j;
    for (int c=threadIdx.x;c<3336;c+=blockDim.x) pf[(int64_t)dst*3336+c]=fused[(int64_t)src*3336+c];
    for (int c=threadIdx.x;c<3072;c+=blockDim.x) pc[(int64_t)dst*3072+c]=conv[(int64_t)src*3072+c];
    for (int c=threadIdx.x;c<1024;c+=blockDim.x) pa[(int64_t)dst*1024+c]=forget[(int64_t)src*1024+c];
}
