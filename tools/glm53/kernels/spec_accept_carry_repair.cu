// GLM-5.3 MTP round commit.
//
// This is a single-CTA, data-flow-only commit kernel. It replaces
// spec_accept, spec_carry_store, and spec_repair_meta. The caller provides
// the round width in rows (3 for k=2, 7 for k=6); nacc selects the committed
// prefix and no host branch is needed. The draft count is rows-1, so all
// pitches and the accept bound derive from rows; k=2 stays bitwise identical.
//
// One CTA is intentional. The prefix scan for advance_cu is exact and
// deterministic, and the hidden-state copy is only 8 * 4096 bf16 values at
// the largest serving batch. A multi-CTA split would require a second
// synchronization point before the carry copy or an atomic prefix scan.
#include <cuda_bf16.h>
#include <stdint.h>
using bf16 = __nv_bfloat16;

extern "C" __global__ void spec_accept_carry_repair(
    const int64_t* drafts, const int64_t* verify, const int* valid,
    const bf16* h, bf16* state, const int* lines,
    int64_t* tokens, int* nacc, int* cu,
    int64_t* repair_ids, int* repair_valid, int seqs, int rows) {
    if (blockIdx.x || blockIdx.y || blockIdx.z) return;

    __shared__ int shared_nacc;
    int offset = 0;
    if (threadIdx.x == 0) cu[0] = 0;
    __syncthreads();

    for (int s = 0; s < seqs; ++s) {
        if (threadIdx.x == 0) {
            int a = 0;
            if (valid[s * rows]) {
                while (a < rows - 1 && drafts[s * (rows - 1) + a] == verify[s * rows + a]) ++a;
                ++a;  // the bonus token is always committed
            }
            shared_nacc = a;
            nacc[s] = a;
            offset += a;
            cu[s + 1] = offset;
            for (int j = 0; j < rows; ++j) {
                tokens[s * rows + j] = verify[s * rows + j];
                repair_ids[s * rows + j] = verify[s * rows + j];
                repair_valid[s * rows + j] = (j < a - 1);
            }
        }
        __syncthreads();

        // Carry the last accepted target hidden state. This is the same
        // row convention as spec_carry_store: h[s*rows + a-1].
        int a = shared_nacc;
        if (a > 0 && valid[s * rows] && lines[s] > 0) {
            int64_t dst = (int64_t)lines[s] * 4096;
            int64_t src = ((int64_t)s * rows + a - 1) * 4096;
            for (int c = threadIdx.x; c < 4096; c += blockDim.x)
                state[dst + c] = h[src + c];
        }
        __syncthreads();
    }
}
