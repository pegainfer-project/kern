// Restore original FP8 block-scaled WO_A weights once, without offline export.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <math.h>
extern "C" __global__ void dsv41_woa_dequant(
    const uint8_t* weight, const uint8_t* scale, __nv_bfloat16* output, int n, int k) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)n * k) return;
    int row = i / k, column = i % k;
    __nv_fp8_e4m3 w; w.__x = weight[i];
    uint8_t e = scale[(int64_t)(row / 32) * (k / 32) + column / 32];
    // E8M0 is a power of two, except 255 (NaN), exactly as checkpoint cast.
    float s = e == 255 ? nanf("") : ldexpf(1.0f, (int)e - 127);
    output[i] = __float2bfloat16_rn((float)w * s);
}
