// Once-only byte permutations for the fused attention tensor layouts.
#include <stdint.h>
__device__ int query_row(int row) {
    // [tile32, head64, lane16] -> [head64, tile32, lane16]
    return ((row / 16) % 64) * 512 + (row / 1024) * 16 + row % 16;
}
__device__ int output_column(int col) {
    // [tile16, head8, lane32] -> [head8, tile16, lane32]
    return ((col / 32) % 8) * 512 + (col / 256) * 32 + col % 32;
}
extern "C" __global__ void dsv41_fused_weight_layout(uint8_t* out,const uint8_t* in,int kind) {
    const int rows=kind ? 8192 : 32768, k=kind ? 4096 : 1280;
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=rows*k)return;
    int row=i/k,col=i%k;
    out[i]=in[k*(kind ? row : query_row(row))+(kind ? output_column(col) : col)];
}
extern "C" __global__ void dsv41_fused_scale_layout(uint32_t* out,const uint8_t* in,int kind) {
    const int n=kind ? 1024 : 32768,k=kind ? 4096 : 1280,groups=kind ? 8 : 1;
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=groups*n*(k/128))return;
    int group=i/(n*(k/128)),row=i%n,word=(i/n)%(k/128);
    int srcrow=kind ? group*n+row : query_row(row);
    uint32_t value=0;
    for(int j=0;j<4;j++) {
        int block=word*4+j;
        int srcblock=kind ? output_column(block*32)/32 : block;
        value|=uint32_t(in[(srcrow/32)*(k/32)+srcblock])<<(j*8);
    }
    out[i]=value;
}
