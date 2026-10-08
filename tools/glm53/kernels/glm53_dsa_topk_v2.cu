// Deterministic single-CTA top-512, ascending pool order, lowest id on ties.
// No atomic allocation of output positions. Integer histogram atomics only.
#include <cub/block/block_scan.cuh>
#include <cuda_runtime.h>
__device__ __forceinline__ unsigned score_key(float v) {
    // Finite logits are the model contract. NaNs rank as -inf; +/-0 tie.
    if(isnan(v)) v=-__int_as_float(0x7f800000);
    if(v==0.f) v=0.f;
    unsigned u=__float_as_uint(v);return u^((u&0x80000000u)?0xffffffffu:0x80000000u);
}
__device__ __forceinline__ int slot_of(const int* bt,int token) {
    return bt[token>>8]*256+(token&255);
}
extern "C" __global__ __launch_bounds__(1024) void glm53_dsa_topk_v2(
    const float* logits,const int* pools,const int* bt,const int* seq,int* dst,
    int logits_stride,int bt_cols) {
    asm volatile("griddepcontrol.wait;":::"memory");
    int t=threadIdx.x,row=blockIdx.x,len=pools[row],tail=seq[row]&3;
    const int* table=bt+(size_t)row*bt_cols;
    int* out=dst+(size_t)row*2051;
    const float* x=logits+(size_t)row*logits_stride;
    if(len<=512) {
        int count=len*4+tail;
        for(int j=t;j<2051;j+=1024) out[j]=j<count?slot_of(table,j):0;
    } else {
        __shared__ unsigned hist[32][256],prefix,mask,kleft;
        if(t==0) {prefix=0;mask=0;kleft=512;}
        __syncthreads();
        for(int shift=24;shift>=0;shift-=8) {
            for(int j=t;j<8192;j+=1024) (&hist[0][0])[j]=0;
            __syncthreads();
            for(int j=t;j<len;j+=1024) {
                unsigned key=score_key(x[j]);
                if((key&mask)==prefix) atomicAdd(&hist[t>>5][(key>>shift)&255],1u);
            }
            __syncthreads();
            if(t<256) {unsigned sum=0;for(int w=0;w<32;++w)sum+=hist[w][t];hist[0][t]=sum;}
            __syncthreads();
            if(t==0) {
                unsigned remain=kleft;int digit=255;
                for(;digit>0;--digit) {unsigned c=hist[0][digit];if(remain<=c)break;remain-=c;}
                prefix|=(unsigned)digit<<shift;mask|=255u<<shift;kleft=remain;
            }
            __syncthreads();
        }
        using Scan=cub::BlockScan<unsigned,1024,cub::BLOCK_SCAN_WARP_SCANS>;
        __shared__ Scan::TempStorage tmp;
        unsigned eqbase=0,gtbase=0;
        // Scan in increasing pool id. Per 1024-item chunk, 11 low bits
        // count ties; upper bits count better scores. Both counts <=1024.
        // A 32-bit warp-scan avoids the spills of a 64-bit raking scan.
        for(int base=0;base<len;base+=1024) {
            int j=base+t;unsigned key=j<len?score_key(x[j]):0;
            bool eq=j<len && key==prefix,gt=j<len && key>prefix;
            unsigned val=(unsigned)eq+((unsigned)gt<<11),before,total;
            Scan(tmp).ExclusiveSum(val,before,total);
            unsigned eqrank=eqbase+(before&2047u),gtrank=gtbase+(before>>11);
            if(gt || (eq && eqrank<kleft)) {
                unsigned rank=gtrank+min(eqrank,kleft);
                #pragma unroll
                for(int k=0;k<4;++k) out[rank*4+k]=slot_of(table,j*4+k);
            }
            eqbase+=total&2047u;gtbase+=total>>11;
            __syncthreads();
        }
        if(t<3) out[2048+t]=t<tail?slot_of(table,len*4+t):0;
    }
    asm volatile("griddepcontrol.launch_dependents;":::"memory");
}
