// Load-time transforms from raw checkpoint rows; no checkpoint export.
// blockIdx.y is the expert: sources and destinations are contiguous per
// expert, so one launch prepares every expert of a layer and a grid of
// height one is the single-expert case.
#include <stdint.h>
extern "C" __global__ void dsv41_interleave_gate_up(uint8_t* out,const uint8_t* gate,const uint8_t* up,int n,int row_bytes){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=2*n*row_bytes)return;
 size_t expert=blockIdx.y;out+=expert*size_t(2*n)*row_bytes;gate+=expert*size_t(n)*row_bytes;up+=expert*size_t(n)*row_bytes;
 int row=i/row_bytes,col=i%row_bytes,srcrow=(row/16)*8+row%8;
 out[i]=(row%16<8?gate:up)[srcrow*row_bytes+col];
}
// Source e8m0 scales [N/row_group,K/32]; destination packed col-major
// UTCCP row order [K/128,N]. Interleave first when gate_up is nonzero.
extern "C" __global__ void dsv41_pack_expert_sf(uint32_t* out,const uint8_t* first,const uint8_t* second,int n,int k,int row_group,int gate_up){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n*(k/128))return;
 size_t expert=blockIdx.y,source_rows=(gate_up?n/2:n)/row_group;out+=expert*size_t(n)*(k/128);first+=expert*source_rows*(k/32);second+=expert*source_rows*(k/32);
 int dstrow=i%n,word=i/n;
 int row=(dstrow/128)*128+(dstrow%4)*32+(dstrow%128)/4;
 const uint8_t* source=first;
 if(gate_up){source=row%16<8?first:second;row=(row/16)*8+row%8;}
 int p=(row/row_group)*(k/32)+word*4;
 out[i]=uint32_t(source[p])|(uint32_t(source[p+1])<<8)|(uint32_t(source[p+2])<<16)|(uint32_t(source[p+3])<<24);
}
// Dense/BMM scales: no UTCCP permutation. Source scales [groups*N/row_group,K/32]
// and destination words [groups,K/128,N]. N must be divisible by 32.
extern "C" __global__ void dsv41_pack_dense_sf(uint32_t* out,const uint8_t* sf,int n,int k,int groups,int row_group){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=groups*n*(k/128))return;int group=i/(n*(k/128)),row=i%n,word=(i/n)%(k/128);int src=((group*n+row)/row_group)*(k/32)+word*4;out[i]=uint32_t(sf[src])|(uint32_t(sf[src+1])<<8)|(uint32_t(sf[src+2])<<16)|(uint32_t(sf[src+3])<<24);
}
