// Engram lookup over a table sharded across the EP group. Rank r holds
// rows [min(r*per_rank, total-per_rank), +per_rank) of the table; the last
// slice overlaps the one before it so every rank's buffer has one shape.
// `tables` / `scales` are the group's addresses of those buffers (a `peer`
// buffer), so each row is a plain load from whichever rank holds it. The
// arithmetic per element is dsv41_lookup's (auxiliary.cu).
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <math.h>
using bf16 = __nv_bfloat16;
// Programmatic dependent launch, as in auxiliary.cu: wait before touching
// anything, trigger right after so the next launch's prologue overlaps.
__device__ __forceinline__ void pdl() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
extern "C" __global__ void dsv41_lookup_peers(bf16* out,const int64_t* ids,const uint8_t* const* tables,const uint8_t* const* scales,int rows,int cols,int dim,int id_stride,int id_offset,int ranks,int64_t per_rank,int64_t total){
 pdl();
 int t=blockIdx.x, head=blockIdx.y;
 int64_t row=ids[(int64_t)t*id_stride+id_offset+head];
 int64_t r=min(row/per_rank,(int64_t)ranks-1);
 int64_t local=row-min(r*per_rank,total-per_rank);
 const uint8_t* table=tables[r]; const uint8_t* scale=scales[r];
 for(int d=threadIdx.x;d<dim;d+=blockDim.x){
  __nv_fp8_e4m3 v; v.__x=table[local*dim+d];
  uint8_t e=scale[local*(dim/32)+d/32];
  float s=e==255?NAN:ldexpf(1.0f,(int)e-127);
  out[((int64_t)t*cols+head)*dim+d]=__float2bfloat16((float)v*s);
 }
}
