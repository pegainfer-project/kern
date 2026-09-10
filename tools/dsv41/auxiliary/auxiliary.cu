// DSV4.1 auxiliary operators. Semantics follow the model's inference reference.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <math.h>
using bf16 = __nv_bfloat16;
__device__ float sum256(float x) {
  __shared__ float sums[8];
  for(int d=16; d; d>>=1) x += __shfl_down_sync(0xffffffff,x,d);
  if((threadIdx.x&31)==0) sums[threadIdx.x>>5]=x;
  __syncthreads();
  x=threadIdx.x<8?sums[threadIdx.x]:0;
  if(threadIdx.x<32) for(int d=16;d;d>>=1)x+=__shfl_down_sync(0xffffffff,x,d);
  if(threadIdx.x==0)sums[0]=x;
  __syncthreads(); return sums[0];
}
extern "C" __global__ void dsv41_history(int64_t* history,const int32_t* ids,const int64_t* slots,const int64_t* map,const uint8_t* mask,int rows){
 int t=blockIdx.x*blockDim.x+threadIdx.x;
 if(t<rows && slots[t]>=0) history[slots[t]]=mask[t]?map[ids[t]]:-1;
}
extern "C" __global__ void dsv41_hash(int64_t* out,const int64_t* history,const int32_t* req,const int32_t* pos,const int32_t* pages,const int64_t* mul,const int64_t* prime,const int64_t* offset,int rows,int page_size,int page_stride,int layers,int heads,int ngram,int64_t pad){
 int t=blockIdx.x; int col=threadIdx.x; int cols=(ngram-1)*heads;
 if(t>=rows||col>=layers*cols)return;
 int layer=col/cols, gram=(col%cols)/heads+2; bool blocked=false; int64_t value=0;
 for(int k=0;k<gram;k++){
   int p=pos[t]-k; int64_t v=p>=0?history[(int64_t)pages[req[t]*page_stride+p/page_size]*page_size+p%page_size]:pad;
   blocked=blocked||p<0||v==-1;
   value^=(blocked?pad:v)*mul[layer*ngram+k];
 }
 out[(int64_t)t*layers*cols+col]=value%prime[col]+offset[col];
}
extern "C" __global__ void dsv41_lookup(bf16* out,const int64_t* ids,const uint8_t* table,const uint8_t* scales,int rows,int cols,int dim,int id_stride,int id_offset){
 int t=blockIdx.x, head=blockIdx.y;
 int64_t row=ids[(int64_t)t*id_stride+id_offset+head];
 for(int d=threadIdx.x;d<dim;d+=blockDim.x){
  __nv_fp8_e4m3 v; v.__x=table[row*dim+d];
  uint8_t e=scales[row*(dim/32)+d/32];
  float s=e==255?NAN:ldexpf(1.0f,(int)e-127);
  out[((int64_t)t*cols+head)*dim+d]=__float2bfloat16((float)v*s);
 }
}
extern "C" __global__ void dsv41_engram_inject(bf16* out,const bf16* residual,const bf16* kv,const bf16* qw,const bf16* kw,const uint8_t* mask,int rows,int copies,int dim,float eps){
 int t=blockIdx.x,c=blockIdx.y; int64_t base=((int64_t)t*copies+c)*dim;
 const bf16* key=kv+((int64_t)t*(copies+1)+c)*dim;
 float hh=0,kk=0,dot=0;
 for(int d=threadIdx.x;d<dim;d+=256){float h=__bfloat162float(residual[base+d]),k=__bfloat162float(key[d]);hh+=h*h;kk+=k*k;dot+=h*(__bfloat162float(qw[c*dim+d])*__bfloat162float(kw[c*dim+d]))*k;}
 float hs=sum256(hh); __syncthreads(); float ks=sum256(kk); __syncthreads(); float ds=sum256(dot);
 float z=ds*rsqrtf(hs/dim+eps)*rsqrtf(ks/dim+eps)*rsqrtf((float)dim);
 float gate=mask[t]?1.0f/(1.0f+expf(-copysignf(sqrtf(fmaxf(fabsf(z),1e-6f)),z))):0;
 for(int d=threadIdx.x;d<dim;d+=256)out[base+d]=__float2bfloat16(__bfloat162float(residual[base+d])+gate*__bfloat162float(kv[((int64_t)t*(copies+1)+copies)*dim+d]));
}
// Inputs are gathered complete groups [groups,ratio,dim]. Staging incomplete
// groups is a separate state operation; this op never mutates accepted history.
extern "C" __global__ void dsv41_compress2(bf16* out,const float* kv,const float* score,const bf16* weight,int groups,int dim,float eps){
 int g=blockIdx.x; __shared__ float values[1024];float ss=0;
 for(int d=threadIdx.x;d<dim;d+=256){int64_t p=(int64_t)g*2*dim+d;float a=score[p],b=score[p+dim],m=fmaxf(a,b),ea=expf(a-m),eb=expf(b-m);float v=__bfloat162float(__float2bfloat16(kv[p]*(ea/(ea+eb))+kv[p+dim]*(eb/(ea+eb))));values[d]=v;ss+=v*v;}
 float s=sum256(ss),inv=rsqrtf(s/dim+eps);
 for(int d=threadIdx.x;d<dim;d+=256)out[(int64_t)g*dim+d]=__float2bfloat16((values[d]*inv)*__bfloat162float(weight[d]));
}
extern "C" __global__ void dsv41_norm(bf16* out,const bf16* x,const bf16* weight,int rows,int dim,float eps){
 int row=blockIdx.x;float ss=0;
 for(int d=threadIdx.x;d<dim;d+=256){float v=__bfloat162float(x[(int64_t)row*dim+d]);ss+=v*v;}
 float s=sum256(ss),inv=rsqrtf(s/dim+eps);
 for(int d=threadIdx.x;d<dim;d+=256)out[(int64_t)row*dim+d]=__float2bfloat16((__bfloat162float(x[(int64_t)row*dim+d])*inv)*__bfloat162float(weight[d]));
}
// The normalized row, rounded to BF16 exactly as dsv41_norm rounds it. Input
// rows may be a slice of a wider projection output, hence the element stride.
__device__ bf16 normed(const bf16* row,const bf16* weight,int d,float inv){
 return __float2bfloat16((__bfloat162float(row[d])*inv)*__bfloat162float(weight[d]));
}
// dsv41_norm followed by the MXFP8 activation quantization of its result, one
// block per token. The normalized rows stay live for the compressor, and the
// quantization re-reads them so its bytes are the ones dsv41_dense_quant_x
// would have produced. Grid covers align4(rows): trailing blocks only clear
// the scale words of the four-row padding, matching that kernel's guard.
//   out [rows,dim] bf16, fp8 [rows,dim] e4m3, sf [dim/128,sf_stride] i32
extern "C" __global__ void dsv41_norm_quant(bf16* out,unsigned char* fp8,int* sf,const bf16* x,const bf16* weight,int rows,int dim,int x_stride,int sf_stride,float eps){
 int token=blockIdx.x,words=dim/128;
 if(token>=rows){for(int word=threadIdx.x;word<words;word+=256)sf[(int64_t)word*sf_stride+token]=0;return;}
 const bf16* src=x+(int64_t)token*x_stride;float ss=0;
 for(int d=threadIdx.x;d<dim;d+=256){float v=__bfloat162float(src[d]);ss+=v*v;}
 float s=sum256(ss),inv=rsqrtf(s/dim+eps);
 bf16* row=out+(int64_t)token*dim;
 for(int d=threadIdx.x;d<dim;d+=256)row[d]=normed(src,weight,d,inv);
 __syncthreads();
 int lane=threadIdx.x%32,group=lane/8,base=(lane%8)*4;
 for(int word=threadIdx.x/32;word<words;word+=8){
  const bf16* in=row+word*128;unsigned char* q=fp8+(int64_t)token*dim+word*128;
  float v[4],amax=0;
  for(int i=0;i<4;i++){v[i]=__bfloat162float(in[group*32+base+i]);amax=fmaxf(amax,fabsf(v[i]));}
  for(int offset=4;offset;offset>>=1)amax=fmaxf(amax,__shfl_xor_sync(0xffffffffu,amax,offset));
  amax=fmaxf(amax,1e-4f);
  unsigned int bits=__float_as_uint(amax/448.0f)&0x7fffffffu;
  int exp=(int)((bits>>23)&0xffu)+((bits&0x7fffffu)!=0u?1:0);exp=exp<1?1:(exp>254?254:exp);
  float inv_sf=1.0f/__uint_as_float((unsigned int)exp<<23);
  for(int i=0;i<4;i++)q[group*32+base+i]=(unsigned char)__nv_cvt_float_to_fp8(v[i]*inv_sf,__NV_SATFINITE,__NV_E4M3);
  unsigned int e=(unsigned int)exp;
  unsigned int e0=__shfl_sync(0xffffffffu,e,0),e1=__shfl_sync(0xffffffffu,e,8),e2=__shfl_sync(0xffffffffu,e,16),e3=__shfl_sync(0xffffffffu,e,24);
  if(lane==0)sf[(int64_t)word*sf_stride+token]=(int)(e0|(e1<<8)|(e2<<16)|(e3<<24));
 }
}
extern "C" __global__ void dsv41_compressor_stage(float* kv_history,float* score_history,const float* kv,const float* score,const int64_t* slots,int rows,int dim){
 int row=blockIdx.x; int64_t dst=slots[row]*dim;
 for(int d=threadIdx.x;d<dim;d+=256){kv_history[dst+d]=kv[(int64_t)row*dim+d];score_history[dst+d]=score[(int64_t)row*dim+d];}
}
extern "C" __global__ void dsv41_compressor_gather(float* kv,float* score,int32_t* valid,const float* kv_history,const float* score_history,const int32_t* req,const int32_t* pos,const int32_t* pages,int rows,int dim,int page_size,int page_stride){
 int row=blockIdx.x; int p=pos[row];bool complete=(p%2)==1;
 if(threadIdx.x==0)valid[row]=complete;
 for(int d=threadIdx.x;d<2*dim;d+=256){
  int source=p-1+d/dim; int64_t slot=complete?(int64_t)pages[req[row]*page_stride+source/page_size]*page_size+source%page_size:0;
  kv[(int64_t)row*2*dim+d]=complete?kv_history[slot*dim+d%dim]:0;
  score[(int64_t)row*2*dim+d]=complete?score_history[slot*dim+d%dim]:0;
 }
}
// Window token index inside a sequence's ring: ring tokens per slot, position
// modulo ring inside it. The slot is the sequence's window line.
__device__ int64_t ring_slot(const int32_t* lines,int seq,int position,int ring){return (int64_t)lines[seq]*ring+position%ring;}
// Logical causal window indices. Each query receives exactly window entries;
// invalid prefix entries are -1. DSpark extends visibility to block_end[row].
extern "C" __global__ void dsv41_window_indices(int32_t* out,const int32_t* req,const int32_t* pos,const int32_t* block_end,const int32_t* lines,int rows,int window,int width,int ring,int noncausal,int draft_rows){
 int row=blockIdx.x;int end=noncausal?block_end[row]:pos[row]+1;
 int start=max(0,noncausal?end-window-draft_rows:pos[row]-window+1);int count=end-start;
 for(int i=threadIdx.x;i<width;i+=256)out[(int64_t)row*width+i]=(i<count)?(int32_t)ring_slot(lines,req[row],start+i,ring):-1;
}
extern "C" __global__ void dsv41_map_indices(int32_t* out,const int32_t* logical,const int32_t* req,const int32_t* pos,const int32_t* pages,int rows,int topk,int ratio,int page_size,int page_stride){
 int row=blockIdx.x;int available=(pos[row]+1)/ratio;
 for(int i=threadIdx.x;i<topk;i+=256){int p=logical[(int64_t)row*topk+i];out[(int64_t)row*topk+i]=(p>=0&&p<available)?pages[req[row]*page_stride+p/page_size]*page_size+p%page_size:-1;}
}
// Page-major FlashMLA cache: all packed values, then all scales in each page.
extern "C" __global__ void dsv41_cache_fp8(uint8_t* cache,const bf16* x,const int64_t* slots,int rows,int page_size){
 int row=blockIdx.x,group=threadIdx.x/32,lane=threadIdx.x%32;
 int64_t slot=slots[row];if(slot<0)return;
 int64_t base=(slot/page_size)*(int64_t)page_size*528;int token=slot%page_size;
 for(int g=group;g<16;g+=8){float v=__bfloat162float(x[(int64_t)row*512+g*32+lane]),amax=fabsf(v);
 for(int d=16;d;d>>=1)amax=fmaxf(amax,__shfl_xor_sync(0xffffffff,amax,d));
 float s=exp2f(ceilf(log2f(fmaxf(amax,1e-4f)*(1.0f/448.0f))));
 __nv_fp8_e4m3 q(v/s);cache[base+token*512+g*32+lane]=q.__x;
 if(lane==0)cache[base+(int64_t)page_size*512+token*16+g]=(__float_as_uint(s)>>23)&255;
 }
}
__device__ float fp4_value(int q){const float table[8]={0,.5f,1,1.5f,2,3,4,6};return (q&8)?-table[q&7]:table[q&7];}
__device__ uint8_t fp4_quant(float x){float a=fminf(fabsf(x),6.0f),best=INFINITY;int q=0;
 for(int i=0;i<8;i++){float d=fabsf(a-fp4_value(i));if(d<best||(d==best&&(i%2)==0)){best=d;q=i;}}
 return q|(signbit(x)?8:0);
}
extern "C" __global__ void dsv41_cache_fp4(uint8_t* cache,const bf16* x,const int64_t* slots,int rows,int page_size){
 int row=blockIdx.x;int64_t slot=slots[row];if(slot<0)return;
 int64_t base=(slot/page_size)*(int64_t)page_size*288;int token=slot%page_size;
 for(int g=threadIdx.x;g<32;g+=256){float amax=6.0f/512;
 for(int i=0;i<16;i++)amax=fmaxf(amax,fabsf(__bfloat162float(x[(int64_t)row*512+g*16+i])));
 __nv_fp8_e4m3 scale(amax/6);float s=(float)scale;
 cache[base+(int64_t)page_size*256+token*32+g]=scale.__x;
 for(int i=0;i<8;i++){float a=__bfloat162float(x[(int64_t)row*512+g*16+2*i])/s,b=__bfloat162float(x[(int64_t)row*512+g*16+2*i+1])/s;
 cache[base+token*256+g*8+i]=fp4_quant(a)|(fp4_quant(b)<<4);}
 }
}
extern "C" __global__ void dsv41_cache_gather(bf16* out,const uint8_t* cache,const int64_t* slots,int rows,int page_size,int fp4){
 int row=blockIdx.x;int64_t slot=slots[row];int bytes=fp4?288:528;int64_t base=slot>=0?(slot/page_size)*(int64_t)page_size*bytes:0;int token=slot>=0?slot%page_size:0;
 for(int d=threadIdx.x;d<512;d+=256){float v=0;
 if(slot>=0){if(fp4){uint8_t q=cache[base+token*256+d/2];__nv_fp8_e4m3 scale;scale.__x=cache[base+(int64_t)page_size*256+token*32+d/16];v=fp4_value((q>>((d%2)*4))&15)*(float)scale;}
 else {__nv_fp8_e4m3 q;q.__x=cache[base+token*512+d];uint8_t s=cache[base+(int64_t)page_size*512+token*16+d/32];v=(float)q*ldexpf(1.0f,(int)s-127);}}
 out[(int64_t)row*512+d]=__float2bfloat16(v);}
}
// Interleaved real/imaginary RoPE tail, supplied cosine/sine preserves YaRN.
extern "C" __global__ void dsv41_rope(bf16* out,const bf16* x,const float* cos_sin,const int32_t* positions,int rows,int heads,int dim,int rope_dim,int inverse){
 int row=blockIdx.x,head=blockIdx.y;
 for(int d=threadIdx.x;d<dim;d+=256){int64_t i=((int64_t)row*heads+head)*dim+d;
 if(d<dim-rope_dim)out[i]=x[i];
 else{int tail=d-(dim-rope_dim),pair=tail/2;if((tail&1)==0){int64_t f=(int64_t)positions[row]*rope_dim+pair*2;float c=cos_sin[f],s=cos_sin[f+1]*(inverse?-1:1);float a=__bfloat162float(x[i]),b=__bfloat162float(x[i+1]);out[i]=__float2bfloat16(a*c-b*s);out[i+1]=__float2bfloat16(a*s+b*c);}}}
}
// dsv41_norm followed by dsv41_rope over its result for a single head, one
// block per row. Nothing reads the normalized rows on their own, so only the
// rotated ones are written; the tail pair is normalized where it is rotated.
extern "C" __global__ void dsv41_norm_rope(bf16* out,const bf16* x,const bf16* weight,const float* cos_sin,const int32_t* positions,int rows,int dim,int rope_dim,int x_stride,int inverse,float eps){
 int row=blockIdx.x;const bf16* src=x+(int64_t)row*x_stride;float ss=0;
 for(int d=threadIdx.x;d<dim;d+=256){float v=__bfloat162float(src[d]);ss+=v*v;}
 float inv=rsqrtf(sum256(ss)/dim+eps);
 for(int d=threadIdx.x;d<dim;d+=256){int64_t i=(int64_t)row*dim+d;
 if(d<dim-rope_dim)out[i]=normed(src,weight,d,inv);
 else{int tail=d-(dim-rope_dim),pair=tail/2;if((tail&1)==0){int64_t f=(int64_t)positions[row]*rope_dim+pair*2;float c=cos_sin[f],s=cos_sin[f+1]*(inverse?-1:1);
 float a=__bfloat162float(normed(src,weight,d,inv)),b=__bfloat162float(normed(src,weight,d+1,inv));out[i]=__float2bfloat16(a*c-b*s);out[i+1]=__float2bfloat16(a*s+b*c);}}}
}
// General MXFP4 preparation for indexer Q/K; BF16 dequant output is optional
// downstream oracle/scoring input, packed bytes and scales feed optimized GEMM.
extern "C" __global__ void dsv41_index_quant(uint8_t* packed,uint8_t* scales,bf16* dequant,const bf16* x,int rows,int dim){
 int row=blockIdx.x;
 for(int g=threadIdx.x;g<dim/32;g+=256){float amax=6*0x1p-126f;
 for(int i=0;i<32;i++)amax=fmaxf(amax,fabsf(__bfloat162float(x[(int64_t)row*dim+g*32+i])));
 float s=exp2f(ceilf(log2f(amax/6)));scales[(int64_t)row*(dim/32)+g]=(__float_as_uint(s)>>23)&255;
 for(int i=0;i<16;i++){int64_t p=(int64_t)row*dim+g*32+2*i;uint8_t a=fp4_quant(__bfloat162float(x[p])/s),b=fp4_quant(__bfloat162float(x[p+1])/s);
 packed[(int64_t)row*(dim/2)+g*16+i]=a|(b<<4);dequant[p]=__float2bfloat16(fp4_value(a)*s);dequant[p+1]=__float2bfloat16(fp4_value(b)*s);}
 }
}
// Bounded ratio2 state: one last-token KV/score pair per sequence line.
// line [2,dim] f32, line_table [seqs,line_width], row_starts [seqs+1].
// Verification never writes committed state. Odd first rows read its KV half;
// all other pairs come exclusively from this invocation's projected workspace.
extern "C" __global__ void dsv41_compressor_short_gather(float* kv_out,float* score_out,int32_t* valid,const float* state,const float* kv,const float* score,const int32_t* lines,const int32_t* starts,const int32_t* req,const int32_t* pos,int rows,int dim,int line_width){
 int row=blockIdx.x,seq=req[row],first=starts[seq];bool complete=(pos[row]&1)!=0;
 if(threadIdx.x==0)valid[row]=complete;
 for(int d=threadIdx.x;d<2*dim;d+=256){float k=0,s=0;
 if(complete){if(d<dim&&row==first){int line=lines[seq*line_width];if(line>0){int64_t base=(int64_t)line*2*dim;k=state[base+d];s=state[base+dim+d];}}
 else{int source=row-1+d/dim;k=kv[(int64_t)source*dim+d%dim];s=score[(int64_t)source*dim+d%dim];}}
 kv_out[(int64_t)row*2*dim+d]=k;score_out[(int64_t)row*2*dim+d]=s;}
}
extern "C" __global__ void dsv41_compressor_commit(float* state,const float* kv,const float* score,const int32_t* lines,const int32_t* starts,const int32_t* accepted,int seqs,int dim,int line_width){
 int seq=blockIdx.x,n=accepted[seq];if(n<=0)return;
 // A prefill has one committed line but can commit many input rows.
 int col=line_width==1?0:n-1;int line=lines[seq*line_width+col];if(line<=0)return;int64_t base=(int64_t)line*2*dim;
 int row=starts[seq]+n-1;
 for(int d=threadIdx.x;d<dim;d+=256){state[base+d]=kv[(int64_t)row*dim+d];state[base+dim+d]=score[(int64_t)row*dim+d];}
}
// Generic serving metadata. phase0 preserves staged rows; phase1 selects the
// five draft rows starting at each six-row verification anchor. `valid` is a caller
// fill; token IDs and leased padding addresses cannot reveal padded rows.
extern "C" __global__ void dsv41_metadata(int32_t* req,int32_t* pos,int64_t* slots,int64_t* window_slots,uint8_t* mask,int32_t* starts,int32_t* seq_valid,int32_t* ends,int32_t* ids32,int32_t* counts,int32_t* window_length,int32_t* compressed_length,const int32_t* input_pos,const int64_t* input_slots,const int32_t* valid,const int64_t* ids,const int32_t* cu,const int32_t* seq_lens,const int32_t* window_lines,int rows,int seqs,int phase,int ring){
 int row=blockIdx.x*256+threadIdx.x;
 if(row<rows){int seq,source;
 if(phase){seq=row/5;source=cu[seq]+row%5;}
 else {int lo=0,hi=seqs;while(lo+1<hi){int mid=(lo+hi)/2;if(cu[mid]<=row)lo=mid;else hi=mid;}seq=lo;source=row;}
 bool live=valid[source]!=0;req[row]=seq;pos[row]=input_pos[source];slots[row]=live?input_slots[source]:-1;mask[row]=live;
 window_slots[row]=live?ring_slot(window_lines,seq,input_pos[source],ring):-1;
 ids32[row]=(int32_t)ids[phase?row:source];ends[row]=seq_lens[seq]-(phase?1:0);window_length[row]=live?(phase?min(seq_lens[seq]-1,133):min(input_pos[source]+1,128)):0;compressed_length[row]=live?512:0;}
 if(row<seqs){bool live=valid[cu[row]]!=0;starts[row]=phase?row*5:cu[row];seq_valid[row]=live;counts[row]=live?(phase?5:cu[row+1]-cu[row]):0;}
 if(row==0)starts[seqs]=rows;
}
extern "C" __global__ void dsv41_compressed_metadata(int64_t* out_slots,int32_t* out_pos,const int64_t* slots,const int32_t* pos,const uint8_t* valid,int rows,int ratio){
 int row=blockIdx.x*256+threadIdx.x;if(row>=rows)return;
 bool complete=valid[row]&&slots[row]>=0&&(pos[row]+1)%ratio==0;
 out_slots[row]=complete?slots[row]/ratio:-1;out_pos[row]=complete?pos[row]+1-ratio:0;
}
extern "C" __global__ void dsv41_indices_mask(int32_t* indices,const uint8_t* valid,int rows,int width){
 int row=blockIdx.x;if(!valid[row])for(int col=threadIdx.x;col<width;col+=256)indices[(int64_t)row*width+col]=-1;
}
extern "C" __global__ void dsv41_accepted_mask(int32_t* out,const int32_t* counts,const int32_t* valid,const int32_t* starts,int seqs){
 int seq=blockIdx.x*256+threadIdx.x;if(seq<seqs)out[seq]=valid[seq]?min(max(counts[seq],0),starts[seq+1]-starts[seq]):0;
}
extern "C" __global__ void dsv41_cache_bf16(bf16* cache,const bf16* x,const int64_t* slots,int rows,int dim){
 int row=blockIdx.x;int64_t slot=slots[row];if(slot<0)return;
 for(int d=threadIdx.x;d<dim;d+=256)cache[slot*dim+d]=x[(int64_t)row*dim+d];
}
extern "C" __global__ void dsv41_gather_bf16(bf16* out,const bf16* cache,const int64_t* slots,int rows,int dim){
 int row=blockIdx.x;int64_t slot=slots[row];
 for(int d=threadIdx.x;d<dim;d+=256)out[(int64_t)row*dim+d]=slot<0?__float2bfloat16(0):cache[slot*dim+d];
}

// Indexer cache page: packed E2M1 keys, E8M0 scales, then padding to 512 bytes.
extern "C" __global__ void dsv41_cache_index(uint8_t* cache,const uint8_t* packed,const uint8_t* scales,const int64_t* slots,int rows,int page_size){
 int row=blockIdx.x;int64_t slot=slots[row];if(slot<0)return;
 int64_t stride=((int64_t)page_size*68+511)/512*512;
 int64_t base=(slot/page_size)*stride;int token=slot%page_size;
 for(int d=threadIdx.x;d<64;d+=256)cache[base+token*64+d]=packed[(int64_t)row*64+d];
 if(threadIdx.x<4)cache[base+(int64_t)page_size*64+token*4+threadIdx.x]=scales[(int64_t)row*4+threadIdx.x];
}

// Query-local causal bounds and physical page IDs for direct paged index scoring.
extern "C" __global__ void dsv41_index_metadata(int* query_pages,int* end,int* request_ids,int* sparse_ends,const int* request,const int* position,const uint8_t* valid,const int* pages,int rows,int page_cols,int compressed_page_size,int ratio){
 int row=blockIdx.x;
 int length=valid[row]?(position[row]+1)/ratio:0;
 if(threadIdx.x==0){end[row]=length;request_ids[row]=row;sparse_ends[row]=valid[row]?16384:0;}
 int used=(length+compressed_page_size-1)/compressed_page_size;
 for(int col=threadIdx.x;col<page_cols;col+=256)
  query_pages[(int64_t)row*page_cols+col]=col<used?pages[(int64_t)request[row]*page_cols+col]:0;
}

extern "C" __global__ void dsv41_context_tap(bf16* out,const bf16* hc,int rows,int tap){
 int64_t i=(int64_t)blockIdx.x*256+threadIdx.x;
 if(i>=(int64_t)rows*5120)return;
 int row=i/5120,d=i%5120;float sum=0;
 for(int h=0;h<4;++h)sum+=__bfloat162float(hc[((int64_t)row*4+h)*5120+d]);
 out[(int64_t)row*15360+tap*5120+d]=__float2bfloat16(sum*0.25f);
}
extern "C" __global__ void dsv41_context_slots(int64_t* out,const int64_t* slots,const int* request,const int* starts,const int* accepted,int rows){
 int row=blockIdx.x*256+threadIdx.x;if(row>=rows)return;
 int seq=request[row],count=accepted[seq];
 out[row]=row-starts[seq]<count?slots[row]:-1;
}
