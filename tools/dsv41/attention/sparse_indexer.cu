#include <deep_gemm/impls/sm100_sparse_mqa_logits.cuh>
#include <deep_gemm/scheduler/sm100_sparse_mqa_logits_metadata.cuh>
using namespace deep_gemm;
static void __instantiate_kernel(){
    auto paged64=reinterpret_cast<void*>(&sm100_paged_sparse_mqa_logits<64,8,2,5,5,5,148,2,true>);
    auto paged128=reinterpret_cast<void*>(&sm100_paged_sparse_mqa_logits<128,8,2,5,5,5,148,2,true>);
    auto meta64=reinterpret_cast<void*>(&sched::sparse_mqa_logits::sm100_sparse_mqa_logits_metadata<true,false,2,640,8,2048,64,8,148,256>);
    auto meta128=reinterpret_cast<void*>(&sched::sparse_mqa_logits::sm100_sparse_mqa_logits_metadata<true,false,2,640,8,2048,128,8,148,256>);
    auto score=reinterpret_cast<void*>(&sm100_sparse_mqa_logits<8,2,5,5,5,148,2,false,true>);
    auto meta=reinterpret_cast<void*>(&sched::sparse_mqa_logits::sm100_sparse_mqa_logits_metadata<false,false,2,640,8,2048,0,8,148,256>);
}
extern "C" __global__ void dsv41_sparse_initialize(unsigned short* logits,unsigned char* workspace,int rows){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<384)workspace[i]=0;
    if(i<(unsigned)rows*16384)logits[i]=0xff80; // BF16 -infinity for unwritten padded candidates
}
extern "C" __global__ void dsv41_sparse_mask(unsigned short* logits,const int* candidates,const int* ends,int rows){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=(unsigned)rows*16384)return;
    unsigned row=i/16384,col=i%16384;
    int block=candidates[row*2048+col/8];
    if(block<0||block*8+(int)(col%8)>=ends[row])logits[i]=0xff80;
}
extern "C" __global__ void dsv41_sparse_positions(const unsigned short* logits,const int* selected,
    const int* candidates,const int* ends,int* output,int rows){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=(unsigned)rows*512)return;
    unsigned row=i/512;int slot=selected[i],position=-1;
    if(slot>=0&&slot<16384&&logits[row*16384+slot]!=0xff80){
        int block=candidates[row*2048+slot/8];
        int absolute=block*8+slot%8;
        if(block>=0&&absolute<ends[row])position=absolute;
    }
    output[i]=position;
}
