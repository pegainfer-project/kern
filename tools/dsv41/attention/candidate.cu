// Adapt the model's candidate block score and invalid-selection conventions.
#include <cuda_runtime.h>
#include <math_constants.h>
extern "C" __global__ void dsv41_candidate_scores(const float* scores,const int* end,
    float* blocks,int* block_ends,int rows,int width,int block_stride){
    int block=blockIdx.x*blockDim.x+threadIdx.x;
    int row=blockIdx.y;
    if(row>=rows||block>=block_stride)return;
    int limit=end[row];if(block==0)block_ends[row]=(limit+7)/8;float score=-CUDART_INF_F;
    #pragma unroll
    for(int i=0;i<8;i++){
        int pos=block*8+i;
        if(pos<width&&pos<limit){
            float value=scores[(long long)row*width+pos];
            score=(isnan(score)||isnan(value))?CUDART_NAN_F:fmaxf(score,value);
        }
    }
    if(limit>0&&block==(limit-1)/8)score=CUDART_INF_F;
    blocks[(long long)row*block_stride+block]=score;
}
extern "C" __global__ void dsv41_selection_filter(const float* scores,const int* selected,
    int* filtered,int rows,int width,int k){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int row=blockIdx.y;
    if(row>=rows||i>=k)return;
    int ix=selected[(long long)row*k+i];
    bool valid=ix>=0&&ix<width;
    if(valid)valid=scores[(long long)row*width+ix]>-CUDART_INF_F;
    filtered[(long long)row*k+i]=valid?ix:-1;
}
