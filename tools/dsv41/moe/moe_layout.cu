#include "mega_config.cuh"
#include <cstdio>
#include <cstdlib>
template<int E,int K,int R,int S>void dump(){using C=dsv41::Config<E,K,R,S>;using namespace deep_gemm;auto b=layout::MegaMoEBuffer(nullptr,5120,2304,R,E,8192,K,C::Ring,C::SfRing,true,1);
 printf("{\"experts\":%d,\"topk\":%d,\"ranks\":%d,\"ring\":%d,\"sf_ring\":%d,\"stages\":%d,\"smem\":%d,\"slab_bytes\":%llu,\"workspace_bytes\":%llu,\"shared_sf_rows\":%d,\"offsets\":{",E,K,R,C::Ring,C::SfRing,C::Stages,C::Smem,(unsigned long long)b.get_num_bytes(),(unsigned long long)b.workspace.get_end_ptr(),layout::get_num_max_shared_sf_tokens(8192));
#define FIELD(NAME,MEMBER) printf("\"" NAME "\":%llu,",(unsigned long long)b.MEMBER.base)
 FIELD("x",input_token_buffer);FIELD("x_sf",input_sf_buffer);FIELD("idx",input_topk_idx_buffer);FIELD("weights",input_topk_weights_buffer);FIELD("shared_x_sf",shared_l1_sf_buffer);FIELD("shared_l2",shared_l2_token_buffer);FIELD("shared_l2_sf",shared_l2_sf_buffer);FIELD("l1",l1_token_buffer);FIELD("l1_sf",l1_sf_buffer);FIELD("l2",l2_token_buffer);
 printf("\"l2_sf\":%llu}}\n",(unsigned long long)b.l2_sf_buffer.base);
}
int main(int argc,char**argv){if(argc!=3)return 2;int e=atoi(argv[1]),s=atoi(argv[2]);
 if(e==384&&s==152)dump<384,6,4,152>();else if(e==128&&s==152)dump<128,3,4,152>();
 else if(e==384&&s==148)dump<384,6,4,148>();else if(e==128&&s==148)dump<128,3,4,148>();else return 2;}
